"""``FileSelection`` — the files of several folders ("sets") as records, limited the same way wherever they are read.

A run over a dataset's files says WHICH of them it reads in one place: the globs it lists, the files it leaves out,
how deep below each set it looks, how many folders of each set and how many files of each folder it keeps, and how
many in all. A quick run is the same selection with smaller limits (``max_folders_per_set: 5``,
``max_files_per_folder: 20``), so every tool that reads the dataset shares one spelling of "a few of them".
"""

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from annotated_types import Interval
from confluid import configurable
from loggair import get_logger
from typing_extensions import Annotated

from recordstream.formats import FileFormat, file_formats
from recordstream.items import Record
from recordstream.sources.files import excluded_by, glob_files

logger = get_logger(__name__)

#: A count a limit takes: 0 keeps every one.
Limit = Annotated[int, Interval(ge=0)]
#: A set: a folder's path, or any object naming one (``os.PathLike`` — asked when the selection is first listed, so a
#: set a download fills is fetched then, never when the config is built).
SetFolder = Union[str, "os.PathLike[str]"]

_LIMITS = ("max_depth", "max_folders_per_set", "max_files_per_folder", "max_files")


@configurable(category="source")
class FileSelection:
    """The files of each set, in turn, as records — ``{"file", "set", "relative"}`` each.

    Each set is listed by ``pattern`` (one glob or several, joined without a file twice), files in name order; a
    dotfile and everything inside a hidden folder (``.sync/``) are never listed, a file ``exclude`` matches is left
    out (a glob without a slash by the file's name, one with by its place under the set, ``*`` crossing folders),
    and so is a companion half a registered format consumes (the data file of a pair: a pair counts once). Then, in
    this order: a file more than ``max_depth`` folders below its set is left out; each set keeps the files of its
    first ``max_folders_per_set`` folders (the folders holding a listed file, in name order); each folder keeps its
    first ``max_files_per_folder`` files; and the whole selection its first ``max_files``. A limit of 0 keeps every
    one. ``relative`` is the file's place under ``relative_to`` (the set's parent folder when empty, or when the file
    lies outside it) less its extension — what a run names what it writes per record by.

    The listing is made on first use and kept while the settings and the sets are the same: a dataset of thousands
    of files is never listed once per record. An empty selection is refused naming the sets and the globs.

    Args:
        sets: The folders read, each a set; a path (``~`` and ``$VAR`` expanded) or an object naming one.
        pattern: The glob listing each set's files, or a list of globs (``["**/*.json", "**/*.mat"]``).
        exclude: A glob of the files left out: by name without a slash (``*.tmp``), by place under the set with one.
        max_depth: The most folders between a set and a file read (0: the set's own files); None reads every depth.
        max_folders_per_set: The folders of each set read, the first in name order (0: every one).
        max_files_per_folder: The files of each folder read, the first in name order (0: every one).
        max_files: The files read in all, the first of the selection (0: every one).
        relative_to: The folder a record's ``relative`` name starts at; empty: each set's parent folder.
        formats: Formats whose ``consumes()`` leaves a companion half out; None = every installed format, an empty
            list the plain listing.
    """

    produces = ("file", "set", "relative")

    def __init__(
        self,
        sets: Optional[Sequence[SetFolder]] = None,
        pattern: Union[str, List[str]] = "**/*",
        exclude: str = "",
        max_depth: Optional[Limit] = None,
        max_folders_per_set: Limit = 0,
        max_files_per_folder: Limit = 0,
        max_files: Limit = 0,
        relative_to: str = "",
        formats: Optional[Sequence[FileFormat]] = None,
    ) -> None:
        # Lazy / zero-arg: store config only — no set is asked for its folder and nothing is listed here.
        self.sets: List[SetFolder] = list(sets or [])
        self.pattern = pattern
        self.exclude = exclude
        self.max_depth = max_depth
        self.max_folders_per_set = max_folders_per_set
        self.max_files_per_folder = max_files_per_folder
        self.max_files = max_files
        self.relative_to = relative_to
        self.formats = formats
        # The listing, keyed on what decides it (the settings and the sets' folders). Runtime state, never dumped.
        self._listing: Optional[Tuple[Tuple[Any, ...], List[Record]]] = None

    # -- the listing ------------------------------------------------------------------------------------------------

    @property
    def patterns(self) -> List[str]:
        """The globs each set is listed by."""
        return [self.pattern] if isinstance(self.pattern, str) else [str(glob) for glob in self.pattern]

    @staticmethod
    def _folder(entry: SetFolder) -> Path:
        return Path(os.path.expanduser(os.path.expandvars(os.fspath(entry))))

    def _listed(self, root: Path, formats: Sequence[FileFormat]) -> List[Path]:
        """One set's files after the globs, the hidden ones, ``exclude`` and the companions, in name order."""
        found = sorted({path for glob in self.patterns for path in glob_files(root, glob)})
        files: List[Path] = []
        for name in found:
            path = Path(name)
            relative = path.relative_to(root)
            if any(part.startswith(".") for part in relative.parts):
                continue  # a hidden folder's files (a download's .sync/) are not data
            if self.exclude and excluded_by(name, self.exclude, root):
                continue
            if formats and any(fmt.consumes(path) for fmt in formats):
                continue
            files.append(path)
        return files

    def _limited(self, root: Path, files: List[Path]) -> List[Path]:
        """One set's files after its depth, folder and per-folder limits."""
        if self.max_depth is not None:
            files = [path for path in files if len(path.relative_to(root).parts) - 1 <= int(self.max_depth)]
        if self.max_folders_per_set:
            folders = sorted({path.parent for path in files})[: int(self.max_folders_per_set)]
            kept = set(folders)
            files = [path for path in files if path.parent in kept]
        if self.max_files_per_folder:
            counts: Dict[Path, int] = {}
            limited: List[Path] = []
            for path in files:
                counts[path.parent] = counts.get(path.parent, 0) + 1
                if counts[path.parent] <= int(self.max_files_per_folder):
                    limited.append(path)
            files = limited
        return files

    def _relative(self, path: Path, root: Path) -> str:
        start = self._folder(self.relative_to) if self.relative_to else root.parent
        try:
            place = path.relative_to(start)
        except ValueError:
            place = path.relative_to(root.parent)
        return place.with_suffix("").as_posix()

    def select(self, sets: Optional[Sequence[SetFolder]] = None) -> List[Tuple[Path, Path]]:
        """Every file selected with its set, ``(file, set)`` in record order — of ``sets`` instead of the selection's
        own when given (a run that reads another copy of its sets, the same limits applied)."""
        entries = list(self.sets if sets is None else sets)
        if any(not os.fspath(entry).strip() for entry in entries):
            raise ValueError("FileSelection: a set is empty — name its folder")
        roots = [self._folder(entry) for entry in entries]
        if not roots:
            raise ValueError("FileSelection: no sets — name the folders to read (sets: [<folder>])")
        for root in roots:
            if not root.is_dir():
                raise ValueError(f"FileSelection: set {str(root)!r} is not a folder")
        formats = file_formats() if self.formats is None else self.formats
        pairs = [(path, root) for root in roots for path in self._limited(root, self._listed(root, formats))]
        if self.max_files:
            pairs = pairs[: int(self.max_files)]
        if not pairs:
            raise ValueError(
                f"FileSelection: {self.patterns} matched no files under {[str(root) for root in roots]} "
                f"(exclude {self.exclude!r}, max_depth {self.max_depth})"
            )
        logger.debug(f"FileSelection: {len(pairs)} file(s) of {len(roots)} set(s)")
        return pairs

    @property
    def _records(self) -> List[Record]:
        roots = tuple(str(self._folder(entry)) for entry in self.sets)
        key = (roots, tuple(self.patterns), self.exclude, *(getattr(self, name) for name in _LIMITS), self.relative_to)
        if self._listing is not None and self._listing[0] == key:
            return self._listing[1]
        # the folders already asked of the sets above: an object naming its folder (a download) is asked once
        named = [root if os.fspath(entry).strip() else "" for entry, root in zip(self.sets, roots)]
        records: List[Record] = [
            {"file": str(path), "set": str(root), "relative": self._relative(path, root)}
            for path, root in self.select(named)
        ]
        self._listing = (key, records)
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Record:
        return dict(self._records[index])

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"FileSelection(sets={len(self.sets)}, pattern={self.pattern!r})"


__all__ = ["FileSelection", "Limit", "SetFolder"]
