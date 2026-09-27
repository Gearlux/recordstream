"""``FilesSource`` — file paths as records: a list you pass, or a folder it lists."""

import os
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from confluid import configurable
from loggair import get_logger

from recordstream.formats import FileFormat, file_formats
from recordstream.items import Record

logger = get_logger(__name__)


@configurable(category="source")
class FilesSource:
    """A list of FILES as records — ``{"file": "<path>"}`` each, nothing more.

    The dataset counterpart of "someone handed me files": no split, no metadata layout, no
    labels — and deliberately NO decoding. The source knows nothing about what a file MEANS;
    turning the path into content is an OP's job (``ReadImage`` for pictures, the
    format-registry ``ReadFile`` for anything a registered format decodes), so the same
    source serves every file kind and the graph shows how a file becomes a record. A
    consuming workspace passes the list (e.g. files a user dropped) and the chain takes it
    from there.

    One listing rule IS format-aware: a PAIRED format's companion half (the file whose
    content rides in with a sibling — decided by the registry's ``consumes()``, a cheap
    name-level test that never decodes the data) is kept out of the listing, so
    ``len()``/ids count one record per pair. With no format packages installed the
    listing is the plain file list.

    Args:
        files: The file paths to serve, in the order to serve them.
        name: Id prefix a consuming viewer derives record ids from.
        exclude: Optional filename glob whose matches are NOT served as records — a
            manual filter on top of the registry rule (name-level, no file reads).
        root: Folder to list instead of passing ``files``. ``~`` and ``$VAR`` are expanded,
            so a config can say ``$DATA_ROOT/captures``. Empty (default) = list ``files``.
            Passing both is refused — two answers to "which files" is a config bug, not a
            merge — and the refusal is repeated at READ time, because a consuming workspace
            may assign ``files`` on an already-built source (filling it from a drag-and-drop)
            where a constructor check cannot see it. The scan is lazy: it happens on first
            use, never in the constructor.
        pattern: Glob applied under ``root`` (default ``"*"``, that folder's own files).
            Ignored when ``root`` is empty. Directories the glob matches are skipped —
            only files become records — and so are dotfiles, because a folder glob picks
            up ``.DS_Store`` and friends, which are not data and fail to decode.
        formats: Formats whose ``consumes()`` decides the companion exclusion. ``None``
            (default) = every installed format from the registry, resolved lazily; an
            empty list restores the plain listing.
    """

    def __init__(
        self,
        files: Optional[List[str]] = None,
        name: str = "files",
        exclude: str = "",
        root: str = "",
        pattern: str = "*",
        formats: Optional[Sequence[FileFormat]] = None,
    ) -> None:
        # Lazy / zero-arg: store config only. A root is not scanned here — the constructor
        # does no I/O, so a folder that does not exist yet is reported on first use.
        if files and root:
            raise ValueError(
                "FilesSource: pass either `files` (an explicit list) or `root` (a folder to "
                f"list), not both — got {len(files)} file(s) and root={root!r}."
            )
        self.files = [str(f) for f in (files or [])]
        # The resolved listing, keyed on what decides it (see `_served`). Runtime state,
        # not configuration — it is never dumped and never a constructor parameter.
        self._served_cache: Optional[Tuple[Any, List[str]]] = None
        self.name = name
        self.exclude = exclude
        self.root = str(root)
        self.pattern = str(pattern)
        self.formats = formats

    @property
    def _listed(self) -> List[str]:
        """The paths before any exclusion — ``files`` as given, or ``root`` scanned now."""
        if not self.root:
            return self.files
        if self.files:
            # Checked HERE as well as in the constructor: a consuming workspace may fill
            # `files` on a live source (annotaide sets it from a drag-and-drop), which the
            # constructor never sees — and silently preferring one over the other would make
            # the drop look like it did nothing.
            raise ValueError(
                f"FilesSource: root={self.root!r} is set AND {len(self.files)} file(s) were "
                "assigned — this source lists a folder, so it cannot also be given a file "
                "list. Clear `root` to accept an explicit list, or clear `files` to list the "
                "folder."
            )
        root = Path(os.path.expanduser(os.path.expandvars(self.root)))
        if not root.is_dir():
            raise ValueError(f"FilesSource: root {str(root)!r} is not a directory")
        found = sorted(str(p) for p in root.glob(self.pattern) if p.is_file() and not p.name.startswith("."))
        if not found:
            # A silent empty listing is the failure that looks like a broken pipeline three
            # ops later, so say which folder and which pattern produced nothing.
            raise ValueError(
                f"FilesSource: pattern {self.pattern!r} matched no files under {str(root)!r} "
                "(a pattern is relative to root: '*' is the folder's own files, '*/*' its "
                "subdirectories, '**/*' everything below it)"
            )
        logger.debug(f"FilesSource: {self.pattern!r} under {str(root)!r} listed {len(found)} file(s)")
        return found

    @property
    def _format_list(self) -> Sequence[FileFormat]:
        return file_formats() if self.formats is None else self.formats

    @property
    def _served(self) -> List[str]:
        """The files this source actually serves — the listing less the excludes and less
        every companion half a format consumes.

        MEMOISED against the inputs that decide it, because deciding is not free: each
        candidate is put to every format's ``consumes``, and a paired format answers by
        STATTING for its sibling. Re-deciding per access made ``source[i]`` O(n) with
        filesystem I/O and ``__iter__`` O(n²) — measured on a real 1430-file library,
        **13.4 ms per index**, so a walk touching each file once spent 19 s inside a listing
        it had already computed.

        The key is the LISTING IT FILTERED — ``_listed``'s own result, with the exclude and
        the formats — never a flag and never an explicit invalidate. That choice is what makes
        the memo safe for both kinds of source, and neither is hypothetical: a consumer FILLS
        ``files`` in place (a drop appends to the very list this source is serving), and a
        ``root`` source is documented to scan the folder NOW, so a file appearing in it must
        still appear here. Keying on the file list alone would have frozen the second kind —
        a root source's ``files`` is empty and its key would never change, so the folder would
        have been scanned once and never again.

        The glob for a root source therefore still runs per access, exactly as before; what
        the memo removes is the part that dominated — a ``consumes`` call per file per format,
        each of them a stat.
        """
        listed = self._listed
        formats = self._format_list
        key = (tuple(listed), self.exclude, id(formats) if formats else 0, len(formats))
        cached = self._served_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        served = listed
        if self.exclude:
            from fnmatch import fnmatch

            served = [f for f in served if not fnmatch(Path(f).name, self.exclude)]
        if formats:
            served = [f for f in served if not any(fmt.consumes(Path(f)) for fmt in formats)]
        self._served_cache = (key, served)
        return served

    def __len__(self) -> int:
        return len(self._served)

    def __getitem__(self, index: int) -> Record:
        return {"file": str(Path(self._served[index]))}

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"FilesSource(files={len(self.files)})"
