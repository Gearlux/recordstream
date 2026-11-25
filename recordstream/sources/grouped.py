"""``GroupedFilesSource`` — a folder of files in which a group folder is ONE record."""

import multiprocessing
import os
import random
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, get_args

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record
from recordstream.sources.files import glob_files

logger = get_logger(__name__)

#: Which file of a group a read takes.
Pick = Literal["next", "first", "seeded"]
PICKS: Tuple[str, ...] = get_args(Pick)


@configurable(category="source")
class GroupedFilesSource:
    """The files under ``root`` as ``{"file", "group_size"}`` records — a GROUP FOLDER is one record, not one per file.

    A folder whose name starts with :attr:`prefix` holds files that stand for each other (near-copies of one
    picture, takes of one scene). Its files are one record; a file outside any such folder is a record of its own.
    So a group of 70 near-copies weighs as much as one lone file, and which of the 70 a read gets is :attr:`pick`:

    - ``next`` (default, for training): another file at every read of the group — its first, then its second, …
      and round again. An epoch reads every record once, so every epoch sees another file of each group.
    - ``first``: always the group's first file (by name).
    - ``seeded``: one file per group, drawn from :attr:`seed` and the group's folder — the same at every read, in
      every worker and in every run. For a validation run that must score the same files throughout.

    The count behind ``next`` lives in memory every worker process SHARES, because a count kept in the source alone
    starts again at zero in each worker a loader starts: with workers restarted per epoch every epoch would read
    the first file. A source handed to a worker when the worker is STARTED (a torch ``DataLoader``'s workers, a
    ``multiprocessing.Process`` argument) keeps counting with the others; a copy made any other way (``pickle``,
    ``copy.deepcopy``) counts on its own from zero.

    The folder is listed ONCE, on first use: files added later are not seen by this object.

    Args:
        root: Folder to list. ``~`` and ``$VAR`` are expanded.
        pattern: Glob applied under ``root`` (default ``"**/*"``, every file below it). Directories and dotfiles
            are never listed.
        prefix: What a group folder's name starts with.
        pick: Which file of a group a read takes: ``next``, ``first`` or ``seeded``.
        seed: What ``seeded`` draws from.
        name: Id prefix a consuming viewer derives record ids from.
    """

    produces = ("file", "group_size")

    def __init__(
        self,
        root: str = "",
        pattern: str = "**/*",
        prefix: str = "group_",
        pick: Pick = "next",
        seed: int = 0,
        name: str = "files",
    ) -> None:
        # Lazy / zero-arg: store config only; the folder is listed on first use.
        self.root = str(root)
        self.pattern = str(pattern)
        self.prefix = prefix
        self.pick = pick
        self.seed = int(seed)
        self.name = name
        # Runtime state, never configuration: the listing and the reads counted per record.
        self._listing: Optional[List[Tuple[str, List[str]]]] = None
        self._reads: Any = None

    def _records(self) -> List[Tuple[str, List[str]]]:
        """Every record as ``(its place under root, its files)``, sorted by place — listed once."""
        if self._listing is not None:
            return self._listing
        root = Path(os.path.expanduser(os.path.expandvars(self.root)))
        if not self.root or not root.is_dir():
            raise ValueError(f"GroupedFilesSource: root {str(root)!r} is not a directory")
        found = glob_files(root, self.pattern)
        if not found:
            raise ValueError(f"GroupedFilesSource: pattern {self.pattern!r} matched no files under {str(root)!r}")
        records: Dict[str, List[str]] = {}
        for path in found:
            parent = Path(path).parent
            grouped = parent != root and parent.name.startswith(self.prefix)
            place = (parent if grouped else Path(path)).relative_to(root).as_posix()
            records.setdefault(place, []).append(path)
        self._listing = sorted(records.items())
        # One count per record, in memory a worker started with this source shares. The spawn context's lock is
        # the one both a spawned and a forked worker can take.
        self._reads = multiprocessing.get_context("spawn").Array("q", len(self._listing))
        logger.debug(f"GroupedFilesSource: {len(found)} file(s) under {str(root)!r} are {len(self._listing)} record(s)")
        return self._listing

    def _picked(self, index: int, place: str, size: int) -> int:
        if size == 1 or self.pick == "first":
            return 0
        if self.pick == "seeded":
            return random.Random(f"{self.seed}:{place}").randrange(size)
        with self._reads.get_lock():
            read = self._reads[index]
            self._reads[index] = read + 1
        return int(read % size)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        if get_spawning_popen() is None:
            # Not a worker being started: shared memory cannot travel, so the copy lists and counts anew.
            state["_listing"] = None
            state["_reads"] = None
        else:
            self._records()  # the worker must get the count the others have, so it exists before it leaves
            state["_listing"] = self._listing
            state["_reads"] = self._reads
        return state

    def __len__(self) -> int:
        return len(self._records())

    def __getitem__(self, index: int) -> Record:
        place, files = self._records()[index]
        return {"file": str(Path(files[self._picked(index, place, len(files))])), "group_size": len(files)}

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"GroupedFilesSource(root={self.root!r}, pick={self.pick!r})"
