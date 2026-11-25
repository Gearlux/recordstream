"""Groups as ONE record: ``GroupedFilesSource`` (a folder per group) and ``GroupedSource`` (a groups file)."""

import json
import multiprocessing
import os
import random
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Literal, Optional, Tuple, get_args

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record
from recordstream.sources.base import _guard_live_source, _pass_through
from recordstream.sources.files import glob_files

logger = get_logger(__name__)

#: Which member of a group a read takes.
Pick = Literal["next", "first", "seeded"]
PICKS: Tuple[str, ...] = get_args(Pick)


class _PickOfGroup:
    """Which member of a group a read takes — the count behind ``next`` in memory every worker shares.

    A count kept in the object alone starts again at zero in each worker a loader starts: with workers restarted
    per epoch every epoch would read the first member. So the count is a ``multiprocessing.Array``, handed to a
    worker when the worker is STARTED (``__getstate__``); a copy made any other way lists and counts anew.
    """

    pick: Pick
    seed: int
    #: The attributes built on first use, dropped from a copy that is not a worker being started.
    _RUNTIME: Tuple[str, ...] = ()
    _reads: Any = None

    def _ready(self) -> None:
        """Build the listing and the count (the subclass's lazy first use)."""
        raise NotImplementedError

    def _count(self, records: int) -> None:
        # The spawn context's lock is the one both a spawned and a forked worker can take.
        self._reads = multiprocessing.get_context("spawn").Array("q", records)

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
        if get_spawning_popen() is not None:
            self._ready()  # the worker must get the count the others have, so it exists before it leaves
            return dict(self.__dict__)
        # Not a worker being started: shared memory cannot travel, so the copy lists and counts anew.
        return {**self.__dict__, **{name: None for name in self._RUNTIME}}


@configurable(category="source")
class GroupedFilesSource(_PickOfGroup):
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
    _RUNTIME = ("_listing", "_reads")

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
        self._count(len(self._listing))
        logger.debug(f"GroupedFilesSource: {len(found)} file(s) under {str(root)!r} are {len(self._listing)} record(s)")
        return self._listing

    def _ready(self) -> None:
        self._records()

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


@configurable(category="source")
class GroupedSource(_PickOfGroup):
    """An indexable source in which a GROUP of records is one record — the groups named by a groups file.

    The view of :class:`GroupedFilesSource` for a dataset that must stay as it is (its labels name its files by
    their place): nothing moves, a small JSON file says which records stand for each other
    (:class:`~recordstream.storage.grouped_files.GroupsFileSink` writes it)::

        {"records": 854, "groups": [[28, 30, 54], [59, 78]]}

    ``records`` is how many records the source had when the file was written; a source of another length is
    refused, because the numbers would name other records. Each group lists its records' indices; a record no
    group names is a record of its own. The view serves one record per group and one per lone record, in the order
    of each one's first index, each exactly as the source gives it. Which record of a group a read gets is
    :attr:`pick` — ``next`` (another one at every read, so every epoch trains on another), ``first`` or ``seeded``
    (one fixed record per group, for a validation run) — counted as in :class:`GroupedFilesSource`.

    A key-restricted walk (``project``) answers with each group's FIRST record and counts nothing.

    Args:
        source: The underlying indexable source (validated lazily on first use).
        groups: Path of the groups file. ``~`` and ``$VAR`` are expanded.
        pick: Which record of a group a read takes: ``next``, ``first`` or ``seeded``.
        seed: What ``seeded`` draws from.
    """

    _RUNTIME = ("_members", "_reads")

    def __init__(self, source: Any = None, groups: str = "", pick: Pick = "next", seed: int = 0) -> None:
        # Lazy / zero-arg: store config only; the groups file is read on first use.
        self.source = source
        self.groups = str(groups)
        self.pick = pick
        self.seed = int(seed)
        self._members: Optional[List[List[int]]] = None
        self._reads: Any = None

    def _records(self) -> List[List[int]]:
        """Every record of the view as the source indices that stand for it — read once."""
        if self._members is not None:
            return self._members
        source = self.source
        _guard_live_source(source, "GroupedSource.source")
        if source is None or not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
            raise TypeError(
                f"GroupedSource requires a source supporting __len__ and __getitem__; got {type(source).__name__}"
            )
        path = Path(os.path.expanduser(os.path.expandvars(self.groups)))
        if not self.groups or not path.is_file():
            raise ValueError(f"GroupedSource: groups file {str(path)!r} does not exist")
        stored = json.loads(path.read_text())
        count = len(source)
        if stored.get("records") != count:
            raise ValueError(
                f"GroupedSource: {str(path)!r} was written for {stored.get('records')} records and the source has "
                f"{count} — its numbers would name other records; write the groups file again for this source"
            )
        group_of: Dict[int, List[int]] = {}
        for group in stored.get("groups", []):
            members = [int(index) for index in group]
            for index in members:
                if not 0 <= index < count or index in group_of:
                    raise ValueError(
                        f"GroupedSource: {str(path)!r} names record {index} twice or outside 0..{count - 1}"
                    )
                group_of[index] = members
        self._members = [
            group_of.get(index, [index]) for index in range(count) if group_of.get(index, [index])[0] == index
        ]
        self._count(len(self._members))
        logger.debug(f"GroupedSource: {count} record(s) are {len(self._members)} record(s) by {str(path)!r}")
        return self._members

    def _ready(self) -> None:
        self._records()

    def project(self, keys: Collection[str]) -> Iterator[Record]:
        """Yield every record of the view carrying only ``keys`` — each group's FIRST record, through the source's
        own walk (:class:`recordstream.projection.SupportsProjection`); nothing is counted. Partial: a generator."""
        from recordstream.projection import project_indices

        yield from project_indices(self.source, keys, [members[0] for members in self._records()])

    def __len__(self) -> int:
        return len(self._records())

    def __getitem__(self, index: int) -> Any:
        members = self._records()[index]
        return _pass_through(self.source[members[self._picked(index, str(members[0]), len(members))]])

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"GroupedSource(groups={self.groups!r}, pick={self.pick!r})"
