"""``DifferenceSource`` — the records of one source whose key no record of another source carries."""

from os import PathLike
from pathlib import PurePath
from typing import Any, Collection, Hashable, Iterator, List, Literal, Optional, Set

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record, item_value
from recordstream.sources.base import _guard_live_source, _pass_through

logger = get_logger(__name__)

#: How two records' keys are compared: the values as they are, the file name a path ends in (any
#: folder), or that file name without its last extension (``x.pickle`` and ``x.json`` agree).
CompareBy = Literal["value", "name", "stem"]


@configurable(category="source")
class DifferenceSource:
    """The records of ``source`` whose ``key`` matches no record of ``remove`` — a set difference.

    The case it exists for: a set of recordings that must never reach training (a test set) is
    a dataset of its own, and the training pool is another; ``DifferenceSource(pool, test)``
    is the pool without them. Records are compared by ONE key, nothing else — two records are
    "the same" when their keys agree, whatever their other entries — and ``compare`` says how
    the key's values agree: whole (``value``), by the file name a path ends in (``name``: the
    same capture in another folder), or by that name less its last extension (``stem``: a
    capture converted on its way into the other set, ``x.pickle`` → ``x.json``). Every copy of
    a removed key goes, and the records kept stay in ``source``'s order.

    Only the keys are read to decide — through each source's projection, so a source of
    recordings lists its file names without decoding a sample — and the decision is taken
    once, on first use; the kept records are then read from ``source`` on demand.

    Args:
        source: The indexable source to keep records from (``__len__`` and ``__getitem__``).
        remove: The source whose records' keys are left out of ``source``; only walked.
        key: The record entry compared (``file``: a file listing's path).
        compare: How the key's values are compared: ``value``, ``name`` or ``stem``.
    """

    def __init__(
        self,
        source: Any = None,
        remove: Any = None,
        key: str = "file",
        compare: CompareBy = "value",
    ) -> None:
        # Lazy / zero-arg: store config only; both sources are walked on first use (`indices`). An unknown
        # `compare` never gets here: @configurable validates the Literal.
        self.source = source
        self.remove = remove
        self.key = key
        self.compare = compare
        self._indices: Optional[List[int]] = None

    def _keys(self, records: Iterator[Record], side: str) -> Iterator[Hashable]:
        """Each record's compared key, in order; a record without one is refused by its index."""
        for index, record in enumerate(records):
            if self.key not in record:  # a projection carries the key alone: the record's other entries are unseen
                raise KeyError(f"DifferenceSource: {side} record {index} has no {self.key!r} entry to compare")
            yield self._compared(item_value(record[self.key]), side, index)

    def _compared(self, value: Any, side: str, index: int) -> Hashable:
        where = f"DifferenceSource: {side} record {index}'s {self.key!r} is {type(value).__name__} {value!r}"
        if self.compare != "value":
            if not isinstance(value, (str, PathLike)):
                raise TypeError(f"{where} — compare={self.compare!r} reads a file path, a string")
            path = PurePath(value)
            return path.name if self.compare == "name" else path.stem
        if not isinstance(value, Hashable):
            raise TypeError(f"{where} — a value compared must be hashable (a string, a number)")
        return value

    @property
    def indices(self) -> List[int]:
        """The indices of ``source`` kept, in its order — decided on first access and cached."""
        if self._indices is None:
            from recordstream.projection import project

            source, remove = self.source, self.remove
            _guard_live_source(source, "DifferenceSource.source")
            _guard_live_source(remove, "DifferenceSource.remove")
            if source is None or not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
                raise TypeError(
                    "DifferenceSource.source must be an indexable source (__len__ and __getitem__); "
                    f"got {type(source).__name__}"
                )
            if remove is None or not hasattr(remove, "__iter__"):
                raise TypeError(f"DifferenceSource.remove must be a source of records; got {type(remove).__name__}")
            removed: Set[Hashable] = set(self._keys(project(remove, (self.key,)), "remove"))
            kept = [i for i, k in enumerate(self._keys(project(source, (self.key,)), "source")) if k not in removed]
            self._indices = kept
            logger.debug(
                f"DifferenceSource: {len(source)} record(s), {len(source) - len(kept)} left out by "
                f"{len(removed)} {self.compare} key(s) of {self.key!r}, {len(kept)} kept"
            )
        return self._indices

    @property
    def removed(self) -> int:
        """How many records of ``source`` were left out."""
        return len(self.source) - len(self.indices)

    def project(self, keys: Collection[str]) -> Iterator[Record]:
        """The kept records carrying only ``keys``, through ``source``'s own projection."""
        from recordstream.projection import project_indices

        yield from project_indices(self.source, keys, self.indices)

    def __iter__(self) -> Iterator[Record]:
        for index in self.indices:
            yield _pass_through(self.source[index])

    def __getitem__(self, index: int) -> Any:
        return _pass_through(self.source[self.indices[index]])

    def __len__(self) -> int:
        return len(self.indices)
