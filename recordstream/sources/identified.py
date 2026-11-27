"""``IdentifiedSource`` — a source's records, each carrying its record id."""

from typing import Any, Collection, Dict, Iterator, List, Optional, Union

from confluid import configurable

from recordstream.items import Record
from recordstream.record_id import RECORD_ID, has_record_keys, record_ids
from recordstream.sources.base import _guard_live_source


@configurable(category="source")
class IdentifiedSource:
    """The records of ``source``, each with its id written under ``key``.

    The id is :func:`recordstream.record_id.record_ids`' answer for the record: the same one through any
    view of its dataset, so an op, a sink or a store further on can name the record without knowing where
    it was read. Nothing else in the record changes, and no record is read to make the ids.

    The ids are listed once, on first use. A folder source that gains a file is read again by a new
    ``IdentifiedSource`` (or after ``reset()``).

    YAML::

        source: !class:recordstream.sources.identified.IdentifiedSource
          source: !class:recordstream.sources.files.FilesSource
            root: $DATA_ROOT/captures
            name: captures

    Args:
        source: The indexable source whose records are named (validated lazily on first use).
        key: The entry each record's id is written under.
    """

    def __init__(self, source: Any = None, key: str = RECORD_ID) -> None:
        # Lazy / zero-arg: store config only; the ids are listed on first use.
        self.source = source
        self.key = key
        self._ids: Optional[List[str]] = None

    def reset(self) -> None:
        """Forget the listed ids, so the next use lists the source again."""
        self._ids = None

    def record_ids(self) -> List[str]:
        """Every record's id, in record order (listed once)."""
        if self._ids is None:
            source = self.source
            _guard_live_source(source, "IdentifiedSource.source")
            if source is None or not hasattr(source, "__getitem__"):
                raise TypeError(
                    "IdentifiedSource requires a source supporting __len__ and __getitem__; "
                    f"got {type(source).__name__}"
                )
            self._ids = record_ids(source)
        return self._ids

    @property
    def record_ids_are_keys(self) -> bool:
        """Whether the ids are the source's own names for its records, not their places."""
        return has_record_keys(self.source)

    @property
    def produces(self) -> Union[Dict[str, str], tuple]:
        """The source's entries plus the id's, in the source's own spelling (a mapping or a tuple of keys)."""
        declared = getattr(self.source, "produces", None)
        if isinstance(declared, dict):
            return {**declared, self.key: "str"}
        return tuple(declared or ()) + (self.key,)

    def project(self, keys: Collection[str]) -> Iterator[Record]:
        """Yield each record carrying only ``keys`` — the id from the listing, the rest through the source's walk."""
        from recordstream.projection import project

        ids = self.record_ids()
        others = [key for key in keys if key != self.key]
        if not others:
            for record_id in ids:
                yield {self.key: record_id}
            return
        wanted = self.key in keys
        for record_id, record in zip(ids, project(self.source, others)):
            yield {**record, self.key: record_id} if wanted else record

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> Record:
        record_id = self.record_ids()[index]
        return {**self.source[index], self.key: record_id}

    def __len__(self) -> int:
        return len(self.record_ids())
