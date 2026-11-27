"""``Derive`` — run ops once per record, and read what they wrote from a store ever after.

A compose-group op (alongside ``Enable`` / ``Parallel``): it wraps an inner op-list whose outputs are worth
keeping — a model's predictions, an embedding — and keeps them in a
:class:`~recordstream.storage.derived.DerivedStore` by record id. Modality-neutral, so it lives in core
recordstream.
"""

from typing import Any, Dict, List, Optional

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record
from recordstream.record_id import RECORD_ID

logger = get_logger(__name__)


@configurable(category="op", group="compose")
class Derive:
    """Wrap ops whose ``outputs`` are kept by record id: computed for a record once, read from then on.

    For each record the wrapper looks its id up in ``store``. When every output is kept, the inner ops do not
    run and the kept values are written onto the record. Otherwise the inner ops run and their outputs are
    kept. What they are kept under is a hash of the inner ops with their settings and of ``files``: change an
    op, a setting or a weights file and the values are computed anew, beside the old ones.

    The record must carry its id (read the dataset through
    :class:`~recordstream.sources.identified.IdentifiedSource`). A record the inner ops drop is dropped, and
    nothing is kept for it.

    YAML::

        - !class:recordstream.ops.derive.Derive
          store: !class:recordstream.storage.derived.DerivedStore {root: $RUN_DIR/derived}
          outputs: [embedding]
          files: [weights/encoder.pt]
          ops:
            - !class:mypackage.Embed {weights: weights/encoder.pt}

    Args:
        ops: Non-empty list of ops whose outputs are kept.
        outputs: The record entries the ops write that are kept.
        store: The DerivedStore the outputs are kept in.
        files: Files whose content decides the outputs (a model's weights); each is part of the hash.
        key: The record entry holding the record's id.
    """

    def __init__(
        self,
        ops: Optional[List] = None,
        outputs: Optional[List[str]] = None,
        store: Any = None,
        files: Optional[List[str]] = None,
        key: str = RECORD_ID,
    ) -> None:
        # Lazy / zero-arg: store config only; the hash is made and the store read on the first record.
        self.ops: List = list(ops) if ops else []
        self.outputs: List[str] = list(outputs) if outputs else []
        self.store = store
        self.files: List[str] = list(files) if files else []
        self.key = key
        self._producer: Optional[str] = None
        self._tables: Optional[Dict[str, Any]] = None

    @property
    def producer(self) -> str:
        """What the outputs are kept under: the hash of the inner ops, their settings and ``files``."""
        if self._producer is None:
            from recordstream.storage.derived import producer_hash

            if not self.ops:
                raise ValueError("Derive requires a non-empty 'ops' list.")
            self._producer = producer_hash(self.ops, self.files)
        return self._producer

    def _ready(self) -> Dict[str, Any]:
        if self._tables is None:
            from confluid import flow
            from confluid.fluid import Fluid

            from recordstream.storage.derived import producer_text

            if not self.outputs:
                raise ValueError("Derive requires `outputs`: the entries of its ops that are kept.")
            if self.store is None:
                raise ValueError("Derive requires a `store` (a DerivedStore).")
            if isinstance(self.store, Fluid):
                self.store = flow(self.store)
            producer = self.producer
            self.store.describe(producer, producer_text(self.ops))
            self._tables = {name: self.store.table(producer, name) for name in self.outputs}
        return self._tables

    def __call__(self, record: Record) -> Optional[Record]:
        tables = self._ready()
        record_id = record.get(self.key)
        if not isinstance(record_id, str) or not record_id:
            raise KeyError(
                f"Derive: the record carries no {self.key!r} entry, so nothing can be kept for it. "
                "Read the dataset through IdentifiedSource."
            )
        if all(record_id in table for table in tables.values()):
            return {**record, **{name: table.get(record_id) for name, table in tables.items()}}

        from confluid import flow
        from confluid.fluid import Fluid

        from recordstream.core import _apply_op

        current: Optional[Record] = record
        for i, op in enumerate(self.ops):
            if current is None:
                return None
            if isinstance(op, Fluid):
                op = flow(op)
                self.ops[i] = op
            if op is None:
                continue
            current = _apply_op(current, op)
        if current is None:
            return None
        absent = [name for name in tables if name not in current]
        if absent:
            raise KeyError(f"Derive: its ops wrote no {absent} entry; `outputs` names what they write.")
        for name, table in tables.items():
            table.put(record_id, current[name])
        return current

    def close(self) -> None:
        """Write what is held in memory, and propagate close to the inner ops."""
        if self._tables is not None:
            for table in self._tables.values():
                table.flush()
        for op in self.ops:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["Derive"]
