"""``RecordSinkOp`` — adapt a :class:`~recordstream.storage.base.DataSink` as a pass-through op.

Lets any storage sink (``HDF5Sink``, ``ZarrGroupSink``, a domain package's JSON
sinks …) slot into a record-based op chain: on first call it opens the
sink, every call writes the record and returns it unchanged, and ``close()``
flushes + closes. Modality-neutral (duck-typed ``open``/``write``/``close``),
so it lives in core recordstream.
"""

from typing import Any

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record

logger = get_logger(__name__)


@configurable(category="op", group="sink")
class RecordSinkOp:
    """Adapter: wrap a :class:`recordstream.storage.base.DataSink` as a pass-through op.

    Sinks implement the ``open()`` / ``write(record)`` / ``close()`` protocol and
    are normally attached to a :class:`recordstream.processing.DatasetProcessor` as
    the stream's terminal sink. This adapter lets the same sinks slot into any
    record-based op chain (e.g. persisting a prediction pipeline's outputs
    mid-chain).

    On the first call the adapter calls ``sink.open()`` (when present); each
    subsequent call forwards the record to ``sink.write(record)`` and returns
    the record unchanged. ``close()`` flushes (when present) and closes the
    underlying sink — propagated by the composing ops at end-of-run.

    YAML::

        - !class:recordstream.ops.sink.RecordSinkOp
          sink: !class:recordstream.storage.hdf5.HDF5Sink
            path: ./records.h5

    Args:
        sink: A DataSink-like object exposing ``write(record)`` (and optionally ``open``/``flush``/``close``).
    """

    def __init__(self, sink: Any = None) -> None:
        # Partial / zero-arg: store config only; a non-None sink is required lazily in __call__.
        self.sink = sink
        self._opened = False

    def __call__(self, record: Record) -> Record:
        if self.sink is None:
            raise ValueError("RecordSinkOp requires a non-None 'sink'.")
        if not self._opened:
            opener = getattr(self.sink, "open", None)
            if callable(opener):
                opener()
            self._opened = True
        self.sink.write(record)
        return record

    def close(self) -> None:
        flush = getattr(self.sink, "flush", None)
        if callable(flush):
            flush()
        closer = getattr(self.sink, "close", None)
        if callable(closer):
            closer()
        self._opened = False


__all__ = ["RecordSinkOp"]
