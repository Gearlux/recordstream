"""``SampleSinkOp`` — adapt a :class:`~dataflux.storage.base.DataSink` as a pass-through op.

Lets any storage sink (``HDF5Sink``, ``ZarrGroupSink``, the waivefront JSON
sinks …) slot into a ``Sample``-based op chain: on first call it opens the
sink, every call writes the sample and returns it unchanged, and ``close()``
flushes + closes. Modality-neutral (duck-typed ``open``/``write``/``close``),
so it lives in core dataflux.
"""

from typing import Any

from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)


@configurable(category="op", group="sink")
class SampleSinkOp:
    """Adapter: wrap a :class:`dataflux.storage.base.DataSink` as a pass-through op.

    Sinks (``JsonPerWindowSink``, ``JsonSink``, ``HDF5Sink`` …) implement the
    ``open()`` / ``write(sample)`` / ``close()`` protocol and are normally
    attached to a :class:`marainer.processing.DatasetProcessor` as the
    flux's terminal sink. This adapter lets the same sinks slot into any
    Sample-based op chain — notably the ``ops`` list of
    :class:`waivefront.sinks.DetectionPredictionsSink`, where the model's
    predictions arrive as a Sample whose metadata carries the new
    ``predicted_regions`` and need to be persisted to disk just like a
    segment-pipeline output.

    On the first call the adapter calls ``sink.open()`` (when present); each
    subsequent call forwards the Sample to ``sink.write(sample)`` and returns
    the Sample unchanged. ``close()`` flushes (when present) and closes the
    underlying sink — propagated by :class:`dataflux.ops.enable.Enable` and
    :class:`waivefront.sinks.DetectionPredictionsSink` at end-of-run.

    YAML::

        - !class:dataflux.ops.sink.SampleSinkOp
          sink: !class:waivefront.sinks.JsonPerWindowSink
            output_dir: ./predictions_per_window

    Args:
        sink: A DataSink-like object exposing ``write(sample)`` (and optionally ``open``/``flush``/``close``).
    """

    def __init__(self, sink: Any = None) -> None:
        # Lazy / zero-arg: store config only; a non-None sink is required lazily in __call__.
        self.sink = sink
        self._opened = False

    def __call__(self, sample: Sample) -> Sample:
        if self.sink is None:
            raise ValueError("SampleSinkOp requires a non-None 'sink'.")
        if not self._opened:
            opener = getattr(self.sink, "open", None)
            if callable(opener):
                opener()
            self._opened = True
        self.sink.write(sample)
        return sample

    def close(self) -> None:
        flush = getattr(self.sink, "flush", None)
        if callable(flush):
            flush()
        closer = getattr(self.sink, "close", None)
        if callable(closer):
            closer()
        self._opened = False


__all__ = ["SampleSinkOp"]
