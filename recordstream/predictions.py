"""Predictions sinks — where a predict/test loop's per-record output goes.

A runnable's predict loop calls ``predictions_sink.write(prediction, metadata)`` once per
record and ``predictions_sink.close()`` at end of run. This module carries the protocol that
states that contract plus the classification sink built on it.

**Why this is a SECOND sink protocol.** :class:`recordstream.storage.base.DataSink` takes a
whole ``record`` (``write(record)``) and is what ``RecordSinkOp`` adapts into an op chain.
A predictions sink instead receives the MODEL's output plus the metadata of the record it came
from, and builds the record itself — the two halves arrive separately because a model emits a
batch while the sink contract is per-record. The split is deliberate and load-bearing
elsewhere: a visual editor surfaces ``category="sink"`` storage sinks as canvas nodes and
excludes prediction sinks precisely because their signature differs. Collapsing them (have the
runnable build the record and write through ``DataSink``) is a real option, tracked in
``TASKS.md`` — until then, do not blur the two.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
from confluid import configurable
from loggair import get_logger

from recordstream.items import Record

logger = get_logger(__name__)


@runtime_checkable
class PredictionsSink(Protocol):
    """What a runnable's ``predictions_sink`` slot must provide.

    Structural, not a base class: a sink is anything that can take one record's prediction plus
    that record's metadata, and be closed at the end. Naming it here — beside the sink this
    package ships — means consuming runnables annotate ``Optional[Lazy[PredictionsSink]]``
    instead of ``Any``, which declared nothing and let a use site call ``.write`` on a slot that
    might still be a deferred marker.

    ``@runtime_checkable`` so an ``isinstance`` guard is available; note that only the METHOD
    NAMES are checked at runtime, never their signatures.
    """

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        """Record one prediction alongside the metadata of the record it came from."""
        ...

    def close(self) -> None:
        """Flush and release whatever the sink holds open."""
        ...


@configurable
class ClassificationPredictionsSink:
    """Lift classification predictions into a record and run ops over it.

    Bridges a predict/test loop's ``predictions_sink`` contract — ``write(prediction, metadata)``
    per record — to record ops, giving the train→eval→predict triad a uniform shape across
    tasks.

    For each call:

    1. Read the model's :class:`~recordstream.outputs.ClassificationOutput` — ``probs`` ``[C]``
       and ``class_idx`` scalar, per record.
    2. Resolve the int class id to a human-readable label via ``class_names``, and build a top-k
       list (the ``top_k`` highest-probability classes with their probabilities + labels).
    3. Build a fresh record carrying the original metadata plus the prediction columns under a
       single ``"metadata"`` key:

       * ``predicted_class_id`` (int)
       * ``predicted_class_label`` (str)
       * ``predicted_confidence`` (float — top-1 probability)
       * ``predicted_top_k`` (list of ``{class_id, label, probability}``, descending)

    4. Thread that record through ``ops`` — typically just ``RecordSinkOp(sink=…)`` to dump JSON.

    Modality-neutral: it reads only the prediction contract and the metadata it is handed, so a
    classifier over images, spectrograms or tabular rows uses it unchanged. Diagnostics identify
    a record by its ORDINAL in the run for the same reason — no domain key names appear here.

    Zero-arg constructible, like every configurable in this package: ``ops`` is required to RUN,
    not to BUILD, so the non-empty check fires on the first :meth:`write` with a clear message
    rather than in ``__init__`` (which would make the class unbuildable by a schema/form
    generator that instantiates with defaults to introspect it).

    Args:
        ops: Ops to run on each per-prediction record. Required by the time the sink is written
            to; validated there, not at construction.
        class_names: Optional ``{int_class_id: str}`` map converting the model's int64 class ids
            back to human-readable strings. Without it, labels become ``str(class_id)``. YAML int
            keys land here as strings (config loaders stringify mapping keys); both forms are
            accepted and normalized to ``str`` internally.
        top_k: Number of top-probability predictions recorded per record. Defaults to ``1``;
            ``top_k > num_classes`` is silently clamped.
        confidence_threshold: Predictions whose top-1 probability is below this are skipped
            entirely (no record produced, no ops run). Default ``0.0`` keeps every prediction.
    """

    def __init__(
        self,
        ops: Optional[List[Any]] = None,
        class_names: Optional[Dict[Any, str]] = None,
        top_k: int = 1,
        confidence_threshold: float = 0.0,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        self.ops: List[Any] = list(ops or [])
        self.top_k = int(top_k)
        self.confidence_threshold = float(confidence_threshold)
        # Normalize keys to `str` so YAML-loaded maps (always stringified) and Python-constructed
        # maps (which may use int keys) are both addressable by the same lookup.
        self.class_names: Dict[str, str] = {str(k): str(v) for k, v in (class_names or {}).items()}
        #: How many predictions have been offered — the ordinal a diagnostic names.
        self._seen = 0

    def _label_for(self, class_id: int) -> str:
        return self.class_names.get(str(int(class_id)), str(int(class_id)))

    def write(self, prediction: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        from confluid import flow
        from confluid.fluid import Fluid

        if not self.ops:
            raise ValueError(
                "ClassificationPredictionsSink: 'ops' is empty — wire at least one op to receive "
                "the prediction records (typically RecordSinkOp(sink=...))."
            )

        where = f"record #{self._seen}"
        self._seen += 1

        probs = prediction.get("probs")
        class_idx = prediction.get("class_idx")
        if probs is None or class_idx is None:
            logger.warning(
                f"ClassificationPredictionsSink: {where} is missing 'probs' or 'class_idx' "
                f"(got keys: {sorted(prediction.keys())}); skipping."
            )
            return

        probs_arr = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)
        # Tolerate both [C] (single-record) and [1, C] (batched-with-1) shapes.
        if probs_arr.ndim == 2 and probs_arr.shape[0] == 1:
            probs_arr = probs_arr[0]
        if probs_arr.ndim != 1:
            logger.warning(
                f"ClassificationPredictionsSink: {where} has probs of shape {probs_arr.shape}, "
                f"expected [C] or [1, C]; skipping."
            )
            return

        n_classes = int(probs_arr.shape[0])
        top1_id = int(class_idx.item()) if hasattr(class_idx, "item") else int(class_idx)
        top1_prob = float(probs_arr[top1_id])
        if top1_prob < self.confidence_threshold:
            logger.debug(
                f"ClassificationPredictionsSink: {where} top1_prob={top1_prob:.3f} "
                f"< threshold={self.confidence_threshold:.3f}; skipping."
            )
            return

        k = min(self.top_k, n_classes)
        top_k_ids = np.argsort(-probs_arr)[:k]  # descending, first k
        top_k_list = [
            {
                "class_id": int(cid),
                "label": self._label_for(int(cid)),
                "probability": float(probs_arr[cid]),
            }
            for cid in top_k_ids
        ]

        new_metadata = dict(metadata)
        new_metadata["predicted_class_id"] = top1_id
        new_metadata["predicted_class_label"] = self._label_for(top1_id)
        new_metadata["predicted_confidence"] = top1_prob
        new_metadata["predicted_top_k"] = top_k_list

        logger.debug(
            f"ClassificationPredictionsSink: {where} top1={new_metadata['predicted_class_label']!r} "
            f"({top1_prob:.3f}), top_k={k}"
        )

        # The prediction metadata rides a single "metadata" key on a plain record. Downstream ops
        # (typically `RecordSinkOp` wrapping a JSON sink) read it via `record["metadata"]`. There
        # is no input/target on a prediction-only record.
        record: Record = {"metadata": new_metadata}
        for i, op in enumerate(self.ops):
            if isinstance(op, Fluid):
                op = flow(op)
                self.ops[i] = op
            record = op(record)

    def close(self) -> None:
        """Propagate ``close()`` to ops that own resources (e.g. wrapped sinks).

        The sink itself buffers nothing, but the wrapped ops typically do (e.g. ``RecordSinkOp``
        wrapping a buffered/file-handle sink).
        """
        for op in self.ops:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["ClassificationPredictionsSink", "PredictionsSink"]
