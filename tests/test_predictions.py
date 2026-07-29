"""Tests for :mod:`recordstream.predictions` — the predictions-sink contract + the classification sink."""

from typing import Any, Dict, List

import pytest
import torch

from recordstream import ClassificationPredictionsSink, PredictionsSink, Record


class _CapturingOp:
    """Tiny op that records every record dict it sees, for assertion purposes."""

    def __init__(self) -> None:
        self.calls: List[Record] = []
        self.closed = False

    def __call__(self, record: Record) -> Record:
        self.calls.append(record)
        return record

    def close(self) -> None:
        self.closed = True


def _make_prediction(
    probs: List[float],
    class_idx: int,
) -> Dict[str, Any]:
    """Build a ClassificationOutput-shaped dict from a flat prob list."""
    p = torch.tensor(probs, dtype=torch.float32)
    return {
        "logits": torch.log(p),  # not used by sink, but realistic
        "probs": p,
        "class_idx": torch.tensor(class_idx, dtype=torch.int64),
    }


def _make_metadata() -> Dict[str, Any]:
    """Arbitrary per-record metadata — the sink must not require any particular key."""
    return {"source_id": "p1", "index": 0}


# --- Happy path ------------------------------------------------------------


def test_top1_path_writes_predicted_columns() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        label_names={0: "DJI MINI3", 1: "DJI AVATA2", 2: "Other"},
        top_k=1,
    )
    sink.write(_make_prediction([0.1, 0.7, 0.2], class_idx=1), _make_metadata())

    assert len(op.calls) == 1
    meta = op.calls[0]["metadata"]
    assert meta["predicted_class_id"] == 1
    assert meta["predicted_class_label"] == "DJI AVATA2"
    assert meta["predicted_confidence"] == pytest.approx(0.7, abs=1e-6)
    assert len(meta["predicted_top_k"]) == 1
    assert meta["predicted_top_k"][0]["label"] == "DJI AVATA2"


def test_top_k_path_returns_descending_probabilities() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        label_names={0: "a", 1: "b", 2: "c", 3: "d"},
        top_k=3,
    )
    sink.write(_make_prediction([0.1, 0.4, 0.45, 0.05], class_idx=2), _make_metadata())

    top_k = op.calls[0]["metadata"]["predicted_top_k"]
    assert len(top_k) == 3
    # Sorted descending by probability.
    probs = [entry["probability"] for entry in top_k]
    assert probs == sorted(probs, reverse=True)
    assert top_k[0]["label"] == "c"
    assert top_k[1]["label"] == "b"
    assert top_k[2]["label"] == "a"


def test_top_k_clamps_when_exceeds_num_classes() -> None:
    """top_k=10 with 3 classes → returns 3, not crash."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=10)
    sink.write(_make_prediction([0.5, 0.3, 0.2], class_idx=0), _make_metadata())
    assert len(op.calls[0]["metadata"]["predicted_top_k"]) == 3


def test_label_names_string_keys_work() -> None:
    """YAML loads int keys as strings; both forms must work."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        label_names={"0": "a", "1": "b"},  # str keys
        top_k=1,
    )
    sink.write(_make_prediction([0.2, 0.8], class_idx=1), _make_metadata())
    assert op.calls[0]["metadata"]["predicted_class_label"] == "b"


def test_label_names_missing_key_falls_back_to_str_class_id() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        label_names={0: "zero"},  # only class 0 mapped
        top_k=1,
    )
    sink.write(_make_prediction([0.1, 0.9], class_idx=1), _make_metadata())
    # Missing class 1 → falls back to "1".
    assert op.calls[0]["metadata"]["predicted_class_label"] == "1"


def test_no_label_names_uses_str_class_id() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=2)
    sink.write(_make_prediction([0.7, 0.3], class_idx=0), _make_metadata())
    meta = op.calls[0]["metadata"]
    assert meta["predicted_class_label"] == "0"
    assert {entry["label"] for entry in meta["predicted_top_k"]} == {"0", "1"}


def test_threaded_record_carries_metadata_key_only() -> None:
    """The threaded record is a plain dict carrying only the prediction metadata key."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.write(_make_prediction([0.6, 0.4], class_idx=0), _make_metadata())
    record = op.calls[0]
    assert isinstance(record, dict)
    assert list(record.keys()) == ["metadata"]


def test_close_propagates_to_ops() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.close()
    assert op.closed is True


# --- Confidence threshold --------------------------------------------------


def test_confidence_threshold_drops_low_predictions() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        top_k=1,
        confidence_threshold=0.5,
    )
    # top1 prob = 0.4, below threshold → no record produced.
    sink.write(_make_prediction([0.4, 0.3, 0.3], class_idx=0), _make_metadata())
    assert op.calls == []


def test_confidence_threshold_keeps_high_predictions() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        top_k=1,
        confidence_threshold=0.5,
    )
    sink.write(_make_prediction([0.7, 0.3], class_idx=0), _make_metadata())
    assert len(op.calls) == 1


# --- Edge cases ------------------------------------------------------------


def test_zero_arg_construction_works_and_ops_is_validated_lazily() -> None:
    """The package-wide lazy-init rule: buildable with no args, validated where it is used."""
    sink = ClassificationPredictionsSink()  # no raise — a form/schema generator can introspect it
    with pytest.raises(ValueError, match="'ops' is empty"):
        sink.write(_make_prediction([0.5, 0.5], class_idx=0), {})


def test_empty_ops_list_raises_on_write_not_construction() -> None:
    sink = ClassificationPredictionsSink(ops=[])
    with pytest.raises(ValueError, match="'ops' is empty"):
        sink.write(_make_prediction([0.5, 0.5], class_idx=0), {})


def test_zero_top_k_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        ClassificationPredictionsSink(ops=[_CapturingOp()], top_k=0)


def test_missing_probs_skips_silently_with_warning() -> None:
    """If the model emits a malformed prediction, the sink skips rather than crashing."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.write({"logits": torch.zeros(3)}, _make_metadata())  # no probs/class_idx
    assert op.calls == []


def test_handles_2d_probs_with_leading_singleton() -> None:
    """Some pipelines emit [1, C] instead of [C]; sink must accept both."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.write(
        {
            "logits": torch.zeros(1, 3),
            "probs": torch.tensor([[0.1, 0.6, 0.3]], dtype=torch.float32),
            "class_idx": torch.tensor(1, dtype=torch.int64),
        },
        _make_metadata(),
    )
    assert len(op.calls) == 1
    assert op.calls[0]["metadata"]["predicted_class_id"] == 1
    assert op.calls[0]["metadata"]["predicted_confidence"] == pytest.approx(0.6, abs=1e-6)


# --- Contract + neutrality -------------------------------------------------


def test_sink_satisfies_the_predictions_sink_protocol() -> None:
    """`isinstance` works because the Protocol is @runtime_checkable (method NAMES only)."""
    assert isinstance(ClassificationPredictionsSink(ops=[_CapturingOp()]), PredictionsSink)


def test_write_works_with_completely_empty_metadata() -> None:
    """No domain key is required — the sink is modality-neutral, diagnostics included.

    The pre-move version built its log line from `pack_id` / `iq_file` / `window_start_sample`,
    signal-domain keys, in a class documented as neutral. A record identified only by ordinal
    keeps that promise.
    """
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.write(_make_prediction([0.2, 0.8], class_idx=1), {})
    assert op.calls[0]["metadata"]["predicted_class_id"] == 1


def test_a_malformed_prediction_is_reported_by_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The diagnostic locates the bad record without naming any domain metadata key."""
    from recordstream import predictions as predictions_module

    warnings: List[str] = []
    monkeypatch.setattr(predictions_module.logger, "warning", lambda msg: warnings.append(str(msg)))

    sink = ClassificationPredictionsSink(ops=[_CapturingOp()], top_k=1)
    sink.write(_make_prediction([0.5, 0.5], class_idx=0), {})  # record #0 — fine
    sink.write({"logits": torch.zeros(2)}, {})  # record #1 — malformed
    assert len(warnings) == 1 and "record #1" in warnings[0]
