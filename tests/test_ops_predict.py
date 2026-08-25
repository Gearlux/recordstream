"""Inference as an op: ``ModelPredict`` stamps a model's output back onto the record.

A pipeline that carries this op IS the predictions flow — a viewer executes it per
record and reads the stamped fields as layers; ``recordstream run`` executes the same
document offline. The model is duck-typed (any callable), so none of this needs torch.
"""

from typing import Any

import numpy as np
import pytest

from recordstream.items import Boxes, Label
from recordstream.ops.predict import ModelPredict


class _Classifier:
    """logits [1, 2] favouring class 1; counts solidify() calls (the lazy checkpoint load)."""

    def __init__(self) -> None:
        self.solidified = 0
        self.calls = 0

    def solidify(self) -> None:
        self.solidified += 1

    def __call__(self, batch: Any) -> np.ndarray:
        self.calls += 1
        return np.array([[0.1, 2.0]])


class TestModelPredict:
    def test_classification_stamps_a_label_and_solidifies_once(self) -> None:
        model = _Classifier()
        op = ModelPredict(model=model, kind="classification")
        record = {"image": np.zeros((8, 12)), "class": 0}
        out = op(record)
        assert isinstance(out["predict"], Label) and out["predict"].value == 1
        assert "predict" not in record  # a new dict — the input record is never mutated
        op(record)
        assert model.solidified == 1 and model.calls == 2  # the checkpoint loads once, not per record

    def test_detection_stamps_boxes_with_the_images_canvas(self) -> None:
        def model(batch: Any) -> Any:
            return {"boxes": [[[1.0, 2.0, 5.0, 6.0]]], "scores": [[0.9]], "labels": [[1]]}

        out = ModelPredict(model=model, kind="detection")({"image": np.zeros((8, 12))})
        stamped = out["predict"]
        assert isinstance(stamped, Boxes)
        assert stamped.boxes == [[1.0, 2.0, 5.0, 6.0]] and stamped.scores == [0.9] and stamped.labels == [1]
        assert stamped.canvas is not None and tuple(stamped.canvas) == (8, 12)

    def test_segmentation_argmaxes_logits_into_an_int_mask(self) -> None:
        logits = np.zeros((1, 2, 4, 6))
        logits[0, 1, :2] = 5.0  # top rows are class 1
        out = ModelPredict(model=lambda b: logits, kind="segmentation")({"image": np.zeros((4, 6))})
        mask = out["predict_mask"]
        assert mask.shape == (4, 6) and mask.dtype == np.int64
        assert mask[0, 0] == 1 and mask[3, 0] == 0

    def test_restoration_moves_channels_last(self) -> None:
        out = ModelPredict(model=lambda b: np.ones((1, 3, 4, 6)), kind="restoration")({"image": np.zeros((4, 6))})
        assert out["predict"].shape == (4, 6, 3)

    def test_bad_kind_and_missing_pieces_are_located_value_errors(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            ModelPredict(kind="nope")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="model"):
            ModelPredict(kind="classification")({"image": np.zeros((2, 2))})
        with pytest.raises(ValueError, match="no field 'image'"):
            ModelPredict(model=_Classifier())({"other": 1})
