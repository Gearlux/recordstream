"""Inference as an op: ``ModelPredict`` stamps a model's output back onto the record.

A pipeline that carries this op IS the predictions flow — a viewer executes it per
record and reads the stamped fields as layers; ``recordstream run`` executes the same
document offline. The model is duck-typed (any callable), so none of this needs torch.
"""

from typing import Any, Dict

import numpy as np
import pytest

from recordstream.items import Boxes, Label
from recordstream.ops.predict import ModelPredict


class _Classifier:
    """logits [1, 2] favouring class 1; counts solidify() calls (the lazy checkpoint load)."""

    seen: Any = None  # the batch the (optionally monkeypatched) forward received

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


class TestClassificationScore:
    def test_the_confidence_rides_beside_the_label(self) -> None:
        out = ModelPredict(model=_Classifier(), kind="classification", output="class")({"image": np.zeros((8, 12))})
        assert out["class"].value == 1
        expected = float(np.exp(2.0) / (np.exp(0.1) + np.exp(2.0)))
        assert isinstance(out["score"], Label) and abs(out["score"].value - expected) < 1e-9

    def test_an_empty_score_key_disables_it(self) -> None:
        out = ModelPredict(model=_Classifier(), kind="classification", score="")({"image": np.zeros((8, 12))})
        assert "score" not in out

    def test_probability_outputs_keep_their_own_confidence(self) -> None:
        out = ModelPredict(model=lambda batch: np.array([[0.25, 0.75]]), kind="classification")(
            {"image": np.zeros((4, 4))}
        )
        assert abs(out["score"].value - 0.75) < 1e-9


class TestTheDeviceIsSelectableAndLimitedToTheMachine:
    def test_an_unknown_device_name_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="device"):
            ModelPredict(device="tpu")  # type: ignore[arg-type]

    def test_auto_resolves_to_an_available_device(self) -> None:
        from recordstream.ops.predict import available_devices

        op = ModelPredict(model=_Classifier(), kind="classification")
        assert op._resolve_device() in available_devices()

    def test_a_device_this_machine_lacks_is_refused_listing_what_it_has(self) -> None:
        from recordstream.ops.predict import available_devices

        machine = available_devices()
        absent = next((d for d in ("cuda", "mps") if d not in machine), None)
        if absent is None:  # pragma: no cover - a machine with every device
            pytest.skip("this machine has every torch device")
        op = ModelPredict(model=_Classifier(), kind="classification", device=absent)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="available"):
            op._resolve_device()

    def test_cpu_is_always_available(self) -> None:
        from recordstream.ops.predict import available_devices

        assert "cpu" in available_devices()

    def test_a_model_without_to_ignores_the_device(self) -> None:
        """The CON case: a plain callable (no ``.to``) runs whatever the device says."""
        out = ModelPredict(model=lambda batch: np.array([[1.0, 0.0]]), kind="classification")(
            {"image": np.zeros((4, 4))}
        )
        assert out["predict"].value == 0


class TestWidgetParamsAreAppendOnly:
    def test_score_comes_AFTER_device_in_the_signature(self) -> None:
        """Canvas widget values are POSITIONAL: a saved canvas converted by an older server
        zips values onto the older parameter list, so a param inserted mid-signature shifts
        every later widget (measured live: ``device: 'score'`` → ConstructionError). New
        params are APPENDED — the old prefix then still zips correctly everywhere."""
        import inspect

        names = list(inspect.signature(ModelPredict.__init__).parameters)
        assert names.index("score") > names.index("device")
        assert names[1:6] == ["model", "kind", "key", "output", "device"], "the pre-existing order is frozen"


class TestPredictBatch:
    """ONE forward per chunk of records — `__call__` is a batch of one, so the two paths
    cannot drift; per-record inference is just `batch_size` 1 (user design 2026-08-29)."""

    def test_one_forward_serves_every_record(self) -> None:
        model = _Classifier()
        op = ModelPredict(model=model, kind="classification", output="class")
        records = [{"image": np.full((4, 4), i, dtype=np.float32), "n": i} for i in range(3)]
        model_out = np.array([[0.1, 2.0], [3.0, 0.1], [0.1, 5.0]])

        def forward(self: Any, batch: Any) -> Any:
            self.seen = batch
            return model_out

        model.__class__.__call__ = forward  # type: ignore[method-assign]
        stamped = op.predict_batch(records)
        assert model.seen.shape == (3, 4, 4), "the records were STACKED into one batch"
        assert [r["class"].value for r in stamped] == [1, 0, 1]
        assert [round(float(r["score"].value), 2) > 0 for r in stamped] == [True, True, True]
        assert [r["n"] for r in stamped] == [0, 1, 2], "every other entry rides along per record"

    def test_call_IS_a_batch_of_one(self) -> None:
        op = ModelPredict(model=_Classifier(), kind="classification")
        single = op({"image": np.zeros((4, 4))})
        batched = op.predict_batch([{"image": np.zeros((4, 4))}])[0]
        assert single["predict"].value == batched["predict"].value

    def test_ragged_records_fail_naming_the_shapes_and_the_fix(self) -> None:
        op = ModelPredict(model=_Classifier(), kind="classification")
        records = [{"image": np.zeros((4, 4))}, {"image": np.zeros((8, 8))}]
        with pytest.raises(ValueError, match="resize"):
            op.predict_batch(records)

    def test_an_empty_chunk_is_a_no_op(self) -> None:
        assert ModelPredict(model=_Classifier()).predict_batch([]) == []

    def test_a_missing_field_names_it_like_the_single_path(self) -> None:
        with pytest.raises(ValueError, match="image"):
            ModelPredict(model=_Classifier()).predict_batch([{"other": 1}])

    def test_batch_size_is_a_declared_knob_appended_LAST(self) -> None:
        """0 (default) = no opinion — the RUNNER's default decides; the widget-order rule
        keeps it after every pre-existing param."""
        import inspect

        assert ModelPredict().batch_size == 0
        names = list(inspect.signature(ModelPredict.__init__).parameters)
        assert names.index("batch_size") > names.index("score"), "appended after the pre-existing params"
        assert names.index("frame") > names.index("batch_size"), "each NEW param appends after the last"

    def test_segmentation_debatches_per_row(self) -> None:
        def model(batch: Any) -> np.ndarray:
            n = batch.shape[0]
            logits = np.zeros((n, 2, 2, 2))
            for i in range(n):
                logits[i, i % 2] = 1.0  # row i argmaxes to class i%2
            return logits

        op = ModelPredict(model=model, kind="segmentation")
        stamped = op.predict_batch([{"image": np.zeros((2, 2))} for _ in range(2)])
        assert int(stamped[0]["predict_mask"][0, 0]) == 0 and int(stamped[1]["predict_mask"][0, 0]) == 1


class TestTheFrameKnob:
    """Detection on a RESIZED copy: `frame` names the entry whose size the boxes map back to.

    A DETR-style wrapper postprocesses to its own square resolution; without the mapping the
    stamped boxes (and every IoU downstream) live in model space, not the image's.
    """

    def test_boxes_scale_from_the_model_input_to_the_named_frame(self) -> None:
        def model(batch: Any) -> Any:
            return [{"boxes": [[96.0, 96.0, 192.0, 192.0]], "scores": [0.9], "labels": [1]}]

        record = {
            "image": np.zeros((768, 1536, 3), dtype=np.uint8),  # the ORIGINAL frame
            "model_input": np.zeros((384, 384, 3), dtype=np.float32),  # what the model saw
        }
        out = ModelPredict(model=model, kind="detection", key="model_input", output="target", frame="image")(record)
        stamped = out["target"]
        # x scales by 1536/384 = 4, y by 768/384 = 2
        assert stamped.boxes == [[384.0, 192.0, 768.0, 384.0]]
        assert stamped.canvas == (768, 1536), "the canvas is the FRAME's, not the model input's"
        assert stamped.labels == [1] and stamped.scores == [0.9]

    def test_an_empty_frame_keeps_todays_behaviour(self) -> None:
        def model(batch: Any) -> Any:
            return [{"boxes": [[10.0, 10.0, 20.0, 20.0]], "scores": [0.5], "labels": [0]}]

        record = {"image": np.zeros((100, 200, 3), dtype=np.uint8)}
        out = ModelPredict(model=model, kind="detection")(record)
        assert out["predict"].boxes == [[10.0, 10.0, 20.0, 20.0]] and out["predict"].canvas == (100, 200)

    def test_a_missing_frame_entry_is_refused_by_name(self) -> None:
        def model(batch: Any) -> Any:
            return [{"boxes": [[0.0, 0.0, 1.0, 1.0]]}]

        with pytest.raises(ValueError, match="frame"):
            ModelPredict(model=model, kind="detection", frame="nope")({"image": np.zeros((4, 4))})

    def test_frame_is_appended_last(self) -> None:
        import inspect

        assert list(inspect.signature(ModelPredict.__init__).parameters)[-1] == "frame"


class TestSolidifyIsAHookNotASwap:
    def test_the_wired_wrapper_stays_the_callable(self) -> None:
        """A wrapper's solidify() may RETURN its inner network (RF-DETR's does) — adopting
        it bypasses the wrapper's own pre/post-processing. The build hook runs; the model
        the graph wired keeps answering."""

        class _Inner:
            def __call__(self, batch: Any) -> Any:  # pragma: no cover - must never be called
                raise AssertionError("the inner network must not replace the wrapper")

        class _Wrapper:
            def __init__(self) -> None:
                self.built = False

            def solidify(self) -> Any:
                self.built = True
                return _Inner()  # returns the inner network, like RF-DETR

            def eval(self) -> None:
                pass

            def __call__(self, batch: Any) -> Any:
                return np.array([[0.2, 0.8]])

        wrapper = _Wrapper()
        out = ModelPredict(model=wrapper, kind="classification")({"image": np.zeros((4, 4))})
        assert wrapper.built and out["predict"].value == 1


class TestDetectionBatchesAreLists:
    def test_the_model_receives_a_list_not_a_stack(self) -> None:
        """The torchvision convention: detection forward(images: List[Tensor])."""
        seen: Dict[str, Any] = {}

        def model(batch: Any) -> Any:
            seen["type"] = type(batch).__name__
            return [{"boxes": [[0.0, 0.0, 1.0, 1.0]], "scores": [0.5], "labels": [0]} for _ in batch]

        records = [{"image": np.zeros((4, 4, 3), dtype=np.float32)} for _ in range(2)]
        out = ModelPredict(model=model, kind="detection").predict_batch(records)
        assert seen["type"] == "list" and len(out) == 2
