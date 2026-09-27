"""``ConvertToBoxes`` — any common box shape into THE canonical ``Boxes`` item.

The record model has ONE box format by contract (absolute-pixel half-open xyxy); this node
is how everything else gets there: an HF detection dict (COCO xywh + category), a plain
[N, 4] array, normalized YOLO rows. The SOURCE format is the knob; the target never varies.
"""

from typing import Any, Dict

import numpy as np
import pytest

from recordstream.items import Boxes, Image, Label
from recordstream.ops.image import ConvertToBoxes


def _record(value: Any, key: str = "class") -> Dict[str, Any]:
    return {"image": Image(np.zeros((100, 200, 3), dtype=np.uint8)), key: value}


class TestTheHuggingFaceObjectsDict:
    """The shape that motivated the node: cppe-5's ``objects`` — COCO bbox + category."""

    OBJECTS = {"bbox": [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 20.0, 10.0]], "category": [2, 0], "area": [1, 2]}

    def test_coco_xywh_becomes_canonical_xyxy(self) -> None:
        out = ConvertToBoxes(format="xywh")(_record(Label(value=self.OBJECTS)))
        boxes = out["target"]
        assert isinstance(boxes, Boxes)
        assert [list(map(float, b)) for b in boxes.boxes] == [[10, 20, 40, 60], [50, 60, 70, 70]]
        assert boxes.labels is not None and list(boxes.labels) == [2, 0]

    def test_the_label_wrapper_is_seen_through(self) -> None:
        """HuggingFaceSource wraps the target column in a Label — the dict is INSIDE it."""
        out = ConvertToBoxes(format="xywh")(_record(Label(value=self.OBJECTS)))
        assert isinstance(out["target"], Boxes)

    def test_the_canvas_comes_from_the_records_image(self) -> None:
        out = ConvertToBoxes(format="xywh")(_record(Label(value=self.OBJECTS)))
        assert out["target"].canvas == (100, 200)

    def test_the_vocabulary_rides_in_when_given(self) -> None:
        out = ConvertToBoxes(format="xywh", classes=["mask", "gloves", "gown"])(_record(Label(value=self.OBJECTS)))
        assert out["target"].classes == ["mask", "gloves", "gown"]

    def test_every_other_entry_travels_untouched(self) -> None:
        record = _record(Label(value=self.OBJECTS))
        record["note"] = "keep"
        out = ConvertToBoxes(format="xywh")(record)
        assert out["note"] == "keep" and "image" in out and "class" in out


class TestCoordinateFormats:
    def test_xyxy_is_the_identity(self) -> None:
        out = ConvertToBoxes(format="xyxy")(_record({"boxes": [[1.0, 2.0, 3.0, 4.0]], "labels": [0]}))
        assert [list(map(float, b)) for b in out["target"].boxes] == [[1, 2, 3, 4]]

    def test_cxcywh_centers_unpack(self) -> None:
        out = ConvertToBoxes(format="cxcywh")(_record({"boxes": [[50.0, 30.0, 20.0, 10.0]]}))
        assert [list(map(float, b)) for b in out["target"].boxes] == [[40, 25, 60, 35]]

    def test_normalized_rows_scale_by_the_canvas(self) -> None:
        """YOLO's classic: normalized cxcywh over a 200x100 image."""
        out = ConvertToBoxes(format="cxcywh", normalized=True)(_record({"boxes": [[0.5, 0.5, 0.2, 0.4]]}))
        assert [list(map(float, b)) for b in out["target"].boxes] == [[80, 30, 120, 70]]

    def test_a_bare_array_is_boxes_without_labels(self) -> None:
        out = ConvertToBoxes(format="xyxy")(_record(np.array([[1.0, 2.0, 3.0, 4.0]])))
        assert isinstance(out["target"], Boxes) and out["target"].labels is None

    def test_an_existing_boxes_item_reconverts_its_rows(self) -> None:
        """Fixing a mis-made Boxes: rows stated in xywh get normalised to the contract."""
        wrong = Boxes(boxes=[[10.0, 20.0, 30.0, 40.0]], labels=[1], scores=[0.5])
        out = ConvertToBoxes(format="xywh")(_record(wrong))
        assert [list(map(float, b)) for b in out["target"].boxes] == [[10, 20, 40, 60]]
        assert list(out["target"].labels) == [1] and list(out["target"].scores) == [0.5]


class TestSelectionAndRefusals:
    def test_the_field_defaults_to_the_first_boxish_entry(self) -> None:
        """Structural, like ConvertToImage's field="": the first entry holding a bbox/boxes
        container or an [N, 4] array qualifies; images never do."""
        record = _record(Label(value={"bbox": [[0.0, 0.0, 1.0, 1.0]], "category": [0]}), key="anything")
        assert isinstance(ConvertToBoxes(format="xywh")(record)["target"], Boxes)

    def test_a_named_field_that_is_not_boxish_is_refused_naming_it(self) -> None:
        with pytest.raises(ValueError, match="image"):
            ConvertToBoxes(field="image")(_record({"boxes": [[0, 0, 1, 1]]}))

    def test_a_record_with_nothing_boxish_is_refused_with_the_fix(self) -> None:
        with pytest.raises(ValueError, match="field"):
            ConvertToBoxes()({"image": Image(np.zeros((4, 4, 3), dtype=np.uint8))})

    def test_an_unknown_format_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="format"):
            ConvertToBoxes(format="yolo")  # type: ignore[arg-type]

    def test_the_output_key_is_a_knob_with_the_contract_default(self) -> None:
        assert ConvertToBoxes().output == "target"


class TestConvertFromBoxes:
    """The INVERSE: canonical ``Boxes`` back into the layout a source delivered — so update
    can write annotations through a sink in the dataset's own format (user design 2026-08-31)."""

    def _boxes(self) -> Boxes:
        return Boxes(
            boxes=[[10.0, 20.0, 40.0, 60.0], [50.0, 60.0, 70.0, 70.0]],
            labels=[2, 0],
            scores=[0.9, 0.4],
            canvas=(100, 200),
            classes=["mask", "gloves", "gown"],
        )

    def test_the_objects_dict_round_trips_coco(self) -> None:
        from recordstream.ops.image import ConvertFromBoxes

        out = ConvertFromBoxes(format="xywh")(_record(self._boxes(), key="target"))
        objects = out["class"]
        assert objects["bbox"] == [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 20.0, 10.0]]
        assert objects["category"] == [2, 0] and objects["score"] == [0.9, 0.4]

    def test_to_boxes_then_from_boxes_is_the_identity_on_the_payload(self) -> None:
        from recordstream.ops.image import ConvertFromBoxes

        objects = {"bbox": [[10.0, 20.0, 30.0, 40.0]], "category": [1]}
        record = _record(Label(value=objects))
        forward = ConvertToBoxes(format="xywh")(record)
        back = ConvertFromBoxes(format="xywh", output="class")(forward)
        assert back["class"]["bbox"] == objects["bbox"] and back["class"]["category"] == objects["category"]

    def test_the_array_container_is_rows_only(self) -> None:
        from recordstream.ops.image import ConvertFromBoxes

        out = ConvertFromBoxes(format="xyxy", container="array")(_record(self._boxes(), key="target"))
        assert out["class"] == [[10.0, 20.0, 40.0, 60.0], [50.0, 60.0, 70.0, 70.0]]

    def test_normalized_scales_back_down_by_the_canvas(self) -> None:
        from recordstream.ops.image import ConvertFromBoxes

        boxes = Boxes(boxes=[[80.0, 30.0, 120.0, 70.0]], canvas=(100, 200))
        out = ConvertFromBoxes(format="cxcywh", normalized=True, container="array")(_record(boxes, key="target"))
        assert out["class"] == [[0.5, 0.5, 0.2, 0.4]]

    def test_a_missing_boxes_entry_is_refused_naming_the_field(self) -> None:
        from recordstream.ops.image import ConvertFromBoxes

        with pytest.raises(ValueError, match="target"):
            ConvertFromBoxes()({"image": Image(np.zeros((4, 4, 3), dtype=np.uint8))})

    def test_defaults_mirror_the_forward_op(self) -> None:
        from recordstream.ops.image import ConvertFromBoxes

        op = ConvertFromBoxes()
        assert op.field == "target" and op.output == "class" and op.container == "objects"


class TestBoxMatching:
    """`box_iou` / `match_boxes` / `size_bucket` — the review pass's arithmetic."""

    def test_iou_of_identical_and_disjoint(self) -> None:
        from recordstream.ops.image import box_iou

        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        iou = box_iou(a, b)
        assert iou.shape == (1, 2)
        assert iou[0, 0] == pytest.approx(1.0) and iou[0, 1] == 0.0

    def test_iou_of_half_overlap(self) -> None:
        from recordstream.ops.image import box_iou

        iou = box_iou(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([[5.0, 0.0, 15.0, 10.0]]))
        assert iou[0, 0] == pytest.approx(50.0 / 150.0)

    def test_greedy_matching_pairs_best_first(self) -> None:
        from recordstream.ops.image import match_boxes

        truth = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
        predicted = [[1.0, 1.0, 11.0, 11.0], [100.0, 100.0, 110.0, 110.0]]
        result = match_boxes(truth, [0, 1], predicted, [0, 0], iou_threshold=0.5)
        assert [(t, p) for t, p, _ in result["matched"]] == [(0, 0)]
        assert result["fn"] == [1], "the second truth box was missed"
        assert result["fp"] == [1], "the far-away detection matches nothing"

    def test_class_aware_matching_refuses_a_cross_class_pair(self) -> None:
        from recordstream.ops.image import match_boxes

        truth, predicted = [[0.0, 0.0, 10.0, 10.0]], [[0.0, 0.0, 10.0, 10.0]]
        strict = match_boxes(truth, [0], predicted, [1], iou_threshold=0.5, class_aware=True)
        assert strict["matched"] == [] and strict["fn"] == [0] and strict["fp"] == [0]
        loose = match_boxes(truth, [0], predicted, [1], iou_threshold=0.5, class_aware=False)
        assert [(t, p) for t, p, _ in loose["matched"]] == [(0, 0)]

    def test_empty_sides_are_all_misses_or_all_false_alarms(self) -> None:
        from recordstream.ops.image import match_boxes

        assert match_boxes([[0, 0, 1, 1]], [0], [], [], iou_threshold=0.5)["fn"] == [0]
        assert match_boxes([], [], [[0, 0, 1, 1]], [0], iou_threshold=0.5)["fp"] == [0]

    def test_size_buckets_follow_the_thresholds(self) -> None:
        from recordstream.ops.image import size_bucket

        assert size_bucket([0, 0, 10, 10]) == "small"  # 100 < 1024
        assert size_bucket([0, 0, 50, 50]) == "medium"  # 2500 < 9216
        assert size_bucket([0, 0, 200, 200]) == "large"
        assert size_bucket([0, 0, 10, 10], thresholds=(50, 200)) == "medium"
