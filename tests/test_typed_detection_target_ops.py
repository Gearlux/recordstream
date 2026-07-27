"""The two detection target-shaping ops over dict records.

Pins the native transforms that build a detection pipeline's torchvision-style
``{boxes, labels}`` target as a :class:`~recordstream.Regions` item:

* :class:`recordstream.ops.target.CocoToTorchVisionDetection` — a HuggingFace / COCO ``objects``
  annotation → a target ``Regions``;
* :class:`recordstream.ops.target.MasksToDetectionBoxes` — a segmentation ``Mask`` → a target ``Regions``.

Each op REUSES its conversion helper, so the op's ``boxes`` / ``labels`` tensors are pinned
byte-identical to the helper (parity). recordstream-only — no domain-package import.
"""

import numpy as np
import pytest
import torch
from confluid.registry import get_registry, resolve_class

from recordstream import Image, Label, Mask, Regions, collate_records
from recordstream.ops.target import (
    CocoToTorchVisionDetection,
    MasksToDetectionBoxes,
    coco_to_detection,
    masks_to_detection,
)

# A COCO / HF objects annotation: two boxes in [x, y, w, h] pixels + integer categories.
_OBJECTS = {"bbox": [[10.0, 20.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]], "category": [1, 3]}


def _instance_mask() -> np.ndarray:
    """A 2-D instance mask with three known objects (pixel values 1/2/3), one per instance."""
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[1:4, 2:5] = 1  # object 1
    mask[6:9, 7:10] = 2  # object 2
    mask[0:2, 9:12] = 3  # object 3
    return mask


# --------------------------------------------------------------------------- #
# CocoToTorchVisionDetection
# --------------------------------------------------------------------------- #
class TestCocoToTorchVisionDetection:
    def test_produces_target_regions(self) -> None:
        out = CocoToTorchVisionDetection(field="objects")({"objects": Label(_OBJECTS)})
        regions = out["target"]
        assert isinstance(regions, Regions)
        assert isinstance(regions.boxes, torch.Tensor)
        assert isinstance(regions.labels, torch.Tensor)
        assert regions.boxes.shape == (2, 4)
        assert regions.labels.shape == (2,)

    def test_parity_with_helper(self) -> None:
        out = CocoToTorchVisionDetection(field="objects")({"objects": Label(_OBJECTS)})
        expected = coco_to_detection(_OBJECTS)
        assert torch.equal(out["target"].boxes, expected["boxes"])
        assert torch.equal(out["target"].labels, expected["labels"])

    def test_parity_xyxy_and_label_offset(self) -> None:
        objects = {"bbox": [[10.0, 20.0, 40.0, 60.0]], "category": [2]}
        out = CocoToTorchVisionDetection(field="objects", bbox_format="xyxy", label_offset=1)(
            {"objects": Label(objects)}
        )
        expected = coco_to_detection(objects, bbox_format="xyxy", label_offset=1)
        assert torch.equal(out["target"].boxes, expected["boxes"])
        assert torch.equal(out["target"].labels, expected["labels"])

    def test_empty_annotation_yields_empty_tensors(self) -> None:
        out = CocoToTorchVisionDetection(field="objects")({"objects": Label({"bbox": [], "category": []})})
        assert out["target"].boxes.shape == (0, 4)
        assert out["target"].labels.shape == (0,)

    def test_default_picks_first_label(self) -> None:
        rec = {"image": Image(np.zeros((2, 2, 3), dtype=np.uint8)), "objects": Label(_OBJECTS)}
        out = CocoToTorchVisionDetection()(rec)
        assert out["target"].boxes.shape == (2, 4)

    def test_new_output_key_keeps_source(self) -> None:
        out = CocoToTorchVisionDetection(field="objects", output="det")({"objects": Label(_OBJECTS)})
        assert isinstance(out["det"], Regions)
        assert out["objects"].value == _OBJECTS  # source left intact

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in record"):
            CocoToTorchVisionDetection(field="nope")({"objects": Label(_OBJECTS)})

    def test_empty_record_raises(self) -> None:
        with pytest.raises(ValueError, match="record is empty"):
            CocoToTorchVisionDetection()({})

    def test_non_dict_source_raises(self) -> None:
        # The shared helper rejects a non-objects-shaped value loudly.
        with pytest.raises(TypeError, match="objects mapping"):
            CocoToTorchVisionDetection(field="objects")({"objects": Label("not a dict")})


# --------------------------------------------------------------------------- #
# MasksToDetectionBoxes
# --------------------------------------------------------------------------- #
class TestMasksToDetectionBoxes:
    def test_instance_mask_produces_target_regions(self) -> None:
        out = MasksToDetectionBoxes(field="mask")({"mask": Mask(_instance_mask())})
        regions = out["target"]
        assert isinstance(regions, Regions)
        assert isinstance(regions.boxes, torch.Tensor)
        assert isinstance(regions.labels, torch.Tensor)
        assert regions.boxes.shape == (3, 4)  # three instances
        assert regions.labels.tolist() == [1, 1, 1]  # every box → foreground class 1

    def test_instance_parity_with_helper(self) -> None:
        mask = _instance_mask()
        out = MasksToDetectionBoxes(field="mask")({"mask": Mask(mask)})
        expected = masks_to_detection(mask)
        assert torch.equal(out["target"].boxes, expected["boxes"])
        assert torch.equal(out["target"].labels, expected["labels"])

    def test_connected_components_parity(self) -> None:
        # A binary/semantic mask (all objects share value 1): connected=True splits into blobs.
        binary = (_instance_mask() != 0).astype(np.uint8)
        out = MasksToDetectionBoxes(field="mask", connected=True, label=2)({"mask": Mask(binary)})
        expected = masks_to_detection(binary, connected=True, label=2)
        assert out["target"].boxes.shape[0] == 3  # three connected blobs
        assert torch.equal(out["target"].boxes, expected["boxes"])
        assert torch.equal(out["target"].labels, expected["labels"])

    def test_min_area_drops_small_instances(self) -> None:
        mask = _instance_mask()
        out = MasksToDetectionBoxes(field="mask", min_area=10)({"mask": Mask(mask)})
        expected = masks_to_detection(mask, min_area=10)
        assert torch.equal(out["target"].boxes, expected["boxes"])

    def test_empty_mask_yields_empty_tensors(self) -> None:
        out = MasksToDetectionBoxes(field="mask")({"mask": Mask(np.zeros((4, 4), dtype=np.uint8))})
        assert out["target"].boxes.shape == (0, 4)
        assert out["target"].labels.shape == (0,)

    def test_default_picks_first_mask(self) -> None:
        rec = {"image": Image(np.zeros((2, 2, 3), dtype=np.uint8)), "seg": Mask(_instance_mask())}
        out = MasksToDetectionBoxes()(rec)
        assert out["target"].boxes.shape == (3, 4)

    def test_new_output_key_keeps_source(self) -> None:
        out = MasksToDetectionBoxes(field="mask", output="det")({"mask": Mask(_instance_mask())})
        assert isinstance(out["det"], Regions)
        assert isinstance(out["mask"], Mask)  # source left intact

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in record"):
            MasksToDetectionBoxes(field="nope")({"mask": Mask(_instance_mask())})

    def test_no_mask_or_array_field_raises(self) -> None:
        with pytest.raises(ValueError, match="no Mask or array-bearing field"):
            MasksToDetectionBoxes()({"lbl": Label("x")})


# --------------------------------------------------------------------------- #
# Collate — per-record Regions gather into a list of detection targets.
# --------------------------------------------------------------------------- #
def test_record_collate_gathers_regions_as_list() -> None:
    a = CocoToTorchVisionDetection(field="objects")({"objects": Label(_OBJECTS)})
    c = CocoToTorchVisionDetection(field="objects")(
        {"objects": Label({"bbox": [[1.0, 2.0, 3.0, 4.0]], "category": [5]})}
    )
    batch = collate_records([a, c])
    # Variable-N boxes can't be stacked → the collate gathers them as a per-record list of tensors.
    assert isinstance(batch["target"], Regions)
    assert isinstance(batch["target"].boxes, list) and len(batch["target"].boxes) == 2
    assert batch["target"].boxes[0].shape == (2, 4)
    assert batch["target"].boxes[1].shape == (1, 4)


# --------------------------------------------------------------------------- #
# Discovery + zero-arg construction.
# --------------------------------------------------------------------------- #
def test_zero_arg_constructible() -> None:
    assert CocoToTorchVisionDetection().output == "target"
    assert CocoToTorchVisionDetection().bbox_format == "xywh"
    assert MasksToDetectionBoxes().output == "target"
    assert MasksToDetectionBoxes().connected is False


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("CocoToTorchVisionDetection", CocoToTorchVisionDetection),
        ("MasksToDetectionBoxes", MasksToDetectionBoxes),
    ],
)
def test_discovery_tags(name: str, cls: type) -> None:
    assert cls.__confluid_category__ == "op"  # type: ignore[attr-defined]
    assert cls.__confluid_group__ == "structure"  # type: ignore[attr-defined]
    assert resolve_class(name) is cls
    registry = get_registry()
    assert name in registry.list_classes(category="op")
    assert name in registry.list_classes(group="structure")
