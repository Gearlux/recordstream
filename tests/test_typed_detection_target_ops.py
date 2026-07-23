"""Typed-bag TWINS of the two detection target-shaping ops.

Pins the native typed transforms that let a ``Sample`` detection pipeline build its
torchvision-style ``{boxes, labels}`` target as a :class:`~sampleflux.Regions` item without the
legacy ``Sample`` path:

* :class:`sampleflux.ops.target.CocoToTorchVisionDetection` — a HuggingFace / COCO ``objects``
  annotation → a target ``Regions``;
* :class:`sampleflux.ops.target.MasksToDetectionBoxes` — a segmentation ``Mask`` → a target ``Regions``.

Each twin REUSES its legacy op's conversion math, so the twin's ``boxes`` / ``labels`` tensors are
pinned byte-identical to a legacy run on the equivalent ``Sample`` (parity). sampleflux-only — no
waivefront import.
"""

import numpy as np
import pytest
import torch
from confluid.registry import get_registry, resolve_class

from sampleflux import Image, Label, Mask, Regions, Sample
from sampleflux.collate import typed_collate
from sampleflux.ops.target import (
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
        s = Sample({"objects": Label(_OBJECTS)}, roles={"objects": "aux"})
        out = CocoToTorchVisionDetection(field="objects")(s)
        regions = out["target"]
        assert isinstance(regions, Regions)
        assert out.role_of("target") == "target"
        assert isinstance(regions.boxes, torch.Tensor)
        assert isinstance(regions.labels, torch.Tensor)
        assert regions.boxes.shape == (2, 4)
        assert regions.labels.shape == (2,)

    def test_parity_with_helper(self) -> None:
        typed = CocoToTorchVisionDetection(field="objects")(Sample({"objects": Label(_OBJECTS)}))
        expected = coco_to_detection(_OBJECTS)
        assert torch.equal(typed["target"].boxes, expected["boxes"])
        assert torch.equal(typed["target"].labels, expected["labels"])

    def test_parity_xyxy_and_label_offset(self) -> None:
        objects = {"bbox": [[10.0, 20.0, 40.0, 60.0]], "category": [2]}
        typed = CocoToTorchVisionDetection(field="objects", bbox_format="xyxy", label_offset=1)(
            Sample({"objects": Label(objects)})
        )
        expected = coco_to_detection(objects, bbox_format="xyxy", label_offset=1)
        assert torch.equal(typed["target"].boxes, expected["boxes"])
        assert torch.equal(typed["target"].labels, expected["labels"])

    def test_empty_annotation_yields_empty_tensors(self) -> None:
        out = CocoToTorchVisionDetection(field="objects")(Sample({"objects": Label({"bbox": [], "category": []})}))
        assert out["target"].boxes.shape == (0, 4)
        assert out["target"].labels.shape == (0,)

    def test_default_picks_first_label(self) -> None:
        s = Sample({"image": Image(np.zeros((2, 2, 3), dtype=np.uint8)), "objects": Label(_OBJECTS)})
        out = CocoToTorchVisionDetection()(s)
        assert out["target"].boxes.shape == (2, 4)

    def test_new_output_field_keeps_source(self) -> None:
        s = Sample({"objects": Label(_OBJECTS)})
        out = CocoToTorchVisionDetection(field="objects", output="det")(s)
        assert isinstance(out["det"], Regions)
        assert out.role_of("det") == "target"
        assert out["objects"].value == _OBJECTS  # source left intact

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in sample"):
            CocoToTorchVisionDetection(field="nope")(Sample({"objects": Label(_OBJECTS)}))

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="sample is empty"):
            CocoToTorchVisionDetection()(Sample({}))

    def test_non_dict_source_raises(self) -> None:
        # The reused legacy op rejects a non-objects-shaped value loudly.
        with pytest.raises(TypeError, match="objects mapping"):
            CocoToTorchVisionDetection(field="objects")(Sample({"objects": Label("not a dict")}))


# --------------------------------------------------------------------------- #
# MasksToDetectionBoxes
# --------------------------------------------------------------------------- #
class TestMasksToDetectionBoxes:
    def test_instance_mask_produces_target_regions(self) -> None:
        s = Sample({"mask": Mask(_instance_mask())}, roles={"mask": "aux"})
        out = MasksToDetectionBoxes(field="mask")(s)
        regions = out["target"]
        assert isinstance(regions, Regions)
        assert isinstance(regions.boxes, torch.Tensor)
        assert isinstance(regions.labels, torch.Tensor)
        assert out.role_of("target") == "target"
        assert regions.boxes.shape == (3, 4)  # three instances
        assert regions.labels.tolist() == [1, 1, 1]  # every box → foreground class 1

    def test_instance_parity_with_helper(self) -> None:
        mask = _instance_mask()
        typed = MasksToDetectionBoxes(field="mask")(Sample({"mask": Mask(mask)}))
        expected = masks_to_detection(mask)
        assert torch.equal(typed["target"].boxes, expected["boxes"])
        assert torch.equal(typed["target"].labels, expected["labels"])

    def test_connected_components_parity(self) -> None:
        # A binary/semantic mask (all objects share value 1): connected=True splits into blobs.
        binary = (_instance_mask() != 0).astype(np.uint8)
        typed = MasksToDetectionBoxes(field="mask", connected=True, label=2)(Sample({"mask": Mask(binary)}))
        expected = masks_to_detection(binary, connected=True, label=2)
        assert typed["target"].boxes.shape[0] == 3  # three connected blobs
        assert torch.equal(typed["target"].boxes, expected["boxes"])
        assert torch.equal(typed["target"].labels, expected["labels"])

    def test_min_area_drops_small_instances(self) -> None:
        mask = _instance_mask()
        typed = MasksToDetectionBoxes(field="mask", min_area=10)(Sample({"mask": Mask(mask)}))
        expected = masks_to_detection(mask, min_area=10)
        assert torch.equal(typed["target"].boxes, expected["boxes"])

    def test_empty_mask_yields_empty_tensors(self) -> None:
        out = MasksToDetectionBoxes(field="mask")(Sample({"mask": Mask(np.zeros((4, 4), dtype=np.uint8))}))
        assert out["target"].boxes.shape == (0, 4)
        assert out["target"].labels.shape == (0,)

    def test_default_picks_first_mask(self) -> None:
        s = Sample({"image": Image(np.zeros((2, 2, 3), dtype=np.uint8)), "seg": Mask(_instance_mask())})
        out = MasksToDetectionBoxes()(s)
        assert out["target"].boxes.shape == (3, 4)

    def test_new_output_field_keeps_source(self) -> None:
        s = Sample({"mask": Mask(_instance_mask())})
        out = MasksToDetectionBoxes(field="mask", output="det")(s)
        assert isinstance(out["det"], Regions)
        assert out.role_of("det") == "target"
        assert isinstance(out["mask"], Mask)  # source left intact

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in sample"):
            MasksToDetectionBoxes(field="nope")(Sample({"mask": Mask(_instance_mask())}))

    def test_no_mask_or_array_field_raises(self) -> None:
        with pytest.raises(ValueError, match="no Mask or array-bearing field"):
            MasksToDetectionBoxes()(Sample({"lbl": Label("x")}))


# --------------------------------------------------------------------------- #
# Collate — per-sample Regions gather into a list of detection targets.
# --------------------------------------------------------------------------- #
def test_typed_collate_gathers_regions_as_list() -> None:
    a = CocoToTorchVisionDetection(field="objects")(Sample({"objects": Label(_OBJECTS)}))
    c = CocoToTorchVisionDetection(field="objects")(
        Sample({"objects": Label({"bbox": [[1.0, 2.0, 3.0, 4.0]], "category": [5]})})
    )
    batch = typed_collate([a, c])
    # Variable-N boxes can't be stacked → the collate gathers them as a per-sample list of tensors.
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
