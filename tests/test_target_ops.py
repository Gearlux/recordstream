"""Tests for the target movers / encoders (``sampleflux.ops.target``)."""

import numpy as np
import pytest
from PIL import Image

from sampleflux.ops.target import (
    CocoToTorchVisionDetectionOp,
    DecodeTargetOp,
    EncodeTargetOp,
    MasksToDetectionBoxesOp,
    MetadataToTargetOp,
)
from sampleflux.sample import Sample


# --------------------------------------------------------------------------- #
# MetadataToTargetOp
# --------------------------------------------------------------------------- #
def test_metadata_to_target_moves_value() -> None:
    out = MetadataToTargetOp(key="drone")(Sample(input=0, metadata={"drone": "DJI MINI3"}))
    assert out.target == "DJI MINI3"


def test_metadata_to_target_leaves_metadata_untouched_without_target_key() -> None:
    sample = Sample(input=0, metadata={"drone": "DJI MINI3"})
    out = MetadataToTargetOp(key="drone")(sample)
    assert set(out.meta) == {"drone"}


def test_metadata_to_target_copies_to_target_key() -> None:
    sample = Sample(input=0, metadata={"drone": "DJI MINI3"})
    out = MetadataToTargetOp(key="drone", target_key="raw_label")(sample)
    assert out.target == "DJI MINI3"
    assert out.meta["raw_label"] == "DJI MINI3"


def test_metadata_to_target_missing_key_raises() -> None:
    with pytest.raises(KeyError, match="no key 'drone'"):
        MetadataToTargetOp(key="drone")(Sample(input=0, metadata={"other": 1}))


# --------------------------------------------------------------------------- #
# EncodeTargetOp
# --------------------------------------------------------------------------- #
def test_encode_target_maps_known_value() -> None:
    op = EncodeTargetOp(mapping={"DJI AVATA2": 2, "DJI MINI3": 5})
    assert op(Sample(input=0, target="DJI MINI3")).target == 5


def test_encode_target_class_zero_allowed() -> None:
    op = EncodeTargetOp(mapping={"first": 0, "second": 1})
    assert op(Sample(input=0, target="first")).target == 0


def test_encode_target_unknown_raises() -> None:
    op = EncodeTargetOp(mapping={"a": 1})
    with pytest.raises(KeyError, match="not in mapping"):
        op(Sample(input=0, target="missing"))


def test_encode_target_unknown_substitutes_default_when_ignored() -> None:
    op = EncodeTargetOp(mapping={"a": 1}, ignore_unknown=True, default=7)
    assert op(Sample(input=0, target="missing")).target == 7


def test_encode_target_empty_mapping_rejected() -> None:
    op = EncodeTargetOp(mapping={})  # lazy: construction succeeds
    with pytest.raises(ValueError, match="at least one entry"):
        op(Sample(input=0, target="x"))


# --------------------------------------------------------------------------- #
# DecodeTargetOp
# --------------------------------------------------------------------------- #
def test_decode_target_inverts_encode() -> None:
    mapping = {"DJI AVATA2": 2, "DJI MINI3": 5}
    inverse = {v: k for k, v in mapping.items()}
    sample = Sample(input=0, target="DJI MINI3")
    encoded = EncodeTargetOp(mapping=mapping)(sample)
    decoded = DecodeTargetOp(mapping=inverse)(encoded)
    assert decoded.target == "DJI MINI3"


def test_decode_target_unknown_default_is_none() -> None:
    op = DecodeTargetOp(mapping={1: "a"}, ignore_unknown=True)
    assert op(Sample(input=0, target=999)).target is None


def test_decode_target_empty_mapping_rejected() -> None:
    op = DecodeTargetOp(mapping={})  # lazy: construction succeeds
    with pytest.raises(ValueError, match="at least one entry"):
        op(Sample(input=0, target=1))


# --------------------------------------------------------------------------- #
# Composed chain (the decomposed classification label path)
# --------------------------------------------------------------------------- #
def test_metadata_to_target_then_encode() -> None:
    label_to_index = {"DJI AVATA2": 2, "DJI MINI3": 5}
    sample = Sample(input=0, metadata={"drone": "DJI AVATA2"})
    sample = MetadataToTargetOp(key="drone", target_key="raw_label")(sample)
    sample = EncodeTargetOp(mapping=label_to_index)(sample)
    assert sample.target == 2
    # raw label preserved for decode/reporting
    assert sample.meta["raw_label"] == "DJI AVATA2"


# --------------------------------------------------------------------------- #
# CocoToTorchVisionDetectionOp (HF / COCO objects -> {boxes xyxy, labels})
# --------------------------------------------------------------------------- #
def _objects_sample(bbox: object, category: object) -> Sample:
    """A Sample shaped like ``HuggingFaceSource(target_feature='objects')`` output."""
    return Sample(input="img", target={"bbox": bbox, "category": category}, metadata={})


def test_objects_to_boxes_xywh_to_xyxy_and_dtypes() -> None:
    op = CocoToTorchVisionDetectionOp()  # default bbox_format="xywh"
    out = op(_objects_sample([[10, 20, 30, 40]], [2]))
    # COCO [x,y,w,h]=[10,20,30,40] -> xyxy [10,20,40,60]
    assert out.target["boxes"].tolist() == [[10.0, 20.0, 40.0, 60.0]]
    assert out.target["labels"].tolist() == [2]
    assert str(out.target["boxes"].dtype) == "torch.float32"
    assert str(out.target["labels"].dtype) == "torch.int64"


def test_objects_to_boxes_label_offset_for_background_class() -> None:
    # label_offset=1 shifts 0-indexed dataset categories to torchvision foreground ids 1..K.
    op = CocoToTorchVisionDetectionOp(label_offset=1)
    out = op(_objects_sample([[0, 0, 4, 4], [1, 1, 2, 2]], [0, 3]))
    assert out.target["labels"].tolist() == [1, 4]


def test_objects_to_boxes_xyxy_passthrough() -> None:
    op = CocoToTorchVisionDetectionOp(bbox_format="xyxy")
    out = op(_objects_sample([[1, 2, 3, 4]], [0]))
    assert out.target["boxes"].tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_objects_to_boxes_cxcywh() -> None:
    op = CocoToTorchVisionDetectionOp(bbox_format="cxcywh")
    # center (50,50), size (20,40) -> [40,30,60,70]
    out = op(_objects_sample([[50, 50, 20, 40]], [1]))
    assert out.target["boxes"].tolist() == [[40.0, 30.0, 60.0, 70.0]]


def test_objects_to_boxes_empty_annotation_yields_empty_tensors() -> None:
    op = CocoToTorchVisionDetectionOp()
    out = op(_objects_sample([], []))
    assert tuple(out.target["boxes"].shape) == (0, 4)
    assert tuple(out.target["labels"].shape) == (0,)


def test_objects_to_boxes_custom_keys() -> None:
    op = CocoToTorchVisionDetectionOp(bbox_key="boxes", category_key="labels")
    out = op(Sample(input="i", target={"boxes": [[0, 0, 2, 2]], "labels": [5]}, metadata={}))
    assert out.target["boxes"].tolist() == [[0.0, 0.0, 2.0, 2.0]]
    assert out.target["labels"].tolist() == [5]


def test_objects_to_boxes_rejects_non_mapping_target() -> None:
    with pytest.raises(TypeError, match="objects mapping"):
        CocoToTorchVisionDetectionOp()(Sample(input="i", target=[1, 2, 3], metadata={}))


# --------------------------------------------------------------------------- #
# MasksToDetectionBoxesOp (segmentation mask -> {boxes xyxy, labels})
# --------------------------------------------------------------------------- #
def _instance_mask() -> np.ndarray:
    """Two objects: instance id 1 at rows 2-4/cols 1-3, id 2 at rows 6-8/cols 7-10."""
    m = np.zeros((10, 12), dtype=np.uint8)
    m[2:5, 1:4] = 1
    m[6:9, 7:11] = 2
    return m


def test_masks_instance_mode_per_id_bbox_and_dtypes() -> None:
    op = MasksToDetectionBoxesOp(label=1)  # default connected=False
    out = op(Sample(input="img", target=Image.fromarray(_instance_mask(), mode="L"), metadata={}))
    # row/col extents → xyxy with exclusive far edge.
    assert sorted(out.target["boxes"].tolist()) == [[1.0, 2.0, 4.0, 5.0], [7.0, 6.0, 11.0, 9.0]]
    assert out.target["labels"].tolist() == [1, 1]
    assert str(out.target["boxes"].dtype) == "torch.float32"
    assert str(out.target["labels"].dtype) == "torch.int64"


def test_masks_label_assigns_class_id() -> None:
    out = MasksToDetectionBoxesOp(label=3)(Sample(input="i", target=_instance_mask(), metadata={}))
    assert out.target["labels"].tolist() == [3, 3]


def test_masks_connected_mode_splits_semantic_blobs() -> None:
    # A SEMANTIC mask (both objects = 1) — connected components separate the two blobs.
    sem = (_instance_mask() > 0).astype(np.uint8)
    out = MasksToDetectionBoxesOp(connected=True)(Sample(input="i", target=sem, metadata={}))
    assert sorted(out.target["boxes"].tolist()) == [[1.0, 2.0, 4.0, 5.0], [7.0, 6.0, 11.0, 9.0]]
    assert out.target["labels"].tolist() == [1, 1]


def test_masks_min_area_drops_small_instances() -> None:
    m = np.zeros((8, 8), dtype=np.uint8)
    m[0, 0] = 1  # area 1
    m[4:7, 4:7] = 2  # area 9
    out = MasksToDetectionBoxesOp(min_area=2)(Sample(input="i", target=m, metadata={}))
    assert out.target["boxes"].tolist() == [[4.0, 4.0, 7.0, 7.0]]


def test_masks_empty_mask_yields_empty_tensors() -> None:
    out = MasksToDetectionBoxesOp()(Sample(input="i", target=np.zeros((5, 5), dtype=np.uint8), metadata={}))
    assert tuple(out.target["boxes"].shape) == (0, 4)
    assert tuple(out.target["labels"].shape) == (0,)


def test_masks_rejects_non_2d_target() -> None:
    with pytest.raises(TypeError, match="2-D segmentation mask"):
        MasksToDetectionBoxesOp()(Sample(input="i", target=np.zeros((4, 4, 3), dtype=np.uint8), metadata={}))
