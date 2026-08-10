"""``ConvertToMask`` — a mask-bearing field to an int64 class-id ``Mask`` item.

The segmentation counterpart of ``ConvertToImage``, and the front half of the target
pipeline a per-pixel task drives: the source hands over a greyscale/paletted PNG, this
turns it into the ``[H, W]`` int64 array a per-pixel loss consumes, and every remaining
step is an op that already existed (``FormulaOp`` remaps the ids, a bare albumentations
transform resizes it jointly with the image, ``collate_records`` stacks it).

Also pins :func:`recordstream.item_value`, extracted here because the op needed the same
"get past a wrapper item" rule ``iter_key`` and ``batch_values`` already had.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest
from PIL import Image as PILImage

from recordstream import Boxes, Image, Label, Mask, MultiLabel, collate_records, item_data, item_value
from recordstream.core import _apply_op
from recordstream.ops import ConvertToMask, DropField, FormulaOp


def _trimap(height: int = 30, width: int = 20) -> PILImage.Image:
    """An Oxford-IIIT-Pet-style trimap: an L-mode PNG whose pixels are 1-based class ids."""
    return PILImage.fromarray(np.random.randint(1, 4, (height, width)).astype(np.uint8), mode="L")


# --------------------------------------------------------------------------- #
# item_value — the unwrapping rule the op shares with iter_key / batch_values
# --------------------------------------------------------------------------- #
class TestItemValue:
    def test_a_label_yields_its_value_where_item_data_yields_the_label(self) -> None:
        # The whole reason the function exists: a Label's payload slot is `value`, not
        # `data`, so `item_data` cannot see into it.
        label = Label("cat")
        assert item_value(label) == "cat"
        assert item_data(label) is label

    def test_a_multilabel_yields_its_values_list(self) -> None:
        assert item_value(MultiLabel(["cat", "dog"])) == ["cat", "dog"]

    def test_an_array_item_yields_its_payload(self) -> None:
        value = item_value(Mask(np.zeros((3, 3), dtype=np.int64)))
        assert type(value) is np.ndarray
        assert value.shape == (3, 3)

    def test_a_plain_value_passes_through(self) -> None:
        assert item_value(30.72e6) == 30.72e6
        assert item_value(None) is None

    def test_iter_key_and_batch_values_agree_with_it(self) -> None:
        """The two former copies now route through it — so they cannot drift from it."""
        from recordstream import batch_values, iter_key

        records = [{"class": Label(0)}, {"class": Label(1)}]
        assert list(iter_key(records, "class")) == [0, 1]
        assert list(batch_values(collate_records(records), "class")) == [0, 1]


# --------------------------------------------------------------------------- #
# ConvertToMask
# --------------------------------------------------------------------------- #
class TestConvertToMask:
    def test_a_pil_mask_becomes_an_int64_mask_item(self) -> None:
        out = ConvertToMask(field="segmentation_mask")({"segmentation_mask": _trimap()})
        mask = out["mask"]
        assert isinstance(mask, Mask)
        assert mask.shape == (30, 20)
        assert mask.dtype == np.int64

    def test_it_reads_a_mask_a_source_wrapped_in_a_label(self) -> None:
        """The real shape: a source that does not know a column is a mask ships it as a Label.

        ``HuggingFaceSource`` does exactly this for every metadata column, so reading through
        ``item_data`` (which hands the Label straight back) found nothing at all.
        """
        record = {"image": Image(np.zeros((30, 20, 3), np.uint8)), "segmentation_mask": Label(_trimap())}
        out = ConvertToMask(field="segmentation_mask")(record)
        assert isinstance(out["mask"], Mask)
        assert set(np.unique(np.asarray(out["mask"]))) <= {1, 2, 3}

    def test_a_blank_field_picks_the_first_array_or_pil_bearing_item(self) -> None:
        out = ConvertToMask()({"note": "text", "seg": np.zeros((4, 5), dtype=np.uint8)})
        assert out["mask"].shape == (4, 5)

    def test_the_output_key_defaults_to_the_albumentations_vocabulary(self) -> None:
        """``mask`` by default — so a bare albumentations transform finds it with no rename."""
        assert ConvertToMask().output == "mask"
        assert "mask" in ConvertToMask()({"seg": np.zeros((4, 5))})

    def test_the_output_key_is_configurable(self) -> None:
        out = ConvertToMask(field="seg", output="target")({"seg": np.zeros((4, 5))})
        assert "target" in out and "mask" not in out

    def test_the_source_field_survives_so_a_later_op_can_read_it(self) -> None:
        out = ConvertToMask(field="seg")({"seg": np.zeros((4, 5))})
        assert "seg" in out  # dropping it is DropField's job, not this op's

    def test_other_keys_pass_through_untouched(self) -> None:
        image = Image(np.zeros((4, 5, 3), np.uint8))
        out = ConvertToMask(field="seg")({"seg": np.zeros((4, 5)), "image": image, "id": 7})
        assert out["image"] is image
        assert out["id"] == 7

    @pytest.mark.parametrize("shape", [(4, 5, 1), (1, 4, 5)])
    def test_singleton_axes_are_squeezed(self, shape: Tuple[int, ...]) -> None:
        assert ConvertToMask()({"seg": np.zeros(shape, dtype=np.uint8)})["mask"].shape == (4, 5)

    def test_an_rgb_mask_is_refused_rather_than_silently_collapsed(self) -> None:
        """Picking one of three channels is a decision the op must not make for the user."""
        with pytest.raises(ValueError, match="2-D"):
            ConvertToMask()({"seg": np.zeros((4, 5, 3), dtype=np.uint8)})

    def test_a_torch_tensor_payload_is_accepted(self) -> None:
        torch = pytest.importorskip("torch")
        out = ConvertToMask()({"seg": torch.randint(0, 3, (4, 5))})
        assert isinstance(out["mask"], Mask)
        assert out["mask"].dtype == np.int64

    # ---- lazy / zero-arg construction (workspace mandate) ------------------- #
    def test_zero_arg_construction_works(self) -> None:
        assert ConvertToMask().field == ""

    def test_a_missing_named_field_is_reported_at_call_time(self) -> None:
        with pytest.raises(ValueError, match="not in record"):
            ConvertToMask(field="nope")({"seg": np.zeros((4, 5))})

    def test_a_record_with_no_candidate_is_reported_at_call_time(self) -> None:
        with pytest.raises(ValueError, match="no array/PIL-bearing field"):
            ConvertToMask()({"note": "text"})


# --------------------------------------------------------------------------- #
# The chain it belongs to — this is what a segmentation config actually writes
# --------------------------------------------------------------------------- #
class TestTheSegmentationTargetPipeline:
    def test_convert_remap_drop_resize_collate(self) -> None:
        """End to end, with every step after the conversion an op that already existed.

        This is the ``preprocess`` chain of a segmentation config, run in Python: the point
        being that ``ConvertToMask`` is the ONLY piece segmentation needed added.
        """
        albumentations = pytest.importorskip("albumentations")
        torch = pytest.importorskip("torch")
        from recordstream import batch_tensor

        record: Dict[str, Any] = {
            "image": Image(np.random.randint(0, 255, (30, 20, 3), dtype=np.uint8)),
            "segmentation_mask": Label(_trimap()),
        }
        chain = [
            ConvertToMask(field="segmentation_mask"),
            FormulaOp(field="mask", formula="a - 1"),  # 1-based trimap -> 0-based class ids
            DropField(key="segmentation_mask"),  # a PIL left in the record would break collate
            albumentations.Resize(height=16, width=16),  # ONE joint draw over image AND mask
        ]
        for op in chain:
            applied = _apply_op(record, op)
            assert applied is not None  # no op in this chain drops a record
            record = applied

        assert set(record) == {"image", "mask"}
        assert record["image"].shape == (16, 16, 3)
        assert record["mask"].shape == (16, 16)
        assert set(np.unique(np.asarray(record["mask"]))) <= {0, 1, 2}

        batch = collate_records([record, record])
        assert batch["image"].shape == (2, 16, 16, 3)
        assert batch["mask"].shape == (2, 16, 16)
        # The dtype the per-pixel CrossEntropy contract needs — named by the CALLER, because
        # albumentations casts the mask to int32 on the way past.
        assert batch_tensor(batch, "mask", dtype=torch.int64).dtype == torch.int64

    def test_a_normalize_leaves_the_mask_alone(self) -> None:
        """Albumentations applies an image-only transform to the image only — pinned, because
        a Normalize that reached the mask would turn class ids into floats silently."""
        albumentations = pytest.importorskip("albumentations")
        record = {
            "image": Image(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)),
            "mask": Mask(np.random.randint(0, 3, (8, 8)).astype(np.int64)),
        }
        out = _apply_op(record, albumentations.Normalize(mean=[0.5] * 3, std=[0.5] * 3))
        assert out is not None
        assert out["image"].dtype == np.float32
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(record["mask"]))


# --------------------------------------------------------------------------- #
# num_mask_classes — the per-pixel twin of num_classes
# --------------------------------------------------------------------------- #
class TestNumMaskClasses:
    def test_it_is_the_largest_id_anywhere_plus_one(self) -> None:
        from recordstream import num_mask_classes

        records = [
            {"mask": Mask(np.array([[0, 1], [1, 0]], dtype=np.int64))},
            {"mask": Mask(np.array([[0, 2], [2, 2]], dtype=np.int64))},  # 2 appears only here
        ]
        assert num_mask_classes(records) == 3

    def test_a_class_in_the_last_record_still_sizes_the_head(self) -> None:
        """The walk covers EVERY record — a rare class must not be missed by an early exit."""
        from recordstream import num_mask_classes

        records = [{"mask": Mask(np.zeros((2, 2), dtype=np.int64))} for _ in range(50)]
        records[-1] = {"mask": Mask(np.array([[0, 7], [0, 0]], dtype=np.int64))}
        assert num_mask_classes(records) == 8

    def test_the_key_is_configurable(self) -> None:
        from recordstream import num_mask_classes

        assert num_mask_classes([{"target": Mask(np.array([[0, 4]], dtype=np.int64))}], key="target") == 5

    def test_an_empty_source_raises_rather_than_guessing(self) -> None:
        from recordstream import num_mask_classes

        with pytest.raises(ValueError, match="no usable"):
            num_mask_classes([])

    def test_a_record_without_a_mask_raises(self) -> None:
        """Silently skipping it would under-count the head against a partially-labelled set."""
        from recordstream import num_mask_classes

        with pytest.raises(ValueError, match="no 'mask' value"):
            num_mask_classes([{"mask": Mask(np.zeros((2, 2), dtype=np.int64))}, {"mask": None}])
        with pytest.raises(ValueError, match="no 'mask' value"):
            num_mask_classes([{"image": Image(np.zeros((2, 2, 3), np.uint8))}])

    def test_num_classes_still_refuses_an_array_target(self) -> None:
        """The reason this is a separate function: the scalar guard must stay strict.

        A classification run accidentally handed masks has to fail loudly rather than report
        whatever the first array's `.item()` would have been.
        """
        from recordstream import num_classes

        with pytest.raises((TypeError, ValueError)):
            num_classes([{"class": Mask(np.array([[0, 1], [1, 0]], dtype=np.int64))}])


@contextmanager
def _captured_warnings() -> Iterator[List[str]]:
    """Collect loggair WARNING records emitted inside the block.

    `caplog` cannot see these: loggair is loguru, which does not propagate to stdlib logging.
    Neither can `capfd` alone — the sink is ENQUEUED, so the write lands on another thread after
    the assertion runs (this looked exactly like "the warning never fired"). `logger.complete()`
    is the deterministic flush the workspace mandates instead of sleeping.
    """
    from loguru import logger

    collected: List[str] = []
    sink_id = logger.add(lambda message: collected.append(str(message)), level="WARNING")
    try:
        yield collected
        logger.complete()
    finally:
        logger.remove(sink_id)


class TestBoxesKnowTheirFrame:
    """`canvas` is the raster a `Boxes`' boxes are stated in — so every op that makes or
    re-frames one records it, and the op that moves pixels ALONE says so.

    Before this, only the coupled resize set `canvas`, which meant the frame was knowable
    exactly when it was least needed (a resize that already moved the boxes correctly) and
    unknown in the chain where boxes and pixels can actually drift apart.
    """

    def test_coco_boxes_record_the_image_they_annotate(self) -> None:
        from recordstream.ops.target import CocoToTorchVisionDetection

        record = {
            "image": Image(np.zeros((300, 400, 3), dtype="uint8")),
            "objects": Label({"bbox": [[10.0, 10.0, 20.0, 20.0]], "category": [1]}),
        }
        out = CocoToTorchVisionDetection(field="objects")(record)
        assert out["target"].canvas == (300, 400), "the annotation's frame is the image's"

    def test_mask_derived_boxes_record_the_mask(self) -> None:
        from recordstream.ops.target import MasksToDetectionBoxes

        mask = np.zeros((64, 96), dtype="int64")
        mask[10:20, 30:40] = 1
        out = MasksToDetectionBoxes()({"mask": Mask(mask)})
        assert out["target"].canvas == (64, 96), "the boxes were derived FROM this raster"

    def test_a_record_with_no_image_still_works(self) -> None:
        """`None` is an ordinary answer — the lookup failing must not fail the op."""
        from recordstream.ops.target import CocoToTorchVisionDetection

        out = CocoToTorchVisionDetection(field="objects")(
            {"objects": Label({"bbox": [[1.0, 2.0, 3.0, 4.0]], "category": [0]})}
        )
        assert out["target"].canvas is None

    def test_a_regions_box_array_is_never_mistaken_for_a_raster(self) -> None:
        """The narrow lookup's whole point: an `[N, 4]` box array is 2-D and must not be read
        as an N x 4 image, which a generic first-array search would do confidently."""
        from recordstream.ops.target import CocoToTorchVisionDetection

        record = {
            "objects": Label({"bbox": [[1.0, 2.0, 3.0, 4.0]], "category": [0]}),
            "other": Boxes(boxes=np.zeros((7, 4), dtype="float32"), labels=np.zeros((7,), dtype="int64")),
        }
        assert CocoToTorchVisionDetection(field="objects")(record)["target"].canvas is None

    def test_an_EMPTY_target_is_re_framed_too(self) -> None:
        """A negative example must not be the one record whose frame is unknown."""
        from recordstream.ops.target import ResizeDetection

        record = {
            "image": Image(np.zeros((100, 100, 3), dtype="uint8")),
            "target": Boxes(boxes=np.zeros((0, 4), dtype="float32"), labels=np.zeros((0,), dtype="int64")),
        }
        out = ResizeDetection(width=64, height=32)(record)
        assert out["target"].canvas == (32, 64)

    def test_an_image_only_resize_WARNS_when_it_desyncs_boxes(self) -> None:
        from recordstream.ops.image import ConvertToImage

        record = {
            "image": Image(np.zeros((200, 200, 3), dtype="uint8")),
            "target": Boxes(boxes=np.array([[10.0, 10.0, 50.0, 50.0]]), labels=np.array([1])),
        }
        op = ConvertToImage(field="image", width=64, height=64)
        with _captured_warnings() as warnings:
            op(record)
        assert len(warnings) == 1
        assert "ResizeDetection" in warnings[0] and "PIXELS ONLY" in warnings[0]
        assert "(200, 200) -> (64, 64)" in warnings[0], "the message states both rasters"
        # Once per op instance: the message is about the CONFIGURATION, not about this record.
        with _captured_warnings() as second:
            op(record)
        assert second == []

    def test_no_warning_without_boxes_or_without_a_resize(self) -> None:
        from recordstream.ops.image import ConvertToImage

        image = Image(np.zeros((200, 200, 3), dtype="uint8"))
        boxes = Boxes(boxes=np.array([[1.0, 2.0, 3.0, 4.0]]), labels=np.array([1]))
        with _captured_warnings() as warnings:
            ConvertToImage(field="image", width=64, height=64)({"image": image})  # resized, no boxes
            ConvertToImage(field="image", max_size=999)({"image": image, "target": boxes})  # boxes, no resize
        assert warnings == []
