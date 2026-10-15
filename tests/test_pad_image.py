"""``PadImage`` — a picture padded to a fixed size with a constant, rows at the bottom and columns at the right.

Its boxes are absolute pixels from the top-left corner, so a pad below and to the right leaves every box and outline
where it was; only the boxes' ``canvas`` (the raster they are stated in) grows to the new size.
"""

from typing import Any

import numpy as np
import pytest
import torch

from recordstream import Boxes, Image
from recordstream.ops.image import PadImage


def _picture(height: int = 3, width: int = 4, channels: int = 0) -> np.ndarray:
    """A uint8 picture whose pixel (r, c) is ``10 * r + c + 1`` — never 0, so a padded pixel is told from a real one."""
    cells = (np.arange(height)[:, None] * 10 + np.arange(width)[None, :] + 1).astype(np.uint8)
    return np.repeat(cells[..., None], channels, axis=2) if channels else cells


def _record(pixels: Any, boxes: Any = ((1.0, 0.0, 3.0, 2.0),)) -> dict:
    target = Boxes(
        boxes=np.asarray(boxes, dtype=np.float32),
        labels=np.arange(len(boxes)),
        canvas=(pixels.shape[0], pixels.shape[1]),
        extras={"polygons": [[[[1.0, 0.0], [3.0, 0.0], [3.0, 2.0]]]]},
    )
    return {"image": Image(pixels), "target": target, "tile": {"row": 0}}


class TestThePad:
    def test_rows_are_added_at_the_bottom_with_the_value_and_the_picture_is_where_it_was(self) -> None:
        out = PadImage(height=6)(_record(_picture()))
        pixels = np.asarray(out["image"])
        assert pixels.shape == (6, 4) and pixels.dtype == np.uint8
        assert np.array_equal(pixels[:3], _picture()), "the picture's own pixels, not resampled"
        assert np.all(pixels[3:] == 0), "the added rows are black"

    def test_columns_are_added_at_the_right(self) -> None:
        pixels = np.asarray(PadImage(width=7, value=255)(_record(_picture()))["image"])
        assert pixels.shape == (3, 7) and np.array_equal(pixels[:, :4], _picture())
        assert np.all(pixels[:, 4:] == 255)

    def test_both_axes_and_a_channel_axis(self) -> None:
        pixels = np.asarray(PadImage(height=5, width=6)(_record(_picture(channels=3)))["image"])
        assert pixels.shape == (5, 6, 3)
        assert np.array_equal(pixels[:3, :4], _picture(channels=3))
        assert np.all(pixels[3:] == 0) and np.all(pixels[:, 4:] == 0)

    def test_a_float_picture_keeps_its_dtype(self) -> None:
        picture = _picture().astype(np.float32) / 255.0
        pixels = np.asarray(PadImage(height=4, value=0.5)(_record(picture))["image"])
        assert pixels.dtype == np.float32 and np.all(pixels[3] == 0.5)

    def test_zero_keeps_an_axis_and_a_picture_already_the_size_is_unchanged(self) -> None:
        record = _record(_picture())
        out = PadImage(height=3, width=0)(record)
        assert np.array_equal(np.asarray(out["image"]), _picture())
        assert out["target"].canvas == (3, 4)

    def test_the_item_type_and_the_rest_of_the_record_come_along(self) -> None:
        out = PadImage(height=6)(_record(_picture()))
        assert isinstance(out["image"], Image) and out["tile"] == {"row": 0}


class TestTheBoxesStay:
    def test_the_boxes_and_outlines_are_unmoved_and_the_canvas_is_the_new_size(self) -> None:
        record = _record(_picture(), boxes=[(1.0, 0.0, 3.0, 2.0), (0.0, 1.0, 2.0, 3.0)])
        out = PadImage(height=8, width=5)(record)
        target = out["target"]
        assert np.array_equal(np.asarray(target.boxes), np.asarray(record["target"].boxes))
        assert np.array_equal(np.asarray(target.labels), [0, 1])
        assert target.extras == record["target"].extras
        assert target.canvas == (8, 5)

    def test_a_record_without_boxes_is_padded_alone(self) -> None:
        out = PadImage(height=5)({"image": Image(_picture())})
        assert np.asarray(out["image"]).shape == (5, 4) and "target" not in out

    def test_torch_boxes_stay_torch(self) -> None:
        record = _record(_picture())
        record["target"] = Boxes(boxes=torch.tensor([[1.0, 0.0, 3.0, 2.0]]), labels=torch.tensor([0]), canvas=(3, 4))
        target = PadImage(height=5)(record)["target"]
        assert isinstance(target.boxes, torch.Tensor) and target.canvas == (5, 4)

    def test_the_keys_are_settings(self) -> None:
        record = {"picture": Image(_picture()), "truth": _record(_picture())["target"]}
        out = PadImage(height=4, input_key="picture", target_key="truth")(record)
        assert np.asarray(out["picture"]).shape == (4, 4) and out["truth"].canvas == (4, 4)


class TestRefusals:
    def test_a_picture_larger_than_the_size_is_refused_never_cut(self) -> None:
        with pytest.raises(ValueError, match=r"PadImage: the picture at 'image' is 3 x 4 \(height x width\), larger"):
            PadImage(height=2)(_record(_picture()))
        with pytest.raises(ValueError, match="larger than the 3 x 3 it is padded to"):
            PadImage(height=3, width=3)(_record(_picture()))

    def test_a_missing_picture_is_refused_naming_the_keys(self) -> None:
        with pytest.raises(ValueError, match=r"PadImage: no 'image' in the record \(keys: \['target'\]\)"):
            PadImage(height=4)({"target": Boxes()})

    def test_a_channels_first_picture_is_refused(self) -> None:
        item = Image(np.zeros((3, 4, 5), dtype=np.uint8), layout="CHW")
        with pytest.raises(ValueError, match="PadImage: a CHW picture"):
            PadImage(height=8)({"image": item})

    def test_a_negative_size_is_refused_when_the_op_is_built(self) -> None:
        with pytest.raises(ValueError, match=r"PadImage: height and width are sizes in pixels \(0 keeps"):
            PadImage(height=-1)
