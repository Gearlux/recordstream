"""A traced node's entry as a PICTURE — ``recordstream.flow.picture`` and ``Tracer.picture``.

The trace answers summaries (type, shape, dtype, range): enough to see that a step ran, not what it made. An entry
that is a raster — an image, a mask, any 2-D array of numbers such as a spectrogram — can also be asked for as a PNG,
and its summary says so (``picture: true``). The boxes the same snapshot holds for a raster of that size are drawn
over it.
"""

import io
from typing import Any, Tuple

import numpy as np
import pytest
from confluid import configurable
from PIL import Image as PILImage

from recordstream import Boxes, Image, Label, Mask, Record, Stream
from recordstream.flow.picture import picture, pictured
from recordstream.flow.trace import Tracer


def _read(png: bytes) -> PILImage.Image:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return PILImage.open(io.BytesIO(png))


def _ramp(height: int = 8, width: int = 32) -> np.ndarray:
    """Rows of a ramp from -40 to -10 (decibels, say): the leftmost column is the lowest value, the rightmost the
    highest."""
    return np.tile(np.linspace(-40.0, -10.0, width, dtype=np.float32), (height, 1))


class TestWhatHasAPicture:
    @pytest.mark.parametrize(
        "value",
        [
            Image(np.zeros((4, 6, 3), dtype=np.uint8)),
            Image(np.zeros((3, 4, 6), dtype=np.uint8), layout="CHW"),
            Image(np.zeros((4, 6), dtype=np.float32)),
            Mask(np.zeros((4, 6), dtype=bool)),
            np.zeros((4, 6), dtype=np.float64),
            np.zeros((4, 6, 1), dtype=np.uint8),
        ],
    )
    def test_a_raster_has_one(self, value: Any) -> None:
        assert pictured(value)

    @pytest.mark.parametrize(
        "value",
        [
            Label(value="cat"),
            "text",
            3.5,
            np.zeros(16, dtype=np.float32),  # one row of numbers is a signal, not a raster
            np.zeros((4, 6), dtype=np.complex64),
            np.zeros((4, 6, 7), dtype=np.float32),  # seven planes are no picture
            np.zeros((0, 6), dtype=np.float32),
            Boxes(boxes=[[0, 0, 1, 1]]),
        ],
    )
    def test_anything_else_has_none_and_asking_says_so(self, value: Any) -> None:
        assert not pictured(value)
        with pytest.raises(ValueError, match="has no picture"):
            picture(value)


class TestThePicture:
    def test_an_rgb_image_is_drawn_as_it_is(self) -> None:
        pixels = np.zeros((4, 6, 3), dtype=np.uint8)
        pixels[1, 2] = (255, 0, 0)
        drawn = _read(picture(Image(pixels)))
        assert drawn.size == (6, 4)
        assert drawn.convert("RGB").getpixel((2, 1)) == (255, 0, 0)

    def test_a_chw_image_is_drawn_upright(self) -> None:
        pixels = np.zeros((3, 4, 6), dtype=np.uint8)
        pixels[1, 3, 5] = 255  # the green plane, row 3, column 5
        drawn = _read(picture(Image(pixels, layout="CHW")))
        assert drawn.size == (6, 4)
        assert drawn.convert("RGB").getpixel((5, 3)) == (0, 255, 0)

    def test_an_array_of_numbers_is_grey_from_its_lowest_to_its_highest_value(self) -> None:
        drawn = _read(picture(_ramp())).convert("L")
        assert drawn.size == (32, 8)
        grey = np.asarray(drawn)
        assert grey[0, 0] == 0 and grey[0, 31] == 255
        assert 120 <= int(grey[0, 16]) <= 136

    def test_values_that_are_not_numbers_never_stretch_the_range(self) -> None:
        """The one quantizer's rule (``normalize_to_uint8``): not-a-number is the lowest, an infinity its bound."""
        values = _ramp()
        values[0, 0], values[0, 1] = np.nan, np.inf
        drawn = _read(picture(values)).convert("L")
        assert drawn.getpixel((0, 0)) == 0 and drawn.getpixel((1, 0)) == 255
        assert drawn.getpixel((0, 3)) == 0 and drawn.getpixel((31, 3)) == 255

    def test_one_value_everywhere_is_drawn_and_is_black(self) -> None:
        assert _read(picture(np.full((4, 6), 7.0))).convert("L").getextrema() == (0, 0)

    def test_a_mask_is_black_and_white(self) -> None:
        mask = np.zeros((4, 6), dtype=bool)
        mask[2, 3] = True
        drawn = _read(picture(Mask(mask))).convert("L")
        assert drawn.getpixel((3, 2)) == 255 and drawn.getpixel((0, 0)) == 0

    def test_a_width_makes_a_wide_picture_narrower_and_keeps_its_shape(self) -> None:
        assert _read(picture(_ramp(height=8, width=32), width=16)).size == (16, 4)

    def test_a_width_never_makes_a_picture_larger(self) -> None:
        assert _read(picture(_ramp(height=8, width=32), width=400)).size == (32, 8)

    def test_a_width_below_one_pixel_is_refused(self) -> None:
        with pytest.raises(ValueError, match="width must be 1 or more"):
            picture(_ramp(), width=0)

    def test_boxes_are_drawn_over_it_and_follow_the_width(self) -> None:
        pixels = np.zeros((40, 80, 3), dtype=np.uint8)
        boxes = Boxes(boxes=[[20, 10, 60, 30]], canvas=(40, 80))
        plain, boxed = _read(picture(Image(pixels))), _read(picture(Image(pixels), boxes=[boxes]))
        assert plain.convert("RGB").getpixel((20, 20)) == (0, 0, 0)
        assert boxed.convert("RGB").getpixel((20, 20)) != (0, 0, 0)  # the left edge
        assert boxed.convert("RGB").getpixel((40, 20)) == (0, 0, 0)  # inside: an outline, never a fill
        half = _read(picture(Image(pixels), boxes=[boxes], width=40)).convert("RGB")
        assert half.size == (40, 20) and half.getpixel((10, 10)) != (0, 0, 0)

    def test_boxes_held_as_a_tensor_are_drawn_too(self) -> None:
        torch = pytest.importorskip("torch")
        pixels = np.zeros((40, 80, 3), dtype=np.uint8)
        boxes = Boxes(boxes=torch.tensor([[20.0, 10.0, 60.0, 30.0]]), canvas=(40, 80))
        assert _read(picture(Image(pixels), boxes=[boxes])).convert("RGB").getpixel((20, 20)) != (0, 0, 0)


@configurable(category="op", group="test")
class PictureBoxIt:
    """Adds one box around the image's centre — the frame the boxes are stated in is the image's."""

    consumes = {"image": "Image"}
    produces = {"target": "Boxes"}

    def __call__(self, record: Record) -> Record:
        height, width = np.asarray(record["image"]).shape[:2]
        box = [width // 4, height // 4, 3 * width // 4, 3 * height // 4]
        return {**record, "target": Boxes(boxes=[box], canvas=(height, width))}


def _traced(shape: Tuple[int, int] = (40, 80)) -> Tracer:
    record: Record = {"image": Image(np.zeros((*shape, 3), dtype=np.uint8)), "class": Label(value="cat")}
    return Tracer(Stream(ops=[PictureBoxIt()])).run(record)


class TestTheTracer:
    def test_a_summary_says_which_entries_have_a_picture(self) -> None:
        (node,) = _traced().to_dict()["nodes"]
        assert node["output"]["image"]["picture"] is True
        assert "picture" not in node["output"]["class"] and "picture" not in node["output"]["target"]

    def test_an_entry_is_answered_as_a_png_of_its_own_size(self) -> None:
        tracer = _traced()
        (name,) = tracer.names
        assert _read(tracer.picture(name, "image")).size == (80, 40)
        assert _read(tracer.picture(name, "image", width=20)).size == (20, 10)

    def test_the_boxes_of_the_same_snapshot_are_drawn_and_the_input_has_none(self) -> None:
        tracer = _traced()
        (name,) = tracer.names
        after = _read(tracer.picture(name, "image", "output")).convert("RGB")
        before = _read(tracer.picture(name, "image", "input")).convert("RGB")
        assert after.getpixel((20, 20)) != (0, 0, 0)  # the box's left edge: x = 80 // 4
        assert before.getextrema() == ((0, 0), (0, 0), (0, 0))  # the input held no boxes yet

    def test_boxes_stated_in_another_frame_are_not_drawn(self) -> None:
        tracer = _traced()
        (name,) = tracer.names
        tracer.value(name, "target").canvas = (400, 800)
        assert _read(tracer.picture(name, "image")).convert("RGB").getextrema() == ((0, 0), (0, 0), (0, 0))

    def test_an_entry_without_a_picture_is_refused_naming_it(self) -> None:
        tracer = _traced()
        (name,) = tracer.names
        with pytest.raises(ValueError, match=r"the output entry 'class' \(Label\) has no picture"):
            tracer.picture(name, "class")

    def test_an_entry_that_is_not_there_is_refused_as_value_refuses_it(self) -> None:
        tracer = _traced()
        (name,) = tracer.names
        with pytest.raises(KeyError, match="has no output entry 'nosuch'"):
            tracer.picture(name, "nosuch")
