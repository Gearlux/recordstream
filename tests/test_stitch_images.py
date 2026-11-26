"""``StitchImages`` — several pictures of a record laid out as ONE picture: in a row, a column or a grid.

Each panel keeps its own pixels (never resampled); a row is as tall as its tallest panel and a column as wide as its
widest, a smaller panel in its cell's top-left corner on the background; ``titles`` puts each panel's name in a strip
above it. A field the record lacks is left out, and a record holding none of them passes through without the output
entry.
"""

from typing import Dict, List, Optional

import numpy as np
import pytest

from recordstream import Image
from recordstream.ops.image import StitchImages


def _panel(height: int, width: int, value: int) -> Image:
    return Image(np.full((height, width, 3), value, dtype=np.uint8), layout="HWC")


def _record(**panels: Image) -> Dict[str, object]:
    return {"other": 7, **panels}


def _pixels(record: Dict[str, object], key: str = "image") -> np.ndarray:
    return np.asarray(record[key])


class TestTheLayouts:
    def test_a_row_puts_the_panels_side_by_side_with_the_gap_between_them(self) -> None:
        out = StitchImages(fields=["a", "b"], layout="row", gap=4)(_record(a=_panel(10, 20, 50), b=_panel(10, 30, 200)))
        pixels = _pixels(out)
        assert pixels.shape == (10, 20 + 4 + 30, 3) and pixels.dtype == np.uint8
        assert np.all(pixels[:, :20] == 50), "the first panel's own pixels, not resampled"
        assert np.all(pixels[:, 20:24] == 0), "the gap is the background"
        assert np.all(pixels[:, 24:] == 200)
        assert out["other"] == 7 and "a" in out and "b" in out, "the rest of the record passes through"

    def test_a_column_stacks_the_panels_top_to_bottom(self) -> None:
        out = StitchImages(fields=["a", "b"], layout="column", gap=2)(_record(a=_panel(5, 8, 50), b=_panel(6, 8, 200)))
        pixels = _pixels(out)
        assert pixels.shape == (5 + 2 + 6, 8, 3)
        assert np.all(pixels[:5] == 50) and np.all(pixels[7:] == 200)

    def test_a_grid_of_three_is_two_by_two_filled_row_by_row(self) -> None:
        panels = {name: _panel(4, 6, value) for name, value in (("a", 10), ("b", 20), ("c", 30))}
        pixels = _pixels(StitchImages(fields=["a", "b", "c"], layout="grid", gap=0)(_record(**panels)))
        assert pixels.shape == (8, 12, 3)
        assert np.all(pixels[:4, :6] == 10) and np.all(pixels[:4, 6:] == 20) and np.all(pixels[4:, :6] == 30)
        assert np.all(pixels[4:, 6:] == 0), "the empty cell is the background"

    def test_a_smaller_panel_sits_in_the_top_left_of_its_cell(self) -> None:
        pixels = _pixels(
            StitchImages(fields=["a", "b"], layout="row", gap=0)(_record(a=_panel(10, 4, 50), b=_panel(6, 4, 200)))
        )
        assert pixels.shape == (10, 8, 3)
        assert np.all(pixels[:6, 4:] == 200) and np.all(pixels[6:, 4:] == 0)

    def test_a_grey_panel_is_laid_out_as_rgb(self) -> None:
        grey = Image(np.full((3, 3), 90, dtype=np.uint8))
        pixels = _pixels(StitchImages(fields=["a", "b"], gap=0)(_record(a=grey, b=_panel(3, 3, 10))))
        assert pixels.shape == (3, 6, 3) and np.all(pixels[:, :3] == 90)

    def test_the_background_is_a_colour_name(self) -> None:
        pixels = _pixels(
            StitchImages(fields=["a", "b"], gap=2, background="white")(_record(a=_panel(2, 2, 0), b=_panel(2, 2, 0)))
        )
        assert np.all(pixels[:, 2:4] == 255)


class TestTitles:
    def test_each_panel_gets_a_strip_above_it_holding_its_name(self) -> None:
        plain = _pixels(StitchImages(fields=["a", "b"], gap=0)(_record(a=_panel(20, 60, 0), b=_panel(20, 60, 0))))
        titled_op = StitchImages(fields=["a", "b"], gap=0, titles=["first", "second"])
        titled = _pixels(titled_op(_record(a=_panel(20, 60, 0), b=_panel(20, 60, 0))))
        strip = titled.shape[0] - plain.shape[0]
        assert strip > 0 and titled.shape[1] == plain.shape[1]
        assert titled[:strip, :60].max() > 0 and titled[:strip, 60:].max() > 0, "each strip holds lettering"
        assert np.all(titled[strip:] == 0), "the panels sit below their strips, unchanged"

    def test_titles_must_name_every_field(self) -> None:
        op = StitchImages(fields=["a", "b"], titles=["only one"])
        with pytest.raises(ValueError, match=r"StitchImages: 1 title\(s\) for 2 field\(s\)"):
            op(_record(a=_panel(1, 1, 0)))


class TestMissingPanels:
    def test_a_field_the_record_lacks_is_left_out(self) -> None:
        pixels = _pixels(
            StitchImages(fields=["a", "missing", "b"], gap=0)(_record(a=_panel(2, 3, 1), b=_panel(2, 3, 2)))
        )
        assert pixels.shape == (2, 6, 3)

    def test_a_record_holding_none_of_them_has_no_output(self) -> None:
        record = _record()
        out = StitchImages(fields=["a", "b"], output="stitched")(record)
        assert "stitched" not in out and out == record

    def test_a_field_that_is_no_picture_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"StitchImages: field 'other' is int, not an Image"):
            StitchImages(fields=["other"])(_record())


class TestTheSettings:
    @pytest.mark.parametrize("fields", [None, []])
    def test_no_fields_is_refused_when_it_runs_not_when_it_is_built(self, fields: Optional[List[str]]) -> None:
        op = StitchImages(fields=fields)
        with pytest.raises(ValueError, match=r"StitchImages: fields is empty"):
            op(_record(a=_panel(1, 1, 0)))

    def test_an_unknown_layout_is_refused_by_its_closed_set(self) -> None:
        with pytest.raises(ValueError, match=r"Input should be 'row', 'column' or 'grid'"):
            StitchImages(fields=["a"], layout="diagonal")  # type: ignore[arg-type]

    def test_a_negative_gap_is_refused_when_it_runs(self) -> None:
        op = StitchImages(fields=["a"], gap=-1)
        with pytest.raises(ValueError, match=r"StitchImages: gap must be 0 or more"):
            op(_record(a=_panel(1, 1, 0)))

    def test_it_builds_with_no_arguments(self) -> None:
        assert StitchImages().fields is None

    def test_the_output_names_the_entry_written(self) -> None:
        out = StitchImages(fields=["a"], output="stitched")(_record(a=_panel(2, 2, 9)))
        assert np.all(np.asarray(out["stitched"]) == 9) and "image" not in out
