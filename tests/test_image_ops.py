"""Tests for :mod:`dataflux.ops.image` — generic value→image conversion.

``ConvertToImageOp`` is the generic image-conversion op (normalize → colormap →
optional flip → resize), and ``value_to_image`` / ``sample_to_image`` back it
(and FluxStudio's preview). The signal-specific overlay drawing lives in
waivefront (``RenderOverlaysOp``) and is tested there.
"""

from typing import get_args

import numpy as np
import pytest
import torch
from PIL import Image

from dataflux.ops.image import (
    COLORMAPS,
    TEXT_POSITIONS,
    Colormap,
    ConvertToImageOp,
    NormalizeToUint8Op,
    TextPosition,
    _apply_colormap,
    array_histogram,
    channel_count,
    draw_text,
    sample_to_image,
    select_channel,
    value_to_image,
)
from dataflux.sample import Sample


def _sample(value: object) -> Sample:
    return Sample(input=value, target=None, metadata={})


# ---------------------------------------------------------------------------
# ConvertToImageOp
# ---------------------------------------------------------------------------


def test_convert_2d_map_to_exact_size_pil_and_publishes_dims() -> None:
    arr = np.linspace(0.0, 1.0, 64 * 32, dtype=np.float32).reshape(64, 32)
    out = ConvertToImageOp(colormap="gray", width=128, height=256)(_sample(arr))
    assert isinstance(out.input, Image.Image)
    assert out.input.size == (128, 256)
    assert out.meta["image_width_px"] == 128
    assert out.meta["image_height_px"] == 256


def test_convert_max_size_path_bounds_longest_side() -> None:
    out = ConvertToImageOp(max_size=256)(_sample(np.zeros((1000, 400), dtype=np.float32)))
    assert max(out.input.size) == 256
    # Dims are published from the actual rendered raster.
    assert out.meta["image_width_px"] == out.input.width
    assert out.meta["image_height_px"] == out.input.height


def test_convert_flip_vertical_mirrors_top_to_bottom() -> None:
    m = np.zeros((10, 4), dtype=np.float32)
    m[0, :] = 1.0  # row 0 bright
    noflip = np.asarray(ConvertToImageOp(colormap="gray", flip_vertical=False)(_sample(m)).input.convert("L"))
    flip = np.asarray(ConvertToImageOp(colormap="gray", flip_vertical=True)(_sample(m)).input.convert("L"))
    assert noflip[0].mean() > noflip[-1].mean(), "no-flip: row 0 stays at the top"
    assert flip[-1].mean() > flip[0].mean(), "flip: row 0 moves to the bottom"


def test_convert_colormap_gray_is_monochrome_color_is_not() -> None:
    arr = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    gray = np.asarray(ConvertToImageOp(colormap="gray", width=16, height=16)(_sample(arr)).input)
    color = np.asarray(ConvertToImageOp(colormap="viridis", width=16, height=16)(_sample(arr)).input)
    assert np.array_equal(gray[..., 0], gray[..., 1]) and np.array_equal(gray[..., 1], gray[..., 2])
    assert not np.array_equal(color[..., 0], color[..., 1])


def test_convert_accepts_torch_chw_tensor() -> None:
    out = ConvertToImageOp(width=64, height=48)(_sample(torch.rand(3, 100, 200)))
    assert isinstance(out.input, Image.Image)
    assert out.input.size == (64, 48)


def test_convert_accepts_pil_passthrough() -> None:
    out = ConvertToImageOp(width=20, height=20)(_sample(Image.new("RGB", (8, 8), color=(10, 20, 30))))
    assert isinstance(out.input, Image.Image)
    assert out.input.size == (20, 20)


# ---------------------------------------------------------------------------
# value_to_image / sample_to_image — generic, modality-agnostic preview
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_input",
    [
        lambda: Image.new("L", (12, 10)),  # PIL greyscale
        lambda: Image.new("RGB", (12, 10)),  # PIL RGB
        lambda: np.random.rand(10, 12).astype(np.float32),  # 2-D float -> colormap
        lambda: (np.random.rand(10, 12, 3) * 255).astype(np.uint8),  # 3-D HWC uint8
        lambda: np.random.rand(3, 10, 12).astype(np.float32),  # 3-D CHW -> transposed
        lambda: np.random.rand(10, 12) > 0.5,  # boolean mask
        lambda: np.random.rand(10, 12, 1).astype(np.float32),  # singleton channel
        lambda: np.random.rand(10, 12, 4).astype(np.float32),  # RGBA -> drop alpha
        lambda: torch.rand(3, 10, 12),  # torch CHW tensor
    ],
)
def test_sample_to_image_returns_hwc_uint8_rgb(make_input) -> None:  # type: ignore[no-untyped-def]
    img = sample_to_image(Sample(input=make_input()))
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3


def test_sample_to_image_falls_back_to_text_for_non_array() -> None:
    img = sample_to_image(Sample(input=[(0, 1, 2, 3), (4, 5, 6, 7)]))
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3


def test_sample_to_image_bounds_longest_side() -> None:
    img = sample_to_image(Sample(input=np.zeros((2000, 500), dtype=np.float32)), max_size=256)
    assert max(img.shape[:2]) <= 256


def test_sample_to_image_gray_is_monochrome_color_is_not() -> None:
    arr = np.linspace(0.0, 1.0, 100).reshape(10, 10).astype(np.float32)
    gray = sample_to_image(Sample(input=arr), colormap="gray")
    color = sample_to_image(Sample(input=arr), colormap="viridis")
    assert np.array_equal(gray[..., 0], gray[..., 1])
    assert not np.array_equal(color[..., 0], color[..., 1])


def test_sample_to_image_flat_array_is_all_zero() -> None:
    img = sample_to_image(Sample(input=np.full((8, 8), 5.0, dtype=np.float32)), colormap="gray")
    assert int(img.max()) == 0


def test_value_to_image_renders_an_arbitrary_value() -> None:
    img = value_to_image(np.eye(12, dtype=bool), colormap="gray")
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3


def test_sample_to_image_delegates_to_value_to_image() -> None:
    arr = np.linspace(0.0, 1.0, 64).reshape(8, 8).astype(np.float32)
    assert np.array_equal(sample_to_image(Sample(input=arr)), value_to_image(arr))


# ---------------------------------------------------------------------------
# NormalizeToUint8Op
# ---------------------------------------------------------------------------


def test_normalize_to_uint8_auto_minmax_spans_full_range() -> None:
    arr = np.linspace(-3.0, 7.0, 100, dtype=np.float32).reshape(10, 10)
    out = NormalizeToUint8Op.normalize_to_uint8(arr)
    assert out.dtype == np.uint8
    assert int(out.min()) == 0 and int(out.max()) == 255


def test_normalize_to_uint8_fixed_range_clamps_outside() -> None:
    arr = np.array([[-10.0, 0.0, 5.0, 20.0]], dtype=np.float32)
    out = NormalizeToUint8Op.normalize_to_uint8(arr, vmin=0.0, vmax=10.0)
    # -10 and 0 clamp to 0; 5 is mid (≈127); 20 clamps to 255.
    assert list(out.ravel()) == [0, 0, 127, 255]


def test_normalize_to_uint8_flat_array_is_all_zero() -> None:
    out = NormalizeToUint8Op.normalize_to_uint8(np.full((4, 4), 9.0, dtype=np.float32))
    assert int(out.max()) == 0


def test_normalize_to_uint8_handles_non_finite() -> None:
    arr = np.array([[0.0, np.nan, np.inf, -np.inf, 4.0]], dtype=np.float32)
    out = NormalizeToUint8Op.normalize_to_uint8(arr)
    # NaN/-inf fold to the low bound (0), +inf to the high bound (4 → 255).
    assert out[0, 0] == 0 and out[0, 1] == 0 and out[0, 3] == 0
    assert out[0, 2] == 255 and out[0, 4] == 255


def test_normalize_to_uint8_op_converts_sample_input() -> None:
    arr = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    out = NormalizeToUint8Op()(Sample(input=arr))
    assert out.input.dtype == np.uint8 and out.input.shape == (8, 8)


def test_normalize_to_uint8_op_accepts_torch_tensor() -> None:
    out = NormalizeToUint8Op()(Sample(input=torch.linspace(0, 1, 16).reshape(4, 4)))
    assert isinstance(out.input, np.ndarray) and out.input.dtype == np.uint8


def test_normalize_to_uint8_op_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="vmin must be < vmax"):
        NormalizeToUint8Op(vmin=10.0, vmax=1.0)(Sample(input=np.zeros((2, 2), dtype=np.float32)))


# ---------------------------------------------------------------------------
# Colormap closed-Literal contract
# ---------------------------------------------------------------------------


def test_colormaps_tuple_is_the_literal_set() -> None:
    assert COLORMAPS == get_args(Colormap)
    assert "viridis" in COLORMAPS and "gray" in COLORMAPS


def test_every_colormap_in_the_literal_set_renders() -> None:
    spec_u8 = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8)
    for cmap in COLORMAPS:
        img = _apply_colormap(spec_u8, cmap)
        assert img.mode == "RGB" and img.size == (8, 8)


# ---------------------------------------------------------------------------
# select_channel — reduce an arbitrary array/tensor to a 2-D map for one channel
# ---------------------------------------------------------------------------


def test_select_channel_2d_passthrough() -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = select_channel(arr, channel=-1)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out, arr)


def test_select_channel_1d_becomes_strip() -> None:
    out = select_channel(np.arange(5, dtype=np.float32))
    assert out.shape == (1, 5)


def test_select_channel_scalar_becomes_cell() -> None:
    assert select_channel(np.float32(3.0)).shape == (1, 1)


def test_select_channel_chw_picks_plane() -> None:
    # 3 channels first (smallest axis) → channel 1 is the middle plane.
    arr = np.stack([np.full((4, 5), c, dtype=np.float32) for c in range(3)], axis=0)
    out = select_channel(arr, channel=1)
    assert out.shape == (4, 5)
    assert float(out.mean()) == 1.0


def test_select_channel_hwc_picks_plane() -> None:
    arr = np.stack([np.full((4, 5), c, dtype=np.float32) for c in range(3)], axis=-1)
    out = select_channel(arr, channel=2)
    assert out.shape == (4, 5)
    assert float(out.mean()) == 2.0


def test_select_channel_all_is_mean_across_channels() -> None:
    arr = np.stack([np.zeros((4, 5), dtype=np.float32), np.full((4, 5), 4.0, dtype=np.float32)], axis=0)
    out = select_channel(arr, channel=-1)
    assert out.shape == (4, 5)
    assert float(out.mean()) == 2.0  # mean of {0, 4}


def test_select_channel_out_of_range_clamps() -> None:
    # Spatial dims (5×4) larger than the channel count (3), so the smallest-axis heuristic
    # unambiguously identifies axis 0 as channels (the channels-are-fewest assumption).
    arr = np.stack([np.full((5, 4), c, dtype=np.float32) for c in range(3)], axis=0)
    # channel 99 clamps to the last channel (index 2).
    assert float(select_channel(arr, channel=99).mean()) == 2.0


def test_select_channel_complex_uses_magnitude() -> None:
    arr = np.array([[3 + 4j, 0]], dtype=np.complex64)  # |3+4j| = 5
    out = select_channel(arr)
    assert out.shape == (1, 2)
    assert float(out[0, 0]) == 5.0


def test_select_channel_torch_tensor() -> None:
    out = select_channel(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert isinstance(out, np.ndarray) and out.shape == (2, 3)


def test_select_channel_non_array_yields_unit_map() -> None:
    assert select_channel("not an array").shape == (1, 1)


def test_channel_count() -> None:
    assert channel_count(np.zeros((4, 5), dtype=np.float32)) == 1  # 2-D → 1
    assert channel_count(np.zeros((3, 4, 5), dtype=np.float32)) == 3  # CHW
    assert channel_count(np.zeros((4, 5, 3), dtype=np.float32)) == 3  # HWC
    assert channel_count("not an array") == 0


# ---------------------------------------------------------------------------
# array_histogram — bin an array's values + summary statistics
# ---------------------------------------------------------------------------


def test_array_histogram_shape_and_stats() -> None:
    arr = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    hist = array_histogram(arr, bins=10)
    assert len(hist["counts"]) == 10
    assert len(hist["bin_edges"]) == 11  # bins + 1
    assert sum(hist["counts"]) == hist["count"] == 100
    assert hist["min"] == 0.0 and hist["max"] == 1.0
    assert hist["channels"] == 1
    assert abs(hist["mean"] - 0.5) < 1e-3


def test_array_histogram_excludes_non_finite() -> None:
    arr = np.array([0.0, 1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
    hist = array_histogram(arr, bins=4)
    # Only the 3 finite values (0, 1, 2) are counted; stats are finite.
    assert hist["count"] == 3
    assert sum(hist["counts"]) == 3
    assert hist["min"] == 0.0 and hist["max"] == 2.0
    assert np.isfinite(hist["mean"]) and np.isfinite(hist["std"])


def test_array_histogram_all_nan_is_empty_but_well_formed() -> None:
    hist = array_histogram(np.full((4,), np.nan, dtype=np.float32), bins=8)
    assert hist["count"] == 0
    assert hist["counts"] == [0] * 8
    assert len(hist["bin_edges"]) == 9
    assert hist["min"] is None and hist["max"] is None and hist["mean"] is None and hist["std"] is None


def test_array_histogram_flat_array_bins_into_first_bin() -> None:
    hist = array_histogram(np.full((10,), 5.0, dtype=np.float32), bins=4)
    assert hist["count"] == 10
    assert sum(hist["counts"]) == 10
    assert hist["min"] == 5.0 and hist["max"] == 5.0


def test_array_histogram_single_channel_vs_all() -> None:
    # channel 0 is all zeros, channel 1 is all ones.
    arr = np.stack([np.zeros((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32)], axis=0)
    only0 = array_histogram(arr, bins=4, channel=0)
    assert only0["count"] == 16 and only0["min"] == 0.0 and only0["max"] == 0.0
    allc = array_histogram(arr, bins=4, channel=-1)
    assert allc["count"] == 32 and allc["min"] == 0.0 and allc["max"] == 1.0


def test_array_histogram_non_array_is_empty() -> None:
    hist = array_histogram("text", bins=8)
    assert hist["count"] == 0 and hist["channels"] == 0


@pytest.mark.parametrize(
    "arr",
    [
        np.tile(np.arange(256, dtype=np.uint8), (256, 1)),  # 256x256 uint8 gray gradient (65536 px)
        np.linspace(-120.0, 0.0, 1024 * 512, dtype=np.float32),  # large float32 dB spectrogram
        (np.random.RandomState(0).rand(512, 512) * 255).astype(np.float32),  # 262144 px random gray
    ],
)
def test_array_histogram_large_array_does_not_raise(arr: np.ndarray) -> None:
    # Regression: numpy 2.2.x's uniform-bins fast path (`bins=<int>, range=(lo,hi)`) block-accumulates
    # via np.bincount for arrays >65536 elements and miscomputes the bincount length on the workspace
    # build — `n += bincount(...)` raised "operands could not be broadcast together with shapes
    # (256,) (257,) (256,)" on any real image. array_histogram uses explicit linspace edges to avoid it.
    finite = arr[np.isfinite(arr)]
    hist = array_histogram(arr, bins=256)
    assert len(hist["counts"]) == 256 and len(hist["bin_edges"]) == 257
    assert sum(hist["counts"]) == hist["count"] == finite.size  # every value still counted


# ---------------------------------------------------------------------------
# draw_text — render text onto an image / a fresh canvas
# ---------------------------------------------------------------------------


def test_text_positions_is_the_literal_set() -> None:
    assert TEXT_POSITIONS == get_args(TextPosition)
    assert "center" in TEXT_POSITIONS and "top-left" in TEXT_POSITIONS and len(TEXT_POSITIONS) == 9


def test_draw_text_blank_canvas_dims_and_dtype() -> None:
    img = draw_text("Hi", None, width=200, height=80, background="black", color="white")
    assert img.shape == (80, 200, 3) and img.dtype == np.uint8
    assert int((img > 0).sum()) > 0  # white text drawn on the black canvas


def test_draw_text_onto_existing_image_preserves_dims_and_copies() -> None:
    base = np.zeros((64, 128, 3), dtype=np.uint8)
    out = draw_text("label", base, color="red", position="center")
    assert out.shape == (64, 128, 3)
    assert (out[..., 0] > 0).any()  # red text pixels present
    assert int(base.sum()) == 0  # the input image is not mutated (drawn on a copy)


def test_draw_text_positions_place_block_differently() -> None:
    tl = draw_text("X", None, width=120, height=120, position="top-left")
    br = draw_text("X", None, width=120, height=120, position="bottom-right")
    assert tl[:60].sum() > tl[60:].sum()  # top-left lights the upper half
    assert br[60:].sum() > br[:60].sum()  # bottom-right lights the lower half


def test_draw_text_wrap_uses_more_vertical_lines() -> None:
    long = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"

    def text_row_span(im: np.ndarray) -> int:
        rows = np.where((im > 0).any(axis=(1, 2)))[0]
        return int(rows.max() - rows.min() + 1) if rows.size else 0

    wrapped = draw_text(long, None, width=120, height=240, wrap=True, position="top-left")
    nowrap = draw_text(long, None, width=120, height=240, wrap=False, position="top-left")
    # Wrapping spreads the text over more rows than a single (unwrapped) line.
    assert text_row_span(wrapped) > text_row_span(nowrap)


def test_draw_text_accepts_torch_tensor_image() -> None:
    out = draw_text("t", torch.zeros(3, 32, 48))  # CHW float tensor
    assert isinstance(out, np.ndarray) and out.ndim == 3 and out.shape[2] == 3
