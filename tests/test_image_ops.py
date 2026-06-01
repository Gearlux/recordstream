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
    Colormap,
    ConvertToImageOp,
    NormalizeToUint8Op,
    _apply_colormap,
    sample_to_image,
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
    assert out.metadata["image_width_px"] == 128
    assert out.metadata["image_height_px"] == 256


def test_convert_max_size_path_bounds_longest_side() -> None:
    out = ConvertToImageOp(max_size=256)(_sample(np.zeros((1000, 400), dtype=np.float32)))
    assert max(out.input.size) == 256
    # Dims are published from the actual rendered raster.
    assert out.metadata["image_width_px"] == out.input.width
    assert out.metadata["image_height_px"] == out.input.height


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
