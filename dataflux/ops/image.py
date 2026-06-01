"""Generic, modality-agnostic image conversion for DataFlux pipelines.

This is the single home for "turn an arbitrary value into an image": the
:class:`ConvertToImageOp` op plus the library functions
(:func:`value_to_image` / :func:`sample_to_image`) that back it and FluxStudio's
sample preview. It lives in dataflux (not waivefront) because the conversion is
fully generic — a 2-D map, a CHW tensor, a PIL image, a boolean mask all render
the same way regardless of domain — so every project (waivefront's spectrogram
render, any image dataset preview, FluxStudio nodes) reuses ONE implementation.

Domain-specific rendering stays in the consuming package: waivefront's
``RenderOverlaysOp`` draws signal-region rectangles on top of the PIL image this
op produces, and ``RenderSignalPlotOp`` builds IQ time/freq/constellation panels.
Those need signal semantics; this op does not.

PIL is a hard dependency here (already used by ``dataflux.typespec``). Matplotlib
is imported lazily inside :func:`_apply_colormap` — only non-``"gray"`` colormaps
need it, so the pure-greyscale path stays matplotlib-free.
"""

from typing import Any, Literal, Optional, Tuple, get_args

import numpy as np
import torch
from confluid import configurable
from logflow import get_logger
from PIL import Image, ImageDraw

from dataflux.sample import Sample
from dataflux.typespec import ArrayType as _ArrayType
from dataflux.typespec import PythonType, SampleType, UnionType

logger = get_logger("dataflux.ops.image")


# Closed set of supported matplotlib colormaps — the SINGLE source of truth for every colormap knob
# across the workspace (``value_to_image`` / ``sample_to_image`` / ``ConvertToImageOp`` and, via
# re-export, waivefront's renderers) AND for FluxStudio's colormap dropdown (which reads ``COLORMAPS``).
# A closed ``Literal`` (never a bare ``str``) makes the choice self-documenting and machine-
# introspectable: the FluxStudio palette, navigaitor's form-spec, and MCP tool schemas enumerate the
# options straight from the annotation via ``typing.get_args`` instead of hard-coding a parallel list
# that silently drifts. ``"gray"`` is the greyscale path (special-cased in ``_apply_colormap``); every
# other name resolves through ``matplotlib.colormaps[name]``. Per the workspace "closed Literal"
# mandate, derive the runtime tuple FROM the Literal (``get_args``) — never restate the values.
Colormap = Literal[
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "gray",
    "hot",
    "cool",
    "jet",
    "turbo",
    "twilight",
    "hsv",
]
COLORMAPS: Tuple[Colormap, ...] = get_args(Colormap)


def _apply_colormap(spec_u8: np.ndarray, colormap: Colormap) -> Image.Image:
    """Turn a ``(H, W)`` uint8 magnitude map into an RGB PIL image.

    ``colormap="gray"`` reproduces the greyscale-to-RGB path (matplotlib-free).
    Any other name is resolved through ``matplotlib.colormaps[name]`` (lazily
    imported) so standard cmaps (``"hot"``, ``"viridis"``, ``"magma"``,
    ``"plasma"``, ``"inferno"``, ``"turbo"``, …) are supported.
    """
    if colormap == "gray":
        return Image.fromarray(spec_u8, mode="L").convert("RGB")
    import matplotlib

    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(spec_u8.astype(np.float32) / 255.0)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _text_to_image(text: str, width: int = 512, height: int = 160) -> np.ndarray:
    """Render a short string to an ``(H, W, 3)`` uint8 image (non-image fallback)."""
    img = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    max_chars = max(1, width // 7)
    lines = [text[i : i + max_chars] for i in range(0, min(len(text), max_chars * 8), max_chars)]
    draw.multiline_text((6, 6), "\n".join(lines) or "<empty>", fill=(220, 220, 220))
    return np.array(img)


def _render_rgb(value: Any, colormap: Colormap) -> np.ndarray:
    """Render an arbitrary value to an ``(H, W, 3)`` uint8 RGB image WITHOUT resizing.

    The core of :func:`value_to_image` factored out so callers that need their
    own resize policy (e.g. :class:`ConvertToImageOp`'s exact ``width``/``height``)
    don't pay a double resize. Handles PIL images, torch tensors, numpy arrays
    (2-D maps → ``colormap``; 3-D → image with channel coercion; bool → 0/255);
    anything else falls back to a text rendering of its ``repr``.
    """
    data: Any = value

    if hasattr(data, "convert"):  # PIL.Image.Image
        data = np.array(data.convert("RGB"))
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if not isinstance(data, np.ndarray):
        return _text_to_image(repr(data))

    arr = np.squeeze(np.asarray(data))

    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255

    if arr.ndim == 2:
        return np.array(_apply_colormap(NormalizeToUint8Op.normalize_to_uint8(arr), colormap))
    if arr.ndim == 3:
        # Normalize channel position to trailing (HWC).
        if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        channels = arr.shape[2]
        if channels == 3:
            pass
        elif channels == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif channels >= 4:
            arr = arr[..., :3]
        else:  # 2 channels (or other) — replicate the first
            arr = np.repeat(arr[..., :1], 3, axis=2)
        return arr if arr.dtype == np.uint8 else NormalizeToUint8Op.normalize_to_uint8(arr)
    return _text_to_image(f"input ndim={arr.ndim}, shape={arr.shape}")


def _bound_longest_side(rgb: np.ndarray, max_size: int) -> np.ndarray:
    """Downscale an ``(H, W, 3)`` image so its longest side is ≤ ``max_size`` (aspect preserved)."""
    height_px, width_px = rgb.shape[:2]
    longest = max(height_px, width_px)
    if max_size <= 0 or longest <= max_size:
        return rgb.astype(np.uint8)
    scale = max_size / longest
    resized = Image.fromarray(rgb).resize(
        (max(1, int(width_px * scale)), max(1, int(height_px * scale))),
        Image.Resampling.BILINEAR,
    )
    return np.array(resized).astype(np.uint8)


def value_to_image(value: Any, colormap: Colormap = "viridis", max_size: int = 512) -> np.ndarray:
    """Render an arbitrary value (a Sample's ``input`` OR ``target``) to an ``(H, W, 3)`` uint8 RGB image.

    A generic, modality-agnostic preview usable from any DataFlux pipeline (and
    by FluxStudio's sample extractor, which renders the selected field). Handles:

    * ``PIL.Image`` — converted to RGB;
    * ``torch.Tensor`` — detached to numpy (CHW collapsed to HWC below);
    * ``np.ndarray`` — 2-D maps go through ``colormap`` (one of the supported
      colormaps — see ``Colormap``; ``"gray"`` for greyscale); 3-D arrays are
      treated as images (a leading channel axis is transposed to trailing,
      1/2/4-channel coerced to 3); boolean masks become 0/255; floating arrays
      are min-max normalized.

    Anything else (e.g. a bbox list) falls back to a text rendering of its
    ``repr`` so the caller still shows *something* rather than erroring.
    ``max_size`` bounds the longest side.

    Args:
        value: The value to render (image / tensor / ndarray / mask, else a text repr of its ``repr``).
        colormap: Colormap applied to 2-D maps — one of the supported names (see ``Colormap``; ``"gray"`` = greyscale).
        max_size: Maximum length in pixels of the longest image side; larger renders are downscaled.
    """
    return _bound_longest_side(_render_rgb(value, colormap), max_size)


def sample_to_image(sample: Sample, colormap: Colormap = "viridis", max_size: int = 512) -> np.ndarray:
    """Render ``sample.input`` to an ``(H, W, 3)`` uint8 RGB image for display.

    Thin wrapper over :func:`value_to_image` (which does the modality-agnostic
    rendering) applied to ``sample.input``. Kept as the canonical "preview a
    sample" entry point for DataFlux pipelines; use :func:`value_to_image`
    directly to render an arbitrary value such as ``sample.target``.

    Args:
        sample: The Sample to preview; its ``input`` field is rendered.
        colormap: Colormap applied to 2-D maps — one of the supported names (see ``Colormap``; ``"gray"`` = greyscale).
        max_size: Maximum length in pixels of the longest image side; larger renders are downscaled.
    """
    return value_to_image(sample.input, colormap=colormap, max_size=max_size)


@configurable(category="op", group="image")
class ConvertToImageOp:
    """Convert ``sample.input`` (array / tensor / 2-D map / PIL image) into a PIL image.

    The generic image-conversion op — normalize → colormap → (flip) → resize.
    It is modality-agnostic: a dB spectrogram, a segmentation logit map, a CHW
    tensor, or an already-PIL image all become a ``PIL.Image.Image`` on
    ``sample.input``. Domain overlays are a SEPARATE concern — chain
    ``waivefront.visualizers.RenderOverlaysOp`` after this op to draw
    signal-region rectangles; this op never draws annotations.

    Rendering uses :func:`value_to_image`'s core (so 2-D maps are colormapped,
    3-D arrays treated as images, bool masks become 0/255, floats min-max
    normalized). Sizing:

    * ``width`` and ``height`` both > 0 → resize to exactly that raster
      (e.g. a spectrogram rendered to ``1024x512`` for downstream detectors).
    * otherwise → bound the longest side by ``max_size``, preserving aspect.

    ``flip_vertical=True`` mirrors the image top-to-bottom — used when the source
    array's row 0 is the *bottom* of the desired image (a spectrogram stores
    row 0 = f_min but display wants f_max at the top, so overlay pixel math
    lines up). The final ``image_width_px`` / ``image_height_px`` are published
    to ``sample.metadata`` so downstream consumers (e.g. a detector
    back-projecting pixel boxes to signal regions) can read the raster size.

    Args:
        colormap: Colormap applied to 2-D maps — a supported ``Colormap`` name (``"gray"`` = greyscale).
        width: Exact output width in pixels; resize to ``(width, height)`` when both width and height are > 0.
        height: Exact output height in pixels; resize to ``(width, height)`` when both width and height are > 0.
        max_size: When ``width``/``height`` aren't both set, bound the longest side to this many pixels (aspect kept).
        flip_vertical: Mirror the image top-to-bottom (e.g. spectrogram row 0 = f_min → display f_max at the top).
    """

    ACCEPTS = SampleType(input=UnionType((PythonType("PIL.Image.Image"), _ArrayType(frameworks={"numpy", "torch"}))))
    PRODUCES = SampleType(input=PythonType("PIL.Image.Image"))

    def __init__(
        self,
        colormap: Colormap = "gray",
        width: int = 0,
        height: int = 0,
        max_size: int = 512,
        flip_vertical: bool = False,
    ) -> None:
        self.colormap: Colormap = colormap
        self.width = int(width)
        self.height = int(height)
        self.max_size = int(max_size)
        self.flip_vertical = bool(flip_vertical)

    def __call__(self, sample: Sample) -> Sample:
        rgb = _render_rgb(sample.input, self.colormap)
        if self.flip_vertical:
            rgb = rgb[::-1, :, :]
        if self.width > 0 and self.height > 0:
            img = Image.fromarray(rgb).resize(
                (self.width, self.height),
                resample=Image.Resampling.BILINEAR,
            )
        else:
            img = Image.fromarray(_bound_longest_side(rgb, self.max_size))

        sample.metadata["image_width_px"] = img.width
        sample.metadata["image_height_px"] = img.height
        return sample._replace(input=img)


@configurable(category="op", group="image")
class NormalizeToUint8Op:
    """Min-max normalize ``sample.input`` to a ``uint8`` array in ``[0, 255]``.

    The generic value→``uint8`` conversion step, decoupled from any colormap or
    PIL rendering (that is :class:`ConvertToImageOp`). Useful as a standalone
    quantization stage — e.g. turning a dB spectrogram or a logit map into a
    display-ready 8-bit grid — and as the shared math behind the renderers in
    this module (:func:`value_to_image` calls :meth:`normalize_to_uint8`
    directly for its 2-D-map and float-array paths).

    By default the scale is taken from the array's own finite min/max (per-array
    auto-contrast). Supply ``vmin`` / ``vmax`` to pin a *fixed* range instead so
    successive samples are quantized on a common scale (e.g. a constant dB window
    across a dataset) — values outside the range clamp to ``0`` / ``255``.

    Non-finite entries (``NaN`` / ``±inf``) are folded to the low / high bound
    before scaling; a degenerate range (``vmax <= vmin``, or a flat array under
    auto bounds) maps to all-zeros to avoid a divide-by-zero.

    Args:
        vmin: Lower bound mapped to ``0``; ``None`` (default) uses the array's finite minimum.
        vmax: Upper bound mapped to ``255``; ``None`` (default) uses the array's finite maximum.
    """

    ACCEPTS = SampleType(input=_ArrayType(frameworks={"numpy", "torch"}))
    PRODUCES = SampleType(input=_ArrayType(dtype="uint8", frameworks={"numpy"}))

    def __init__(self, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
        self.vmin = None if vmin is None else float(vmin)
        self.vmax = None if vmax is None else float(vmax)

    @staticmethod
    def normalize_to_uint8(
        arr: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:
        """Min-max normalize ``arr`` to ``uint8`` in ``[0, 255]``.

        ``vmin`` / ``vmax`` pin the scale when given (clamping out-of-range
        values); otherwise the array's finite min / max are used. Non-finite
        entries are folded to the bounds; a degenerate range yields all-zeros.
        """
        arr = np.asarray(arr).astype(np.float32)
        finite = arr[np.isfinite(arr)]
        lo = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
        hi = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else 0.0)
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        filled = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
        norm = (filled - lo) / (hi - lo)
        return (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)

    def __call__(self, sample: Sample) -> Sample:
        if self.vmin is not None and self.vmax is not None and self.vmin >= self.vmax:
            raise ValueError(f"NormalizeToUint8Op: vmin must be < vmax; got vmin={self.vmin!r}, vmax={self.vmax!r}")
        arr = sample.input
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        return sample._replace(input=self.normalize_to_uint8(arr, self.vmin, self.vmax))


__all__ = [
    "Colormap",
    "COLORMAPS",
    "ConvertToImageOp",
    "NormalizeToUint8Op",
    "value_to_image",
    "sample_to_image",
]
