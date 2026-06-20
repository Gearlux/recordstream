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

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, get_args

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


# --------------------------------------------------------------------------- #
# Array introspection helpers — channel selection + histogram.
#
# These back FluxStudio's "Array / Tensor Histogram" viewer node (and are usable
# from any pipeline / notebook): a generic, modality-agnostic way to look at the
# RAW numeric values of an array/tensor — pick a channel, render it, and bin its
# values. Pure functions (NOT @configurable ops): they measure/derive, they don't
# transform a Sample, so they're library helpers like value_to_image — not canvas
# nodes. They live here (not in the FluxStudio node) so the computation is reusable
# and unit-tested, per the workspace "rendering/analysis lives in dataflux" mandate.
# --------------------------------------------------------------------------- #


def _coerce_to_ndarray(value: Any) -> Optional[np.ndarray]:
    """Best-effort view of an arbitrary value as a numeric ``np.ndarray`` for analysis.

    PIL image → RGB array, ``torch.Tensor`` → detached numpy, complex array →
    magnitude (``abs``), list/scalar → ``np.asarray``. Returns ``None`` when the
    value cannot sensibly be viewed as a numeric array (string/bytes/None, or an
    object-dtype array such as a list of ragged things).
    """
    data: Any = value
    if data is None or isinstance(data, (str, bytes)):
        return None
    if hasattr(data, "convert"):  # PIL.Image.Image
        data = np.array(data.convert("RGB"))
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    try:
        arr = np.asarray(data)
    except Exception:  # pragma: no cover - defensive: exotic objects np can't view
        return None
    if arr.dtype == object:
        return None
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr


def _squeeze_to_3d(arr: np.ndarray) -> np.ndarray:
    """Squeeze size-1 axes, then drop leading axes until at most 3-D (a ``[B,C,H,W]`` → first item)."""
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def _channel_axis(shape: Tuple[int, ...]) -> int:
    """Index of the channel axis of a 3-D shape: the SMALLEST axis (channels-are-fewest convention).

    Deliberately distinct from the other two channel heuristics in this workspace, each scoped to a
    narrower job: :func:`_render_rgb`'s ``{1,3,4}``-membership test is RGB-render-specific (it only
    recognises 1/3/4-channel *images*), and ``fluxstudio.nodes.SampleExtractorNode._as_2d`` is
    mask-specific (float-only). For a general N-channel feature map (e.g. an 8-channel tensor) the
    smallest-axis rule is the most defensible default; documented here so the three never look like an
    accidental disagreement.
    """
    return int(np.argmin(shape))


def select_channel(value: Any, channel: int = -1) -> np.ndarray:
    """Reduce an arbitrary array/tensor to a single 2-D ``float32`` map for the given channel.

    The view used both for rendering one channel and for the per-pixel hover readout:

    * a 2-D array passes through; a 1-D array becomes a ``(1, N)`` strip; a scalar a ``(1, 1)`` cell;
    * a 3-D array selects ``channel`` along its channel axis (the smallest axis — see
      :func:`_channel_axis`); ``channel < 0`` collapses that axis by **mean** (an "all channels" view);
    * higher-rank arrays drop leading axes to 3-D first; complex data is magnitude (``abs``).

    Out-of-range ``channel`` is clamped into ``[0, channels-1]``. A non-numeric value yields a ``(1, 1)``
    zero map (so callers always get a real 2-D array).

    Args:
        value: The array / tensor / PIL image / scalar to view.
        channel: Channel index to select; ``-1`` (default) means "all" → mean across the channel axis.
    """
    arr = _coerce_to_ndarray(value)
    if arr is None:
        return np.zeros((1, 1), dtype=np.float32)
    arr = _squeeze_to_3d(np.asarray(arr, dtype=np.float32))
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    # 3-D: the channel axis is the smallest axis.
    caxis = _channel_axis(arr.shape)
    n_channels = arr.shape[caxis]
    if channel is None or channel < 0:
        return np.asarray(arr.mean(axis=caxis), dtype=np.float32)
    idx = min(max(int(channel), 0), n_channels - 1)
    return np.asarray(np.take(arr, idx, axis=caxis), dtype=np.float32)


def channel_count(value: Any) -> int:
    """Number of channels of an array/tensor: 1 for ≤2-D data, the smallest-axis size for 3-D, 0 for non-arrays."""
    arr = _coerce_to_ndarray(value)
    if arr is None:
        return 0
    sq = _squeeze_to_3d(np.asarray(arr))
    return int(sq.shape[_channel_axis(sq.shape)]) if sq.ndim == 3 else 1


def array_histogram(value: Any, bins: int = 256, channel: int = -1) -> Dict[str, Any]:
    """Bin the values of an array/tensor into a histogram + summary statistics.

    Counts and statistics are taken over **finite** values only (``NaN`` / ``±inf`` are dropped, so
    the result is always JSON-safe — no non-finite floats leak into ``min``/``max``/``bin_edges``).
    When ``channel >= 0`` the histogram is of that single channel's plane; ``channel < 0`` histograms
    **every** element across all channels.

    Returns a dict with ``counts`` (length ``bins``), ``bin_edges`` (length ``bins+1``), ``min`` /
    ``max`` / ``mean`` / ``std`` (``None`` when there are no finite values), ``count`` (number of
    finite values) and ``channels`` (detected channel count). A degenerate all-equal array bins into
    the first bin over a unit-wide range.

    Args:
        value: The array / tensor / PIL image / scalar to histogram.
        bins: Number of histogram bins (clamped to at least 1).
        channel: Channel to histogram; ``-1`` (default) histograms all elements across channels.
    """
    bins = max(1, int(bins))
    arr = _coerce_to_ndarray(value)
    channels = channel_count(value)
    if arr is None:
        flat = np.empty((0,), dtype=np.float32)
    elif channel is not None and channel >= 0:
        flat = select_channel(value, channel).astype(np.float32).ravel()
    else:
        flat = np.asarray(arr, dtype=np.float32).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        edges = np.linspace(0.0, 1.0, bins + 1)
        return {
            "counts": [0] * bins,
            "bin_edges": edges.tolist(),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "count": 0,
            "channels": channels,
        }
    lo = float(finite.min())
    hi = float(finite.max())
    # A flat array (all values equal) has a zero-width range — pin a deterministic unit range so the
    # single populated bin is predictable (np.histogram would otherwise auto-pad to lo±0.5).
    hi_edge = hi if hi > lo else lo + 1.0
    # Pass EXPLICIT bin edges (np.linspace), NOT `bins=<int>, range=(lo, hi)`. numpy 2.2.x's uniform
    # fast path block-accumulates with `np.bincount(...)` for arrays larger than its 65536-element
    # block, and on the workspace build that miscomputes the bincount length so `n += bincount(...)`
    # dies with "operands could not be broadcast together with shapes (256,) (257,) (256,)" — i.e. it
    # fails on any real image/spectrogram (>65536 px) while passing on the small arrays unit tests use.
    # The explicit-edges path (searchsorted) sidesteps that bug and is otherwise identical: the last
    # bin is closed, so values == hi are still counted (sum(counts) == finite.size).
    edges = np.linspace(lo, hi_edge, bins + 1)
    counts, edges = np.histogram(finite, bins=edges)
    return {
        "counts": counts.astype(int).tolist(),
        "bin_edges": edges.astype(float).tolist(),
        "min": lo,
        "max": hi,
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "count": int(finite.size),
        "channels": channels,
    }


def confusion_matrix_payload(
    matrix: Any,
    class_names: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Structure a confusion matrix + class names into a JSON-safe payload for a GUI viewer.

    Backs FluxStudio's *Confusion Matrix* viewer node (``fluxstudio.nodes.ConfusionMatrixViewerNode``).
    The MATH that lives here is the three normalizations (the viewer toggles between them WITHOUT a
    re-run — the JS only colours + labels + hovers): ``true`` (each row / actual-class sums to 1),
    ``pred`` (each column / predicted-class sums to 1) and ``all`` (the whole matrix sums to 1). Every
    float is finite-checked (``NaN``/``±inf`` → ``None``, never a misleading substitute) so the payload
    survives ComfyUI's ``json.dumps`` websocket encoding — mirroring :func:`array_histogram`. A row /
    column whose count-sum is ``0`` normalises to ``None`` (undefined, not ``0``).

    Args:
        matrix: A square ``N×N`` confusion matrix (integer counts) as an array / tensor / nested list.
        class_names: Optional length-``N`` class labels; defaults to ``["0", "1", …, "N-1"]``.

    Returns a dict with ``counts`` (``N×N`` ints), ``normalized`` (``{"true","pred","all"}``, each
    ``N×N`` floats or ``None``), ``class_names`` (length ``N``), ``n_classes``, and ``total``. A
    non-square / empty / non-2-D input yields ``{"n_classes": 0, ...}`` + a ``message``.
    """
    arr = _coerce_to_ndarray(matrix)
    if arr is None or arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        shape = None if arr is None else tuple(int(d) for d in arr.shape)
        return {
            "counts": [],
            "normalized": {"true": [], "pred": [], "all": []},
            "class_names": [],
            "n_classes": 0,
            "total": 0,
            "message": f"not a square 2-D confusion matrix (shape {shape})",
        }

    counts = np.asarray(arr).astype(np.int64)
    n = int(counts.shape[0])
    total = int(counts.sum())
    row_sums = counts.sum(axis=1)  # per true class
    col_sums = counts.sum(axis=0)  # per predicted class

    def _normed(divisor: np.ndarray) -> list:
        # Element-wise count / divisor; a 0 divisor (empty row/col/matrix) -> None (undefined).
        out: list = []
        for i in range(n):
            row: list = []
            for j in range(n):
                d = float(divisor[i, j])
                row.append(_sanitize_finite(counts[i, j] / d) if d != 0.0 else None)
            out.append(row)
        return out

    names = [str(c) for c in class_names] if class_names is not None else [str(i) for i in range(n)]
    # Pad / trim to exactly N so the viewer always has one label per row/column.
    names = (names + [str(i) for i in range(len(names), n)])[:n]

    return {
        "counts": [[int(c) for c in row] for row in counts.tolist()],
        "normalized": {
            "true": _normed(np.broadcast_to(row_sums.reshape(n, 1), (n, n))),
            "pred": _normed(np.broadcast_to(col_sums.reshape(1, n), (n, n))),
            "all": _normed(np.full((n, n), float(total))),
        },
        "class_names": names,
        "n_classes": n,
        "total": total,
    }


def _is_square_2d(value: Any) -> bool:
    """True when ``value`` views as a square ``N×N`` (``N>=1``) numeric array — confusion-matrix shape."""
    arr = _coerce_to_ndarray(value)
    return arr is not None and arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= 1


def confusion_matrices_payload(metrics: Any, class_names: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    """Extract EVERY confusion matrix from a metrics result and build a render payload for each.

    The generic counterpart to :func:`confusion_matrix_payload`: a model evaluator emits its FULL
    metric results (``name -> value``; scalars, vectors, AND `N×N` matrices) with NO knowledge of which
    is a confusion matrix — this scans them and renders all CONFUSION-MATRIX-SHAPED entries (square 2-D,
    ``_is_square_2d``, by SHAPE not name), returning one :func:`confusion_matrix_payload` per match
    (each tagged with its metric ``name``) in dict order, or ``[]`` when none. A bare square-2D
    ``metrics`` (not a dict) is treated as a single matrix named ``"confusion_matrix"``. This is what
    lets FluxStudio's *Confusion Matrix* viewer render ALL matrices from one all-metrics output (there
    can be several). ``class_names`` labels every matrix the same way (they share the class set).

    Args:
        metrics: An evaluator's metric results — a ``dict`` of ``name -> value`` (the usual form), or a
            single ``N×N`` matrix.
        class_names: Optional length-``N`` class labels applied to each matrix; defaults to indices.
    """
    items = list(metrics.items()) if isinstance(metrics, dict) else [("confusion_matrix", metrics)]
    out: List[Dict[str, Any]] = []
    for name, value in items:
        if _is_square_2d(value):
            payload = confusion_matrix_payload(value, class_names=class_names)
            payload["name"] = str(name)
            out.append(payload)
    return out


def _sanitize_finite(x: float) -> Optional[float]:
    """A finite float rounded for compactness, or ``None`` for ``NaN``/``±inf`` (JSON-safe)."""
    v = float(x)
    return round(v, 6) if np.isfinite(v) else None


# --------------------------------------------------------------------------- #
# Text → image rendering — draw text onto an image (or a fresh canvas).
# --------------------------------------------------------------------------- #

# Closed 9-grid set of text anchor positions (a closed Literal per the workspace mandate, so the
# choice is a dropdown in FluxStudio / navigaitor enumerated from one source of truth).
TextPosition = Literal[
    "top-left",
    "top",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom",
    "bottom-right",
]
TEXT_POSITIONS: Tuple[TextPosition, ...] = get_args(TextPosition)


def _text_anchor_xy(position: str, block_w: int, block_h: int, img_w: int, img_h: int, margin: int) -> Tuple[int, int]:
    """Top-left ``(x, y)`` for a ``block_w × block_h`` text block per the 9-grid ``position`` + margin."""
    if "left" in position:
        x: float = margin
    elif "right" in position:
        x = img_w - block_w - margin
    else:  # "top" / "bottom" / "center" (no left/right) → horizontally centered
        x = (img_w - block_w) / 2
    if position.startswith("top"):
        y: float = margin
    elif position.startswith("bottom"):
        y = img_h - block_h - margin
    else:  # "center-*" / left / right (no top/bottom) → vertically centered
        y = (img_h - block_h) / 2
    return int(round(x)), int(round(y))


def _wrap_text(draw: "ImageDraw.ImageDraw", text: str, font: Any, max_width: int) -> str:
    """Greedy word-wrap so each line fits ``max_width`` px (explicit newlines preserved)."""
    out: list = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}".strip()
            if line and draw.textlength(trial, font=font) > max_width:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
    return "\n".join(out)


def draw_text(
    text: str,
    image: Optional[Any] = None,
    *,
    width: int = 512,
    height: int = 256,
    font_size: int = 24,
    color: str = "white",
    background: str = "black",
    position: TextPosition = "top-left",
    margin: int = 8,
    wrap: bool = True,
) -> np.ndarray:
    """Render ``text`` onto ``image`` (or a fresh ``background`` canvas) → an ``(H, W, 3)`` uint8 RGB array.

    The single, modality-agnostic "draw text on an image" renderer (FluxStudio's *Draw Text to Image*
    node is thin glue over it). When ``image`` is ``None`` a blank ``(height, width)`` canvas of color
    ``background`` is created; otherwise the value is coerced to an RGB image (via :func:`_render_rgb`,
    so PIL / ndarray / tensor / 2-D maps all work) and drawn on a copy. The text is word-wrapped to the
    image width (``wrap``; explicit newlines kept) and anchored per the 9-grid ``position`` with a
    ``margin`` inset. Uses PIL's sized default bitmap font.

    Args:
        text: The text to draw (multi-line allowed).
        image: Background image (PIL / ndarray / tensor / 2-D map); ``None`` makes a blank canvas.
        width: Blank-canvas width in pixels (used only when ``image`` is ``None``).
        height: Blank-canvas height in pixels (used only when ``image`` is ``None``).
        font_size: Font size in points.
        color: Text color — any PIL color name or hex (``"white"`` / ``"#ffcc00"`` / ...).
        background: Canvas color when ``image`` is ``None`` — any PIL color name or hex.
        position: Anchor of the text block — one of the 9-grid ``TextPosition`` values.
        margin: Inset in pixels from the edges for non-centered anchors.
        wrap: Word-wrap long lines to fit the image width.
    """
    from PIL import ImageFont

    if image is None:
        img = Image.new("RGB", (max(1, int(width)), max(1, int(height))), color=background)
    else:
        img = Image.fromarray(_render_rgb(image, "gray"))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=int(font_size))
    except TypeError:  # Pillow < 10.1 has no sized default font — fall back to the fixed bitmap font
        font = ImageFont.load_default()
    rendered = _wrap_text(draw, text, font, max(1, img.width - 2 * margin)) if wrap else text
    # multiline_textbbox is relative to the anchor; subtract its offset so the block's top-left lands at (x, y).
    bbox = draw.multiline_textbbox((0, 0), rendered, font=font)
    block_w, block_h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    x, y = _text_anchor_xy(position, block_w, block_h, img.width, img.height, margin)
    draw.multiline_text((x - bbox[0], y - bbox[1]), rendered, fill=color, font=font)
    return np.array(img)


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
    to ``sample.meta`` so downstream consumers (e.g. a detector
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

        sample.meta["image_width_px"] = img.width
        sample.meta["image_height_px"] = img.height
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
        # np.asarray (not .astype) so the return type is ndarray under stub
        # versions where clip-arithmetic degrades to Any.
        return np.asarray(np.clip(norm, 0.0, 1.0) * 255.0, dtype=np.uint8)

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
    "select_channel",
    "channel_count",
    "array_histogram",
    "confusion_matrix_payload",
    "confusion_matrices_payload",
    "draw_text",
    "TextPosition",
    "TEXT_POSITIONS",
]
