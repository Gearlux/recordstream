"""A record value as a PICTURE — the PNG a debugger shows beside a traced node's summary.

A summary (type, shape, dtype, range) says that a step ran; a picture says what it made. A value has one when it is a
RASTER: an image, a mask, or any 2-D array of numbers (a spectrogram, a heat map). The drawing itself is
:func:`recordstream.ops.image.value_to_image`'s — greyscale, quantized by the one quantizer — so a picture here and a
preview anywhere else agree; this module adds the question "does this value have one", the width, the boxes drawn over
it and the PNG.
"""

import io
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from recordstream._compat import is_torch_tensor
from recordstream.items import Boxes, Image

__all__ = ["picture", "pictured"]

#: The planes a 3-D array may hold to be drawn as an image: grey, RGB, RGBA.
_PLANES = (1, 3, 4)
#: The outline of a box: a colour a greyscale raster never holds, one pixel wide at any width.
_BOX_COLOUR = (255, 64, 64)
#: zlib's fastest level: a 12-megapixel raster is written in a fraction of the default's time, a few percent larger.
_PNG_LEVEL = 1


def _raster(value: Any) -> Optional[np.ndarray]:  # any record value
    """``value`` as the ``(H, W)`` or ``(H, W, planes)`` array a picture is drawn of, or ``None``. An
    :class:`~recordstream.Image` is read in its DECLARED layout; nothing is guessed from the shape."""
    if not isinstance(value, np.ndarray):
        return None
    data = np.asarray(value)
    if data.size == 0 or not (data.dtype == bool or np.issubdtype(data.dtype, np.number)):
        return None
    if np.issubdtype(data.dtype, np.complexfloating):
        return None
    if data.ndim == 3 and isinstance(value, Image) and getattr(value, "layout", "HWC") == "CHW":
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2 or (data.ndim == 3 and data.shape[2] in _PLANES):
        return data
    return None


def pictured(value: Any) -> bool:  # any record value
    """Whether ``value`` has a picture: an image, a mask, or a 2-D array of numbers."""
    return _raster(value) is not None


def _corners(boxes: Boxes) -> np.ndarray:
    """A :class:`~recordstream.Boxes`' rows as an ``[N, 4]`` float array, whatever type holds them."""
    held = boxes.boxes
    if is_torch_tensor(held):
        held = held.detach().cpu().numpy()
    return np.asarray(held, dtype=np.float64).reshape(-1, 4)


def picture(value: Any, *, width: Optional[int] = None, boxes: Sequence[Boxes] = ()) -> bytes:  # any record value
    """``value`` as a PNG: an image as it is, a mask black and white, an array of numbers grey from its lowest to
    its highest finite value.

    Args:
        value: The record value — refused with a ``ValueError`` when it has no picture (:func:`pictured`).
        width: The widest the picture may be, in pixels: a wider raster is scaled down, its shape kept; a narrower
            one is never enlarged. ``None`` keeps every pixel.
        boxes: Boxes stated in the raster's own frame, outlined over it (the caller checks the frame).
    """
    from recordstream.ops.image import value_to_image  # the one renderer; imported at use, like its other callers

    raster = _raster(value)
    if raster is None:
        raise ValueError(
            f"{type(value).__name__} has no picture — one is drawn of an image, a mask or a 2-D array of numbers"
        )
    if width is not None and width < 1:
        raise ValueError(f"width must be 1 or more, got {width}")
    drawn = PILImage.fromarray(value_to_image(raster, colormap="gray", max_size=0))
    scale = 1.0
    if width is not None and drawn.width > width:
        scale = width / drawn.width
        drawn = drawn.resize((width, max(1, round(drawn.height * scale))), PILImage.Resampling.BILINEAR)
    outlined = False
    pen = ImageDraw.Draw(drawn)
    for held in boxes:
        for x0, y0, x1, y1 in _corners(held):
            # half-open pixel boxes: the last pixel INSIDE is x1 - 1, y1 - 1
            right, bottom = max(x0, x1 - 1) * scale, max(y0, y1 - 1) * scale
            pen.rectangle((x0 * scale, y0 * scale, right, bottom), outline=_BOX_COLOUR, width=1)
            outlined = True
    if not outlined and raster.ndim == 2:
        drawn = drawn.convert("L")  # a grey raster is one plane on the wire: a third of the bytes
    out = io.BytesIO()
    drawn.save(out, format="PNG", compress_level=_PNG_LEVEL)
    return out.getvalue()
