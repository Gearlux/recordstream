from typing import Any

import numpy as np
import torch
from confluid import configurable

from recordstream.items import NDArrayItem, Record, item_data
from recordstream.transform import Transform


def to_tensor(img: Any) -> torch.Tensor:
    """Convert a PIL image / NumPy array to a CHW ``torch.Tensor`` of the SAME element type.

    An ``[H, W, C]`` array is transposed to ``[C, H, W]`` (a 2-D array gets a leading channel
    axis); a PIL image is read as its array first. Nothing else happens: ``uint8`` pixels stay
    ``uint8`` ``0..255``. The value changes are ops of their own —
    :class:`~recordstream.ops.image.ConvertMode` (channel layout), :class:`~recordstream.ops.numpy.Scale`
    (range) and :class:`~recordstream.ops.numpy.ToType` (element type) — because a conversion that
    also rescaled had to GUESS the input's range, and guessed an already-standardized image into
    ``[-0.008, 0.010]``.
    """
    if hasattr(img, "convert"):
        img = np.array(img)
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        elif img.ndim == 2:
            img = img[np.newaxis, :]
        return torch.from_numpy(img)
    return torch.as_tensor(img)


@configurable(category="op", group="torch")
class ToTensor(Transform):
    """An array-bearing field → a LIVE CHW ``torch.Tensor`` record value, same element type.

    Reads the payload of an array-bearing field (blank ``field`` picks the first array/PIL-bearing
    item — typically the :class:`~recordstream.Image` a :class:`~recordstream.ops.image.ConvertToImage`
    produced), runs the HWC→CHW conversion (:func:`to_tensor`) and writes the resulting
    ``torch.Tensor`` back AS-IS. It CONVERTS and nothing else: ``uint8`` pixels arrive as a
    ``uint8`` tensor. Put :class:`~recordstream.ops.image.ConvertMode`,
    :class:`~recordstream.ops.numpy.Scale` or :class:`~recordstream.ops.numpy.ToType` before it for
    a channel layout, a value range or an element type. By default it REPLACES the resolved field in place
    (``output`` blank); set ``output`` to write a NEW key instead. Any other key passes
    through untouched.

    The output is a PLAIN record value (a record holds arbitrary values — the ``"plain"`` codec
    tag covers storage): ``collate_records`` stacks torch tensors natively (``torch.stack``), a
    torchvision ``transforms.v2`` op downstream transforms it as-is, and array sinks convert via
    ``to_numpy`` on write. It is deliberately NOT wrapped in an :class:`~recordstream.Image` — an
    ``NDArrayItem`` coerces through ``np.asarray`` and cannot hold a live tensor.

    Args:
        field: Name of the source field to tensorize; blank (default) picks the first array/PIL-bearing item.
        output: Key the tensor is written to; blank (default) replaces the source field in place.
    """

    handles = (NDArrayItem,)
    consumes = (NDArrayItem,)
    produces = (torch.Tensor,)

    def __init__(self, field: str = "", output: str = "") -> None:
        super().__init__()
        self.field = field
        self.output = output

    def _find_field(self, record: Record) -> str:
        """Resolve the KEY of the field to tensorize (``self.field`` or the first array/PIL item)."""
        if self.field:
            if self.field not in record:
                raise ValueError(f"ToTensor: field {self.field!r} not in record (keys: {list(record)})")
            return self.field
        for key, item in record.items():
            data = item_data(item)
            if isinstance(data, np.ndarray) or hasattr(data, "convert"):
                return key
        raise ValueError(f"ToTensor: no array-bearing field in record (keys: {list(record)})")

    def __call__(self, record: Record) -> Record:
        key = self._find_field(record)
        data = item_data(record[key])
        tensor = to_tensor(data)
        out_key = self.output or key
        return {**record, out_key: tensor}


__all__ = ["ToTensor", "to_tensor"]
