from typing import Any, Optional

import numpy as np
import torch
from confluid import configurable

from sampleflux.items import NDArrayItem, Record, item_data
from sampleflux.transform import Transform


def to_tensor(img: Any, normalize: bool = True, mode: Optional[str] = None) -> torch.Tensor:
    """Convert a PIL image / NumPy array to a CHW ``torch.Tensor``.

    A PIL image is optionally mode-coerced (``mode="RGB"`` forces 3 channels) then arrayed;
    an ``[H, W, C]`` array is transposed to ``[C, H, W]`` (a 2-D array gets a leading channel
    axis). With ``normalize`` an integer / 0-255-float payload is scaled into ``[0, 1]``.
    """
    if hasattr(img, "convert"):
        if mode is not None:
            img = img.convert(mode)
        img = np.array(img)

    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        elif img.ndim == 2:
            img = img[np.newaxis, :]
        tensor = torch.from_numpy(img)
    else:
        tensor = torch.as_tensor(img)

    if normalize and tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    elif normalize and tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


@configurable(category="op", group="torch")
class ToTensor(Transform):
    """An array-bearing field → a LIVE CHW-float ``torch.Tensor`` record value.

    Reads the payload of an array-bearing field (blank ``field`` picks the first array/PIL-bearing
    item — typically the :class:`~sampleflux.Image` a :class:`~sampleflux.ops.image.ConvertToImage`
    produced), runs the HWC→CHW transpose + ``normalize`` conversion (:func:`to_tensor`), and writes
    the resulting ``torch.Tensor`` back AS-IS. By default it REPLACES the resolved field in place
    (``output`` blank); set ``output`` to write a NEW key instead. Any other key passes
    through untouched.

    The output is a PLAIN record value (a record holds arbitrary values — the ``"plain"`` codec
    tag covers storage): ``collate_records`` stacks torch tensors natively (``torch.stack``), a
    torchvision ``transforms.v2`` op downstream transforms it as-is, and array sinks convert via
    ``to_numpy`` on write. It is deliberately NOT wrapped in an :class:`~sampleflux.Image` — an
    ``NDArrayItem`` coerces through ``np.asarray`` and cannot hold a live tensor.

    Args:
        normalize: When ``True`` (default), scale integer pixel inputs into the ``[0, 1]`` float range.
        mode: Optional PIL mode to convert a PIL payload to (e.g. ``"RGB"`` forces 3 channels); ``None`` = as-is.
        field: Name of the source field to tensorize; blank (default) picks the first array/PIL-bearing item.
        output: Key the tensor is written to; blank (default) replaces the source field in place.
    """

    handles = (NDArrayItem,)
    consumes = (NDArrayItem,)
    produces = (torch.Tensor,)

    def __init__(
        self,
        normalize: bool = True,
        mode: Optional[str] = None,
        field: str = "",
        output: str = "",
    ) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.mode = mode
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
        tensor = to_tensor(data, self.normalize, self.mode)
        out_key = self.output or key
        return {**record, out_key: tensor}


__all__ = ["ToTensor", "to_tensor"]
