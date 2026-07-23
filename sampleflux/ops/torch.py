from typing import Any, Optional

import numpy as np
import torch
from confluid import configurable

from sampleflux.bag.items import Image as ImageItem
from sampleflux.bag.items import NDArrayItem, item_data
from sampleflux.bag.sample import Sample
from sampleflux.bag.transform import Transform


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
    """An array-bearing field → a CHW-float ``Image`` item.

    Reads the payload of an array-bearing field (blank ``field`` picks the first array/PIL-bearing
    item — typically the :class:`~sampleflux.Image` a :class:`~sampleflux.ops.image.ConvertToImage`
    produced), runs the HWC→CHW transpose + ``normalize`` conversion (:func:`to_tensor`), and writes
    a CHW-layout :class:`~sampleflux.Image` back. By default it REPLACES the resolved field in place
    (``output`` blank), so the field's role is preserved; set ``output`` to write a NEW field
    (tagged ``input``) instead. Any other field passes through untouched.

    IMPORTANT — payload dtype. A :class:`~sampleflux.NDArrayItem` (which ``Image`` is) coerces its
    payload through ``np.asarray`` on construction, so it CANNOT hold a live ``torch.Tensor``: the
    stored payload is a CHW ``float32`` **numpy** array. The typed collate stacks these payloads with
    ``np.stack``; the numpy→``torch.Tensor`` conversion happens at the collate / model boundary.

    Args:
        normalize: When ``True`` (default), scale integer pixel inputs into the ``[0, 1]`` float range.
        mode: Optional PIL mode to convert a PIL payload to (e.g. ``"RGB"`` forces 3 channels); ``None`` = as-is.
        field: Name of the source field to tensorize; blank (default) picks the first array/PIL-bearing item.
        output: Field the CHW ``Image`` is written to; blank (default) replaces the source field in place
            (role preserved). A non-blank name writes a new field tagged ``input``.
    """

    handles = (NDArrayItem,)
    consumes = (NDArrayItem,)
    produces = (ImageItem,)

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

    def _find_field(self, sample: Sample) -> str:
        """Resolve the KEY of the field to tensorize (``self.field`` or the first array/PIL item)."""
        if self.field:
            if self.field not in sample.keys():
                raise ValueError(f"ToTensor: field {self.field!r} not in sample (fields: {list(sample.keys())})")
            return self.field
        for key, item in sample.items():
            data = item_data(item)
            if isinstance(data, np.ndarray) or hasattr(data, "convert"):
                return key
        raise ValueError(f"ToTensor: no array-bearing field in sample (fields: {list(sample.keys())})")

    def __call__(self, sample: Sample) -> Sample:
        key = self._find_field(sample)
        data = item_data(sample[key])
        tensor = to_tensor(data, self.normalize, self.mode)
        arr = tensor.detach().cpu().numpy()
        out_key = self.output or key
        out = sample.replace_field(out_key, ImageItem(arr, layout="CHW"))
        if self.output:
            out = out.set_role(out_key, "input")
        return out


__all__ = ["ToTensor", "to_tensor"]
