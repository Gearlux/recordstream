"""``TorchvisionV2Adapter`` — run a torchvision ``transforms.v2`` transform over a typed bag.

The typed bag and torchvision's ``tv_tensors`` are the SAME shape — a heterogeneous structure
of typed leaves — so this adapter is thin: it maps our items to ``tv_tensors``
(:class:`~sampleflux.bag.items.Image`\\ →``Image``,
:class:`~sampleflux.bag.items.Mask`\\ →``Mask``,
:class:`~sampleflux.bag.items.Regions`\\ →``BoundingBoxes``), hands the WHOLE dict to the v2
transform (v2 draws its random parameters once and applies them across every leaf, so a
geometric augmentation stays consistent across image / mask / boxes), and maps the result
back into typed items with their metadata preserved.

torchvision is lazy-imported; this module imports without it installed (a missing install
raises a clear error pointing at the ``sampleflux[vision]`` extra).
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from sampleflux.bag.items import Image, Mask, Regions, item_data, with_data
from sampleflux.bag.sample import TypedSample
from sampleflux.bag.transform import Transform, register_adapter


def _import_v2() -> Any:
    try:
        from torchvision.transforms import v2
    except ImportError as exc:  # pragma: no cover - exercised only without torchvision
        raise ImportError(
            "TorchvisionV2Adapter requires torchvision (transforms.v2 / tv_tensors). "
            'Install it via `pip install "sampleflux[vision]"`.'
        ) from exc
    return v2


class TorchvisionV2Adapter(Transform):
    """Wrap one ``transforms.v2`` transform (or ``v2.Compose``) as a typed-bag transform.

    Args:
        transform: A ``transforms.v2`` transform / ``v2.Compose``. Validated lazily on first call.
        only: Restrict to these field keys (still type-gated).
    """

    handles = (Image, Mask, Regions)
    consumes = (Image,)
    optional = (Mask, Regions)
    produces = (Image, Mask, Regions)

    def __init__(self, transform: Optional[Any] = None, only: Optional[List[str]] = None) -> None:
        super().__init__(only=only)
        self.transform = transform

    def __call__(self, sample: TypedSample) -> TypedSample:
        import torch
        from torchvision import tv_tensors

        _import_v2()  # raise the actionable extra hint before any torchvision use
        if self.transform is None:
            raise ValueError("TorchvisionV2Adapter: 'transform' must be set before calling.")

        canvas = _canvas_size(sample)
        structure: dict = {}
        for key, item in sample.items():
            if self.only is not None and key not in self.only:
                continue
            wrapped = _wrap(item, tv_tensors, torch, canvas)
            if wrapped is not None:
                structure[key] = wrapped
        if not structure:
            return sample

        out_structure = self.transform(structure)
        out = sample
        for key, wrapped_out in out_structure.items():
            out = out.replace_field(key, _unwrap(sample[key], wrapped_out, torch))
        return out


def _canvas_size(sample: TypedSample) -> Optional[Tuple[int, int]]:
    """``(H, W)`` from the first Image/Mask field — the reference frame for bounding boxes."""
    for _, item in sample.items():
        if isinstance(item, (Image, Mask)):
            arr = item_data(item)
            if isinstance(item, Image) and getattr(item, "layout", "HWC") == "CHW" and arr.ndim == 3:
                return int(arr.shape[1]), int(arr.shape[2])
            if arr.ndim >= 2:
                return int(arr.shape[0]), int(arr.shape[1])
    return None


def _wrap(item: Any, tv_tensors: Any, torch: Any, canvas: Optional[Tuple[int, int]]) -> Any:
    """Our item → a ``tv_tensors`` carrier (``None`` for a type torchvision does not handle)."""
    if isinstance(item, Image):
        arr = item_data(item)
        tensor = torch.as_tensor(np.ascontiguousarray(arr))
        if getattr(item, "layout", "HWC") == "HWC" and tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tv_tensors.Image(tensor)
    if isinstance(item, Mask):
        return tv_tensors.Mask(torch.as_tensor(np.ascontiguousarray(item_data(item))))
    if isinstance(item, Regions):
        size = item.canvas or canvas
        if size is None:
            raise ValueError(
                "TorchvisionV2Adapter: Regions need a canvas (H, W) — set Regions.canvas or include an Image field."
            )
        boxes = torch.as_tensor(np.asarray(item.boxes, dtype=np.float32).reshape(-1, 4))
        return tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=size)
    return None


def _unwrap(original: Any, wrapped_out: Any, torch: Any) -> Any:
    """A ``tv_tensors`` result → our item type, metadata preserved."""
    if isinstance(original, Image):
        tensor = wrapped_out.as_subclass(torch.Tensor)
        arr = tensor.detach().cpu().numpy()
        if getattr(original, "layout", "HWC") == "HWC" and arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return with_data(original, arr)
    if isinstance(original, Mask):
        return with_data(original, wrapped_out.as_subclass(torch.Tensor).detach().cpu().numpy())
    if isinstance(original, Regions):
        boxes = wrapped_out.as_subclass(torch.Tensor).detach().cpu().numpy().reshape(-1, 4).tolist()
        return Regions(boxes=boxes, labels=original.labels, scores=original.scores, canvas=original.canvas)
    return original  # pragma: no cover - only wrapped types reach here


def is_torchvision_v2_transform(obj: Any) -> bool:
    """True for a torchvision ``transforms.v2`` transform / ``Compose`` — by MRO module name.

    Inspects the object's own class MRO (which the caller already imported), so it recognises v2
    objects WITHOUT importing torchvision here; v1 ``torchvision.transforms.transforms`` objects do
    not match (they don't handle ``tv_tensors``).
    """
    return any(getattr(cls, "__module__", "").startswith("torchvision.transforms.v2") for cls in type(obj).__mro__)


# Drop a bare v2 transform straight into a Pipeline — it is wrapped in a TorchvisionV2Adapter.
register_adapter(is_torchvision_v2_transform, TorchvisionV2Adapter)

__all__ = ["TorchvisionV2Adapter", "is_torchvision_v2_transform"]
