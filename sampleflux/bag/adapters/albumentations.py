"""``AlbumentationsAdapter`` — run an albumentations transform over a typed bag.

albumentations dispatches by keyword NAME (``image=`` / ``mask=`` / ``bboxes=``) rather than by
type, so this adapter maps typed items to those named arguments, calls the transform once (one
draw applied jointly), and maps the result back. It targets ONE image field (the first, or the
one selected via ``only``), plus an optional mask and an optional regions field.

albumentations operates on numpy HWC images and stays numpy HWC. It is a hard dependency but
imported lazily so module import stays light.
"""

from typing import Any, List, Optional

import numpy as np

from sampleflux.bag.items import Image, Mask, Regions, item_data, with_data
from sampleflux.bag.sample import TypedSample
from sampleflux.bag.transform import Transform, register_adapter


class AlbumentationsAdapter(Transform):
    """Wrap one albumentations transform (or ``A.Compose``) as a typed-bag transform.

    Args:
        transform: An albumentations transform / ``A.Compose``. Validated lazily on first call.
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
        if self.transform is None:
            raise ValueError("AlbumentationsAdapter: 'transform' must be set before calling.")

        img_key = self._pick(sample, Image)
        if img_key is None:
            return sample  # albumentations needs an image; nothing to do
        mask_key = self._pick(sample, Mask)
        reg_key = self._pick(sample, Regions)

        kwargs: dict = {"image": np.asarray(item_data(sample[img_key]))}
        if mask_key is not None:
            kwargs["mask"] = np.asarray(item_data(sample[mask_key]))
        if reg_key is not None:
            regions = sample[reg_key]
            kwargs["bboxes"] = [list(box) for box in regions.boxes]
            kwargs["labels"] = list(regions.labels) if regions.labels is not None else [0] * len(regions.boxes)

        out = self._compose(need_bbox=reg_key is not None)(**kwargs)

        result = sample.replace_field(img_key, with_data(sample[img_key], out["image"]))
        if mask_key is not None:
            result = result.replace_field(mask_key, with_data(sample[mask_key], out["mask"]))
        if reg_key is not None:
            regions = sample[reg_key]
            result = result.replace_field(
                reg_key,
                Regions(
                    boxes=[list(box) for box in out["bboxes"]],
                    labels=list(out["labels"]),
                    scores=regions.scores,
                    canvas=regions.canvas,
                ),
            )
        return result

    def _pick(self, sample: TypedSample, item_type: type) -> Optional[str]:
        """The first field of ``item_type`` (honoring ``only``), or ``None``."""
        for key, item in sample.items():
            if self.only is not None and key not in self.only:
                continue
            if isinstance(item, item_type):
                return key
        return None

    def _compose(self, need_bbox: bool) -> Any:
        """The live ``A.Compose`` — a prebuilt Compose is used as-is; a bare transform is wrapped."""
        import albumentations as A
        from albumentations.core.composition import BaseCompose

        if isinstance(self.transform, BaseCompose):
            return self.transform
        bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"]) if need_bbox else None
        return A.Compose([self.transform], bbox_params=bbox_params)


def is_albumentations_transform(obj: Any) -> bool:
    """True for an albumentations transform / ``Compose`` — by MRO module name (no import here)."""
    return any(getattr(cls, "__module__", "").startswith("albumentations") for cls in type(obj).__mro__)


# Drop a bare albumentations transform straight into a Pipeline — wrapped in an AlbumentationsAdapter.
register_adapter(is_albumentations_transform, AlbumentationsAdapter)

__all__ = ["AlbumentationsAdapter", "is_albumentations_transform"]
