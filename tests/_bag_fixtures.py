"""Test-local typed-bag fixtures.

``FixtureFlip`` is the former native ``HorizontalFlip`` kept ONLY as a test fixture: sampleflux
ships no native augmentation transforms (geometric/photometric augmentation comes from
torchvision v2 / albumentations through adapter coercion), but the kernel-dispatch machinery
(once-per-sample params, per-type kernels, MRO resolution, ``only=`` filter) still needs a
fully native transform to pin — and the fixture doubles as the ADAPTER-PARITY reference (a
`v2.RandomHorizontalFlip(p=1.0)` through the adapter must move image/mask/boxes exactly like
this native implementation does).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sampleflux import Image, Mask, Regions, Transform, TypedSample, item_data, with_data


class FixtureFlip(Transform):
    """Horizontal flip with ONE shared decision across Image + Mask + Regions (test fixture)."""

    handles = (Image, Mask, Regions)
    consumes = (Image,)
    optional = (Mask, Regions)
    produces = (Image, Mask, Regions)

    def __init__(self, p: float = 0.5, only: Optional[List[str]] = None) -> None:
        super().__init__(only=only)
        self.p = p

    def get_params(self, sample: TypedSample) -> Dict[str, Any]:
        do = float(np.random.random()) < self.p
        return {"do": do, "width": _reference_width(sample)}


@FixtureFlip.kernel(Image)
def _flip_image(item: Image, params: Dict[str, Any]) -> Image:
    if not params["do"]:
        return item
    axis = 2 if getattr(item, "layout", "HWC") == "CHW" else 1
    return with_data(item, np.flip(item_data(item), axis=axis).copy())


@FixtureFlip.kernel(Mask)
def _flip_mask(item: Mask, params: Dict[str, Any]) -> Mask:
    if not params["do"]:
        return item
    return with_data(item, np.flip(item_data(item), axis=1).copy())


@FixtureFlip.kernel(Regions)
def _flip_regions(item: Regions, params: Dict[str, Any]) -> Regions:
    if not params["do"]:
        return item
    width = params.get("width") or (item.canvas[1] if item.canvas else None)
    if width is None:
        raise ValueError("FixtureFlip: no reference width to flip Regions")
    boxes = [[width - box[2], box[1], width - box[0], box[3]] for box in item.boxes]
    return Regions(boxes=boxes, labels=item.labels, scores=item.scores, canvas=item.canvas)


def _reference_width(sample: TypedSample) -> Optional[int]:
    """The horizontal extent to flip boxes against — from the first Image/Mask, or a Regions canvas."""
    for _, item in sample.items():
        if isinstance(item, Image):
            arr = item_data(item)
            axis = 2 if getattr(item, "layout", "HWC") == "CHW" else 1
            if arr.ndim > axis:
                return int(arr.shape[axis])
        if isinstance(item, Mask):
            arr = item_data(item)
            if arr.ndim >= 2:
                return int(arr.shape[1])
    for _, item in sample.items():
        if isinstance(item, Regions) and item.canvas:
            return int(item.canvas[1])
    return None
