"""``TorchvisionTransformOp`` — run torchvision ``transforms.v2`` transforms as a SampleFlux op.

v2 transforms draw their random parameters ONCE per call and apply them to every
``tv_tensors`` carrier passed in, so a geometric augmentation moves ``sample.input`` AND
(per the ``target`` mode) its segmentation mask / detection boxes consistently; metadata
passes through untouched.

Transforms are authored **Confluid-natively** as nested ``!class:`` nodes::

    - !class:sampleflux.ops.torchvision.TorchvisionTransformOp
      target: mask
      transforms:
        - !class:torchvision.transforms.v2.RandomHorizontalFlip
          p: 0.5

For per-transform graph nodes (one op per v2 transform, e.g. ``TvRandomHorizontalFlip``)
see :mod:`sampleflux.ops.torchvision_transforms`.

Layout contract: torchvision operates on **CHW torch tensors** (PIL images pass through
natively; numpy HWC arrays are converted on entry) and the output is CHW tensors — no
:class:`~sampleflux.ops.torch.ToTensorOp` needed downstream. Contrast with
:class:`~sampleflux.ops.albumentations.AlbumentationsOp`, which stays numpy HWC.

This module imports WITHOUT torchvision installed (it is entry-pointed for discovery);
torchvision is lazy-imported on first call and a missing install raises a clear error
pointing at the ``sampleflux[vision]`` extra.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from confluid import configurable
from loggair import get_logger

from sampleflux.bag.items import Mask, item_data, with_data
from sampleflux.bag.sample import Sample, primary
from sampleflux.ops.albumentations import TargetMode, _resolve_transform

logger = get_logger(__name__)


def _import_v2() -> Any:
    """The ``torchvision.transforms.v2`` module, or a clear error naming the extra."""
    try:
        from torchvision.transforms import v2
    except ImportError as exc:  # pragma: no cover - exercised only without torchvision
        raise ImportError(
            "TorchvisionTransformOp requires torchvision (transforms.v2 / tv_tensors). "
            'Install it via `pip install "sampleflux[vision]"`.'
        ) from exc
    return v2


@configurable(category="op", group="augment", random=True)
class TorchvisionTransformOp:
    """Apply torchvision ``transforms.v2`` transforms to ``sample.input`` (and optionally the target).

    Pass EITHER ``transform`` (one v2 transform, or a prebuilt ``v2.Compose``) OR
    ``transforms`` (a list composed into a ``v2.Compose`` lazily) — never both. Entries
    may be live v2 objects, Confluid ``!class:`` markers, or generated per-transform ops
    (:mod:`sampleflux.ops.torchvision_transforms`), which unwrap to their inner library
    transform.

    Target modes (the ``target`` knob):

    * ``"none"`` — input-only augmentation; the sample's target passes through untouched.
    * ``"mask"`` — ``sample.target`` is a segmentation mask (2-D array / tensor or PIL
      ``L`` image), wrapped as ``tv_tensors.Mask`` so image and mask receive the SAME
      spatial transform.
    * ``"boxes"`` — ``sample.target`` is the torchvision detection dict
      ``{"boxes": [N,4] xyxy-pixel, "labels": [N]}`` (what
      :class:`~sampleflux.ops.target.CocoToTorchVisionDetectionOp` /
      :class:`~sampleflux.ops.target.MasksToDetectionBoxesOp` emit); boxes are wrapped as
      ``tv_tensors.BoundingBoxes(format="XYXY", canvas_size=(H, W))`` and come back as
      plain float32 / int64 tensors.

    Stochasticity lives in the library: v2 transforms draw from torch's global RNG
    (``torch.manual_seed(N)`` pins it); gate per sample via
    :class:`~sampleflux.ops.random_apply.RandomApply`.

    YAML:

    .. code-block:: yaml

        - !class:sampleflux.ops.torchvision.TorchvisionTransformOp
          target: mask
          transforms:
            - !class:torchvision.transforms.v2.RandomHorizontalFlip
              p: 0.5

    Args:
        transform: ONE ``transforms.v2`` transform or a prebuilt ``v2.Compose``. Validated lazily on first call.
        transforms: List of v2 transforms composed lazily into a ``v2.Compose``.
        target: Joint-augmentation mode — ``none`` (input-only, default), ``mask``, or ``boxes``.
    """

    def __init__(
        self,
        transform: Optional[object] = None,
        transforms: Optional[List[Any]] = None,
        target: TargetMode = "none",
    ) -> None:
        # Lazy / zero-arg: store config only; transforms are resolved/validated on first call.
        self.transform = transform
        self.transforms: List[Any] = list(transforms) if transforms else []
        self.target = target
        self._pipeline: Optional[object] = None
        self._pipeline_key: Optional[tuple] = None

    @property
    def pipeline(self) -> Any:
        """The live v2 transform — built lazily, cached until the configuration changes."""
        if self.transform is not None and self.transforms:
            raise ValueError("TorchvisionTransformOp: pass either 'transform' or 'transforms', not both.")
        key = (id(self.transform), tuple(id(t) for t in self.transforms))
        if self._pipeline is None or self._pipeline_key != key:
            if self.transform is not None:
                self._pipeline = _resolve_transform(self.transform)
            elif self.transforms:
                v2 = _import_v2()
                self._pipeline = v2.Compose([_resolve_transform(entry) for entry in self.transforms])
            else:
                raise ValueError(
                    "TorchvisionTransformOp requires 'transform' (one v2 transform / v2.Compose) or "
                    "'transforms' (a list) to be set before calling."
                )
            self._pipeline_key = key
        return self._pipeline

    def __call__(self, sample: Sample) -> Sample:
        import torch

        _import_v2()  # raise the actionable extra hint before any torchvision use
        from torchvision import tv_tensors

        pipeline = self.pipeline
        key, item = primary(sample, "input")
        image = self._wrap_image(item_data(item), tv_tensors, torch)

        if self.target == "mask":
            mask_field = next(iter(sample.items_of_type(Mask)), None)
            if mask_field is None:
                raise ValueError(
                    "TorchvisionTransformOp(target='mask'): no Mask field in the sample to transform jointly."
                )
            mkey, mitem = mask_field
            mask = self._wrap_mask(item_data(mitem), tv_tensors, torch)
            out_image, out_mask = pipeline(image, mask)
            result = sample.replace_field(key, with_data(item, self._unwrap(out_image, torch).detach().cpu().numpy()))
            return result.replace_field(mkey, with_data(mitem, self._unwrap(out_mask, torch).detach().cpu().numpy()))
        if self.target == "boxes":
            raise NotImplementedError(
                "TorchvisionTransformOp(target='boxes') is not yet ported to the typed-bag Regions target "
                "(migration follow-up); use target='none' or 'mask'."
            )
        out_image = self._unwrap(pipeline(image), torch)
        return sample.replace_field(key, with_data(item, out_image.detach().cpu().numpy()))

    @staticmethod
    def _wrap_image(value: Any, tv_tensors: Any, torch: Any) -> Any:
        """``value`` as a v2 carrier: PIL passes through, tensors/arrays become CHW ``tv_tensors.Image``."""
        if hasattr(value, "convert"):  # PIL image — v2 transforms handle it natively
            return value
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            array = np.asarray(value)
            tensor = torch.as_tensor(np.ascontiguousarray(array))
            if tensor.ndim == 3:  # numpy convention is HWC; torchvision wants CHW
                tensor = tensor.permute(2, 0, 1)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tv_tensors.Image(tensor)

    @staticmethod
    def _wrap_mask(value: Any, tv_tensors: Any, torch: Any) -> Any:
        """``value`` as a ``tv_tensors.Mask`` (PIL ``L`` images and 2-D arrays alike)."""
        if hasattr(value, "convert"):
            value = np.asarray(value)
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.ascontiguousarray(value))
        return tv_tensors.Mask(tensor)

    @staticmethod
    def _canvas_size(image: Any) -> Tuple[int, int]:
        """``(H, W)`` of the wrapped input — the reference frame for bounding boxes."""
        if hasattr(image, "convert"):  # PIL
            return int(image.height), int(image.width)
        return int(image.shape[-2]), int(image.shape[-1])

    @staticmethod
    def _unwrap(value: Any, torch: Any) -> Any:
        """Strip the ``tv_tensors`` subclass so plain tensors flow downstream."""
        if isinstance(value, torch.Tensor):
            return value.as_subclass(torch.Tensor)
        return value


__all__ = ["TorchvisionTransformOp"]
