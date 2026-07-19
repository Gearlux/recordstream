"""``AlbumentationsOp`` — run `albumentations <https://albumentations.ai>`_ transforms as a SampleFlux op.

One random draw is applied jointly to ``sample.input`` and (per the ``target`` mode) its
segmentation mask / detection boxes, so a geometric augmentation moves image AND target
consistently; metadata passes through untouched.

Transforms are authored **Confluid-natively** — nested ``!class:`` nodes, never
albumentations' own ``to_dict`` format::

    - !class:sampleflux.ops.albumentations.AlbumentationsOp
      target: mask
      seed: 0
      transforms:
        - !class:albumentations.HorizontalFlip
          p: 0.5
        - !class:albumentations.Affine
          translate_percent: 0.1

For per-transform graph nodes (one op per albumentations transform, e.g.
``AlbHorizontalFlip``) see :mod:`sampleflux.ops.albumentations_transforms`.

Layout contract: albumentations operates on **numpy HWC** images (PIL inputs are converted
via ``np.asarray``) and the output stays numpy HWC — tensorize downstream with
:class:`~sampleflux.ops.torch.ToTensorOp`. Contrast with
:class:`~sampleflux.ops.torchvision.TorchvisionTransformOp`, which emits CHW torch tensors.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from confluid import configurable
from loggair import get_logger

from sampleflux.sample import Sample

logger = get_logger(__name__)

#: Which part of the ``Sample`` rides through the library jointly with the input. Closed
#: set so a typo fails at the call site and UIs / form-specs enumerate the choices.
TargetMode = Literal["none", "mask", "boxes"]


def _as_array(value: Any) -> np.ndarray:
    """``value`` as a numpy array (PIL images and array-likes alike)."""
    return np.asarray(value)


def _resolve_transform(entry: Any) -> Any:
    """A raw library transform from a wired entry.

    Confluid ``!class:`` / ``!lazy:`` markers are flowed lazily (the RandomApply
    paradigm), and a generated per-transform op (``AlbHorizontalFlip`` …) wired on a
    visual canvas unwraps to its inner library transform via ``raw_transform``.
    """
    from confluid import flow
    from confluid.fluid import Fluid

    if isinstance(entry, Fluid):
        entry = flow(entry)
    return getattr(entry, "raw_transform", entry)


@configurable(category="op", group="augment", random=True)
class AlbumentationsOp:
    """Apply albumentations transforms to ``sample.input`` (and optionally the target).

    Pass EITHER ``transform`` (one transform, or a prebuilt ``A.Compose``) OR
    ``transforms`` (a list composed into an ``A.Compose`` lazily) — never both. Entries
    may be live albumentations objects, Confluid ``!class:`` markers, or generated
    per-transform ops (:mod:`sampleflux.ops.albumentations_transforms`), which unwrap to
    their inner library transform.

    Target modes (the ``target`` knob):

    * ``"none"`` — input-only augmentation (color jitter, noise, blur); the sample's
      target passes through untouched.
    * ``"mask"`` — ``sample.target`` is a segmentation mask (2-D array or PIL ``L``
      image); image and mask receive the SAME spatial transform.
    * ``"boxes"`` — ``sample.target`` is the torchvision detection dict
      ``{"boxes": [N,4] xyxy-pixel, "labels": [N]}`` (what
      :class:`~sampleflux.ops.target.CocoToTorchVisionDetectionOp` /
      :class:`~sampleflux.ops.target.MasksToDetectionBoxesOp` emit). When the op builds
      the Compose itself the required ``bbox_params`` are added automatically
      (``pascal_voc`` = absolute-pixel xyxy); a prebuilt Compose must carry its own.

    Stochasticity lives in the library: ``seed`` maps onto ``A.Compose(seed=...)``; gate
    per sample via :class:`~sampleflux.ops.random_apply.RandomApply` (each albumentations
    transform also carries its own ``p``).

    YAML:

    .. code-block:: yaml

        - !class:sampleflux.ops.albumentations.AlbumentationsOp
          target: mask
          seed: 0
          transforms:
            - !class:albumentations.HorizontalFlip
              p: 0.5

    Args:
        transform: ONE albumentations transform or a prebuilt ``A.Compose``. Validated lazily on first call.
        transforms: List of albumentations transforms composed lazily into an ``A.Compose``.
        target: Joint-augmentation mode — ``none`` (input-only, default), ``mask``, or ``boxes``.
        seed: ``A.Compose`` seed for deterministic draws. ``None`` = non-deterministic (default).
    """

    def __init__(
        self,
        transform: Optional[object] = None,
        transforms: Optional[List[Any]] = None,
        target: TargetMode = "none",
        seed: Optional[int] = None,
    ) -> None:
        # Lazy / zero-arg: store config only; transforms are resolved/validated on first call.
        self.transform = transform
        self.transforms: List[Any] = list(transforms) if transforms else []
        self.target = target
        self.seed = seed
        self._pipeline: Optional[object] = None
        self._pipeline_key: Optional[tuple] = None

    def _entries(self) -> List[Any]:
        """The configured raw transforms (markers flowed, generated ops unwrapped)."""
        if self.transform is not None and self.transforms:
            raise ValueError("AlbumentationsOp: pass either 'transform' or 'transforms', not both.")
        entries = [self.transform] if self.transform is not None else list(self.transforms)
        if not entries:
            raise ValueError(
                "AlbumentationsOp requires 'transform' (one transform / A.Compose) or "
                "'transforms' (a list) to be set before calling."
            )
        return [_resolve_transform(entry) for entry in entries]

    @property
    def pipeline(self) -> Any:
        """The live ``A.Compose`` — built lazily, cached until the configuration changes."""
        key = (id(self.transform), tuple(id(t) for t in self.transforms), self.target, self.seed)
        if self._pipeline is None or self._pipeline_key != key:
            # Albumentations is a hard dependency but slow to import — keep it lazy so
            # module import (entry-point discovery) stays light (the target.py precedent).
            import albumentations as A
            from albumentations.core.composition import BaseCompose

            entries = self._entries()
            if len(entries) == 1 and isinstance(entries[0], BaseCompose):
                if self.seed is not None:
                    raise ValueError(
                        "AlbumentationsOp: 'seed' only applies when the op builds the Compose itself; "
                        "put the seed on your prebuilt A.Compose(..., seed=...) instead."
                    )
                self._pipeline = entries[0]
            else:
                bbox_params = (
                    A.BboxParams(format="pascal_voc", label_fields=["labels"]) if self.target == "boxes" else None
                )
                self._pipeline = A.Compose(entries, seed=self.seed, bbox_params=bbox_params)
            self._pipeline_key = key
        return self._pipeline

    def __call__(self, sample: Sample) -> Sample:
        image = _as_array(sample.input)
        if self.target == "mask":
            out = self.pipeline(image=image, mask=_as_array(sample.target))
            return sample._replace(input=out["image"], target=out["mask"])
        if self.target == "boxes":
            new_input, new_target = self._apply_boxes(image, sample.target)
            return sample._replace(input=new_input, target=new_target)
        out = self.pipeline(image=image)
        return sample._replace(input=out["image"])

    def _apply_boxes(self, image: np.ndarray, target: Any) -> Tuple[Any, Dict[str, Any]]:
        """Route the torchvision detection dict through albumentations' bbox machinery."""
        import torch

        pipeline = self.pipeline
        if not isinstance(target, dict) or "boxes" not in target or "labels" not in target:
            raise TypeError(
                f"AlbumentationsOp(target='boxes'): sample.target must be the torchvision detection "
                f"dict {{'boxes': [N,4] xyxy, 'labels': [N]}}; got {type(target).__name__}. Wire "
                "CocoToTorchVisionDetectionOp / MasksToDetectionBoxesOp upstream."
            )
        if "bboxes" not in getattr(pipeline, "processors", {}):
            raise ValueError(
                "AlbumentationsOp(target='boxes'): the prebuilt Compose was built without bbox_params. "
                "Construct it as A.Compose([...], bbox_params=A.BboxParams(format='pascal_voc', "
                "label_fields=['labels'])) — or pass 'transforms' and let the op add them."
            )
        boxes = np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = [int(v) for v in np.asarray(target["labels"]).reshape(-1)]
        out = pipeline(image=image, bboxes=boxes.tolist(), labels=labels)
        out_boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
        new_target = dict(target)
        new_target["boxes"] = torch.as_tensor(out_boxes, dtype=torch.float32)
        new_target["labels"] = torch.as_tensor(list(out["labels"]), dtype=torch.int64).reshape(-1)
        return out["image"], new_target


__all__ = ["AlbumentationsOp", "TargetMode"]
