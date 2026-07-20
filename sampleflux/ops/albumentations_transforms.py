"""Auto-generated ops: every public albumentations transform as its own SampleFlux op.

Generated at import time by :mod:`sampleflux.ops._augment_bridge` from albumentations'
public namespace — one ``Alb<Name>`` op per concrete transform (``AlbHorizontalFlip``,
``AlbAffine``, ``AlbRandomBrightnessContrast``, …), each a subclass of
:class:`~sampleflux.ops.albumentations.AlbumentationsOp` mirroring the transform's own
constructor parameters plus the adapter's ``target`` / ``seed`` knobs.

YAML (the registered short name resolves via the confluid registry):

.. code-block:: yaml

    - !class:AlbHorizontalFlip
      p: 0.5
      target: mask

The ``Alb`` prefix is MANDATORY — albumentations and torchvision share many bare class
names (``ColorJitter``, ``Normalize``, ``Resize``, …) and confluid's registry is flat and
name-keyed, so unprefixed names would silently clobber each other.

Composition transforms (``Compose`` / ``OneOf`` / ``SomeOf`` …) are deliberately NOT
generated — chaining ops is native SampleFlux (``ops:`` lists, ``TransformChain``,
``RandomApply``), and an ``A.Compose`` still wires verbatim into
``AlbumentationsOp(transform=...)``.
"""

from typing import Any, List, Tuple

from loggair import get_logger

from sampleflux.ops._augment_bridge import generate_transform_ops
from sampleflux.ops.albumentations import AlbumentationsOp

logger = get_logger(__name__)


def __getattr__(name: str) -> Any:
    # Generated names live in module globals; this fallback only fires for genuinely
    # missing ones — and tells mypy the dynamic attributes exist (module-__getattr__ rule).
    raise AttributeError(
        f"module {__name__!r} has no generated op {name!r} — "
        "the albumentations transform may not exist in the installed version."
    )


def _library_classes() -> List[Tuple[str, type]]:
    """The concrete, generatable albumentations transform classes, sorted by name."""
    import albumentations as A
    from albumentations.core.composition import BaseCompose
    from albumentations.core.transforms_interface import BasicTransform

    out: List[Tuple[str, type]] = []
    for name in sorted(vars(A)):
        obj = getattr(A, name)
        if not (isinstance(obj, type) and issubclass(obj, BasicTransform)):
            continue
        if issubclass(obj, BaseCompose) or obj.__name__ != name or name.startswith("_"):
            continue
        if obj.__module__.startswith("albumentations.core"):
            continue  # the abstract bases (BasicTransform / DualTransform / ImageOnlyTransform)
        if name == "Lambda":
            continue  # callable-valued params — not representable as config
        out.append((name, obj))
    return out


__all__ = generate_transform_ops(
    classes=_library_classes(),
    base=AlbumentationsOp,
    prefix="Alb",
    group="augment/albumentations",
    module_globals=globals(),
    seed_param=True,
)
