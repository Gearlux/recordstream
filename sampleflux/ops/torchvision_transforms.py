"""Auto-generated ops: every torchvision ``transforms.v2`` transform as its own SampleFlux op.

Generated at import time by :mod:`sampleflux.ops._augment_bridge` from the
``torchvision.transforms.v2`` namespace — one ``Tv<Name>`` op per concrete transform
(``TvRandomHorizontalFlip``, ``TvColorJitter``, …), each a subclass of
:class:`~sampleflux.ops.torchvision.TorchvisionTransformOp` mirroring the transform's own
constructor parameters plus the adapter's ``target`` knob.

YAML (the registered short name resolves via the confluid registry):

.. code-block:: yaml

    - !class:TvRandomHorizontalFlip
      p: 0.5
      target: mask

The ``Tv`` prefix is MANDATORY — torchvision and albumentations share many bare class
names (``ColorJitter``, ``Normalize``, ``Resize``, …) and confluid's registry is flat and
name-keyed, so unprefixed names would silently clobber each other.

The module imports WITHOUT torchvision installed (entry-point discovery stays safe): it
then generates zero ops and logs a debug note pointing at the ``sampleflux[vision]``
extra. Container transforms (``Compose`` / ``RandomApply`` / ``RandomChoice`` /
``RandomOrder``) are deliberately NOT generated — chaining ops is native SampleFlux.
"""

from typing import Any, List, Tuple

from loggair import get_logger

from sampleflux.ops._augment_bridge import generate_transform_ops
from sampleflux.ops.torchvision import TorchvisionTransformOp

logger = get_logger(__name__)


def __getattr__(name: str) -> Any:
    # Generated names live in module globals; this fallback only fires for genuinely
    # missing ones — and tells mypy the dynamic attributes exist (module-__getattr__ rule).
    raise AttributeError(
        f"module {__name__!r} has no generated op {name!r} — torchvision may be missing "
        '(install via `pip install "sampleflux[vision]"`) or the transform may not exist '
        "in the installed version."
    )


#: v2 names not generated: containers (native chaining) + the deprecated v1 shim ToTensor.
_EXCLUDED = frozenset({"Compose", "RandomApply", "RandomChoice", "RandomOrder", "ToTensor", "Transform"})


def _library_classes() -> List[Tuple[str, type]]:
    """The concrete, generatable v2 transform classes, sorted by name (empty without torchvision)."""
    try:
        from torchvision.transforms import v2
    except ImportError:
        logger.debug(
            "torchvision not installed — no Tv* transform ops generated "
            '(install via `pip install "sampleflux[vision]"`).'
        )
        return []

    out: List[Tuple[str, type]] = []
    for name in sorted(vars(v2)):
        obj = getattr(v2, name)
        if not (isinstance(obj, type) and issubclass(obj, v2.Transform)):
            continue
        if name in _EXCLUDED or obj.__name__ != name or name.startswith("_"):
            continue
        out.append((name, obj))
    return out


__all__ = generate_transform_ops(
    classes=_library_classes(),
    base=TorchvisionTransformOp,
    prefix="Tv",
    group="augment/torchvision",
    module_globals=globals(),
    seed_param=False,
)
