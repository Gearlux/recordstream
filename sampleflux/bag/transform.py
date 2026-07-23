"""``Transform`` — type-dispatched sample transforms with once-per-sample parameters.

A transform samples its random / configured parameters ONCE per sample (:meth:`Transform.get_params`),
then walks the bag and, for each field whose item type it handles, applies the registered
kernel (:mod:`sampleflux.bag.dispatch`). Fields it does not handle pass through untouched.

Two properties fall out of this shape for free:

* **Cross-field consistency.** Because params are sampled once and shared, one transform
  moves every spatial field with the SAME decision (a torchvision-v2 flip dropped into a
  :class:`Pipeline` flips an :class:`~sampleflux.bag.items.Image`, its
  :class:`~sampleflux.bag.items.Mask`, and its :class:`~sampleflux.bag.items.Regions`
  together) — the thing the old flat-metadata model could not express.
* **Open extension.** A new item type is taught to an existing transform with one
  ``@Transform.kernel(NewType)`` registration and no core edit.

Targeting is by TYPE, with an optional ``only=[keys]`` filter for surgical control (touch
only the named fields even if others share a handled type).

sampleflux ships NO native augmentation transforms — geometric/photometric augmentation
comes from the libraries (torchvision ``transforms.v2`` / albumentations) through the
adapter coercion registry below; a domain package registers its own transforms (e.g. a
signal FFT) via the same ``Transform`` + kernel machinery from outside.

Graph annotations (``consumes`` / ``optional`` / ``produces`` — item-type tuples) describe a
transform's item-level inputs/outputs for a visual editor's typed side sockets; they are
declarative metadata, not enforced at runtime here.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sampleflux.bag.dispatch import Kernel, dispatch, register_kernel
from sampleflux.bag.items import item_data, with_data
from sampleflux.bag.sample import Sample

__all__ = [
    "Transform",
    "Pipeline",
    "FunctionTransform",
    "as_transform",
    "register_adapter",
    "coerce_transform",
]


class Transform:
    """Base class for type-dispatched transforms (see the module docstring).

    Subclasses declare ``handles`` (the item types they process) and register a kernel per
    type via ``@MyTransform.kernel(ItemType)``. Override :meth:`get_params` to sample shared
    parameters once per sample.

    Args:
        only: Restrict the transform to these field keys (still type-gated). ``None`` = every
            field of a handled type.
    """

    #: Item types this transform processes (a field of another type passes through).
    handles: Tuple[type, ...] = ()
    #: Graph metadata — required input item types (defaults to ``handles`` when empty).
    consumes: Tuple[type, ...] = ()
    #: Graph metadata — optional input item types.
    optional: Tuple[type, ...] = ()
    #: Graph metadata — item types this transform adds or changes.
    produces: Tuple[type, ...] = ()

    def __init__(self, only: Optional[List[str]] = None) -> None:
        self.only = list(only) if only else None

    @classmethod
    def kernel(cls, item_type: type) -> Callable[[Kernel], Kernel]:
        """Register a kernel for ``item_type`` on this transform (decorator over :func:`register_kernel`)."""
        return register_kernel(cls, item_type)

    def get_params(self, sample: Sample) -> Dict[str, Any]:
        """Sample the shared parameters for one call. Default: no params."""
        return {}

    def __call__(self, sample: Sample) -> Sample:
        params = self.get_params(sample)
        out = sample
        for key, item in sample.items():
            if self.only is not None and key not in self.only:
                continue
            kernel = dispatch(type(self), type(item))
            if kernel is None:
                continue
            out = out.replace_field(key, kernel(item, params))
        return out

    def decode(self, sample: Sample) -> Sample:
        """The inverse transform (for visualization / back-projection). Not defined by default."""
        raise NotImplementedError(f"{type(self).__name__} defines no decode (inverse)")


# ---------------------------------------------------------------------------
# Adapter coercion registry — drop a FOREIGN transform (a torchvision v2 transform,
# an albumentations transform, a user library object) straight into a Pipeline and the
# right adapter wraps it. Open for extension: register a matcher + factory for any type.
# ---------------------------------------------------------------------------
#: A matcher decides whether an object is adaptable; a factory wraps it into a Transform.
AdapterMatcher = Callable[[Any], bool]
AdapterFactory = Callable[[Any], "Transform"]

_ADAPTERS: List[Tuple[AdapterMatcher, AdapterFactory]] = []


def register_adapter(matcher: AdapterMatcher, factory: AdapterFactory) -> AdapterFactory:
    """Teach :func:`coerce_transform` (and thus ``Pipeline``) to adapt a foreign transform type.

    ``matcher(obj) -> bool`` recognises the objects this adapter handles (keep it import-free —
    inspect ``type(obj).__mro__`` module names rather than importing the library); ``factory(obj)``
    returns a :class:`Transform` wrapping it. Later registrations win on ties (checked last-first).

    Example — make a user's library transforms droppable into a ``Pipeline``::

        register_adapter(
            lambda o: type(o).__module__.startswith("mylib"),
            lambda o: MyLibAdapter(o),
        )
    """
    _ADAPTERS.append((matcher, factory))
    return factory


def coerce_transform(obj: Any) -> "Transform":
    """Return ``obj`` if it is already a :class:`Transform`, else adapt it via a registered adapter.

    Raises a clear ``TypeError`` naming the object when no adapter matches (wrap it with
    :func:`as_transform` / an explicit adapter, or :func:`register_adapter`).
    """
    if isinstance(obj, Transform):
        return obj
    for matcher, factory in reversed(_ADAPTERS):
        try:
            matched = matcher(obj)
        except Exception:  # pragma: no cover - a defensive matcher never breaks coercion
            matched = False
        if matched:
            return factory(obj)
    raise TypeError(
        f"Pipeline: don't know how to adapt {type(obj).__module__}.{type(obj).__name__} into a "
        "Transform. Wrap it with as_transform(...) or an adapter, or register one via "
        "sampleflux.bag.register_adapter(matcher, factory)."
    )


class Pipeline:
    """Sequential application of transforms — ``Pipeline([a, b, c])(sample)`` is ``c(b(a(sample)))``.

    Elements are COERCED (:func:`coerce_transform`): a :class:`Transform` is used as-is, and a
    foreign transform (a torchvision ``transforms.v2`` transform, an albumentations transform, a
    registered user type) is wrapped by its adapter automatically — so libraries drop straight in::

        Pipeline([v2.RandomHorizontalFlip(p=0.5), v2.Normalize(mean, std), A.GaussNoise(p=1.0)])(sample)

    For surgical control (targeting one field key), construct the adapter explicitly with ``only=``.
    """

    def __init__(self, transforms: Sequence[Any]) -> None:
        self.transforms: List[Transform] = [coerce_transform(t) for t in transforms]

    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        return f"Pipeline([{', '.join(type(t).__name__ for t in self.transforms)}])"


class FunctionTransform(Transform):
    """A transform that applies one plain function ``fn(data) -> data`` to every handled field.

    The escape hatch for custom transforms: no kernel registration, no subclass — wrap a
    function and say which item types it applies to (via :func:`as_transform`).
    """

    def __init__(self, fn: Callable[[Any], Any], handles: Sequence[type], only: Optional[List[str]] = None) -> None:
        super().__init__(only=only)
        self._fn = fn
        self.handles = tuple(handles)

    def __call__(self, sample: Sample) -> Sample:
        out = sample
        for key, item in sample.items():
            if self.only is not None and key not in self.only:
                continue
            if isinstance(item, self.handles):
                out = out.replace_field(key, with_data(item, self._fn(item_data(item))))
        return out


def as_transform(
    fn: Callable[[Any], Any], handles: Sequence[type], only: Optional[List[str]] = None
) -> FunctionTransform:
    """Wrap a plain ``fn(data) -> data`` as a :class:`FunctionTransform` over ``handles``."""
    return FunctionTransform(fn, handles, only=only)
