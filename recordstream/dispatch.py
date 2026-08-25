"""The kernel registry — type dispatch for ops (the torchvision-v2 ``_KERNEL_REGISTRY`` pattern).

An op does not hard-code how to handle each value type. Instead a kernel is registered
per ``(transform class, item type)`` pair, and :func:`dispatch` looks one up — walking the
value's MRO so a kernel registered for a base item type also serves its subclasses. This is
the same registry idea as :mod:`recordstream.collate` (batching keyed by representation),
applied to per-type op behaviour.

Registration is open: a downstream package teaches an existing op about a new value
type with one decorator and NO core edit —

    from mypkg.transforms import Denoise      # any Transform subclass
    from mypkg.items import IQSignal          # any registered item type

    @Denoise.kernel(IQSignal)
    def _(value, params):
        return denoise_iq(value, strength=params["strength"])

The transform base exposes ``.kernel(item_type)`` as a thin wrapper over
:func:`register_kernel`; both are documented so either entry point works.
"""

from typing import Any, Callable, Dict, Optional, Tuple

__all__ = ["Kernel", "register_kernel", "get_kernel", "dispatch", "registered_kernels"]

#: A kernel maps ``(value, params) -> value`` — the per-type behaviour of one op.
Kernel = Callable[[Any, Dict[str, Any]], Any]

_KERNEL_REGISTRY: Dict[Tuple[type, type], Kernel] = {}
# Memoized MRO-resolution results (``None`` = a resolved miss). Cleared on every registration.
_DISPATCH_CACHE: Dict[Tuple[type, type], Optional[Kernel]] = {}


def register_kernel(transform_cls: type, item_cls: type) -> Callable[[Kernel], Kernel]:
    """Register a kernel for ``(transform_cls, item_cls)`` (usable as a decorator).

    Re-registering the same pair overwrites (a consumer may deliberately replace a kernel).
    """

    def _register(fn: Kernel) -> Kernel:
        _KERNEL_REGISTRY[(transform_cls, item_cls)] = fn
        _DISPATCH_CACHE.clear()  # a new registration may change what an MRO walk resolves
        return fn

    return _register


def get_kernel(transform_cls: type, item_cls: type) -> Optional[Kernel]:
    """The kernel registered EXACTLY for ``(transform_cls, item_cls)`` (no MRO walk); ``None`` if absent."""
    return _KERNEL_REGISTRY.get((transform_cls, item_cls))


def dispatch(transform_cls: type, item_cls: type) -> Optional[Kernel]:
    """The kernel handling ``item_cls`` for ``transform_cls``, resolved by MRO, or ``None``.

    Resolution walks the transform's MRO (a subclass transform inherits its base's kernels
    unless it overrides them) and, for each, the item's MRO (a kernel on a base item type
    serves subclasses). The MOST specific transform wins; within a transform, the most
    specific item type wins. ``None`` means "this op does not handle this value" —
    the caller passes the entry through untouched. Results are memoized (see
    :data:`_DISPATCH_CACHE`), invalidated on every :func:`register_kernel`.
    """
    key = (transform_cls, item_cls)
    if key in _DISPATCH_CACHE:
        return _DISPATCH_CACHE[key]
    resolved: Optional[Kernel] = None
    for t_cls in transform_cls.__mro__:
        for i_cls in item_cls.__mro__:
            kernel = _KERNEL_REGISTRY.get((t_cls, i_cls))
            if kernel is not None:
                resolved = kernel
                break
        if resolved is not None:
            break
    _DISPATCH_CACHE[key] = resolved
    return resolved


def registered_kernels() -> Tuple[Tuple[str, str], ...]:
    """Every registered ``(transform-name, item-name)`` pair (sorted) — for introspection / tests."""
    return tuple(sorted((t.__name__, i.__name__) for (t, i) in _KERNEL_REGISTRY))
