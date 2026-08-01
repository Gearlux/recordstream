"""Applying ONE op to ONE record — the op-family registry and the dispatch chokepoint.

The bottom of the op-facing layer: every composing op in :mod:`recordstream.ops` reaches
:func:`_apply_op` from here, and nothing here imports back out (see ``docs/architecture.md``
§5 on the import direction this protects). Also home to the ``EXPANDS`` protocol
(:func:`_op_expands` / :func:`_expand`), because "does this op expand, and how do I run it?"
is the same question as "how do I apply this op" — the graph kernel asks all three together.
"""

from typing import Any, Callable, List, Optional, Tuple, cast

from loggair import get_logger

from recordstream.items import NDArrayItem, Record, with_data

logger = get_logger(__name__)


def _op_expands(op: Any) -> bool:
    """True when an op is a 1→N expanding op (explicit ``EXPANDS = True`` class attribute)."""
    return bool(getattr(op, "EXPANDS", False))


def _expand(op: Any, record: Any) -> List[Any]:
    """Run a 1→N EXPANDING op and return its flattened children."""
    raw = op(record)
    if raw is None:
        return []
    return [child for child in raw if child is not None]


#: An op-family MATCHER recognises a library's op objects. Keep it IMPORT-FREE — inspect
#: ``type(op).__mro__`` module names rather than importing the library.
OpMatcher = Callable[[Any], bool]
#: An op-family INVOKER applies one foreign op with its library's native calling
#: convention: ``(record, op) -> Optional[Record]`` (``None`` = drop the record).
OpInvoker = Callable[[Record, Any], Optional[Record]]

#: The registered op families, in registration order. Dispatch checks LAST-registered
#: first, so a later (more specific) family can shadow an earlier one.
_OP_FAMILIES: List[Tuple[str, OpMatcher, OpInvoker]] = []


def register_op_family(name: str, matcher: OpMatcher, invoker: OpInvoker) -> None:
    """Teach the engine to invoke a NEW library's ops natively — the open extension point.

    ``matcher(op) -> bool`` recognises the family's op objects (keep it import-free —
    inspect ``type(op).__mro__`` module names); ``invoker(record, op)`` applies one op with
    the library's own calling convention and returns the new record (``None`` drops it).
    Re-registering a ``name`` REPLACES that family in place; otherwise the family is
    appended, and dispatch checks last-registered first (a more specific family shadows an
    earlier one — register yours after the built-ins to win an overlap).

    Both callables MUST be module-level functions (picklable by reference): the engine's
    spawn-parallel routes ship non-builtin families to worker processes by pickling them.

    Example — kornia augmentations (``nn.Module``s over batched BCHW tensors)::

        def is_kornia(op) -> bool:
            return any(c.__module__.startswith("kornia.augmentation") for c in type(op).__mro__)

        def invoke_kornia(record, op):
            img = record["image"]                    # a CHW torch.Tensor (e.g. after ToTensor)
            out = op(img.unsqueeze(0)).squeeze(0)    # kornia draws once per batch call
            return {**record, "image": out}

        register_op_family("kornia", is_kornia, invoke_kornia)
    """
    entry = (str(name), matcher, invoker)
    for i, (existing, _, _) in enumerate(_OP_FAMILIES):
        if existing == name:
            _OP_FAMILIES[i] = entry
            return
    _OP_FAMILIES.append(entry)


def registered_op_families() -> Tuple[str, ...]:
    """The registered op-family names, in registration/dispatch-precedence order."""
    return tuple(name for name, _, _ in _OP_FAMILIES)


def _sync_op_families(families: Optional[List[Tuple[str, OpMatcher, OpInvoker]]]) -> None:
    """Merge families shipped from the parent process into this process's registry.

    Spawn workers import this module (built-ins present) but never re-run the user's
    registration side effects — the parallel routes therefore pass the parent's
    non-builtin entries along and merge them here (idempotent by name).
    """
    for name, matcher, invoker in families or []:
        register_op_family(name, matcher, invoker)


def _extra_op_families() -> List[Tuple[str, OpMatcher, OpInvoker]]:
    """The non-builtin registry entries — what a spawn worker cannot rebuild by import alone."""
    return [entry for entry in _OP_FAMILIES if entry[0] not in _BUILTIN_FAMILIES]


#: The record keys albumentations understands — its OWN target vocabulary. An albumentations
#: op receives exactly these keys (the ones present) and nothing else, so extra record
#: entries (scalars, domain items) never reach a library that would reject them.
_ALB_KEYS: Tuple[str, ...] = ("image", "mask", "masks", "bboxes", "keypoints", "labels")


def _is_albumentations(op: Any) -> bool:
    """True for an albumentations transform / ``Compose`` — by MRO module name (no import here)."""
    return any(getattr(cls, "__module__", "").startswith("albumentations") for cls in type(op).__mro__)


def _invoke_albumentations(record: Record, op: Any) -> Optional[Record]:
    """albumentations dispatches by KWARG NAME: hand the op exactly its own target keys
    present in the record (one call = one joint draw across them); array outputs are
    re-wrapped in the incoming value's item type (``with_data``) so ``Image``/``Mask``
    keep their type and metadata. Box-carrying augmentation belongs in albumentations' own
    ``A.Compose(..., bbox_params=...)`` — format handling is Compose's job in that library.
    """
    kwargs = {k: record[k] for k in _ALB_KEYS if k in record}
    if not kwargs:
        logger.debug(
            f"albumentations op {type(op).__name__} received no known keys "
            f"({', '.join(_ALB_KEYS)}) — record keys: {list(record)}; passing through."
        )
        return record
    out = op(**kwargs)
    merged = dict(record)
    for key, value in out.items():
        original = record.get(key)
        if isinstance(original, NDArrayItem) and not isinstance(value, NDArrayItem):
            value = with_data(original, value)
        merged[key] = value
    return merged


def _is_torchvision_v2(op: Any) -> bool:
    """True for a torchvision ``transforms.v2`` transform — by MRO module name (no import here)."""
    return any(getattr(cls, "__module__", "").startswith("torchvision.transforms.v2") for cls in type(op).__mro__)


def _invoke_torchvision_v2(record: Record, op: Any) -> Optional[Record]:
    """torchvision v2 natively walks a dict: params sampled once, tensor/tv_tensor/PIL
    leaves transformed, everything else passed through — called as-is."""
    return cast(Record, op(record))


# The built-in families register through the SAME open registry third parties use —
# one mechanism, no privileged code path. Registered at import, so spawn workers
# rebuild them by importing this module.
register_op_family("albumentations", _is_albumentations, _invoke_albumentations)
register_op_family("torchvision_v2", _is_torchvision_v2, _invoke_torchvision_v2)
_BUILTIN_FAMILIES: Tuple[str, ...] = ("albumentations", "torchvision_v2")


def _apply_op(record: Record, op: Any) -> Optional[Record]:
    """Apply one op to the record dict — the engine's op-FAMILY dispatch.

    The single op-application chokepoint shared by the sequential, parallel (via
    :func:`~recordstream.core.stream._worker_task`), streamed, and random-access
    (``__getitem__``) paths; composing ops (``Pipeline`` / ``Parallel`` / ``Enable`` /
    ``RandomApply`` / ``ConfigureOp``) route their inner ops through here so every op is
    applied identically. Each op family is invoked the way its library expects — no
    wrapper/adapter classes: the registered families (:func:`register_op_family`; built-ins
    ``albumentations`` / ``torchvision_v2``) are checked LAST-registered first, and an op
    matching none of them is a native/wiring op called ``op(record) -> Optional[Record]``
    (``None`` drops the record — filter semantics).
    """
    for _name, matcher, invoker in reversed(_OP_FAMILIES):
        try:
            matched = matcher(op)
        except Exception:  # pragma: no cover - a defensive matcher never breaks dispatch
            matched = False
        if matched:
            return invoker(record, op)
    return cast(Optional[Record], op(record))
