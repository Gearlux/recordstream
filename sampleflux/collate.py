"""The pluggable collate registry — batch builders keyed by representation.

Batching in sampleflux is two-stage: the engine groups carriers (``Flux.batch`` /
``FlowGraph.batch`` yield ``list``\\ s of N items) and a COLLATE function stacks a group
into one batched carrier. Historically each consumer package shipped its own collate
(classification / segmentation / detection, with two divergent metadata conventions);
this registry gives them ONE addressable home without changing any of them — consumers
``register_collate`` their task collates additively, and callers dispatch by key or by
the DETECTED carrier kind (:func:`sampleflux.kinds.classify_carrier`).

The string keys primarily serve AI-callable (MCP) tool surfaces, which pass
JSON-serializable names — never function objects — and enumerate the legal values via
:func:`registered_collates`; in Python (and in YAML via a dotted ``!ref:`` to the
function), passing a collate function directly remains the normal path.

Defaults registered here:

- ``"sample"`` — stacks ``input``/``target`` (torch-first, numpy fallback, else kept as a
  list) and gathers per-item metadata into the LIST form (``Sample.is_batched`` True —
  the marainer/sonair convention).
- ``"pair"`` — a metadata-free 2-tuple batch: ``(stacked inputs, stacked targets)``.
- ``"value"`` — bare values stacked directly.

Consumer conventions are deliberately NOT unified here (deltaid/raidar's
``{"per_sample": [...]}`` nesting stays theirs — see TASKS.md); the registry is additive.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from loggair import get_logger

from sampleflux.kinds import classify_carrier
from sampleflux.sample import InputMeta, Sample, TargetMeta

logger = get_logger(__name__)

CollateFn = Callable[[Sequence[Any]], Any]

_REGISTRY: Dict[str, CollateFn] = {}

__all__ = ["CollateFn", "collate", "get_collate", "register_collate", "registered_collates"]


def register_collate(key: str) -> Callable[[CollateFn], CollateFn]:
    """Register a collate function under ``key`` (a kind name or a task alias).

    Usable as a decorator::

        @register_collate("yolo")
        def yolo_collate(items): ...

    Re-registering a key overwrites it (logged at debug — consumers may deliberately
    replace a default).
    """

    def _register(fn: CollateFn) -> CollateFn:
        if key in _REGISTRY:
            logger.debug(f"collate registry: overwriting existing collate for key {key!r}")
        _REGISTRY[key] = fn
        return fn

    return _register


def get_collate(key: str) -> CollateFn:
    """The collate registered under ``key``; a miss names the known keys."""
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"no collate registered for {key!r} (known: {known})") from None


def registered_collates() -> Tuple[str, ...]:
    """The registered keys (sorted)."""
    return tuple(sorted(_REGISTRY))


def collate(items: Sequence[Any], key: Optional[str] = None) -> Any:
    """Collate ``items`` into one batched carrier.

    ``key`` picks a registered collate explicitly; omitted, the DETECTED kind of the
    first item dispatches — a ``TypedSample`` batch routes to ``"typed"``, everything
    else through the classic carrier classifier (``sample`` / ``pair`` / ``value``).
    An empty batch raises.
    """
    from sampleflux.bag.sample import TypedSample

    if not items:
        raise ValueError("collate: cannot collate an empty batch")
    if key is None:
        key = "typed" if isinstance(items[0], TypedSample) else classify_carrier(items[0])
    return get_collate(key)(items)


def _stack(values: List[Any]) -> Any:
    """Best-effort stacking: torch tensors -> stacked tensor, numpy -> stacked array, else a list."""
    first = values[0]
    try:
        import torch

        if isinstance(first, torch.Tensor):
            return torch.stack(list(values))
    except ImportError:  # pragma: no cover - torch is a hard dep today, defensive only
        pass
    try:
        import numpy as np

        if isinstance(first, np.ndarray):
            return np.stack(list(values))
        if isinstance(first, (int, float)) and all(isinstance(v, (int, float)) for v in values):
            return np.asarray(values)
    except ImportError:  # pragma: no cover - numpy is a hard dep, defensive only
        pass
    return list(values)


@register_collate("sample")
def sample_collate(items: Sequence[Any]) -> Sample:
    """Default Sample collate: stacked input/target + LIST-form metadata (``is_batched`` True)."""
    samples = [item if isinstance(item, Sample) else Sample.from_any(item) for item in items]
    return Sample(
        input=_stack([s.input for s in samples]),
        target=_stack([s.target for s in samples]),
        metadata=[dict(s.meta) for s in samples],
    )


@register_collate("pair")
def pair_collate(items: Sequence[Any]) -> Tuple[Any, Any]:
    """Default pair collate: ``(stacked inputs, stacked targets)`` — no metadata anywhere."""
    return _stack([item[0] for item in items]), _stack([item[1] for item in items])


@register_collate("value")
def value_collate(items: Sequence[Any]) -> Any:
    """Default value collate: the bare values stacked."""
    return _stack(list(items))


@register_collate("input_meta")
def input_meta_collate(items: Sequence[Any]) -> InputMeta:
    """Default InputMeta collate: stacked inputs + the per-item metadata dicts as a list."""
    return InputMeta(_stack([item.input for item in items]), [dict(item.metadata) for item in items])


@register_collate("target_meta")
def target_meta_collate(items: Sequence[Any]) -> TargetMeta:
    """Default TargetMeta collate: stacked targets + the per-item metadata dicts as a list."""
    return TargetMeta(_stack([item.target for item in items]), [dict(item.metadata) for item in items])


@register_collate("typed")
def typed_collate(items: Sequence[Any]) -> Any:
    """The typed-bag collate: N ``TypedSample``\\ s → ONE batched ``TypedSample``.

    Per field (union of keys is NOT taken — every sample must carry the same fields, a
    mismatch raises): payloads are stacked via :func:`_stack` (torch → stacked tensor,
    numpy → stacked array, else a list) and each declared item attr becomes a LIST of
    per-item values. Array items come back as the SAME item type over the stacked payload;
    wrapper items likewise (attrs as lists). Roles are preserved. This single convention
    replaces both classic batch shapes (the list-form batched metadata and the
    ``{"per_sample": [...]}`` dict-nest) — trainers read ``primary(batch)`` /
    ``batch.targets()``.
    """
    from sampleflux.bag.io import EncodedItem, decode_item, encode_item
    from sampleflux.bag.sample import TypedSample

    if not items:
        raise ValueError("typed_collate: cannot collate an empty batch")
    first = items[0]
    if not isinstance(first, TypedSample):
        raise TypeError(f"typed_collate: expected TypedSample items, got {type(first).__name__}")
    keys = list(first.keys())
    for i, sample in enumerate(items):
        if not isinstance(sample, TypedSample) or list(sample.keys()) != keys:
            raise ValueError(
                f"typed_collate: item {i} fields {list(sample.keys()) if isinstance(sample, TypedSample) else '?'} "
                f"do not match the batch fields {keys} — collate requires a homogeneous batch."
            )
    fields: Dict[str, Any] = {}
    for key in keys:
        encoded = [encode_item(sample[key]) for sample in items]
        type_name = encoded[0].type_name
        stacked_payload = _stack([e.payload for e in encoded]) if encoded[0].payload is not None else None
        batched_attrs = {name: [e.attrs.get(name) for e in encoded] for name in encoded[0].attrs}
        fields[key] = decode_item(EncodedItem(type_name=type_name, payload=stacked_payload, attrs=batched_attrs))
    return TypedSample(fields, {key: first.role_of(key) for key in keys})
