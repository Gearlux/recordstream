"""The pluggable collate registry — batch builders keyed by representation.

Batching in sampleflux is two-stage: the engine groups carriers (``Flux.batch`` /
``FlowGraph.batch`` yield ``list``\\ s of N items) and a COLLATE function stacks a group
into one batched carrier. This registry gives consumer packages ONE addressable home for
their task collates — consumers ``register_collate`` their task collates additively, and
callers dispatch by key or by the default typed collate.

The string keys primarily serve AI-callable (MCP) tool surfaces, which pass
JSON-serializable names — never function objects — and enumerate the legal values via
:func:`registered_collates`; in Python (and in YAML via a dotted ``!ref:`` to the
function), passing a collate function directly remains the normal path.

The default registered here is ``"typed"`` — N :class:`~sampleflux.bag.sample.Sample` bags
collated into ONE batched bag (payloads stacked per field, per-item attrs as lists, roles
preserved). Consumer conventions are deliberately NOT unified here; the registry is additive.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from loggair import get_logger

from sampleflux.bag.sample import Sample

logger = get_logger(__name__)

CollateFn = Callable[[Sequence[Any]], Any]

_REGISTRY: Dict[str, CollateFn] = {}

__all__ = ["CollateFn", "collate", "get_collate", "register_collate", "registered_collates", "typed_collate"]


def register_collate(key: str) -> Callable[[CollateFn], CollateFn]:
    """Register a collate function under ``key`` (a task alias).

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

    ``key`` picks a registered collate explicitly; omitted, the default ``"typed"`` collate
    is used (every carrier is a :class:`~sampleflux.bag.sample.Sample` bag). An empty batch raises.
    """
    if not items:
        raise ValueError("collate: cannot collate an empty batch")
    return get_collate(key or "typed")(items)


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


@register_collate("typed")
def typed_collate(items: Sequence[Any]) -> Any:
    """The typed-bag collate: N ``Sample``\\ s → ONE batched ``Sample``.

    Per field (union of keys is NOT taken — every sample must carry the same fields, a
    mismatch raises): payloads are stacked via :func:`_stack` (torch → stacked tensor,
    numpy → stacked array, else a list) and each declared item attr becomes a LIST of
    per-item values. Array items come back as the SAME item type over the stacked payload;
    wrapper items likewise (attrs as lists). Roles are preserved. Trainers read
    ``primary(batch)`` / ``batch.targets()``.
    """
    from sampleflux.bag.io import EncodedItem, decode_item, encode_item

    if not items:
        raise ValueError("typed_collate: cannot collate an empty batch")
    first = items[0]
    if not isinstance(first, Sample):
        raise TypeError(f"typed_collate: expected Sample items, got {type(first).__name__}")
    keys = list(first.keys())
    for i, sample in enumerate(items):
        if not isinstance(sample, Sample) or list(sample.keys()) != keys:
            raise ValueError(
                f"typed_collate: item {i} fields {list(sample.keys()) if isinstance(sample, Sample) else '?'} "
                f"do not match the batch fields {keys} — collate requires a homogeneous batch."
            )
    fields: Dict[str, Any] = {}
    for key in keys:
        encoded = [encode_item(sample[key]) for sample in items]
        type_name = encoded[0].type_name
        stacked_payload = _stack([e.payload for e in encoded]) if encoded[0].payload is not None else None
        batched_attrs = {name: [e.attrs.get(name) for e in encoded] for name in encoded[0].attrs}
        fields[key] = decode_item(EncodedItem(type_name=type_name, payload=stacked_payload, attrs=batched_attrs))
    return Sample(fields, {key: first.role_of(key) for key in keys})
