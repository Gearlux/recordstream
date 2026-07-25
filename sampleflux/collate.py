"""The pluggable collate registry — batch builders keyed by representation.

Batching in sampleflux is two-stage: the engine groups carriers (``Flux.batch`` /
``FlowGraph.batch`` yield ``list``\\ s of N items) and a COLLATE function stacks a group
into one batched carrier. This registry gives consumer packages ONE addressable home for
their task collates — consumers ``register_collate`` their task collates additively, and
callers dispatch by key or by the default record collate.

The string keys primarily serve AI-callable (MCP) tool surfaces, which pass
JSON-serializable names — never function objects — and enumerate the legal values via
:func:`registered_collates`; in Python (and in YAML via a dotted ``!ref:`` to the
function), passing a collate function directly remains the normal path.

The default registered here is ``"record"`` — N plain record dicts collated into ONE
batched record (array payloads stacked per key, per-record item attrs as lists, plain
values gathered into lists). Consumer conventions are deliberately NOT unified here; the
registry is additive.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from loggair import get_logger

from sampleflux.io import PLAIN_TYPE, EncodedItem, decode_item, encode_item
from sampleflux.items import Record

logger = get_logger(__name__)

CollateFn = Callable[[Sequence[Any]], Any]

_REGISTRY: Dict[str, CollateFn] = {}

__all__ = ["CollateFn", "collate", "collate_records", "get_collate", "register_collate", "registered_collates"]


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

    ``key`` picks a registered collate explicitly; omitted, the default ``"record"`` collate
    is used (every carrier is a plain record dict). An empty batch raises.
    """
    if not items:
        raise ValueError("collate: cannot collate an empty batch")
    return get_collate(key or "record")(items)


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


@register_collate("record")
def collate_records(items: Sequence[Record]) -> Record:
    """The record collate: N record dicts → ONE batched record dict.

    Per key (union of keys is NOT taken — every record must carry the same keys, a
    mismatch raises): typed values encode through :func:`~sampleflux.io.encode_item`,
    payloads are stacked via :func:`_stack` (torch → stacked tensor, numpy → stacked
    array, else a list) and each declared item attr becomes a LIST of per-record values,
    decoding back into ONE batched item of the same type. A ``"plain"``-tagged value
    (a scalar / string / bare value) batches as the plain LIST of per-record values.
    """
    if not items:
        raise ValueError("collate_records: cannot collate an empty batch")
    first = items[0]
    if not isinstance(first, dict):
        raise TypeError(f"collate_records: expected record dicts, got {type(first).__name__}")
    keys = list(first.keys())
    for i, record in enumerate(items):
        if not isinstance(record, dict) or list(record.keys()) != keys:
            raise ValueError(
                f"collate_records: item {i} keys {list(record.keys()) if isinstance(record, dict) else '?'} "
                f"do not match the batch keys {keys} — collate requires a homogeneous batch."
            )
    batched: Record = {}
    for key in keys:
        encoded = [encode_item(record[key]) for record in items]
        type_name = encoded[0].type_name
        if type_name == PLAIN_TYPE:
            batched[key] = [e.payload for e in encoded]
            continue
        stacked_payload = _stack([e.payload for e in encoded]) if encoded[0].payload is not None else None
        batched_attrs = {name: [e.attrs.get(name) for e in encoded] for name in encoded[0].attrs}
        batched[key] = decode_item(EncodedItem(type_name=type_name, payload=stacked_payload, attrs=batched_attrs))
    return batched
