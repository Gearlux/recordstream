"""The pluggable collate registry — batch builders keyed by representation.

Batching in recordstream is two-stage: the engine groups carriers (``Stream.batch`` /
``FlowGraph.batch`` yield ``list``\\ s of N items) and a COLLATE function stacks a group
into one batched carrier. The registry keys serve surfaces that pass JSON-serializable
NAMES rather than function objects (an AI-callable tool, ``RecordSequence``'s ``collate=``
string form), enumerating the legal values via :func:`registered_collates`; in Python (and
in YAML via a dotted ``!ref:`` to the function), passing a collate function directly is the
normal path.

TWO collates are registered here, and they differ in ONE decision — whether array payloads
are STACKED — because that decision belongs to the model, not to the data:

* ``"record"`` (the default) — array payloads stacked per key, per-record item attrs as
  lists, plain values gathered into lists. What a classifier or a dense-target detector wants.
* ``"list"`` — nothing stacked; every key becomes a per-record list, items kept as items.
  What a model taking variable-size inputs wants (a torchvision detector's ``List[Tensor]``).

**The registry is engine-internal — only shape-generic, parameter-free collates register
(decided 2026-08-06, closing the open question).** A TASK's batch shape (a fastai
``(x, y)`` tuple, a keras ``(inputs, targets)`` array pair) is parameterized by task-decided
state — which keys are input and target, an int-id vs multi-hot target, a channels-last
transpose — that a bare registry string cannot carry: a registered ``"tuple"`` with baked
default keys would be the silent-wrong-keys trap. Consumers therefore pass their callable
straight to the slot that takes one (``DataLoader(collate_fn=...)``,
``RecordSequence(transform=...)``) and never register it. Measured before deciding: no
workspace consumer had ever registered one.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from loggair import get_logger

from recordstream.io import PLAIN_TYPE, EncodedItem, decode_item, encode_item
from recordstream.items import Record

logger = get_logger(__name__)

CollateFn = Callable[[Sequence[Any]], Any]

#: Bound to `CollateFn` but PRESERVED through the decorator, so registering a collate does
#: not erase its own signature — `collate_records`' `stack=` stays visible to callers and
#: to a type checker (a plain `-> CollateFn` return flattened every registered collate to
#: the loose one-argument protocol).
F = TypeVar("F", bound=CollateFn)

_REGISTRY: Dict[str, CollateFn] = {}

__all__ = [
    "CollateFn",
    "collate",
    "collate_list",
    "collate_records",
    "get_collate",
    "register_collate",
    "registered_collates",
]


def register_collate(key: str) -> Callable[[F], F]:
    """Register a collate function under ``key``.

    For SHAPE-GENERIC, parameter-free collates only (see the module docstring) — a
    task-shaped collate carries task state a key cannot, and is passed as a callable to
    the slot that takes one instead of being registered. Usable as a decorator::

        @register_collate("record")
        def collate_records(items): ...

    Re-registering a key overwrites it (logged at debug — a deliberate replacement of a
    default is legal).
    """

    def _register(fn: F) -> F:
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
    is used (every carrier is a plain dict). An empty batch raises.
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
def collate_records(items: Sequence[Record], stack: bool = True) -> Record:
    """The record collate: N record dicts → ONE batched record dict.

    Per key (union of keys is NOT taken — every record must carry the same keys, a
    mismatch raises): typed values encode through :func:`~recordstream.io.encode_item`,
    payloads are stacked via :func:`_stack` (torch → stacked tensor, numpy → stacked
    array, else a list) and each declared item attr becomes a LIST of per-record values,
    decoding back into ONE batched item of the same type. A ``"plain"``-tagged value
    (a scalar / string / bare value) batches as the plain LIST of per-record values.

    Args:
        items: The records to batch.
        stack: Whether array payloads are STACKED into one array/tensor. ``False`` is the
            registered ``"list"`` collate (:func:`collate_list`) — see it for why the choice
            belongs to the caller rather than to the data.
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
        if not stack:
            # The values VERBATIM, one per record — items stay items, so per-record metadata
            # (an `Image`'s layout) survives instead of being flattened into one batched item.
            batched[key] = [record[key] for record in items]
            continue
        encoded = [encode_item(record[key]) for record in items]
        type_name = encoded[0].type_name
        if type_name == PLAIN_TYPE:
            batched[key] = [e.payload for e in encoded]
            continue
        payloads = [e.payload for e in encoded]
        stacked_payload = _stack_or_explain(payloads, key) if encoded[0].payload is not None else None
        batched_attrs = {name: [e.attrs.get(name) for e in encoded] for name in encoded[0].attrs}
        batched[key] = decode_item(EncodedItem(type_name=type_name, payload=stacked_payload, attrs=batched_attrs))
    return batched


@register_collate("list")
def collate_list(items: Sequence[Record]) -> Record:
    """The UNSTACKED collate: N record dicts → one record whose every key is a per-record LIST.

    The sibling of :func:`collate_records`, and the reason the registry exists: **the batch
    SHAPE is a model requirement, not a property of the data.** A torchvision detector takes
    ``List[Tensor]`` because its images differ in size; a classifier takes one ``[N, C, H, W]``
    tensor. Left implicit, that choice is made by accident — under the default collate a column
    stacks if it holds array ITEMS and stays a list if it holds PLAIN values, so the shape ends
    up decided by whether an op like ``ToTensor`` happened to run. This key lets the consumer
    DECLARE it instead.

    Two consequences worth knowing:

    * a variable-size column is fine here (nothing is stacked), where the default collate
      raises — and the type does not depend on the data, because you asked for lists;
    * items stay ITEMS (a list of :class:`~recordstream.Image`, not a list of bare arrays), so
      per-record metadata survives. The read-back helpers (:func:`~recordstream.batch_values`,
      :func:`~recordstream.batch_regions`, :func:`~recordstream.batch_metadata`) accept BOTH
      shapes, so a consumer reads the batch the same way under either collate.

    Example::

        DataLoader(stream, collate_fn=collate_list)      # or: collate(items, key="list")
    """
    return collate_records(items, stack=False)


def _stack_or_explain(payloads: List[Any], key: str) -> Any:
    """:func:`_stack`, but a stacking failure names the KEY, the shapes and the way out.

    Raw, the failure is ``ValueError: all input arrays must have the same shape`` from inside
    numpy — which says nothing about which column, what shapes, or what to do. A variable-size
    column is a legitimate batch; it just is not a STACKED one.
    """
    try:
        return _stack(payloads)
    except (ValueError, RuntimeError) as exc:
        shapes = [getattr(p, "shape", None) for p in payloads]
        raise ValueError(
            f"collate_records: cannot stack the {key!r} column — its payloads have differing "
            f"shapes {shapes}. Either make them uniform upstream (a resize op), or batch with "
            f'the "list" collate (`collate_list` / `collate(items, key="list")`), which keeps '
            f"every column as a per-record list."
        ) from exc
