"""Key projection for RecordStream sources — read only the record keys you need.

Walking a source for a single key (the canonical case: counting classes from the label
key) should not pay for constructing the values you don't need — e.g. decoding image
inputs you are about to throw away. This module adds an **opt-in** projection protocol
plus walk helpers that any consumer can use against any source, with a correct (if
unoptimized) fallback for sources that don't implement the protocol.

The primitive is deliberately general (any subset of record KEYS); :func:`num_classes`
is one helper built on top of it.

Design notes
------------
* :class:`SupportsProjection` is a ``Protocol`` (never a base class), so a source
  opts in by *defining* ``project``, not by inheriting.
* Every public function is a lazy generator (**Lazy Evaluation** mandate) —
  nothing materializes the whole source.
* :func:`num_classes` (integer class-id semantics) is a free function, *not* a
  method on the generic :class:`~recordstream.core.Stream` engine — counting classes is
  a classification concern, and bolting it onto the task-agnostic engine would
  make every ``Stream`` look classification-capable to duck-typed consumers.
"""

from typing import Any, Collection, Iterator, Protocol, runtime_checkable

from recordstream.items import Label, MultiLabel, Record, is_item, item_data


@runtime_checkable
class SupportsProjection(Protocol):
    """A source that can yield partial records restricted to the requested keys.

    Implementers SHOULD avoid building unrequested values — e.g. skip decoding the
    input image when only the label key is asked for; that efficiency is the whole
    point of the protocol. ``keys`` is a subset of the source's record keys.
    """

    def project(self, keys: Collection[str]) -> Iterator[Record]: ...


def project(source: Any, keys: Collection[str]) -> Iterator[Record]:
    """Yield partial records from ``source`` carrying only ``keys``.

    Uses the source's own ``project`` when it implements :class:`SupportsProjection` (the
    efficient path that skips building unrequested values); otherwise falls back to a full
    iteration that keeps only the requested keys. Lazy: a generator.
    """
    want = frozenset(keys)
    if isinstance(source, SupportsProjection):
        yield from source.project(want)
        return
    for record in source:
        yield {k: v for k, v in record.items() if k in want}


def iter_key(source: Any, key: str) -> Iterator[Any]:
    """Lazily yield each record's ``key`` VALUE (skipping other-key construction when supported).

    A :class:`~recordstream.items.Label` unwraps to its ``.value`` (the class id / name), a
    :class:`~recordstream.items.MultiLabel` to its ``.values`` list; any
    other registered item unwraps to its payload via :func:`~recordstream.items.item_data`; a
    plain value passes through verbatim. A record without ``key`` yields ``None``.
    """
    for record in project(source, (key,)):
        value = record.get(key)
        if isinstance(value, MultiLabel):
            yield value.values
        elif isinstance(value, Label):
            yield value.value
        elif is_item(value):
            yield item_data(value)
        else:
            yield value


def _to_int(value: Any) -> int:
    """Coerce a single target into a Python ``int`` class id.

    Handles plain ``int``, numpy scalars, and 0-d / single-element torch tensors
    (via ``.item()``). Rejects ``bool`` (an ``int`` subclass — accepting it would
    silently turn a boolean target into class 0/1) and anything that isn't a
    scalar so callers fail loudly instead of miscounting.
    """
    if isinstance(value, bool):
        raise TypeError(f"target {value!r} is a bool, not a class id")
    if isinstance(value, int):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            result = item()
        except Exception as exc:  # pragma: no cover - exotic array/tensor types
            raise TypeError(f"could not read a scalar class id from target {value!r}: {exc}") from exc
        if isinstance(result, bool):
            raise TypeError(f"target {value!r} resolved to a bool, not a class id")
        if isinstance(result, int):
            return result
        if isinstance(result, float) and result.is_integer():
            return int(result)
        raise TypeError(f"target {value!r} did not yield an integer class id (got {result!r})")
    raise TypeError(f"target {value!r} of type {type(value).__name__} is not a scalar class id")


def num_classes(source: Any, key: str = "class") -> int:
    """Derive the number of classes by walking **every** ``key`` value in ``source``.

    Always walks the full label stream (key-restricted, so other values are never
    constructed when the source supports projection) and returns
    ``max(class_id) + 1`` — the classifier-head size needed to cover the largest
    label, robust to a class id that happens not to appear in this split. Raises
    ``ValueError`` if the source yields no values (or a ``None`` value) under ``key``.

    This is the engine behind a dataset's lazy ``num_classes()`` method.

    Args:
        source: The source to walk (any iterable of records; projection-aware when supported).
        key: The record key holding the class label. Defaults to ``"label"``.
    """
    highest = -1
    for target in iter_key(source, key):
        if target is None:
            raise ValueError(f"num_classes: encountered a record with no {key!r} value — cannot derive a class count.")
        cid = _to_int(target)
        if cid > highest:
            highest = cid
    if highest < 0:
        raise ValueError(f"num_classes: source yielded no {key!r} values — cannot derive a class count.")
    return highest + 1


__all__ = [
    "SupportsProjection",
    "project",
    "iter_key",
    "num_classes",
]
