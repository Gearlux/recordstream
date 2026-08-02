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
  method on the generic :class:`~recordstream.core.stream.Stream` engine — counting classes is
  a classification concern, and bolting it onto the task-agnostic engine would
  make every ``Stream`` look classification-capable to duck-typed consumers.
"""

from typing import Any, Collection, Iterator, List, Optional, Protocol, runtime_checkable

from recordstream.items import Record, item_value


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

    A DEFERRED source (a ``!class:`` marker straight out of a config) is materialized first,
    so a caller never has to remember which entry point flows and which does not —
    :meth:`~recordstream.LabelMap.encode` already did, and every consumer of this one was
    writing ``flow(source)`` at the call site to compensate. Flowing a live object is a no-op.
    """
    from confluid import flow

    source = flow(source)
    want = frozenset(keys)
    if isinstance(source, SupportsProjection):
        yield from source.project(want)
        return
    for record in source:
        yield {k: v for k, v in record.items() if k in want}


def iter_key(source: Any, key: str) -> Iterator[Any]:
    """Lazily yield each record's ``key`` VALUE (skipping other-key construction when supported).

    Unwrapping is :func:`~recordstream.items.item_value`'s rule, not a second spelling of it: a
    :class:`~recordstream.items.Label` yields its ``.value`` (the class id / name), a
    :class:`~recordstream.items.MultiLabel` its ``.values`` list, any other registered item its
    payload, and a plain value passes through verbatim. What is THIS function's own is only the
    projection — a record without ``key`` yields ``None``.
    """
    for record in project(source, (key,)):
        yield item_value(record.get(key))


def first_value(source: Any, key: str) -> Any:
    """The first non-``None`` value under ``key`` in ``source`` — ``None`` when there is none.

    The cheapest possible question about a column: ONE peek, which is all a consumer needs to
    learn the KIND of the values without walking the set. The canonical use is a label column,
    where the peek decides whether the targets are class NAMES needing a
    :class:`~recordstream.labels.LabelMap` or ids that pass straight through (ask
    :func:`~recordstream.items.is_class_id` of the answer), and whether the column is
    multi-label (a :class:`~recordstream.items.MultiLabel` arrives here as its ``values``
    LIST, so a sequence IS multi-label — the item type, not a guess about what a list means).

    Built on :func:`iter_key`, so all three of its properties carry over: a
    projection-aware source never builds the values this does not ask for, a deferred source
    is materialized first, and the walk stops at the first hit — a missing or all-``None``
    column costs one full pass and answers ``None`` rather than raising.
    """
    for value in iter_key(source, key):
        if value is not None:
            return value
    return None


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


def class_names(*sources: Any) -> Optional[List[str]]:
    """The class vocabulary carried by the first of ``sources`` that has one.

    The naming counterpart of :func:`num_classes`: that one WALKS a source to count classes,
    this one READS the vocabulary a source already carries — set by
    :meth:`~recordstream.LabelMap.encode` when it wrapped the source, so the names travel with
    the encoded data rather than in a LabelMap the consumer has to keep alongside it.

    Takes several sources because a vocabulary is a property of the RUN, not of whichever
    split happens to carry it: a config may encode only the train set, or hand the eval path a
    pre-encoded test set. ``None`` entries are skipped, so the common
    ``class_names(train_set, val_set, test_set)`` needs no guards at the call site.

    Args:
        *sources: Datasets / streams to consult, in priority order. ``None`` values are ignored.

    Returns:
        The names as a list of ``str``, or ``None`` when no source carries a usable vocabulary
        — a source with no labels, or an unencoded one, is not an error.

    Example::

        names = class_names(train_set, val_set, test_set)   # ['bird', 'cat', 'dog'] or None
    """
    for source in sources:
        names = getattr(source, "class_names", None)
        if not names:
            continue
        try:
            return [str(n) for n in names]
        except TypeError:  # not iterable — a source using the name for something else
            continue
    return None


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


def num_mask_classes(source: Any, key: str = "mask") -> int:
    """The per-pixel twin of :func:`num_classes` — walk a MASK column and return ``max + 1``.

    A per-pixel target is an ``[H, W]`` array of class ids rather than one id, so the count is
    the largest id anywhere in the column plus one. Same contract as :func:`num_classes`
    otherwise: the walk is key-restricted (a projection-aware source never decodes the image
    beside the mask), it covers EVERY record so a class appearing only in the last one still
    sizes the head, and an empty column raises rather than returning a plausible number.

    It is a separate function rather than a widening of :func:`num_classes`, and the reason is
    that function's strictness: ``_to_int`` REJECTS a non-scalar target on purpose, so a
    classification run that is accidentally handed arrays fails loudly instead of miscounting.
    Teaching it to take the max of whatever it gets would trade that guard away for both tasks.

    Args:
        source: The source to walk (any iterable of records; projection-aware when supported).
        key: The record key holding the per-pixel mask. Defaults to ``"mask"``.
    """
    import numpy as np

    highest = -1
    for mask in iter_key(source, key):
        if mask is None:
            raise ValueError(f"num_mask_classes: a record has no {key!r} value — cannot derive a class count.")
        arr = np.asarray(mask)
        if arr.size == 0:
            continue  # an empty mask constrains nothing; a column of them raises below
        highest = max(highest, int(arr.max()))
    if highest < 0:
        raise ValueError(f"num_mask_classes: source yielded no usable {key!r} masks — cannot derive a class count.")
    return highest + 1


__all__ = [
    "SupportsProjection",
    "project",
    "iter_key",
    "num_classes",
    "num_mask_classes",
]
