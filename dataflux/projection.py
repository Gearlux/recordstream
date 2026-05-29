"""Field projection for DataFlux sources — read only the input or only the target.

Walking a source for a single field (the canonical case: counting classes from
*targets*) should not pay for constructing the fields you don't need — e.g.
decoding image inputs you are about to throw away. This module adds an **opt-in**
projection protocol plus walk helpers that any consumer can use against any
source, with a correct (if unoptimized) fallback for sources that don't
implement the protocol.

The primitive is deliberately general (``input`` / ``target`` / ``metadata``
selection); :func:`num_classes` is one helper built on top of it.

Design notes
------------
* :class:`SupportsProjection` is a ``Protocol`` (never a base class), so it
  composes with the DataFlux **Functional Purity** mandate — a source opts in by
  *defining* ``project``, not by inheriting.
* Every public function is a lazy generator (**Lazy Evaluation** mandate) —
  nothing materializes the whole source.
* :func:`num_classes` (integer class-id semantics) is a free function, *not* a
  method on the generic :class:`~dataflux.core.Flux` engine — counting classes is
  a classification concern, and bolting it onto the task-agnostic engine would
  make every ``Flux`` look classification-capable to duck-typed consumers.
"""

from typing import Any, Collection, Iterator, Protocol, runtime_checkable

from dataflux.sample import Sample

INPUT = "input"
TARGET = "target"
METADATA = "metadata"
_FIELDS = (INPUT, TARGET, METADATA)


@runtime_checkable
class SupportsProjection(Protocol):
    """A source that can yield partial :class:`~dataflux.sample.Sample` records.

    Implementers SHOULD avoid building unrequested fields — e.g. skip decoding the
    input image when only ``target`` is asked for; that efficiency is the whole
    point of the protocol. ``fields`` is a subset of
    ``{"input", "target", "metadata"}``; unrequested fields come back as ``None``
    (``{}`` for ``metadata``).
    """

    def project(self, fields: Collection[str]) -> Iterator[Sample]: ...


def project(source: Any, fields: Collection[str]) -> Iterator[Sample]:
    """Yield :class:`Sample` records from ``source`` carrying only ``fields``.

    Uses the source's own ``project`` when it implements
    :class:`SupportsProjection` (the efficient path that skips building
    unrequested fields); otherwise falls back to a full iteration that builds
    every field and nulls the unrequested ones — always correct, just not faster.
    Lazy: a generator that never materializes the source.
    """
    want = frozenset(fields)
    unknown = want - frozenset(_FIELDS)
    if unknown:
        raise ValueError(f"Unknown projection field(s): {sorted(unknown)}; valid fields are {list(_FIELDS)}.")
    if isinstance(source, SupportsProjection):
        yield from source.project(want)
        return
    for raw in source:
        s = Sample.from_any(raw)
        yield Sample(
            input=s.input if INPUT in want else None,
            target=s.target if TARGET in want else None,
            metadata=s.metadata if METADATA in want else {},
        )


def iter_inputs(source: Any) -> Iterator[Any]:
    """Lazily yield each sample's ``input`` (skipping target construction when supported)."""
    for s in project(source, (INPUT,)):
        yield s.input


def iter_targets(source: Any) -> Iterator[Any]:
    """Lazily yield each sample's ``target`` (skipping input construction when supported)."""
    for s in project(source, (TARGET,)):
        yield s.target


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


def num_classes(source: Any) -> int:
    """Derive the number of classes by walking **every** target in ``source``.

    Always walks the full target stream (target-only, so inputs are never
    constructed when the source supports projection) and returns
    ``max(class_id) + 1`` — the classifier-head size needed to cover the largest
    label, robust to a class id that happens not to appear in this split. Raises
    ``ValueError`` if the source yields no targets (or a ``None`` target).

    This is the engine behind a dataset's lazy ``num_classes()`` method.
    """
    highest = -1
    for target in iter_targets(source):
        if target is None:
            raise ValueError("num_classes: encountered a sample with no target — cannot derive a class count.")
        cid = _to_int(target)
        if cid > highest:
            highest = cid
    if highest < 0:
        raise ValueError("num_classes: source yielded no targets — cannot derive a class count.")
    return highest + 1


__all__ = [
    "SupportsProjection",
    "project",
    "iter_inputs",
    "iter_targets",
    "num_classes",
]
