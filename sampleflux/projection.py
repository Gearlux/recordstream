"""Field projection for SampleFlux sources — read only the input or only the target.

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
* :class:`SupportsProjection` is a ``Protocol`` (never a base class), so a source
  opts in by *defining* ``project``, not by inheriting.
* Every public function is a lazy generator (**Lazy Evaluation** mandate) —
  nothing materializes the whole source.
* :func:`num_classes` (integer class-id semantics) is a free function, *not* a
  method on the generic :class:`~sampleflux.core.Flux` engine — counting classes is
  a classification concern, and bolting it onto the task-agnostic engine would
  make every ``Flux`` look classification-capable to duck-typed consumers.
"""

from typing import Any, Collection, Dict, Iterator, Literal, Protocol, Tuple, get_args, runtime_checkable

from sampleflux.bag.items import Label, item_data
from sampleflux.bag.sample import Role, Sample, primary

#: The projectable :class:`~sampleflux.bag.sample.Sample` roles, as a *closed*
#: ``Literal`` rather than a bare ``str``. Typing the field set this way lets UIs,
#: form-spec builders, and MCP tool schemas enumerate the allowed values straight
#: from the annotation (``typing.get_args(ProjectionField)``) and lets a type
#: checker reject a typo at the call site. ``metadata`` maps onto the bag's ``aux`` role.
ProjectionField = Literal["input", "target", "metadata"]

INPUT: ProjectionField = "input"
TARGET: ProjectionField = "target"
METADATA: ProjectionField = "metadata"
_FIELDS: Tuple[ProjectionField, ...] = get_args(ProjectionField)
_FIELD_ROLES: Dict[str, str] = {"input": "input", "target": "target", "metadata": "aux"}


@runtime_checkable
class SupportsProjection(Protocol):
    """A source that can yield partial :class:`~sampleflux.bag.sample.Sample` records.

    Implementers SHOULD avoid building unrequested fields — e.g. skip decoding the
    input image when only ``target`` is asked for; that efficiency is the whole
    point of the protocol. ``fields`` is a subset of ``{"input", "target", "metadata"}``.
    """

    def project(self, fields: Collection[ProjectionField]) -> Iterator[Sample]: ...


def _carrier_field(sample: Sample, field: ProjectionField) -> Any:
    """Read one field's VALUE from a typed :class:`Sample`.

    The value of the ``input`` / ``target`` role is the FIRST field of that role — a
    ``Label``'s ``.value`` (the class id / scalar), else the item's raw payload
    (:func:`item_data`). A missing role yields ``None``, so a target-only walk feeds
    :func:`num_classes`.
    """
    role: Role = "input" if field == INPUT else "target"
    try:
        _key, item = primary(sample, role)
    except KeyError:
        return None
    return item.value if isinstance(item, Label) else item_data(item)


def project(source: Any, fields: Collection[ProjectionField]) -> Iterator[Sample]:
    """Yield partial records from ``source`` carrying only ``fields``.

    Uses the source's own ``project`` when it implements :class:`SupportsProjection` (the
    efficient path that skips building unrequested fields); otherwise falls back to a full
    iteration that keeps only the fields whose role matches the request. Lazy: a generator.
    """
    want = frozenset(fields)
    unknown = want - frozenset(_FIELDS)
    if unknown:
        raise ValueError(f"Unknown projection field(s): {sorted(unknown)}; valid fields are {list(_FIELDS)}.")
    if isinstance(source, SupportsProjection):
        yield from source.project(want)
        return
    want_roles = {_FIELD_ROLES[f] for f in want}
    for sample in source:
        keep = [k for k in sample.keys() if sample.role_of(k) in want_roles]
        yield Sample({k: sample[k] for k in keep}, {k: sample.role_of(k) for k in keep})


def iter_inputs(source: Any) -> Iterator[Any]:
    """Lazily yield each sample's ``input`` value (skipping target construction when supported)."""
    for s in project(source, (INPUT,)):
        yield _carrier_field(s, INPUT)


def iter_targets(source: Any) -> Iterator[Any]:
    """Lazily yield each sample's ``target`` value (skipping input construction when supported)."""
    for s in project(source, (TARGET,)):
        yield _carrier_field(s, TARGET)


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
    "ProjectionField",
    "SupportsProjection",
    "project",
    "iter_inputs",
    "iter_targets",
    "num_classes",
]
