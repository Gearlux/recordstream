"""The step MODEL — what a parsed flow step is, and how its references are spelled.

Pure data + string grammar: no execution, no confluid, no op dispatch. Everything else in
:mod:`recordstream.flow` builds on this, so it deliberately sits at the bottom and imports
nothing from its siblings.

It also holds the few things every layer above must agree on WITHOUT importing each other: what
an op declares (:func:`_declaration`, read by the tracer's check and by a subgraph's derived
boundary), and how a SUBGRAPH op is recognised and opened (:func:`_is_subgraph`,
:func:`_open_subgraph`) — by duck typing, because ``subgraph.py`` imports ``parse.py`` and
``parse.py`` must open a subgraph right after building it.
"""

import inspect
from typing import Any, Dict, Literal, NamedTuple, Optional, Sequence, Tuple

RESERVED_STEP_KEYS = ("from", "merge_from", "bind")
"""Step-grammar keys stripped from a step mapping before the op is constructed."""

SUBGRAPH_SEPARATOR = "/"
"""Joins a subgraph step and one of its inner steps into a node name: ``prep/grey``. Because of it
a step name may not contain ``/`` (as ``.`` is reserved for ``step.attr`` references)."""

_Declaration = Literal["none", "names", "types"]


def _declaration(op: Any) -> _Declaration:  # an op of any family, or None for a fan-in step
    """How an op declares its interface: by record-entry NAME (``{key: type}``), by TYPE (a
    ``Transform``'s non-empty tuple of item classes), or not at all (absent or empty).

    The distinction matters because ``check_chain`` reads the declaration as ``{record key: type}``
    and, handed a tuple of classes, takes each CLASS for a key (root ``TASKS.md``; measured:
    ``check_chain([Threshold()], provided={"image"})`` refuses on the entry
    ``"<class 'recordstream.items.NDArrayItem'>"``). A types-only op is therefore never handed over.
    """
    if op is None:
        return "none"
    declared = (getattr(op, "consumes", None), getattr(op, "produces", None))
    if any(isinstance(value, (tuple, list)) and value for value in declared):
        return "types"
    if any(isinstance(value, dict) for value in declared):
        return "names"
    return "none"


def _is_subgraph(op: Any) -> bool:  # an op of any family
    """True for an op whose body is a flow — ``recordstream.flow.subgraph.Subgraph`` or a subclass.

    Read off the op's CLASS attribute ``FLOW_SUBGRAPH`` (never the instance: a tracer's probe forwards
    attribute reads to the op it wraps, and must not be mistaken for the subgraph itself). Such an op
    exposes ``flow_steps`` (its parsed inner steps) and ``output_step``.
    """
    return getattr(type(op), "FLOW_SUBGRAPH", False) is True


#: The attribute a subgraph refusal carries once it names a location, so an outer level adds none.
_LOCATED = "subgraph_location"


def _open_subgraph(op: Any, label: str, where: str = "") -> None:  # op: a Subgraph (see _is_subgraph)
    """Parse a subgraph's inside NOW, so a mistake in it is refused before the first record runs.

    A refusal from inside names the outer step — ``flow step 'prep' (a subgraph): <inner message>``
    (``label`` is ``flow step 'prep'`` or ``Stream.ops[1]``) — plus ``(at file:line:col)`` when the
    caller knows where the subgraph is written. A NESTED refusal already located at the subgraph whose
    refusal it is keeps that one location: each outer level adds its step name, never a second
    ``(at …)`` pointing further out (the marker of ``outer`` says nothing about a mistake in ``deep``).
    The exception keeps its kind (``TypeError`` for an expanding inner op, ``ValueError`` for
    everything else).
    """
    try:
        op.flow_steps
    except (TypeError, ValueError) as refusal:
        located = getattr(refusal, _LOCATED, "")
        at = f" (at {where})" if where and not located else ""
        kind = TypeError if isinstance(refusal, TypeError) else ValueError
        raised = kind(f"{label} (a subgraph): {refusal}{at}")
        setattr(raised, _LOCATED, located or where)
        raise raised from refusal


class StepReferenceError(ValueError):
    """A step reference (``from:`` / ``merge_from:`` / ``bind:``) naming no EARLIER step.

    Raised by :func:`recordstream.flow.parse.parse_flow` with the flow grammar's own message
    (``flow step 'm': from: 'zz' does not name an EARLIER step …``); catch it as the ``ValueError``
    it is. The attributes let a caller say something more useful — a subgraph re-words it when the
    name is not one of its inner steps at all (a step outside it): ``key`` is the reserved step key,
    ``ref`` the reference as written, ``target`` the step it names; ``step`` and ``param`` (for a
    ``bind:``) are the step that wrote it and the parameter it binds.
    """

    def __init__(self, message: str, *, key: str, ref: str, target: str, step: str = "", param: str = "") -> None:
        super().__init__(message)
        self.key = key
        self.ref = ref
        self.target = target
        self.step = step
        self.param = param


_MISSING = object()


def _read_output(op: Any, name: str) -> Any:
    """Read attribute ``name`` off ``op``, looking through ``target``/``op`` wrapper chains.

    Backs the ``bind: {param: "step.attr"}`` grammar — the step op's live ``@output`` after it
    ran. The wrapper walk matters because a step op may be a composing op (``ConfigureOp``
    wrapping the real op in ``target``). Returns ``_MISSING`` when absent.
    """
    cur, seen = op, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        value = getattr(cur, name, _MISSING)
        if value is not _MISSING:
            return value
        cur = getattr(cur, "target", None) or getattr(cur, "op", None)
    return _MISSING


class FlowStep(NamedTuple):
    """One parsed step of a flow document."""

    name: str
    op: Optional[Any]  # live op callable; None = pure fan-in / identity step
    from_: Optional[str]  # None = previous step (first step: the source record)
    bind: Dict[str, str]  # param -> "step" | "step.attr" | "step[key]"
    merge_from: Tuple[str, ...] = ()  # typed fan-in: union these steps' FIELDS, in slot order


class _BindRef(NamedTuple):
    """A parsed ``bind:`` reference."""

    step: str
    attr: Optional[str]  # "step.attr" = the step op's @output attribute
    key: Optional[str]  # "step[key]" = the named ENTRY of the step's record result


def _split_bind_ref(ref: str) -> _BindRef:
    """Split a bind reference into its three shapes: ``step`` / ``step.attr`` / ``step[key]``."""
    text = str(ref)
    if text.endswith("]") and "[" in text:
        head, _, inner = text[:-1].partition("[")
        if head and inner and "." not in head:
            return _BindRef(head, None, inner)
    head, dot, attr = text.partition(".")
    return _BindRef(head, attr if dot else None, None)


def _parse_bind_ref(ref: str, known: Sequence[str]) -> _BindRef:
    parsed = _split_bind_ref(ref)
    if parsed.step not in known:
        raise StepReferenceError(
            f"flow: bind reference {ref!r} does not name an earlier step "
            f"(known steps at this point: {list(known)!r})",
            key="bind",
            ref=ref,
            target=parsed.step,
        )
    return parsed


def _check_reserved_collision(op: Any, step_name: str) -> None:
    """Raise if the op's constructor has a param named like a reserved step key.

    Reserved keys are stripped from the step mapping before the op is built, so such a
    param could never be configured inline — fail loudly instead of silently stealing it.
    """
    try:
        params = inspect.signature(type(op).__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-extension ctor
        return
    clash = [k for k in RESERVED_STEP_KEYS if k in params]
    if clash:
        raise ValueError(
            f"flow step {step_name!r}: op {type(op).__name__!r} has constructor parameter(s) "
            f"{clash!r} that collide with reserved flow step keys {RESERVED_STEP_KEYS!r} — "
            "such an op cannot be configured in a flow document; rename the parameter or "
            "wire the op in the flat ops form instead."
        )
