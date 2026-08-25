"""The step MODEL — what a parsed flow step is, and how its references are spelled.

Pure data + string grammar: no execution, no confluid, no op dispatch. Everything else in
:mod:`recordstream.flow` builds on this, so it deliberately sits at the bottom and imports
nothing from its siblings.
"""

import inspect
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

RESERVED_STEP_KEYS = ("from", "merge_from", "bind")
"""Step-grammar keys stripped from a step mapping before the op is constructed."""

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
        raise ValueError(
            f"flow: bind reference {ref!r} does not name an earlier step "
            f"(known steps at this point: {list(known)!r})"
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
