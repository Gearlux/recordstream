"""Parsing a ``flow:`` mapping into ordered, validated :class:`FlowStep`\\ s.

The only module that knows the DOCUMENT form. It builds ops (flowing confluid markers per
step, because confluid does not auto-flow two-levels-nested markers) and enforces the
grammar's one structural rule: a reference must name an EARLIER step, so document order IS
the schedule and cycles are inexpressible.

A step whose op is a SUBGRAPH (``recordstream.flow.subgraph.Subgraph``) is opened right after it is
built — its inside parsed — so a mistake inside it is refused here, before the first record, naming
the outer step. The subgraph is recognised by duck typing (:func:`_is_subgraph`): ``subgraph.py``
imports this module, so this one must not import it.
"""

from copy import copy
from typing import Any, Dict, List, Optional, Tuple

from confluid import flow
from confluid.fluid import Fluid as _ConfluidFluid
from confluid.fluid import format_yaml_loc

from recordstream.flow.steps import (
    _MISSING,
    RESERVED_STEP_KEYS,
    SUBGRAPH_SEPARATOR,
    FlowStep,
    StepReferenceError,
    _check_reserved_collision,
    _is_subgraph,
    _open_subgraph,
    _parse_bind_ref,
    _read_output,
    _split_bind_ref,
)


def _location(value: Any) -> str:  # a step value: a marker, a mapping holding one under 'op', a live op
    """Where a step is written — ``file:line:col`` of its marker (or its ``op:`` marker), ``""`` when unknown.

    A live SUBGRAPH op answers with the location the parser that built it noted
    (:func:`_note_where_inner_subgraphs_are_written`) — the only trace of its marker once confluid has
    built it inside an outer subgraph.
    """
    if isinstance(value, dict):
        value = value.get("op")
    if isinstance(value, _ConfluidFluid):
        return format_yaml_loc(value)
    if value is not None and _is_subgraph(value):
        return str(getattr(value, "_written_at", "") or "")
    return ""


def _without_step_keys(marker: _ConfluidFluid) -> Tuple[_ConfluidFluid, Dict[str, Any]]:
    """A bare step marker split into (the op's marker, the reserved step keys it carried).

    The keys are read off a COPY: the caller's marker stays as it was written, so parsing the same steps
    again (a subgraph re-parsed after its ``result`` changed, a tracer rebuilding it) wires them the same
    way. Popping them off the caller's marker made every later parse read ``from:`` as absent.
    """
    reserved = {key: marker.kwargs[key] for key in RESERVED_STEP_KEYS if key in marker.kwargs}
    if not reserved:
        return marker, reserved
    stripped = copy(marker)  # its location and target stay; only the kwargs mapping is its own
    stripped.kwargs = {key: value for key, value in marker.kwargs.items() if key not in RESERVED_STEP_KEYS}
    return stripped, reserved


def _note_where_inner_subgraphs_are_written(marker: Any, built: Any) -> None:  # a step's marker, its built op
    """Hand each subgraph built INSIDE ``marker`` the location of the marker it was built from.

    confluid builds a subgraph's inner markers when it builds the subgraph, and a built op keeps no trace
    of where it was written — so a refusal in a nested subgraph could only point at the OUTERMOST marker.
    The locations are read off the marker tree here, before they are lost, and kept on each built
    subgraph as ``_written_at`` for :func:`_location`.
    """
    if not isinstance(marker, _ConfluidFluid) or not _is_subgraph(built):
        return
    written, made = marker.kwargs.get("steps"), getattr(built, "steps", None)
    if not isinstance(written, dict) or not isinstance(made, dict):
        return
    for name, value in written.items():
        inner_marker = value.get("op") if isinstance(value, dict) else value
        held = made.get(name)
        inner = held.get("op") if isinstance(held, dict) else held
        if isinstance(inner_marker, _ConfluidFluid) and inner is not None and _is_subgraph(inner):
            inner._written_at = format_yaml_loc(inner_marker)
            _note_where_inner_subgraphs_are_written(inner_marker, inner)


def _refuse_read_into_subgraph(step: str, key: str, ref: str, producers: Dict[str, FlowStep], where: str) -> None:
    """Refuse an OUTER ``from:`` / ``merge_from:`` naming a step INSIDE a subgraph (``prep/grey``).

    The flow grammar's own message ("does not name an EARLIER step") is true but sends the reader
    looking for a typo; the fix is to move the step out of the subgraph, so that is what this says.
    """
    head, slash, inner = ref.partition(SUBGRAPH_SEPARATOR)
    producer = producers.get(head)
    if not slash or producer is None or not _is_subgraph(producer.op):
        return
    at = f" (at {where})" if where else ""
    raise ValueError(
        f"flow step {step!r}: {key}: {ref!r} reads the inner step {inner!r} of {head!r}, which is a subgraph — a "
        f"step outside a subgraph cannot read a step inside it; move that step out of the subgraph{at}"
    )


def _refuse_bind_into_subgraph(step: str, param: str, ref: str, producers: Dict[str, FlowStep], where: str) -> None:
    """Refuse an OUTER ``bind:`` that reads a value of a step INSIDE a subgraph (DESIGN decision 2).

    Two spellings reach for it: ``measure.level`` (``level`` is not an attribute of the subgraph
    ``measure`` itself — it is an inner step's ``@output``) and ``measure/level.level``. Either would
    fail at the first record (an ``AttributeError``) or read nothing; the answer is to move the step
    out of the subgraph, so that is what the refusal says, before anything runs.
    """
    parsed = _split_bind_ref(ref)
    head, slash, inner = parsed.step.partition(SUBGRAPH_SEPARATOR)
    producer = producers.get(head)
    if producer is None or not _is_subgraph(producer.op):
        return
    if slash:
        what = f"reads the inner step {inner!r} of {head!r}"
    elif parsed.attr is not None and _read_output(producer.op, parsed.attr) is _MISSING:
        what = f"reads {parsed.attr!r} of {head!r}"
    else:
        return
    at = f" (at {where})" if where else ""
    raise ValueError(
        f"flow step {step!r}: bind {param}={ref!r} {what}, which is a subgraph — a step outside a subgraph "
        f"cannot read a value of a step inside it; move that step out of the subgraph{at}"
    )


def parse_flow(flow_doc: Any, outputs: str = "", build: bool = True) -> Tuple[List[FlowStep], str]:
    """Parse a flow mapping into ordered :class:`FlowStep`\\ s + the resolved output step name.

    ``flow_doc`` is the ``flow:`` mapping — step values may be confluid markers (from
    ``resolve()``/``load()``), plain dicts (pure fan-in steps, or programmatic
    ``{"op": <op>, "from": ...}`` form), or live op callables. Reserved keys are read off
    (the mapping and its markers are never changed, so a second parse wires the same graph);
    markers are flowed per step (confluid does not auto-flow two-levels-nested markers).
    Validates: step names carry no dots (reserved for ``step.attr`` references) and no slashes
    (reserved for a subgraph's inner nodes, ``prep/grey``), every reference points to an EARLIER
    step, a subgraph's inside holds together, and no ``bind:`` reaches into a subgraph.

    ``build=False`` keeps a marker step UNBUILT (the op stays a Fluid marker) — for
    structural consumers (converters/importers) that must not materialize ops; a subgraph is then
    not opened either.
    """
    if not isinstance(flow_doc, dict) or not flow_doc:
        raise ValueError("flow: expected a non-empty mapping of step-name -> op")

    steps: List[FlowStep] = []
    seen: List[str] = []
    producers: Dict[str, FlowStep] = {}
    for name, value in flow_doc.items():
        name = str(name)
        if "." in name:
            raise ValueError(f"flow: step name {name!r} may not contain '.' (reserved for @output refs)")
        if SUBGRAPH_SEPARATOR in name:
            raise ValueError(
                f"flow: step name {name!r} may not contain '/' (reserved for the nodes inside a subgraph: "
                "'prep/grey' is the step 'grey' inside the subgraph 'prep')"
            )
        if name in seen:
            raise ValueError(f"flow: duplicate step name {name!r}")
        where = _location(value)

        reserved: Dict[str, Any] = {}
        op: Optional[Any]
        if isinstance(value, _ConfluidFluid):
            marker, reserved = _without_step_keys(value)
            op = flow(marker) if build else marker
            if build:
                _note_where_inner_subgraphs_are_written(value, op)
        elif isinstance(value, dict):
            extra = value.get("op")
            reserved = {k: v for k, v in value.items() if k in RESERVED_STEP_KEYS}
            unknown = [k for k in value if k not in RESERVED_STEP_KEYS and k != "op"]
            if unknown:
                raise ValueError(
                    f"flow step {name!r}: unknown step key(s) {unknown!r} — a plain-mapping step "
                    f"accepts only {RESERVED_STEP_KEYS!r} and 'op'"
                )
            op = flow(extra) if (build and isinstance(extra, _ConfluidFluid)) else extra
            if op is not extra:
                _note_where_inner_subgraphs_are_written(extra, op)
        elif callable(value):
            op = value
        elif value is None:
            op = None
        else:
            raise TypeError(f"flow step {name!r}: expected an op, a marker, or a mapping — got {type(value).__name__}")

        if op is not None and not isinstance(op, _ConfluidFluid) and not callable(op):
            raise TypeError(f"flow step {name!r}: op is not callable ({type(op).__name__})")
        if op is not None and not isinstance(op, _ConfluidFluid):
            _check_reserved_collision(op, name)
        if op is not None and not isinstance(op, _ConfluidFluid) and _is_subgraph(op):
            if where:
                op._written_at = where
            _open_subgraph(op, f"flow step {name!r}", where)

        from_ = reserved.get("from")
        if from_ is not None and str(from_) not in seen:
            _refuse_read_into_subgraph(name, "from", str(from_), producers, where)
            raise StepReferenceError(
                f"flow step {name!r}: from: {from_!r} does not name an EARLIER step "
                f"(document order is the schedule; steps so far: {seen!r})",
                key="from",
                ref=str(from_),
                target=str(from_),
                step=name,
            )
        merge_raw = reserved.get("merge_from")
        merge_from: Tuple[str, ...] = ()
        if merge_raw is not None:
            merge_from = (str(merge_raw),) if isinstance(merge_raw, str) else tuple(str(r) for r in merge_raw)
            for ref in merge_from:
                if ref not in seen:
                    _refuse_read_into_subgraph(name, "merge_from", ref, producers, where)
                    raise StepReferenceError(
                        f"flow step {name!r}: merge_from: {ref!r} does not name an EARLIER step "
                        f"(document order is the schedule; steps so far: {seen!r})",
                        key="merge_from",
                        ref=ref,
                        target=ref,
                        step=name,
                    )
        bind_raw = reserved.get("bind") or {}
        if not isinstance(bind_raw, dict):
            raise TypeError(f"flow step {name!r}: bind must be a mapping of param -> step[.output]")
        bind: Dict[str, str] = {}
        for param, ref in bind_raw.items():
            _refuse_bind_into_subgraph(name, str(param), str(ref), producers, where)
            try:
                _parse_bind_ref(str(ref), seen)  # validates
            except StepReferenceError as missing:
                missing.step, missing.param = name, str(param)
                raise
            bind[str(param)] = str(ref)
        if bind and op is None:
            raise ValueError(f"flow step {name!r}: bind requires an op to configure")

        step = FlowStep(
            name=name,
            op=op,
            from_=None if from_ is None else str(from_),
            bind=bind,
            merge_from=merge_from,
        )
        steps.append(step)
        producers[name] = step
        seen.append(name)

    out = str(outputs) if outputs else steps[-1].name
    if out not in seen:
        raise ValueError(f"flow: outputs {out!r} does not name a step (steps: {seen!r})")
    return steps, out
