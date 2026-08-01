"""Parsing a ``flow:`` mapping into ordered, validated :class:`FlowStep`\\ s.

The only module that knows the DOCUMENT form. It builds ops (flowing confluid markers per
step, because confluid does not auto-flow two-levels-nested markers) and enforces the
grammar's one structural rule: a reference must name an EARLIER step, so document order IS
the schedule and cycles are inexpressible.
"""

from typing import Any, Dict, List, Optional, Tuple

from confluid import flow
from confluid.fluid import Fluid as _ConfluidFluid

from recordstream.flow.steps import RESERVED_STEP_KEYS, FlowStep, _check_reserved_collision, _parse_bind_ref


def parse_flow(flow_doc: Any, outputs: str = "", build: bool = True) -> Tuple[List[FlowStep], str]:
    """Parse a flow mapping into ordered :class:`FlowStep`\\ s + the resolved output step name.

    ``flow_doc`` is the ``flow:`` mapping — step values may be confluid markers (from
    ``resolve()``/``load()``), plain dicts (pure fan-in steps, or programmatic
    ``{"op": <op>, "from": ...}`` form), or live op callables. Reserved keys are popped;
    markers are flowed per step (confluid does not auto-flow two-levels-nested markers).
    Validates: step names carry no dots, every reference points to an EARLIER step.

    ``build=False`` keeps a marker step UNBUILT (the op stays a Fluid marker) — for
    structural consumers (converters/importers) that must not materialize ops.
    """
    if not isinstance(flow_doc, dict) or not flow_doc:
        raise ValueError("flow: expected a non-empty mapping of step-name -> op")

    steps: List[FlowStep] = []
    seen: List[str] = []
    for name, value in flow_doc.items():
        name = str(name)
        if "." in name:
            raise ValueError(f"flow: step name {name!r} may not contain '.' (reserved for @output refs)")
        if name in seen:
            raise ValueError(f"flow: duplicate step name {name!r}")

        reserved: Dict[str, Any] = {}
        op: Optional[Any]
        if isinstance(value, _ConfluidFluid):
            for key in RESERVED_STEP_KEYS:
                if key in value.kwargs:
                    reserved[key] = value.kwargs.pop(key)
            op = flow(value) if build else value
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

        from_ = reserved.get("from")
        if from_ is not None and str(from_) not in seen:
            raise ValueError(
                f"flow step {name!r}: from: {from_!r} does not name an EARLIER step "
                f"(document order is the schedule; steps so far: {seen!r})"
            )
        merge_raw = reserved.get("merge_from")
        merge_from: Tuple[str, ...] = ()
        if merge_raw is not None:
            merge_from = (str(merge_raw),) if isinstance(merge_raw, str) else tuple(str(r) for r in merge_raw)
            for ref in merge_from:
                if ref not in seen:
                    raise ValueError(
                        f"flow step {name!r}: merge_from: {ref!r} does not name an EARLIER step "
                        f"(document order is the schedule; steps so far: {seen!r})"
                    )
        bind_raw = reserved.get("bind") or {}
        if not isinstance(bind_raw, dict):
            raise TypeError(f"flow step {name!r}: bind must be a mapping of param -> step[.output]")
        bind: Dict[str, str] = {}
        for param, ref in bind_raw.items():
            _parse_bind_ref(str(ref), seen)  # validates
            bind[str(param)] = str(ref)
        if bind and op is None:
            raise ValueError(f"flow step {name!r}: bind requires an op to configure")

        steps.append(
            FlowStep(
                name=name,
                op=op,
                from_=None if from_ is None else str(from_),
                bind=bind,
                merge_from=merge_from,
            )
        )
        seen.append(name)

    out = str(outputs) if outputs else steps[-1].name
    if out not in seen:
        raise ValueError(f"flow: outputs {out!r} does not name a step (steps: {seen!r})")
    return steps, out
