"""``Subgraph`` — a ``flow:`` mapping used as ONE op, written inline in the step that uses it.

.. code-block:: yaml

    flow:
      prep:
        op: !class:recordstream.flow.subgraph.Subgraph
          steps:
            grey:
              op: !class:recordstream.ops.image.ConvertMode {mode: L}
            scaled:
              op: !class:recordstream.ops.numpy.Scale {source_min: 0.0, source_max: 255.0}
          result: scaled
      mask:
        op: !class:recordstream.ops.numpy.Threshold {low_level: 0.5}
    outputs: mask

It is an ordinary ``@configurable`` op — no new YAML syntax, no confluid change — whose body is
parsed by :func:`recordstream.flow.parse.parse_flow` and run by the SAME kernel a ``FlowGraph`` runs
(:func:`recordstream.flow.execute.run_steps`). There is no second executor: a subgraph is the flow
grammar applied to the record one step receives.

The rules that make it safe to use, each one refused BEFORE the first record (the outer
``parse_flow`` opens a subgraph right after building it, a ``Stream`` when it compiles its ops):

* ``result:`` names an inner step (blank = the last one) — it is ``result`` and never ``outputs``,
  because confluid broadcasts a document's own top-level ``outputs:`` into a constructor parameter
  of that name (measured: the outer flow's ``outputs: boxes`` overwrote it);
* no inner op expands 1→N — a subgraph returns one record per record it receives;
* an inner ``from:`` / ``merge_from:`` / ``bind:`` names a step INSIDE — the inside reads only the
  record the subgraph receives and the steps before it there;
* a reserved step key is not written inside an inner op's ``!class:`` marker, where confluid sets it
  as a plain attribute and the step never sees it (measured: silently wrong, mask mean 0.0 instead
  of 0.2855).

Its inner nodes are named ``prep/grey`` (the tracer's descent, a visual editor), so a step name may
not contain ``/``. Rationale: ``docs/architecture.md`` §23; usage: ``docs/graph.md`` → "Subgraphs".
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple

from confluid import configurable
from confluid.fluid import Fluid as _ConfluidFluid

from recordstream.core.families import _op_expands
from recordstream.flow.execute import _result_readers, run_steps
from recordstream.flow.parse import parse_flow
from recordstream.flow.steps import RESERVED_STEP_KEYS, FlowStep, StepReferenceError, _declaration
from recordstream.items import Record

#: The reader accounting the kernel takes (see :func:`recordstream.flow.execute._result_readers`).
_Readers = Dict[str, List[Tuple[int, str]]]

#: A subgraph's ``steps``: step name -> step value. ``Any`` because a step value is a ``!class:`` marker,
#: a mapping ``{op, from, merge_from, bind}`` or a live op of any op family — the flow grammar admits all
#: three, and no narrower type does.
Steps = Dict[str, Any]


def _as_yaml(value: Any) -> str:  # a reserved step key's value: a step name or a list of them
    """A reserved key's value spelled as it is written in a flow document (for the refusal's fix).

    A ``bind:`` mapping never reaches here: confluid reads a mapping under a ``!class:`` marker as
    addressed configuration and drops it without a trace (measured), so only ``from:`` and
    ``merge_from:`` survive onto the op to be refused.
    """
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_as_yaml(item) for item in value) + "]"
    return str(value)


def _is_subclass(made: Any, needed: Any) -> bool:  # two entries of a types-only declaration
    return isinstance(made, type) and isinstance(needed, type) and issubclass(made, needed)


@configurable(category="op", group="compose")
class Subgraph:
    """A flow used as ONE op: its inner steps run on the engine's kernel against the record it receives.

    The inner steps are the flow grammar — a step is an op, a ``!class:`` marker, or a mapping
    ``{op, from, merge_from, bind}`` — and they read only the record the subgraph receives and each
    other. ``consumes`` / ``produces`` / ``flags`` are DERIVED from the inner ops, so a chain check
    sees a subgraph as the steps it holds.

    Args:
        steps: The inner steps, step name -> op (the same grammar as a flow: mapping); an included steps file works.
        result: The inner step whose record the subgraph returns. Blank (the default) = the last inner step.
    """

    #: Marks the class for the engine's duck-typed checks (``recordstream.flow.steps._is_subgraph``):
    #: ``parse.py`` and ``core.stream`` must recognise a subgraph without importing this module.
    FLOW_SUBGRAPH: ClassVar[bool] = True

    def __init__(self, steps: Optional[Steps] = None, result: str = "") -> None:
        # Stores only: parsing BUILDS the inner ops (a marker is a construction), so it waits for first use.
        self.steps = steps
        self.result = str(result or "")
        # (the steps mapping it was parsed from, the result it was parsed with, steps, output step)
        self._parsed: Optional[Tuple[Optional[Steps], str, List[FlowStep], str]] = None
        self._readers: Optional[_Readers] = None
        # Where its marker is written (``file:line:col``), noted by the parser that built it — confluid
        # keeps no location on a built op, so a refusal inside a NESTED subgraph can still point here.
        self._written_at = ""

    # -- the inside, parsed on first use --------------------------------------------------------

    def _ensure(self) -> Tuple[List[FlowStep], str]:
        """The parsed inside — cached against the ``steps`` object and ``result`` it was parsed from,
        so a reassigned ``steps`` (a host configuring the op after construction) is parsed again."""
        cached = self._parsed
        if cached is None or cached[0] is not self.steps or cached[1] != self.result:
            steps, output = self._parse()
            self._parsed = (self.steps, self.result, steps, output)
            self._readers = None
            return steps, output
        return cached[2], cached[3]

    def _parse(self) -> Tuple[List[FlowStep], str]:
        """Parse ``steps`` with the flow grammar and refuse what a subgraph cannot hold (module docstring)."""
        steps = self.steps
        if not steps:
            raise ValueError("Subgraph: steps is empty — give it at least one step")
        if not isinstance(steps, dict):
            raise ValueError(f"Subgraph: steps must be a mapping of step name -> op, got {type(steps).__name__}")
        names = [str(name) for name in steps]
        if self.result and self.result not in names:
            raise ValueError(f"Subgraph: result {self.result!r} does not name an inner step (the steps are: {names})")
        for name, value in steps.items():
            _refuse_swallowed_keys(str(name), value)
        try:
            parsed, output = parse_flow(steps, self.result)
        except StepReferenceError as missing:
            if missing.target in names:  # an inner step written LATER: the flow grammar's own message says it
                raise
            how = (
                f"bind: {missing.param}={missing.ref!r}" if missing.key == "bind" else f"{missing.key}: {missing.ref!r}"
            )
            raise ValueError(
                f"Subgraph: step {missing.step!r} reads {missing.target!r} ({how}), which is not a step inside this "
                "subgraph — a step inside a subgraph reads only the record the subgraph receives and the steps "
                f"before it inside (the steps are: {names})"
            ) from missing
        for step in parsed:
            if step.op is not None and _op_expands(step.op):
                raise TypeError(
                    f"Subgraph: step {step.name!r} ({type(step.op).__name__}) is a 1→N expanding op, and a "
                    "subgraph returns one record per record it receives — move the step out of the subgraph"
                )
        return parsed, output

    @property
    def flow_steps(self) -> List[FlowStep]:
        """The parsed inner steps, their ops built (parsed on first read; a refusal raises here)."""
        return self._ensure()[0]

    @property
    def output_step(self) -> str:
        """The inner step whose record the subgraph returns — ``result``, or the last inner step."""
        return self._ensure()[1]

    @property
    def readers(self) -> _Readers:
        """The kernel's reader accounting for the inside, computed ONCE (see ``run_steps_multi``)."""
        steps, output = self._ensure()
        if self._readers is None:
            self._readers = _result_readers(steps, output)
        return self._readers

    def __call__(self, record: Record) -> Optional[Record]:
        """Run the inner steps on ``record`` on the engine's kernel; ``None`` = an inner step dropped it."""
        steps, output = self._ensure()
        return run_steps(record, steps, output, self.readers)

    # -- what the inside declares, read off the inner ops (never written twice) --------------------

    def _returned(self) -> List[FlowStep]:
        """The inner steps on the lineage of the result step — the ones whose writes reach the record returned.

        A step's record comes from its ``from:`` step (or the step before it) and every ``merge_from:``
        step; the result step's record carries what those wrote and nothing else. A sibling branch still
        RUNS (its needs count in ``consumes``) but its entries and flags never come back.
        """
        steps, output = self._ensure()
        by_name = {step.name: position for position, step in enumerate(steps)}
        wanted, pending = set(), [output]
        while pending:
            name = pending.pop()
            if name in wanted:
                continue
            wanted.add(name)
            position = by_name[name]
            step = steps[position]
            source = step.from_ or (steps[position - 1].name if position else None)
            pending += ([source] if source is not None else []) + list(step.merge_from)
        return [step for step in steps if step.name in wanted]

    def _boundary(self) -> Tuple[Any, Any]:  # (consumes, produces): both {entry: type} or both type tuples
        """What the inside needs from the record it receives and what it adds, derived from the inner ops.

        By NAME when every inner op declares names (or nothing): ``consumes`` is ``{entry: type}`` an
        inner op needs that no EARLIER inner step produces, ``produces`` what the steps on the result's
        lineage (:meth:`_returned`) write. By TYPE when an inner op declares only types (a
        ``Transform``'s tuple): entry names cannot be known then, so both are tuples, read the way a flat
        chain of the same ops would be. "Earlier" is document order, so a sibling branch's product counts
        as available for ``consumes`` (the tracer's descent checks each inner node on its own lineage),
        while ``produces`` is exact: a sibling branch's entries are not in the record returned.
        """
        if not self.steps:
            return {}, {}
        steps = self.flow_steps
        returned = {id(step) for step in self._returned()}
        if any(_declaration(step.op) == "types" for step in steps):
            needed: List[Any] = []
            made: List[Any] = []
            for step in steps:
                consumes = getattr(step.op, "consumes", None) or ()
                produces = getattr(step.op, "produces", None) or ()
                if isinstance(consumes, (tuple, list)):
                    needed += [kind for kind in consumes if not any(_is_subclass(m, kind) for m in made)]
                elif isinstance(consumes, dict):
                    needed += [str(kind) for kind in consumes.values()]
                if id(step) not in returned:
                    continue
                if isinstance(produces, (tuple, list)):
                    made += list(produces)
                elif isinstance(produces, dict):
                    made += [str(kind) for kind in produces.values()]
            return tuple(dict.fromkeys(needed)), tuple(dict.fromkeys(made))
        consumes_by_name: Dict[str, str] = {}
        available: set = set()
        produces_by_name: Dict[str, str] = {}
        for step in steps:
            for entry, kind in (getattr(step.op, "consumes", None) or {}).items():
                if str(entry) not in available and str(entry) not in consumes_by_name:
                    consumes_by_name[str(entry)] = str(kind)
            for entry, kind in (getattr(step.op, "produces", None) or {}).items():
                available.add(str(entry))
                if id(step) in returned:
                    produces_by_name[str(entry)] = str(kind)
        return consumes_by_name, produces_by_name

    @property
    def consumes(self) -> Any:  # {entry: type} by name, or a tuple of item types (see _boundary)
        """What the inside needs from the record the subgraph receives (derived, see :meth:`_boundary`)."""
        return self._boundary()[0]

    @property
    def produces(self) -> Any:  # the same shape as consumes
        """What the inside adds to the record (derived, see :meth:`_boundary`)."""
        return self._boundary()[1]

    @property
    def flags(self) -> Tuple[str, ...]:
        """The flags the steps on the result's lineage raise — so a gate AFTER the subgraph sees them
        (measured: without it, a graph that runs correctly was refused with "gated on the flag 'bright',
        which no node before it raises"). A flag raised on a sibling branch is not in the record the
        subgraph returns, so it is not here either — the same steps written flat are refused the same way."""
        if not self.steps:
            return ()
        raised: List[str] = []
        for step in self._returned():
            raised += [str(flag) for flag in (getattr(step.op, "flags", None) or ())]
        return tuple(dict.fromkeys(raised))


def _refuse_swallowed_keys(name: str, value: Any) -> None:  # an inner step value: marker, mapping or live op
    """Refuse a reserved step key (``from:`` …) written INSIDE an inner op's ``!class:`` marker.

    confluid builds a subgraph's inner markers when it builds the subgraph, and a key the op's
    constructor does not take is set as a plain attribute (with a warning) and recorded on
    ``__confluid_extra__`` — so the step never sees it, and the graph runs on the wrong input without
    an error. A bare MARKER still unbuilt is fine (``parse_flow`` pops its reserved keys); a marker
    under a mapping's ``op:`` is not (``parse_flow`` builds it with every key it carries).
    """
    op = value.get("op") if isinstance(value, dict) else value
    if op is None:
        return
    if isinstance(op, _ConfluidFluid):
        if op is value:
            return
        written = {key: op.kwargs[key] for key in RESERVED_STEP_KEYS if key in op.kwargs}
        target = getattr(op, "target", None)
        path = f"{target.__module__}.{target.__qualname__}" if isinstance(target, type) else str(target)
        cls_name = path.rsplit(".", 1)[-1]
    else:
        extra = getattr(op, "__confluid_extra__", None) or ()
        written = {key: getattr(op, key) for key in RESERVED_STEP_KEYS if key in extra}
        cls_name = type(op).__name__
        path = f"{type(op).__module__}.{type(op).__qualname__}"
    if not written:
        return
    key, text = next(iter(written.items()))
    raise ValueError(
        f"Subgraph: step {name!r} carries {key!r} inside its op's !class: marker, where {cls_name} takes it as a "
        f"plain attribute and the step never reads it — write the step as a mapping: "
        f"{name}: {{op: !class:{path} {{...}}, {key}: {_as_yaml(text)}}}"
    )
