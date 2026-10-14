"""Trace ONE record through a ``Stream`` or a ``FlowGraph``: a snapshot per node; stop before a
node, step, resume; rerun one node with new parameters and recompute only from there on.

:class:`Tracer` is the engine behind a graph debugger — a page that shows what each node received
and produced, a per-node tool an LLM calls with new settings, a test that pins a pipeline node by
node::

    tracer = Tracer(graph)                        # a Stream or a FlowGraph; nothing built until first use
    tracer.check(seed)                            # static: each node's needs against what reaches it
    tracer.run(seed, until="gated")               # pauses BEFORE 'gated', everything so far recorded
    tracer.step(); tracer.resume()
    tracer.rerun_from("floor", percentile=75.0)   # rebuilds ONE node through its constructor, reruns from there
    tracer.value("gated", "mask")                 # the real array, only when asked
    tracer.to_dict()                              # JSON: arrays summarised, never dumped

Three decisions shape it — the seed of its record in ``docs/architecture.md``.

**A probe around each op, not a second executor.** The run IS the kernel's run. Every node's op is
wrapped in a :class:`_Probe` that records what went in and what came out, and the wrapped step list
is handed to the SAME kernel a plain run uses (``run_steps_multi`` / ``_run_from`` in
:mod:`recordstream.flow.execute`): the kernel keeps deciding when a fan-out read copies and a last
read moves, in which order ``bind:`` sets a step's parameters, how a 1→N step forks the remaining
subgraph. A second executor would have to restate every one of those rules and would drift from
them — the lowering pass deleted 2026-07-30 was that mistake in another shape. The probe forwards
everything the kernel touches on an op (``setattr`` for ``bind:``, ``getattr`` for a ``step.attr``
output and for ``EXPANDS``) and adds one thing: the record around ``_apply_op``. Measured on 200
records of 64x64 through three ops: 0.107 ms per record plain, 0.115 ms traced.

**Snapshots are references by default.** A node's input snapshot is the very object the kernel
handed it. Deep-copying every input and output costs memory in proportion to the record — measured
for one 1024x1024 float32 record through four nodes: 5 MiB retained by reference, 37 MiB by copy.
An op that edits its record in place makes a reference snapshot lie about what the node received,
so the trace FLAGS it (``in_place``), and ``copy_snapshots=True`` is the remedy for exactly those
chains.

**A rerun goes through the constructor, from the node's CURRENT values.**
``rerun_from(node, **params)`` rebuilds the op as ``type(op)(**{**current, **params})``, where
``current`` is read off the live op for each constructor parameter, because the constructor is
where confluid's validation lives: a value outside a ``Literal`` or an ``Interval`` is refused THERE,
before anything in the trace changes. Setting the attribute directly would bypass that validation
and leave the live op poisoned; rebuilding from the kwargs captured at construction would drop what
the host set afterwards (a viewer writes its window into an op by ``setattr``). Measured both ways
before choosing.

**A subgraph step is traced by a child tracer, on the same kernel.** A step whose op is a
``Subgraph`` (:mod:`recordstream.flow.subgraph`) gets a CHILD tracer over the subgraph's own parsed
inner steps; the node's probe runs the child (``run_steps_multi`` over the child's probes) instead of
calling the op, and the inner nodes appear as ``<step>/<inner>`` (``prep/grey``). Flattening the inside
into this graph's step list would be a lowered second execution of the same steps. A pause inside is
the same breakpoint exception, re-raised by each parent with its node prefixed (``prep/scaled``), and
``step`` / ``resume`` / ``rerun_from`` re-enter the child where it stopped, then this graph after the
subgraph. The child's run is not checked on its own: the parent's :meth:`Tracer.check` checks the
inside with what reaches the subgraph and the flags raised before it, which the child cannot know.
"""

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
)

import numpy as np
from confluid import input_specs

from recordstream._compat import is_torch_tensor
from recordstream.core.families import _apply_op, _op_expands
from recordstream.flow.execute import _result_readers, _run_from, run_steps_multi
from recordstream.flow.graph import FlowGraph
from recordstream.flow.steps import (
    _MISSING,
    SUBGRAPH_SEPARATOR,
    FlowStep,
    _declaration,
    _Declaration,
    _is_subgraph,
    _open_subgraph,
)
from recordstream.items import NDArrayItem, Record, is_item
from recordstream.ops.contract import ChainContractError, check_chain

if TYPE_CHECKING:  # the annotation only — a module-level import of core.stream would invert the core -> flow layering
    from recordstream.core.stream import Stream

NodeStatus = Literal["not reached", "paused", "ok", "dropped", "error"]
"""What the trace says about a node: never reached; paused BEFORE it ran (its input recorded, the op
not called); ran and returned a record; ran and dropped the record (returned ``None``); raised."""

Verdict = Literal["ok", "unverifiable", "refused"]
"""The static check's word on a node: its needs are met by name; an earlier node declares TYPES rather
than entry names, so the question cannot be answered by name; an unmet need — the check raises."""

Side = Literal["input", "output"]
"""Which snapshot of a node :meth:`Tracer.value` reads."""

SIDES: Tuple[str, ...] = get_args(Side)

#: An op is whatever the op-family dispatch invokes — a native callable, a bare albumentations or
#: torchvision transform, ``None`` for a fan-in step. ``FlowStep.op`` is ``Any`` for the same reason:
#: no narrower type admits every family.
Op = Any

#: Arrays of this many elements or fewer are listed in full in a summary; anything larger is
#: described (shape, dtype, range) and never dumped.
_MAX_LISTED = 8
#: A non-JSON value is represented by its ``repr``, clipped to this many characters.
_MAX_REPR = 120


class _Breakpoint(Exception):
    """Raised INSIDE the kernel by a node's probe to stop the run there, after recording its input."""

    def __init__(self, node: str) -> None:
        super().__init__(node)
        self.node = node


class _Probe:
    """Stands in for one node's op inside the kernel: records around the call, forwards everything else.

    The kernel touches a step's op in four ways, and the probe is transparent to all of them:
    ``bind:`` does ``setattr(op, param, value)``; a ``step.attr`` bind reads ``getattr(op, attr)``
    through ``_read_output``; ``_op_expands`` reads ``EXPANDS``; ``_apply_op`` calls the op. A fan-in
    step has no op — its probe is the identity, so the fan-in result is snapshotted like any other
    node. The probe's own class carries no library's MRO, so the op-family dispatch treats it as a
    native callable; the wrapped op is then dispatched by its REAL family inside :meth:`Tracer._visit`.
    """

    __slots__ = ("_inner", "_node", "_tracer")
    _inner: Op
    _node: str
    _tracer: "Tracer"

    def __init__(self, inner: Op, node: str, tracer: "Tracer") -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_tracer", tracer)

    def __getattr__(self, attr: str) -> Any:  # the wrapped op's attributes are its own — no narrower type
        if self._inner is None:
            raise AttributeError(attr)
        return getattr(self._inner, attr)

    def __setattr__(self, attr: str, value: Any) -> None:  # a bound value is whatever the producer wrote
        setattr(self._inner, attr, value)

    def __call__(self, record: Any) -> Any:  # a record, or the non-dict carrier a source may yield
        return self._tracer._visit(self._node, self._inner, record)


@dataclass
class _NodeTrace:
    """What one visit of a node recorded."""

    node: str
    op: Optional[str]
    generation: int
    params: Dict[str, Any]
    bound: Dict[str, Any]
    input: Any  # the record the node received — or the non-dict carrier a source may yield
    status: NodeStatus = "paused"
    output: Any = None
    ms: Optional[float] = None
    in_place: bool = False
    changed: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class _Opened:
    """A subgraph node whose inside has not finished: what its visit started with, for the node's entry."""

    record: Any  # the record the subgraph received — the very object, for the in-place comparison
    before: Optional[Dict[str, Any]]  # its entries before the inside ran (None for a non-dict carrier)
    seconds: float = 0.0  # time spent in earlier legs (a run paused inside, then continued)


class _Plan(NamedTuple):
    """The parsed graph, built on first use: the steps with their LIVE ops, and the same steps probed."""

    steps: List[FlowStep]  # a rerun replaces one op in place
    outputs: str
    names: List[str]
    positions: Dict[str, int]  # node -> its place in the schedule
    readers: Dict[str, List[Tuple[int, str]]]
    probes: List[FlowStep]  # what the kernel runs
    param_names: Dict[str, Tuple[str, ...]]  # node -> its op's constructor parameters


# --------------------------------------------------------------------------------------------
# what an op declares, and what its constructor takes
# --------------------------------------------------------------------------------------------


def _declared_entries(op: Op, name: str) -> Optional[Union[Dict[str, str], List[str]]]:
    """One declaration as JSON: ``{entry: type}`` by name, a list of type names for a types-only op,
    ``None`` when nothing is declared."""
    value = None if op is None else getattr(op, name, None)
    if isinstance(value, dict):
        return {str(key): str(kind) for key, kind in value.items()}
    if isinstance(value, (tuple, list)) and value:
        return [getattr(kind, "__name__", str(kind)) for kind in value]
    return None


def _parameter_names(op: Op) -> Tuple[str, ...]:
    """The op's constructor parameters, in signature order — what a rerun may change."""
    if op is None:
        return ()
    try:
        return tuple(str(spec["name"]) for spec in input_specs(type(op)))
    except (TypeError, ValueError):  # a constructor without an introspectable signature (a C extension)
        return ()


def _constructor_values(op: Op, names: Sequence[str]) -> Dict[str, Any]:
    """The node's CURRENT constructor-parameter values.

    The live attribute of the parameter's name when the op has one — what the host set after
    construction is what runs, so it is what a rebuild must start from — else the kwarg confluid
    captured at construction (a parameter the constructor stores under another name). A parameter
    found neither way is left to the constructor's default.
    """
    captured = getattr(op, "__confluid_kwargs__", None) or {}
    values: Dict[str, Any] = {}
    for name in names:
        try:
            value = getattr(op, name, _MISSING)
        except Exception:  # a property that cannot answer right now — the captured kwarg still can
            value = _MISSING
        if value is _MISSING and name in captured:
            value = captured[name]
        if value is not _MISSING:
            values[name] = value
    return values


def _rebuild(op: Op, names: Sequence[str], params: Dict[str, Any]) -> Op:
    """The op again, through its constructor: its current values with ``params`` written over them.

    The constructor is the validation authority (confluid wraps it), so a refused value raises here
    and the caller has changed nothing yet.
    """
    return type(op)(**{**_constructor_values(op, names), **params})


def _mutated(before: Dict[str, Any], record: Dict[str, Any]) -> bool:
    """True when the record's entries are not the ones it held before the op ran (by identity)."""
    if len(before) != len(record) or any(key not in record for key in before):
        return True
    return any(record[key] is not value for key, value in before.items())


def _lineage(
    step: FlowStep, previous: Optional[str], lineage: Dict[str, List[str]], index: Dict[str, int]
) -> List[str]:
    """The nodes whose RECORDS reach ``step``, in schedule order.

    Its ``from:`` step (or the previous step) with that step's own lineage, plus every ``merge_from:``
    step and theirs. A ``bind:`` reference is deliberately not in it: it hands ONE value to a
    parameter, not a record to the input, so the producer's entries do not arrive at this node.
    """
    source = step.from_ or previous
    names: List[str] = [] if source is None else [*lineage[source], source]
    for ref in step.merge_from:
        for name in [*lineage[ref], ref]:
            if name not in names:
                names.append(name)
    names.sort(key=index.__getitem__)
    return names


# --------------------------------------------------------------------------------------------
# the tracer
# --------------------------------------------------------------------------------------------


class Tracer:
    """Run ONE record through a ``Stream`` or a ``FlowGraph`` on the engine's own kernel, one snapshot per node.

    Node names: a ``Stream``'s ops are ``ops[0]``, ``ops[1]``, … (their position in the list — the
    same op twice is two nodes); a ``FlowGraph``'s steps are their document keys. A step whose op is a
    ``Subgraph`` is followed by its inner nodes, named ``<step>/<inner>`` (``prep/grey``, a Stream
    member's ``ops[1]/grey``, nested ``prep/inner/x``) — :attr:`names` lists them in that order, and
    every method that takes a node name takes these too. An unknown name is refused with ``KeyError``
    naming it and the nodes there are.

    Args:
        graph: The ``Stream`` or ``FlowGraph`` to trace. Stored as given; parsed and built on first use.
        where: What refusals name as the location (a file, a graph name); a node is appended as ``where:node``.
        copy_snapshots: Deep-copy each node's input and output instead of keeping references (see the module docstring).
    """

    def __init__(self, graph: Union["Stream", FlowGraph], *, where: str = "", copy_snapshots: bool = False) -> None:
        # Stores only. Parsing a flow document BUILDS its ops (a `!class:` marker is a construction),
        # so it waits for the first use — the same laziness FlowGraph.steps has.
        self._graph = graph
        self._where = str(where)
        self._copy = bool(copy_snapshots)
        self._built: Optional[_Plan] = None
        self._trace: Dict[str, _NodeTrace] = {}
        self._report: List[Dict[str, Any]] = []
        self._seed: Any = None  # the record run() was given — or the non-dict carrier a source may yield
        self._breakpoint: Optional[str] = None
        self._paused_at: Optional[str] = None
        self._result: Optional[List[Record]] = None
        self._generation = 0
        self._total_ms = 0.0
        # The descent: node -> (the op it was built for, the parsed inside it was built for, its child
        # tracer — None for an ordinary op); a subgraph node whose child has not finished -> what its
        # visit started with.
        self._children: Dict[str, Tuple[Op, Optional[List[FlowStep]], Optional["Tracer"]]] = {}
        self._open: Dict[str, _Opened] = {}
        self._nested = False  # a child tracer names its nodes where/node: 'demo.yaml:prep/grey'

    # -- the plan: parsed once, on first use ----------------------------------------------------

    def _plan(self) -> _Plan:
        if self._built is None:
            try:
                steps, outputs = self._steps_of(self._graph)
            except (TypeError, ValueError) as refusal:  # a structural refusal: located like every other one
                if not self._where or self._nested:
                    raise
                kind = TypeError if isinstance(refusal, TypeError) else ValueError
                raise kind(f"{self._where}: {refusal}") from refusal
            names = [step.name for step in steps]
            self._built = _Plan(
                steps=list(steps),
                outputs=outputs,
                names=names,
                positions={name: position for position, name in enumerate(names)},
                readers=_result_readers(steps, outputs),
                probes=[step._replace(op=_Probe(step.op, step.name, self)) for step in steps],
                param_names={step.name: _parameter_names(step.op) for step in steps},
            )
        return self._built

    def _steps_of(self, graph: Union["Stream", FlowGraph]) -> Tuple[List[FlowStep], str]:
        """Both facades as the one step list the kernel runs — a FlowGraph's steps as they are, a Stream's ops
        compiled the way the Stream itself compiles them (``linear_steps``), renamed ``ops[i]``."""
        if isinstance(graph, FlowGraph):
            return list(graph.steps), graph.output_step
        # Body-local on purpose: core.stream reaches flow through body-local imports, and this direction
        # stays module-level-free the same way (tests/test_module_layout.py).
        from recordstream.core.stream import Stream, _check_ops_materialized, linear_steps

        if isinstance(graph, Stream):
            _check_ops_materialized(graph.ops)
            positional, _ = linear_steps(graph.ops)
            steps = [cast(FlowStep, step)._replace(name=f"ops[{i}]") for i, step in enumerate(positional)]
            return steps, (steps[-1].name if steps else "")
        raise TypeError(f"Tracer: expected a Stream or a FlowGraph, got {type(graph).__name__}")

    def _locate(self, node: Optional[str] = None) -> str:
        if node is None:
            return self._where or "Tracer"
        if self._nested:
            return f"{self._where}{SUBGRAPH_SEPARATOR}{node}"
        return f"{self._where}:{node}" if self._where else node

    # -- the descent: a child tracer per subgraph step ---------------------------------------------

    def _child(self, node: str) -> Optional["Tracer"]:
        """The child tracer of ``node`` — ``None`` for an ordinary op; rebuilt when the node's op was replaced
        or its inside parsed again.

        The child traces the subgraph's own parsed inner steps (the SAME live op objects the op's
        ``__call__`` runs) through the same kernel: a subgraph node is not flattened into its parent,
        which would be a second, lowered execution of the same steps. A ``bind:`` onto the subgraph's own
        ``result`` (or ``steps``) makes the op parse its inside again when the record arrives — the child
        follows that parse, so the trace runs what the plain run runs.
        """
        plan = self._plan()
        op = plan.steps[plan.positions[node]].op
        inside = op.flow_steps if op is not None and _is_subgraph(op) else None
        held = self._children.get(node)
        if held is not None and held[0] is op and held[1] is inside:
            return held[2]
        child: Optional[Tracer] = None
        if op is not None and inside is not None:
            graph = FlowGraph(flow=list(inside), outputs=op.output_step)
            child = Tracer(graph, where=self._locate(node), copy_snapshots=self._copy)
            child._nested = True
        self._children[node] = (op, inside, child)
        return child

    def _current_inside(self) -> Dict[str, Any]:  # step name -> {op, from, merge_from, bind}: the flow grammar
        """This graph's steps as a flow mapping, each op the one it runs NOW — a rerun's rebuild included.

        What a rebuild of the subgraph node around this child starts from (:meth:`rerun_from`): the
        subgraph's own ``steps`` still hold the ops it was built with, and rebuilding from them would
        silently undo every rerun made inside. A nested subgraph step is rebuilt the same way, from its
        own child's current inside.
        """
        plan = self._plan()
        mapping: Dict[str, Any] = {}
        for step in plan.steps:
            op = step.op
            held = self._children.get(step.name)
            if held is not None and held[0] is op and held[2] is not None:
                inner = held[2]._current_inside()
                op = type(op)(**{**_constructor_values(op, plan.param_names[step.name]), "steps": inner})
            entry: Dict[str, Any] = {"op": op}
            if step.from_ is not None:
                entry["from"] = step.from_
            if step.merge_from:
                entry["merge_from"] = list(step.merge_from)
            if step.bind:
                entry["bind"] = dict(step.bind)
            mapping[step.name] = entry
        return mapping

    def _resolve(self, node: str) -> Tuple["Tracer", str]:
        """The tracer that owns ``node`` and its name there: ``'prep/grey'`` -> (the child of prep, ``'grey'``)."""
        tracer, rest = self, node
        while True:
            plan = tracer._plan()
            if rest in plan.positions:
                return tracer, rest
            head, separator, tail = rest.partition(SUBGRAPH_SEPARATOR)
            child = tracer._child(head) if separator and head in plan.positions else None
            if child is None:
                raise KeyError(f"{self._locate()}: no node named {node!r} — the nodes are {self.names}")
            tracer, rest = child, tail

    def _reached(self, node: str) -> bool:
        """Whether ``node`` is recorded — for an inner node, every subgraph around it too."""
        tracer, rest = self, node
        while rest not in tracer._plan().positions:
            head, _, rest = rest.partition(SUBGRAPH_SEPARATOR)
            if head not in tracer._trace:
                return False
            tracer = cast(Tracer, tracer._child(head))
        return rest in tracer._trace

    def _forget(self, node: str) -> None:
        """Drop what a node recorded — and, for a subgraph node, everything its inside recorded."""
        self._trace.pop(node, None)
        self._open.pop(node, None)
        held = self._children.get(node)
        if held is not None and held[2] is not None:
            held[2]._clear()

    def _clear(self) -> None:
        self._trace.clear()
        self._open.clear()
        self._paused_at = None
        self._result = None
        for _op, _inside, child in self._children.values():
            if child is not None:
                child._clear()

    def _inner_until(self, node: str) -> Optional[str]:
        """The part of the breakpoint inside ``node``: ``'prep/scaled'`` at ``prep`` -> ``'scaled'``."""
        prefix = f"{node}{SUBGRAPH_SEPARATOR}"
        breakpoint_ = self._breakpoint
        return breakpoint_[len(prefix) :] if breakpoint_ is not None and breakpoint_.startswith(prefix) else None

    # -- the static check -------------------------------------------------------------------------

    def check(self, seed: Record) -> List[Dict[str, Any]]:
        """Refuse an unmet need BEFORE anything runs; one JSON row per node, in schedule order.

        Each node is checked with :func:`check_chain` over the nodes whose records reach it — its
        lineage, in schedule order — and itself. That is the whole-chain check made graph-aware: a
        flag raised three nodes earlier is carried to the gate (checked one node at a time, every
        gated node would be refused), while a fork's sibling branch does not count, because its
        entries and flags never arrive. A refusal is raised as the engine's LOCATED
        :class:`ChainContractError` — ``where:node: <check_chain's message>`` — so a canvas can mark
        the node.

        An op that declares TYPES rather than entry names (a ``Transform``'s tuple) cannot be handed to
        ``check_chain`` (see :func:`_declaration`): its own verdict is ``unverifiable``, and every node
        downstream of it is checked on incomplete knowledge (``complete: false``) — a refusal there is
        reported as ``unverifiable`` instead of raised, and the run shows the truth.

        A SUBGRAPH node is checked as the op it is (its declarations are derived from its inside), and
        then its inside is checked, each inner node over its own lineage, with what reaches the subgraph
        as the entries provided and the flags raised before it as raised. A refusal at the subgraph node
        is re-asked of its inside, so it names the inner node that needs the entry.

        Row: ``{node, op, available, consumes, produces, complete, verdict}`` plus ``note`` (why a
        node is unverifiable) or ``error`` (the refusal); an inner node's row carries ``parent`` and a
        subgraph's row ``inner`` (its inner nodes).
        """
        provided: Optional[FrozenSet[str]] = frozenset(str(key) for key in seed) if isinstance(seed, dict) else None
        return self._check(provided, frozenset(), provided is not None)

    def _check(self, provided: Optional[FrozenSet[str]], raised: FrozenSet[str], known: bool) -> List[Dict[str, Any]]:
        """:meth:`check` over what reaches this graph: ``provided`` entries (``None`` = unknown), the flags
        ``raised`` before it, and whether that much is ``known`` completely (a parent's lineage may not be)."""
        plan = self._plan()
        report: List[Dict[str, Any]] = []
        lineage: Dict[str, List[str]] = {}
        # Read once per node, reused by every node downstream: an op's declaration may be a PROPERTY
        # (an Algorithm derives its consumes/produces on each read), and each lineage re-visits it.
        declared: Dict[str, _Declaration] = {}
        produced: Dict[str, FrozenSet[str]] = {}
        previous: Optional[str] = None
        for step in plan.steps:
            upstream = _lineage(step, previous, lineage, plan.positions)
            lineage[step.name] = upstream
            declared[step.name] = _declaration(step.op)
            produced[step.name] = (
                frozenset(str(key) for key in (getattr(step.op, "produces", None) or {}))
                if declared[step.name] == "names"
                else frozenset()
            )
            with_op = [name for name in upstream if plan.steps[plan.positions[name]].op is not None]
            by_name = [plan.steps[plan.positions[name]].op for name in with_op if declared[name] != "types"]
            complete = known and len(by_name) == len(with_op)
            available = set(provided or ()).union(*(produced[name] for name in with_op))
            verdict: Verdict = "ok"
            row: Dict[str, Any] = {
                "node": step.name,
                "op": None if step.op is None else type(step.op).__name__,
                "available": sorted(available),
                "consumes": _declared_entries(step.op, "consumes"),
                "produces": _declared_entries(step.op, "produces"),
                "complete": complete,
                "verdict": verdict,
            }
            refusal: Optional[ChainContractError] = None
            if declared[step.name] == "types":
                row["verdict"] = "unverifiable"
                row["note"] = (
                    f"{type(step.op).__name__} declares the TYPES it consumes and produces, not the record "
                    "entries — nothing can be checked by name here; the run shows what it wrote"
                )
            elif step.op is not None:
                try:
                    check_chain(
                        [*by_name, step.op], provided=provided or (), raised=raised, where=self._locate(step.name)
                    )
                except ChainContractError as error:
                    if complete:
                        row["verdict"] = "refused"
                        row["error"] = str(error)
                        refusal = error
                    else:
                        row["verdict"] = "unverifiable"
                        row["note"] = str(error)
            child = self._child(step.name)
            inner_rows: List[Dict[str, Any]] = []
            if child is not None:
                row["inner"] = [f"{step.name}{SUBGRAPH_SEPARATOR}{name}" for name in child._plan().names]
                before = raised.union(str(flag) for op in by_name for flag in (getattr(op, "flags", None) or ()))
                try:
                    child._check(frozenset(available) if provided is not None else None, frozenset(before), complete)
                except ChainContractError:
                    # the inside names the node that needs it — that refusal is the one raised
                    self._report = [*report, row, *_nested_rows(step.name, child._report)]
                    raise
                inner_rows = _nested_rows(step.name, child._report)
            report.append(row)
            report.extend(inner_rows)
            if refusal is not None:
                self._report = report
                raise refusal
            previous = step.name
        self._report = report
        return report

    # -- running --------------------------------------------------------------------------------

    def run(self, seed: Record, *, until: Optional[str] = None) -> "Tracer":
        """Check, then run the seed through every node; ``until`` pauses BEFORE the named node.

        ``until`` may name a node inside a subgraph (``prep/scaled``): the run stops INSIDE it, the
        subgraph node and the inner node both ``paused``. A refused check raises before any node runs
        and leaves an earlier run's trace as it was. A node that raises is recorded with the error
        (later nodes stay ``not reached``) and the error propagates; :meth:`rerun_from` that node recovers.
        """
        plan = self._plan()
        if until is not None:
            self._resolve(until)
        self.check(seed)
        self._clear()
        self._seed = deepcopy(seed) if self._copy else seed  # a rerun from the first node starts from THIS
        self._generation = 0
        self._breakpoint = until
        self._drive(lambda: run_steps_multi(seed, plan.probes, plan.outputs, plan.readers))
        return self

    def step(self) -> "Tracer":
        """From a pause: run the paused node and pause before the next one of :attr:`names`.

        :attr:`names` lists a subgraph's inner nodes right after it, so paused before a subgraph a step
        enters it (pauses before its first inner node), and paused before its last inner node a step
        finishes the subgraph and pauses before the next outer node. The last node completes the run.
        """
        paused = self._paused_at
        if paused is None:
            raise RuntimeError(f"{self._locate()}: nothing is paused — run(seed, until=<node>) first")
        names = self.names
        position = names.index(paused)
        until = names[position + 1] if position + 1 < len(names) else None
        self._drive(lambda: self._continue_at(paused, until))
        return self

    def resume(self) -> "Tracer":
        """From a pause: run to the end — through the rest of a subgraph the run paused in, then the steps after it."""
        paused = self._paused_at
        if paused is None:
            raise RuntimeError(f"{self._locate()}: nothing is paused — run(seed, until=<node>) first")
        self._drive(lambda: self._continue_at(paused, None))
        return self

    def rerun_from(self, node: str, **params: Any) -> "Tracer":
        """Rebuild ``node`` with ``params`` written over its current values and recompute from there on.

        The rebuild goes through the constructor (see the module docstring): a value it refuses
        (a pydantic ``ValidationError``, a ``TypeError``) propagates and the trace is untouched. The
        rebuilt node is checked against what reaches it before anything runs. The generation counter
        advances for this node and everything after it; the nodes before keep theirs and are not run
        again — their recorded outputs feed the rerun.

        ``node`` may be inside a subgraph (``prep/scaled``): the inside reruns from that node, then the
        steps after the subgraph; nothing before the inner node, and nothing before the subgraph, runs
        again. ``params`` are the node's constructor parameters; their values are whatever that
        constructor accepts. A node never reached has no recorded input to start from and is refused.
        """
        owner, local = self._resolve(node)
        if not self._reached(node):
            raise RuntimeError(
                f"{self._locate(node)}: {node!r} was never reached (its status is 'not reached') — run() first"
            )
        if params:
            plan = owner._plan()
            index = plan.positions[local]
            step, probed = plan.steps[index], plan.probes[index]
            if step.op is None:
                raise TypeError(f"{self._locate(node)}: {node!r} is a fan-in step without an op — no parameters to set")
            held = owner._children.get(local)
            if held is not None and held[0] is step.op and held[2] is not None and "steps" not in params:
                # a subgraph starts from the inside its child runs NOW, not the steps it was built with
                params = {**params, "steps": held[2]._current_inside()}
            rebuilt = _rebuild(step.op, plan.param_names[local], params)  # a refusal raises HERE: nothing changed yet
            if _is_subgraph(rebuilt):
                _open_subgraph(rebuilt, self._locate(node))  # a refused inside raises HERE too
            children, report = dict(owner._children), self._report
            plan.steps[index] = step._replace(op=rebuilt)
            plan.probes[index] = probed._replace(op=_Probe(rebuilt, local, owner))
            try:
                self.check(self._seed)
            except ChainContractError:
                plan.steps[index], plan.probes[index] = step, probed
                owner._children, self._report = children, report
                raise
        self._generation += 1
        self._drive(lambda: self._continue_at(node, None))
        return self

    def _continue_at(self, node: str, until: Optional[str], seed: Any = _MISSING) -> List[Record]:
        """Re-enter the run AT ``node`` (it runs again) and carry on, pausing before ``until`` if it is reached.

        A node of this graph: the kernel re-enters at it (:meth:`_kernel_from`). A node inside a
        subgraph step: the subgraph's child carries on from there, then — once the inside has finished
        — this graph carries on from the step AFTER the subgraph. A pause anywhere below raises the
        internal breakpoint up to :meth:`_drive`. ``seed`` is the record this graph's first node reads
        (a parent hands its child the record the subgraph step received); left out, it is this tracer's own.
        """
        plan = self._plan()
        if node in plan.positions:
            self._breakpoint = until
            return self._kernel_from(plan.positions[node], seed)
        head, _, rest = node.partition(SUBGRAPH_SEPARATOR)
        index = plan.positions[head]
        child = cast(Tracer, self._child(head))
        self._breakpoint = until
        self._paused_at = None
        if head not in self._open:  # a rerun inside a finished subgraph: its recorded input is where it started
            started_with = self._trace[head].input
            if self._copy:  # the recorded snapshot stays as recorded; the rerun edits a copy of it
                started_with = deepcopy(started_with)
            self._open[head] = _Opened(started_with, dict(started_with) if isinstance(started_with, dict) else None)
        record = self._open[head].record  # the very object the inside runs on, for the in-place comparison
        result = self._leg(head, child, lambda inner_until: child._continue_at(rest, inner_until, record))
        if result is None:  # the inside dropped the record: nothing after the subgraph runs
            for name in plan.names[index + 1 :]:
                self._forget(name)
            self._result = []
            return []
        return self._kernel_from(index + 1)

    def _kernel_from(self, index: int, seed: Any = _MISSING) -> List[Record]:
        """Re-enter the kernel at ``steps[index]`` with the step environment rebuilt from the recorded outputs.

        ``_run_from`` needs the outputs of the earlier steps that steps from ``index`` on still read,
        with their remaining read counts (the kernel frees a result after its last read). Each is a
        COPY: the kernel MOVES a last read, so the node about to run may edit what it gets, and the
        recorded output of the earlier node must survive that or a second rerun starts from a changed
        input. The seed the first node reads is ``seed`` when a parent hands one down, else this tracer's
        own — a COPY of it with ``copy_snapshots`` (the first node may edit its record in place, and a
        second rerun must start from the record as it arrived, not from the one the first rerun edited).
        """
        if seed is _MISSING:
            seed = deepcopy(self._seed) if self._copy else self._seed
        plan = self._plan()
        for name in plan.names[index:]:
            self._forget(name)
        self._paused_at = None
        env: Dict[str, Any] = {}
        remaining: Dict[str, int] = {}
        for name, reads in plan.readers.items():
            later = sum(1 for consumer, _slot in reads if consumer >= index)
            remaining[name] = later
            if not later or plan.positions[name] >= index:
                continue
            producer = plan.steps[plan.positions[name]]
            if producer.op is not None and _op_expands(producer.op):
                raise RuntimeError(
                    f"{self._locate(name)}: {name!r} is a 1→N expanding node and the trace keeps only its last "
                    f"branch — rerun from {name!r} or earlier instead"
                )
            recorded = self._trace.get(name)
            if recorded is None or recorded.output is None:
                raise RuntimeError(f"{self._locate(name)}: {name!r} has no recorded output to continue from")
            env[name] = deepcopy(recorded.output)
        previous = plan.names[index - 1] if index > 0 else None
        out: List[Record] = []
        _run_from(index, seed, plan.probes, plan.outputs, env, remaining, previous, out)
        self._result = out
        return out

    def _enter(self, record: Any, until: Optional[str]) -> List[Record]:  # the record a subgraph step received
        """A child's run: the subgraph's inside on the record its step received, on the kernel.

        No check here — the parent's :meth:`check` already checked the inside with what reaches it and
        the flags raised before it, which a check from here could not know.
        """
        plan = self._plan()
        self._clear()
        self._seed = deepcopy(record) if self._copy else record  # the record as it arrived (see _kernel_from)
        self._breakpoint = until
        out = run_steps_multi(record, plan.probes, plan.outputs, plan.readers)
        self._result = out
        return out

    def _drive(self, kernel: Callable[[], List[Record]]) -> None:
        """Run the kernel; a pause is not an error, everything else propagates after the timing is kept."""
        self._result = None
        started = time.perf_counter()
        try:
            self._result = kernel()
        except _Breakpoint:
            self._result = None  # the probe recorded the pause; there is no result yet
        finally:
            self._total_ms = (time.perf_counter() - started) * 1e3

    def _visit(self, node: str, inner: Op, record: Any) -> Any:
        """Called by the node's probe INSIDE the kernel: record, pause or run, record again."""
        plan = self._plan()
        step = plan.steps[plan.positions[node]]
        entry = _NodeTrace(
            node=node,
            op=None if inner is None else type(inner).__name__,
            generation=self._generation,
            params=_constructor_values(inner, plan.param_names[node]) if inner is not None else {},
            bound={param: getattr(inner, param, None) for param in step.bind},
            input=deepcopy(record) if self._copy else record,
        )
        self._trace[node] = entry
        if node == self._breakpoint:
            self._paused_at = node
            raise _Breakpoint(node)
        child = self._child(node) if inner is not None else None
        if child is not None:  # a subgraph: its inside runs through the child, on the same kernel
            self._open[node] = _Opened(record, dict(record) if isinstance(record, dict) else None)
            return self._leg(node, child, lambda until: child._enter(record, until))
        before = dict(record) if isinstance(record, dict) else None
        started = time.perf_counter()
        try:
            result = record if inner is None else _apply_op(record, inner)
        except Exception as exc:
            entry.ms = (time.perf_counter() - started) * 1e3
            entry.status = "error"
            entry.error = f"{type(exc).__name__}: {exc}"
            raise
        entry.ms = (time.perf_counter() - started) * 1e3
        entry.output = deepcopy(result) if self._copy else result
        entry.status = "dropped" if result is None else "ok"
        if before is not None:
            entry.in_place = _mutated(before, record)
            if isinstance(result, dict):
                entry.changed = [key for key, value in result.items() if key not in before or before[key] is not value]
                entry.removed = [key for key in before if key not in result]
        return result

    def _leg(self, node: str, child: "Tracer", run_child: Callable[[Optional[str]], List[Record]]) -> Any:
        """Run (part of) a subgraph node's inside through its child and record the node around it.

        A pause inside leaves the node ``paused`` and raises the breakpoint on, named ``node/<inner>``; the
        leg that finishes the inside records the node's output like any other node's. Returns the one
        record the subgraph yields, or ``None`` when its inside dropped it.
        """
        entry, opened = self._trace[node], self._open[node]
        child._generation = self._generation
        started = time.perf_counter()
        try:
            results = run_child(self._inner_until(node))
        except _Breakpoint:
            opened.seconds += time.perf_counter() - started
            entry.status = "paused"
            self._paused_at = f"{node}{SUBGRAPH_SEPARATOR}{child._paused_at}"
            raise _Breakpoint(self._paused_at) from None
        except Exception as exc:
            entry.ms = (opened.seconds + time.perf_counter() - started) * 1e3
            entry.status = "error"
            entry.error = f"{type(exc).__name__}: {exc}"
            self._open.pop(node, None)
            raise
        entry.ms = (opened.seconds + time.perf_counter() - started) * 1e3
        self._open.pop(node, None)
        if len(results) > 1:  # pragma: no cover - an expanding inner op is refused when the subgraph is parsed
            raise TypeError(f"{self._locate(node)}: the subgraph yielded {len(results)} records — a step op is 1→1")
        result = results[0] if results else None
        entry.generation = self._generation
        entry.output = deepcopy(result) if self._copy else result
        entry.status = "dropped" if result is None else "ok"
        entry.error = None
        if opened.before is not None:
            entry.in_place = _mutated(opened.before, opened.record)
            if isinstance(result, dict):
                entry.changed = [
                    key for key, value in result.items() if key not in opened.before or opened.before[key] is not value
                ]
                entry.removed = [key for key in opened.before if key not in result]
        return result

    # -- reading the trace ------------------------------------------------------------------------

    def value(self, node: str, entry: str, side: Side = "output") -> Any:  # the real object, of whatever type
        """The REAL object under ``entry`` in ``node``'s input or output snapshot — not a summary."""
        owner, local = self._resolve(node)  # an unknown name is refused here, listing every node
        if owner is not self:
            return owner.value(local, entry, side)
        if side not in SIDES:
            raise ValueError(f"{self._locate(node)}: side must be one of {SIDES}, got {side!r}")
        recorded = self._trace.get(node)
        if recorded is None:
            raise RuntimeError(f"{self._locate(node)}: {node!r} was never reached — nothing is recorded for it")
        record = recorded.input if side == "input" else recorded.output
        if record is None:
            raise RuntimeError(f"{self._locate(node)}: {node!r} has no {side} yet (its status is {recorded.status!r})")
        if not isinstance(record, dict):
            raise KeyError(f"{self._locate(node)}: the {side} of {node!r} is a {type(record).__name__}, not a record")
        if entry not in record:
            raise KeyError(f"{self._locate(node)}: {node!r} has no {side} entry {entry!r} — it has {sorted(record)}")
        return record[entry]

    @property
    def names(self) -> List[str]:
        """The node names, in schedule order, each subgraph's inner nodes right after it (parses on first use)."""
        names: List[str] = []
        for name in self._plan().names:
            names.append(name)
            child = self._child(name)
            if child is not None:
                names.extend(f"{name}{SUBGRAPH_SEPARATOR}{inner}" for inner in child.names)
        return names

    @property
    def where(self) -> str:
        """The location refusals name."""
        return self._where

    @property
    def paused_at(self) -> Optional[str]:
        """The node the run paused BEFORE (``prep/scaled`` inside a subgraph), or ``None``."""
        return self._paused_at

    @property
    def result(self) -> Optional[List[Record]]:
        """What the graph yielded for the seed — ``None`` until a run completes (paused, refused, raised)."""
        return self._result

    @property
    def generation(self) -> int:
        """The current generation: 0 after ``run``, +1 per ``rerun_from``. Nodes carrying it were computed last."""
        return self._generation

    @property
    def statuses(self) -> Dict[str, NodeStatus]:
        """``{node: status}`` for every node, inner nodes included. A subgraph node is ``paused`` while the run
        is paused before it OR anywhere inside it."""
        statuses: Dict[str, NodeStatus] = {}
        for name in self._plan().names:
            statuses[name] = self._trace[name].status if name in self._trace else "not reached"
            child = self._child(name)
            if child is not None:
                statuses.update({f"{name}{SUBGRAPH_SEPARATOR}{key}": value for key, value in child.statuses.items()})
        return statuses

    @property
    def generations(self) -> Dict[str, Optional[int]]:
        """``{node: generation}`` — the generation each node's entry was recorded in; ``None`` until reached."""
        generations: Dict[str, Optional[int]] = {}
        for name in self._plan().names:
            generations[name] = self._trace[name].generation if name in self._trace else None
            child = self._child(name)
            if child is not None:
                generations.update(
                    {f"{name}{SUBGRAPH_SEPARATOR}{key}": value for key, value in child.generations.items()}
                )
        return generations

    @property
    def report(self) -> List[Dict[str, Any]]:
        """The rows of the last :meth:`check`."""
        return list(self._report)

    def to_dict(self) -> Dict[str, Any]:
        """The trace as plain JSON — arrays summarised (type, shape, dtype, range), never dumped.

        ``{where, check, nodes, paused_at, result, total_ms}``; each node:
        ``{node, op, status, generation, ms, params, bound, input, output, changed, removed, in_place, error}``
        — a node never reached carries only ``node``, ``op``, ``status`` and ``generation: None``; a
        paused node has its ``input`` and no ``output`` yet. A subgraph node carries ``inner`` (its inner
        nodes) and is followed by their entries, each with ``parent`` (the subgraph node).
        """
        plan = self._plan()
        nodes: List[Dict[str, Any]] = []
        for step in plan.steps:
            nodes.append(self._node_entry(step))
            child = self._child(step.name)
            if child is not None:
                inner = child.to_dict()["nodes"]
                nodes[-1]["inner"] = [
                    f"{step.name}{SUBGRAPH_SEPARATOR}{row['node']}" for row in inner if "parent" not in row
                ]
                nodes.extend(_nested_rows(step.name, inner))
        return {
            "where": self._where,
            "check": self._report,
            "nodes": nodes,
            "paused_at": self._paused_at,
            "result": None if self._result is None else [_summarize_record(record) for record in self._result],
            "total_ms": round(self._total_ms, 3),
        }

    def _node_entry(self, step: FlowStep) -> Dict[str, Any]:
        recorded = self._trace.get(step.name)
        op_name = None if step.op is None else type(step.op).__name__
        if recorded is None:
            return {"node": step.name, "op": op_name, "status": "not reached", "generation": None}
        node: Dict[str, Any] = {
            "node": step.name,
            "op": recorded.op,
            "status": recorded.status,
            "generation": recorded.generation,
            "ms": None if recorded.ms is None else round(recorded.ms, 3),
            "params": _jsonable(recorded.params),
            "bound": _jsonable(recorded.bound),
            "input": _summarize_record(recorded.input),
        }
        if recorded.status in ("ok", "dropped"):
            node["output"] = None if recorded.output is None else _summarize_record(recorded.output)
            node["changed"] = list(recorded.changed)
            node["removed"] = list(recorded.removed)
            node["in_place"] = recorded.in_place
        if recorded.error is not None:
            node["error"] = recorded.error
        return node


def _nested_rows(head: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A child's rows (check rows or node entries) as the parent lists them: names qualified ``head/…``,
    ``parent`` naming the subgraph node they sit in, ``inner`` qualified the same way."""
    nested: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row, node=f"{head}{SUBGRAPH_SEPARATOR}{row['node']}")
        row["parent"] = f"{head}{SUBGRAPH_SEPARATOR}{row['parent']}" if "parent" in row else head
        if "inner" in row:
            row["inner"] = [f"{head}{SUBGRAPH_SEPARATOR}{name}" for name in row["inner"]]
        nested.append(row)
    return nested


# --------------------------------------------------------------------------------------------
# summaries: JSON, never the arrays
# --------------------------------------------------------------------------------------------


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_REPR else text[:_MAX_REPR] + "…"


def _number(value: float) -> Union[float, str]:
    """A float as JSON: itself, or its name (``"nan"``, ``"-inf"``) — JSON has no spelling for those."""
    return value if math.isfinite(value) else str(value)


def _jsonable(value: Any) -> Any:  # any record value in, plain JSON out
    """A plain-JSON rendering of any value: scalars as they are, arrays SUMMARISED, the rest by ``repr``."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _summarize_array(value)
    if is_torch_tensor(value):
        return _summarize_tensor(value)
    return _clip(repr(value))


def _summarize_array(data: np.ndarray) -> Dict[str, Any]:
    """Shape, dtype and range of an array; the values themselves only when there are very few."""
    plain = np.asarray(data)  # an item is an ndarray subclass — the summary is about the payload
    summary: Dict[str, Any] = {"shape": list(plain.shape), "dtype": str(plain.dtype)}
    if plain.size == 0:
        return summary
    if plain.dtype == bool:
        summary["true_fraction"] = round(float(plain.mean()), 4)
    elif np.issubdtype(plain.dtype, np.complexfloating):
        magnitude = np.abs(plain)
        summary["abs_min"], summary["abs_max"] = _number(float(magnitude.min())), _number(float(magnitude.max()))
    elif np.issubdtype(plain.dtype, np.number):
        finite = plain[np.isfinite(plain)] if np.issubdtype(plain.dtype, np.floating) else plain.ravel()
        if finite.size:
            summary["min"], summary["max"] = float(finite.min()), float(finite.max())
        if finite.size < plain.size:
            summary["non_finite"] = int(plain.size - finite.size)
    if plain.size <= _MAX_LISTED:
        summary["values"] = _jsonable(plain.tolist())
    return summary


def _summarize_tensor(value: Any) -> Dict[str, Any]:  # a torch tensor, described without importing torch
    summary: Dict[str, Any] = {"shape": [int(n) for n in value.shape], "dtype": str(value.dtype)}
    device = getattr(value, "device", None)
    if device is not None:
        summary["device"] = str(device)
    return summary


def _summarize_value(value: Any) -> Dict[str, Any]:  # any record value
    """One record value as JSON: its type, an item's declared attributes, and its payload summarised."""
    summary: Dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, NDArrayItem):
        for attr in type(value)._item_attrs:
            declared = getattr(value, attr, None)
            if declared is not None:
                summary[attr] = _jsonable(declared)
        summary.update(_summarize_array(value))
        return summary
    if is_item(value) and is_dataclass(value):  # Label / MultiLabel / Boxes / a domain package's wrapper
        for slot in fields(value):
            held = getattr(value, slot.name)
            if held is not None:
                summary[slot.name] = _jsonable(held)
        return summary
    if isinstance(value, np.ndarray):
        summary.update(_summarize_array(value))
        return summary
    if is_torch_tensor(value):
        summary.update(_summarize_tensor(value))
        return summary
    if value is None or isinstance(value, (bool, str, int, float, np.generic, dict, list, tuple)):
        summary["value"] = _jsonable(value)
        return summary
    summary["repr"] = _clip(repr(value))
    return summary


def _summarize_record(record: Any) -> Any:  # a record dict, or the non-dict carrier a source may yield
    if isinstance(record, dict):
        return {str(key): _summarize_value(value) for key, value in record.items()}
    return _summarize_value(record)
