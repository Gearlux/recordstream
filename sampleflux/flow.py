"""The ``flow:`` document, the :class:`FlowGraph` engine, and the flow⇄ops converters.

A **flow document** is the readable, named-step form of a graph-shaped pipeline: a
mapping of ``step-name → op``, where a step's name is also the name later steps use to
reference its result. It is the authoring format (humans and the FluxStudio exporter
write it); the flat context-ops form (:mod:`sampleflux.ops.context`) is the serial
execution format the plain :class:`~sampleflux.core.Flux` engine runs. The two convert
**bidirectionally**: :func:`to_ops` lowers a flow into a flat op list, :func:`from_ops`
lifts a flat op list back into a flow — with execution parity in both directions.

.. code-block:: yaml

    flow:
      spec:     !class:waivefront.SpectrogramOp()             # input: the source sample
      rescaled: !class:sampleflux.ops.numpy.RescaleOp()       # input: previous step
      masked:   !class:waivefront.SegmentOp() {from: spec}    # 2nd reader of spec = fan-out
      thresh:   !class:sampleflux.ops.formula.FormulaOp(formula="a*0.5") {from: masked}
      denoised: !class:waivefront.torchsig.processing.NoiseFloorOp()
        from: rescaled
        bind: {low_level: thresh}       # per-sample param := thresh's result
      out: {from: denoised, target_from: masked}              # pure fan-in (no op)
    outputs: out

Step grammar (the four RESERVED step keys, stripped before the op is built):

- ``from:`` — the step supplying this step's input sample. Omitted = the previous step
  (the first step reads the source sample). Must name an EARLIER step: document order is
  the schedule, so forward references are errors and cycles are inexpressible.
- ``target_from:`` / ``metadata_from:`` — fan-in: compose the incoming sample's target /
  metadata from another step's result before the op runs (the ``Mix`` slot semantics —
  a Sample result contributes its corresponding field, metadata merges last-write-wins).
- ``bind:`` — ``{param: ref}`` per-sample parameters: ``ref`` is a step name (its result
  sample's ``input``, or the raw value) or ``step.attr`` (the step op's live ``@output``
  after it ran — lowered through ``Capture``; stochastic-correct).

A step may be a plain mapping with no op (``out: {from: a, target_from: b}``) — a pure
fan-in/identity step; ``{}`` is the identity (used to give the source a referable name).
``outputs:`` names the step whose result the pipeline yields (default: the last step).

Cell-lifetime management is AUTOMATIC in both forms: :class:`FlowGraph` frees each step
result after its last reader, and :func:`to_ops` computes the same liveness into
``drop`` flags on the emitted context ops.
"""

import inspect
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union, cast

import torch.utils.data
from confluid import configurable, flow
from confluid import resolve as _confluid_resolve
from confluid.fluid import Fluid as _ConfluidFluid
from loggair import get_logger

from sampleflux.ops.context import _MISSING, Apply, Capture, Drop, Mix, Save, Use, _read_output
from sampleflux.sample import Sample

logger = get_logger(__name__)

RESERVED_STEP_KEYS = ("from", "target_from", "metadata_from", "bind")
"""Step-grammar keys stripped from a step mapping before the op is constructed."""

__all__ = ["FlowGraph", "FlowStep", "from_ops", "parse_flow", "to_ops", "RESERVED_STEP_KEYS"]


class FlowStep(NamedTuple):
    """One parsed step of a flow document."""

    name: str
    op: Optional[Any]  # live op callable; None = pure fan-in / identity step
    from_: Optional[str]  # None = previous step (first step: the source sample)
    target_from: Optional[str]
    metadata_from: Optional[str]
    bind: Dict[str, str]  # param -> "step" | "step.attr"


class _BindRef(NamedTuple):
    """A parsed ``bind:`` reference."""

    step: str
    attr: Optional[str]  # None = the step's result; else the step op's @output attribute


def _parse_bind_ref(ref: str, known: Sequence[str]) -> _BindRef:
    head, dot, attr = str(ref).partition(".")
    if head not in known:
        raise ValueError(
            f"flow: bind reference {ref!r} does not name an earlier step "
            f"(known steps at this point: {list(known)!r})"
        )
    return _BindRef(head, attr if dot else None)


def _check_reserved_collision(op: Any, step_name: str) -> None:
    """Raise if the op's constructor has a param named like a reserved step key.

    Reserved keys are stripped from the step mapping before the op is built, so such a
    param could never be configured inline — fail loudly instead of silently stealing it.
    (``from`` is a Python keyword and can never be a param, but the others could.)
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


def parse_flow(flow_doc: Any, outputs: str = "", build: bool = True) -> Tuple[List[FlowStep], str]:
    """Parse a flow mapping into ordered :class:`FlowStep`\\ s + the resolved output step name.

    ``flow_doc`` is the ``flow:`` mapping — step values may be confluid markers (from
    ``resolve()``/``load()``), plain dicts (pure fan-in steps, or programmatic
    ``{"op": <op>, "from": ...}`` form), or live op callables. Reserved keys are popped;
    markers are flowed per step (confluid does not auto-flow two-levels-nested markers).
    Validates: step names carry no dots, every reference points to an EARLIER step.

    ``build=False`` keeps a marker step UNBUILT (the op stays a Fluid marker) — for
    structural consumers (converters/importers) that must not materialize ops (hoisted
    dotted ``!ref:`` values would resolve outside their document); such steps skip the
    reserved-ctor-param check and their live construction happens at first call.
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
        target_from = reserved.get("target_from")
        metadata_from = reserved.get("metadata_from")
        for key, ref in (("from", from_), ("target_from", target_from), ("metadata_from", metadata_from)):
            if ref is not None and str(ref) not in seen:
                raise ValueError(
                    f"flow step {name!r}: {key}: {ref!r} does not name an EARLIER step "
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
                target_from=None if target_from is None else str(target_from),
                metadata_from=None if metadata_from is None else str(metadata_from),
                bind=bind,
            )
        )
        seen.append(name)

    out = str(outputs) if outputs else steps[-1].name
    if out not in seen:
        raise ValueError(f"flow: outputs {out!r} does not name a step (steps: {seen!r})")
    return steps, out


def _result_readers(steps: Sequence[FlowStep], outputs: str) -> Dict[str, List[Tuple[int, str]]]:
    """Step-result cell -> ordered ``(consumer_index, slot)`` reads.

    Slot granularity matters: one consumer step may read the SAME producer through several
    slots (its input AND a ``bind`` param), and only the ``"in"`` slot of the immediately
    following step can ride the linear stream. Slots: ``"in"`` (input), ``"target"``,
    ``"meta"``, ``"bind"``, and the final ``"out"`` read at index ``len(steps)``. A
    ``bind`` step-result reference counts; an ``@output`` (``step.attr``) reference does
    NOT (it reads the op instance, not the result cell).
    """
    readers: Dict[str, List[Tuple[int, str]]] = {s.name: [] for s in steps}
    for i, step in enumerate(steps):
        implicit = steps[i - 1].name if i > 0 else None
        source = step.from_ or implicit
        if source is not None:
            readers[source].append((i, "in"))
        if step.target_from is not None:
            readers[step.target_from].append((i, "target"))
        if step.metadata_from is not None:
            readers[step.metadata_from].append((i, "meta"))
        for ref in step.bind.values():
            parsed = _BindRef(*ref.partition(".")[::2]) if "." in ref else _BindRef(ref, None)
            if parsed.attr is None:
                readers[parsed.step].append((i, "bind"))
    readers[outputs].append((len(steps), "out"))
    return readers


# ---------------------------------------------------------------------------
# The FlowGraph engine
# ---------------------------------------------------------------------------


@configurable(category="engine")
class FlowGraph(torch.utils.data.Dataset[Sample]):
    """Named-step graph engine — executes a ``flow:`` document natively.

    The readable twin of :class:`~sampleflux.core.Flux`: steps run in document order over
    a per-sample environment of named results, with fan-out isolation (copy-on-read, move
    on last read) and automatic cell lifetimes. Any FlowGraph converts to a flat op list
    for the serial engine (:func:`to_ops`) and back (:func:`from_ops`) — execution parity
    between the two is a pinned contract.

    Args:
        source: Any iterable or indexable dataset (duck-typed) to wrap; ``None`` yields an empty stream.
        flow: The flow mapping (step-name -> op / marker / step mapping) or a parsed list of FlowStep.
        outputs: Name of the step whose result is yielded. Blank (default) = the last step.
        chunk_size: Batch size for chunked iteration; ``0`` (the default) yields single samples.
    """

    def __init__(
        self,
        source: Optional[Any] = None,
        flow: Optional[Union[Dict[str, Any], List[FlowStep]]] = None,
        outputs: str = "",
        chunk_size: int = 0,
    ) -> None:
        # Lazy / zero-arg: store config only; parsing/validation happen in the cached property.
        self.source = source
        self.flow = flow
        self.outputs = str(outputs)
        self._chunk_size = int(chunk_size)
        self._workers = 1
        self._parsed: Optional[Tuple[List[FlowStep], str]] = None

    # -- parsing -----------------------------------------------------------

    @property
    def steps(self) -> List[FlowStep]:
        """The parsed, validated steps (cached; recomputed only if ``flow`` is reassigned)."""
        return self._ensure_parsed()[0]

    @property
    def output_step(self) -> str:
        """The resolved output step name."""
        return self._ensure_parsed()[1]

    def _ensure_parsed(self) -> Tuple[List[FlowStep], str]:
        if self._parsed is None:
            if self.flow is None:
                raise ValueError("FlowGraph.flow is not set — provide a flow mapping or FlowStep list.")
            if isinstance(self.flow, list) and all(isinstance(s, FlowStep) for s in self.flow):
                names = [s.name for s in self.flow]
                out = self.outputs or (names[-1] if names else "")
                if out not in names:
                    raise ValueError(f"FlowGraph: outputs {out!r} does not name a step ({names!r})")
                self._parsed = (list(self.flow), out)
            else:
                self._parsed = parse_flow(cast(Dict[str, Any], self.flow), self.outputs)
        return self._parsed

    @classmethod
    def from_yaml(cls, path: str, source: Optional[Any] = None) -> "FlowGraph":
        """Build a FlowGraph from a ``{flow: {...}, outputs: ...}`` YAML document (or inline string).

        Uses ``confluid.resolve`` so step markers stay UNbuilt until :func:`parse_flow`
        pops the reserved step keys and flows each op itself.
        """
        doc = _confluid_resolve(path)
        if not isinstance(doc, dict) or "flow" not in doc:
            raise ValueError(f"FlowGraph.from_yaml: {path!r} has no 'flow:' mapping")
        return cls(source=source, flow=doc["flow"], outputs=str(doc.get("outputs", "") or ""))

    @classmethod
    def from_ops_yaml(cls, path: str, source: Optional[Any] = None) -> "FlowGraph":
        """Lift a flat ``{ops: [...]}`` YAML document into a FlowGraph (via :func:`from_ops`)."""
        from sampleflux.core import Flux

        flux = Flux.from_ops_yaml(path, source=source)
        flow_doc, outputs = from_ops(flux.ops)
        return cls(source=source, flow=flow_doc, outputs=outputs)

    # -- execution ---------------------------------------------------------

    def _run(self, seed: Sample) -> Optional[Sample]:
        """Run one sample through the steps; ``None`` = filtered (an op returned None)."""
        steps, outputs = self._ensure_parsed()
        readers = _result_readers(steps, outputs)
        env: Dict[str, Any] = {}
        remaining = {name: len(idx) for name, idx in readers.items()}

        def read_result(name: str, *, copy: bool) -> Any:
            value = env[name]
            remaining[name] -= 1
            if remaining[name] <= 0:
                del env[name]
            elif copy:
                from copy import deepcopy

                value = deepcopy(value)
            return value

        prev: Optional[str] = None
        for step in steps:
            # 1. the input sample (implicit stream reads move; explicit fan-out reads copy)
            if step.from_ is not None:
                base = read_result(step.from_, copy=True)
            elif prev is not None:
                base = read_result(prev, copy=False)
            else:
                base = seed
            sample = Sample.from_any(base)

            # 2. fan-in slots (Mix semantics)
            if step.target_from is not None or step.metadata_from is not None:
                metadata = dict(sample.meta)
                target = sample.target
                if step.target_from is not None:
                    value = read_result(step.target_from, copy=True)
                    target = value.target if isinstance(value, Sample) else value
                    if isinstance(value, Sample):
                        metadata.update(value.meta)
                if step.metadata_from is not None:
                    value = read_result(step.metadata_from, copy=True)
                    extra = value.meta if isinstance(value, Sample) else value
                    if not isinstance(extra, dict):
                        raise TypeError(
                            f"flow step {step.name!r}: metadata_from holds {type(extra).__name__}, "
                            "expected a Sample or a dict"
                        )
                    metadata.update(extra)
                sample = sample._replace(target=target, metadata=metadata)

            # 3. per-sample parameter binds
            if step.op is not None:
                op = step.op
                from sampleflux.kinds import op_contract as _op_contract

                if _op_contract(op).expands:
                    raise NotImplementedError(
                        f"flow step {step.name!r}: {type(op).__name__!r} is a 1→N expanding op — "
                        "FlowGraph steps are strictly 1→1 (a named-step env has one result per step). "
                        "Run expanding pipelines through the Flux engine (iterable-only)."
                    )
                for param, ref in step.bind.items():
                    if "." in ref:
                        head, _, attr = ref.partition(".")
                        producer = next(s for s in steps if s.name == head)
                        value = _read_output(producer.op, attr)
                        if value is _MISSING:
                            raise AttributeError(
                                f"flow step {step.name!r}: bind {param}={ref!r} — "
                                f"step {head!r} op has no @output attribute {attr!r}"
                            )
                    else:
                        value = read_result(ref, copy=False)
                        if isinstance(value, Sample):
                            value = value.input
                    setattr(op, param, value)
                result = op(sample)
                if result is None:
                    return None
                sample = cast(Sample, result)

            env[step.name] = sample
            prev = step.name

        return cast(Optional[Sample], env.get(outputs)) if outputs in env else None

    def __iter__(self) -> Iterator[Any]:
        if self.source is None:
            return
        it = self._iter_samples()
        if self._chunk_size > 0:
            batch: List[Sample] = []
            for sample in it:
                batch.append(sample)
                if len(batch) == self._chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            yield from it

    def _iter_samples(self) -> Iterator[Sample]:
        if self._workers > 1:
            yield from self._iter_parallel()
            return
        assert self.source is not None
        for item in self.source:
            result = self._run(Sample.from_any(item))
            if result is not None:
                yield result

    def _iter_parallel(self) -> Iterator[Sample]:
        """Multiprocess execution — delegates to the serial engine over the LOWERED op list.

        Lowering + Flux's spawn pool is the sanctioned parallel path (one worker
        implementation, guaranteed parity by the to_ops contract); a native process pool
        here would duplicate it for no gain.
        """
        from sampleflux.core import Flux

        assert self.source is not None
        flux = Flux(source=self.source, ops=to_ops(self.steps, self.output_step)).parallel(self._workers)
        yield from flux

    def __len__(self) -> int:
        from collections.abc import Sized

        if isinstance(self.source, Sized):
            return len(self.source)
        return 0

    def __getitem__(self, index: int) -> Sample:
        if self.source is None:
            raise TypeError("FlowGraph source is None — cannot index.")
        if hasattr(self.source, "__getitem__"):
            raw = self.source[index]
        else:
            raise TypeError(
                f"FlowGraph source {type(self.source).__name__} does not support indexing; "
                "wrap it in a list or use iteration."
            )
        result = self._run(Sample.from_any(raw))
        if result is None:
            raise IndexError(f"Sample {index} filtered out by the flow")
        return result

    def parallel(self, workers: int = 4) -> "FlowGraph":
        """Enable multiprocess execution (spawn, via the lowered serial form)."""
        self._workers = workers
        return self

    def batch(self, chunk_size: int) -> "FlowGraph":
        """Group yielded samples into lists of ``chunk_size``."""
        self._chunk_size = chunk_size
        return self

    def collect(self) -> List[Any]:
        """Materialize the full stream into a list."""
        return list(self)

    def to_flux(self) -> Any:
        """The serial-engine twin: a Flux running the LOWERED flat op list (same results)."""
        from sampleflux.core import Flux

        return Flux(source=self.source, ops=to_ops(self.steps, self.output_step))


# ---------------------------------------------------------------------------
# Lowering: flow -> flat context-ops list
# ---------------------------------------------------------------------------


def to_ops(steps: Union[Sequence[FlowStep], Dict[str, Any]], outputs: str = "") -> List[Any]:
    """Lower a flow (parsed steps or a raw flow mapping) into a flat context-ops list.

    The result runs on the plain serial :class:`~sampleflux.core.Flux` engine and is the
    serialization form FluxStudio's ``--serial`` export emits. Cell names are the step
    names (deterministic, diffable); liveness is compiled into ``drop`` flags so a
    well-formed graph leaves the Context empty. A purely linear flow lowers to the bare
    op list — zero context ops.
    """
    if isinstance(steps, dict):
        parsed, outputs = parse_flow(steps, outputs)
    else:
        parsed = list(steps)
        outputs = outputs or (parsed[-1].name if parsed else "")

    readers = _result_readers(parsed, outputs)
    # Which step results must live in a cell? Every read EXCEPT the one that can ride the
    # linear stream: the immediately-next step's INPUT slot, or the final output read when
    # this is the last step. Slot granularity matters — a consumer may read the same
    # producer through its input slot AND a bind slot (only the input slot can stream).
    needs_cell: Dict[str, bool] = {}
    cell_reads_left: Dict[str, int] = {}
    for i, step in enumerate(parsed):
        consumers = list(readers[step.name])
        stream_read: Optional[Tuple[int, str]] = None
        if i + 1 < len(parsed) and (parsed[i + 1].from_ or step.name) == step.name:
            stream_read = (i + 1, "in")
        elif i == len(parsed) - 1:
            stream_read = (len(parsed), "out")
        cell_reads = [c for c in consumers if c != stream_read]
        needs_cell[step.name] = bool(cell_reads)
        cell_reads_left[step.name] = len(cell_reads)

    ops: List[Any] = []
    attr_cells: Dict[str, str] = {}  # "step.attr" -> cell name

    # Pre-scan @output refs: the producer op must be wrapped in Capture at ITS step.
    attr_refs: Dict[str, List[str]] = {}
    for step in parsed:
        for ref in step.bind.values():
            if "." in ref:
                head, _, attr = ref.partition(".")
                attr_refs.setdefault(head, [])
                if attr not in attr_refs[head]:
                    attr_refs[head].append(attr)

    def take_cell(name: str) -> Tuple[str, bool]:
        """(cell, is_last_read) — decrement the read counter."""
        cell_reads_left[name] -= 1
        return name, cell_reads_left[name] <= 0

    prev_name: Optional[str] = None
    for i, step in enumerate(parsed):
        # 1. input slot (explicit from == previous step consumes the stream — no Use)
        if step.from_ is not None and step.from_ != prev_name:
            cell, last = take_cell(step.from_)
            ops.append(Use(name=cell, drop=last))

        # 2. fan-in slots
        if step.target_from is not None or step.metadata_from is not None:
            drops: List[str] = []
            kwargs: Dict[str, Any] = {}
            if step.target_from is not None:
                cell, last = take_cell(step.target_from)
                kwargs["target_from"] = cell
                if last:
                    drops.append(cell)
            if step.metadata_from is not None:
                cell, last = take_cell(step.metadata_from)
                kwargs["metadata_from"] = cell
                if last:
                    drops.append(cell)
            ops.append(Mix(drop=drops, **kwargs))

        # 3. the op, wrapped for binds (Apply) and @output captures (Capture)
        emitted: Optional[Any] = step.op
        if emitted is not None:
            for param, ref in step.bind.items():
                if "." in ref:
                    cell = attr_cells[ref]
                    cell_reads_left.setdefault(cell, 1)
                    cell_reads_left[cell] -= 1
                    emitted = Apply(op=emitted, param=param, source=cell, drop=cell_reads_left[cell] <= 0)
                else:
                    cell, last = take_cell(ref)
                    emitted = Apply(op=emitted, param=param, source=cell, drop=last)
            captures = attr_refs.get(step.name, [])
            if captures:
                for attr in captures:
                    cell = f"{step.name}.{attr}"
                    attr_cells[cell] = cell
                    cell_reads_left[cell] = sum(
                        1 for s in parsed for r in s.bind.values() if r == f"{step.name}.{attr}"
                    )
                if len(captures) == 1:
                    emitted = Capture(op=emitted, output=captures[0], name=f"{step.name}.{captures[0]}")
                else:
                    emitted = Capture(op=emitted, captures={a: f"{step.name}.{a}" for a in captures})
            ops.append(emitted)
        elif step.target_from is None and step.metadata_from is None and step.from_ is None and i == 0:
            # identity first step ({}: names the source) — nothing to run
            pass

        # 4. persist the result for non-stream readers
        if needs_cell[step.name]:
            ops.append(Save(name=step.name))

        prev_name = step.name

    # 5. the output: if it is not the final stream, fetch it.
    if parsed and outputs != parsed[-1].name:
        cell, last = take_cell(outputs)
        ops.append(Use(name=cell, drop=last))

    # 6. safety net: any cells the liveness pass left alive get an explicit Drop.
    leftovers = [name for name, left in cell_reads_left.items() if left > 0 and needs_cell.get(name, True)]
    if leftovers:
        ops.append(Drop(names=sorted(leftovers)))

    return ops


# ---------------------------------------------------------------------------
# Lifting: flat context-ops list -> flow
# ---------------------------------------------------------------------------


_CONTEXT_OP_CLASSES = (Save, Use, Drop, Apply, Capture, Mix)


def _ctx_view(raw: Any) -> Optional[type]:
    """The context-op class ``raw`` represents, live instance OR confluid marker; else None."""
    if isinstance(raw, _ConfluidFluid):
        target = getattr(raw, "target", None)
        return target if isinstance(target, type) and target in _CONTEXT_OP_CLASSES else None
    return type(raw) if isinstance(raw, _CONTEXT_OP_CLASSES) else None


def _ctx_field(raw: Any, name: str, default: Any = None) -> Any:
    """Read a context-op field off a live instance OR a marker's kwargs."""
    if isinstance(raw, _ConfluidFluid):
        return raw.kwargs.get(name, default)
    return getattr(raw, name, default)


def _capture_items(raw: Any) -> Dict[str, str]:
    """A Capture's ``{output_attr: cell}`` map, live instance or marker."""
    if not isinstance(raw, _ConfluidFluid):
        return cast(Capture, raw)._items()
    items = dict(raw.kwargs.get("captures") or {})
    output = str(raw.kwargs.get("output", "") or "")
    if output:
        items.setdefault(output, str(raw.kwargs.get("name", "") or "") or output)
    return items


def _auto_name(op: Any, index: int, taken: Dict[str, int]) -> str:
    if op is None:
        base = "step"
    elif isinstance(op, _ConfluidFluid):
        target = getattr(op, "target", None)
        base = getattr(target, "__name__", str(target)).lower()
    else:
        base = type(op).__name__.lower()
    taken[base] = taken.get(base, 0) + 1
    return base if taken[base] == 1 else f"{base}_{taken[base]}"


def from_ops(ops: Sequence[Any], outputs: str = "") -> Tuple[Dict[str, Any], str]:
    """Lift a flat op list into a ``(flow_mapping, outputs)`` pair.

    Context ops are absorbed into step grammar: ``Save`` names the preceding step (or an
    identity first step for a source fork), ``Use`` starts a branch (``from:``), ``Mix``
    becomes ``target_from``/``metadata_from`` on the following step (or a pure fan-in
    step), ``Apply``/``Capture`` unwrap into ``bind:`` references, and ``Drop`` vanishes
    (liveness is recomputed on lowering). A plain linear list lifts to a linear flow with
    auto-generated step names. The result round-trips: ``to_ops(from_ops(ops))`` is
    execution-equivalent to ``ops``.

    Accepts LIVE ops or confluid ``Instance``/``Class`` MARKERS interchangeably (the
    FluxStudio exporter lifts compiled marker lists without materializing them, keeping
    hoisted-constant ``!ref:``\\ s intact); a real op arrives in the flow mapping verbatim
    (marker in, marker out).
    """
    flow_map: Dict[str, Dict[str, Any]] = {}
    taken: Dict[str, int] = {}
    prev_name: Optional[str] = None
    capture_cells: Dict[str, str] = {}  # cell -> "step.attr" bind ref
    pending: Dict[str, Any] = {}  # accumulating step grammar (from/target_from/...)

    def cell_ref(cell: str) -> str:
        """Map a cell name to its bind reference (an @output capture or a step result)."""
        return capture_cells.get(cell, cell)

    def flush_step(op: Optional[Any], explicit_name: Optional[str] = None) -> str:
        nonlocal prev_name, pending
        name = explicit_name or _auto_name(op, len(flow_map), taken)
        entry: Any
        if op is not None and not pending:
            entry = op  # a grammar-less step is just its op (the compact document form)
        else:
            entry = dict(pending)
            if op is not None:
                entry["op"] = op
        flow_map[name] = entry
        pending = {}
        prev_name = name
        return name

    for raw in ops:
        view = _ctx_view(raw)
        if view is Save:
            save_name = str(_ctx_field(raw, "name", ""))
            if prev_name is not None:
                # rename the just-flushed step to the cell name
                entry = flow_map.pop(prev_name)
                # keep bind refs pointing at the old auto name consistent
                for e in flow_map.values():
                    b = e.get("bind") if isinstance(e, dict) else None
                    if b:
                        for p, r in list(b.items()):
                            head, dot, attr = r.partition(".")
                            if head == prev_name:
                                b[p] = save_name + (dot + attr if dot else "")
                flow_map[save_name] = entry
                for cell, ref in list(capture_cells.items()):
                    head, dot, attr = ref.partition(".")
                    if head == prev_name:
                        capture_cells[cell] = save_name + (dot + attr if dot else "")
                prev_name = save_name
            else:
                # Save before any op: an identity step naming the source
                flow_map[save_name] = {}
                prev_name = save_name
            continue
        if view is Use:
            pending["from"] = cell_ref(str(_ctx_field(raw, "name", "")))
            continue
        if view is Mix:
            mix_grammar: Dict[str, Any] = {}
            if _ctx_field(raw, "input_from", ""):
                mix_grammar["from"] = cell_ref(str(_ctx_field(raw, "input_from")))
            if _ctx_field(raw, "target_from", ""):
                mix_grammar["target_from"] = cell_ref(str(_ctx_field(raw, "target_from")))
            if _ctx_field(raw, "metadata_from", ""):
                mix_grammar["metadata_from"] = cell_ref(str(_ctx_field(raw, "metadata_from")))
            pending.update(mix_grammar)
            pending["__mix_pending__"] = True
            continue
        if view is Drop:
            continue  # liveness is recomputed on lowering

        # A real op (possibly Apply/Capture-wrapped): unwrap into bind grammar.
        bind: Dict[str, str] = {}
        captures: Dict[str, str] = {}
        op: Any = raw
        while _ctx_view(op) in (Apply, Capture):
            if _ctx_view(op) is Capture:
                for attr, cell in _capture_items(op).items():
                    captures[cell] = attr
            else:
                bind[str(_ctx_field(op, "param", ""))] = cell_ref(str(_ctx_field(op, "source", "")))
            op = _ctx_field(op, "op")
        pending.pop("__mix_pending__", None)
        if bind:
            pending["bind"] = bind
        name = flush_step(op)
        for cell, attr in captures.items():
            capture_cells[cell] = f"{name}.{attr}"

    # A trailing Mix (or Use) with no following op = a pure fan-in step.
    if pending:
        pending.pop("__mix_pending__", None)
        flush_step(None)

    if not flow_map:
        raise ValueError("from_ops: no steps could be lifted (empty op list?)")
    out = outputs or prev_name or next(reversed(flow_map))
    return flow_map, out


def flow_yaml_to_flux(path: str, source: Optional[Any] = None) -> Any:
    """Convenience: load a ``flow:`` YAML and return the SERIAL engine (lowered Flux)."""
    from sampleflux.core import Flux

    doc = _confluid_resolve(path)
    if not isinstance(doc, dict) or "flow" not in doc:
        raise ValueError(f"flow_yaml_to_flux: {path!r} has no 'flow:' mapping")
    parsed, outputs = parse_flow(doc["flow"], str(doc.get("outputs", "") or ""))
    return Flux(source=source, ops=to_ops(parsed, outputs))
