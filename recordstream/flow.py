"""The ``flow:`` document and the :class:`FlowGraph` engine.

A **flow document** is the named-step form of a pipeline: a mapping of ``step-name → op``,
where a step's name is how later steps reference its result. It is the spelling to reach for
when a pipeline BRANCHES; a straight chain is written as a plain ``ops:`` list, which the
engine compiles to positional steps (``recordstream.core.linear_steps``). Both parse to the
same :class:`FlowStep` list and run through the same per-record kernel — there is ONE
execution model, and no lowering pass between the two forms (the flow⇄ops converters and the
per-record context ops they emitted were deleted 2026-07-30; see ``docs/architecture.md`` §3).

.. code-block:: yaml

    flow:
      spec:     !class:mypkg.MakeSpectrogram()                # input: the source record
      rescaled: !class:recordstream.ops.numpy.Threshold()     # input: previous step
      masked:   !class:mypkg.Segment() {from: spec}           # 2nd reader of spec = fan-out
      out: {from: masked, merge_from: [rescaled]}             # fan-in (no op)
    outputs: out

Step grammar (the three RESERVED step keys, stripped before the op is built):

- ``from:`` — the step supplying this step's input record. Omitted = the previous step
  (the first step reads the source record). Must name an EARLIER step: document order is
  the schedule, so forward references are errors and cycles are inexpressible.
- ``merge_from:`` — fan-in: UNION the named steps' record entries into this step's incoming
  record before the op runs (listed order, last-write-wins on a key collision).
- ``bind:`` — ``{param: ref}`` per-record parameters: ``ref`` is a step name (the step's
  whole result record), ``step[key]`` (one entry of it), or ``step.attr`` (the step op's
  live ``@output`` after it ran — read through wrapper chains by :func:`_read_output`).

A step may be a plain mapping with no op (``out: {from: a, merge_from: [b]}``) — a pure
fan-in/identity step; ``{}`` is the identity (used to give the source a referable name).
``outputs:`` names the step whose result the pipeline yields (default: the last step).

Step results are freed automatically: :func:`_result_readers` counts each step's readers
slot-granularly and the kernel drops a result after its last one. A straight chain needs no
environment at all — :func:`is_linear` routes it to :func:`_run_linear`.
"""

import concurrent.futures
import inspect
import multiprocessing
from copy import deepcopy
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union, cast

from confluid import configurable, flow
from confluid import resolve as _confluid_resolve
from confluid.fluid import Fluid as _ConfluidFluid
from loggair import get_logger

from recordstream.core import (
    OpInvoker,
    OpMatcher,
    _apply_op,
    _expand,
    _extra_op_families,
    _op_expands,
    _sync_op_families,
)
from recordstream.items import Record

logger = get_logger(__name__)

RESERVED_STEP_KEYS = ("from", "merge_from", "bind")
"""Step-grammar keys stripped from a step mapping before the op is constructed."""

__all__ = ["FlowGraph", "FlowStep", "parse_flow", "run_steps", "run_steps_multi", "RESERVED_STEP_KEYS"]

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


def _result_readers(steps: Sequence[FlowStep], outputs: str) -> Dict[str, List[Tuple[int, str]]]:
    """Step-result cell -> ordered ``(consumer_index, slot)`` reads.

    Slot granularity matters: one consumer step may read the SAME producer through several
    slots (its input AND a ``bind`` param), and only the ``"in"`` slot of the immediately
    following step can ride the linear stream. Slots: ``"in"`` (input), ``"merge"``,
    ``"bind"``, and the final ``"out"`` read at index ``len(steps)``. A ``bind`` step-result
    reference counts; an ``@output`` (``step.attr``) reference does NOT.

    NO steps is the identity graph (a bare ``ops: []``): nothing is produced, so nothing is
    read — and there is no output step to account for.
    """
    if not steps:
        return {}
    readers: Dict[str, List[Tuple[int, str]]] = {s.name: [] for s in steps}
    for i, step in enumerate(steps):
        implicit = steps[i - 1].name if i > 0 else None
        source = step.from_ or implicit
        if source is not None:
            readers[source].append((i, "in"))
        for ref in step.merge_from:
            readers[ref].append((i, "merge"))
        for ref in step.bind.values():
            parsed = _split_bind_ref(ref)
            if parsed.attr is None:
                readers[parsed.step].append((i, "bind"))
    readers[outputs].append((len(steps), "out"))
    return readers


# ---------------------------------------------------------------------------
# The FlowGraph engine
# ---------------------------------------------------------------------------


def run_steps_multi(
    seed: Any,
    steps: Sequence[FlowStep],
    outputs: str,
    readers: Optional[Dict[str, List[Tuple[int, str]]]] = None,
) -> List[Record]:
    """Run ONE source record through the parsed steps, returning EVERY resulting record.

    The engine's per-record kernel, module-level so a spawn worker can pickle a reference to
    it. Usually one record back, zero when a step filtered (an op returned ``None``), several
    when a 1→N EXPANDING step fired.

    ``readers`` is the slot-granular reader accounting from :func:`_result_readers`; it
    depends only on ``(steps, outputs)``, so a caller running many records MUST compute it
    once and pass it in — recomputing per record is an O(steps²) tax on every record (it was
    measured at 3.4 µs/record on a 23-step pipeline, roughly half the graph engine's total
    overhead over a flat op list).

    EXPANSION semantics: a step whose op carries ``EXPANDS`` yields N children, and the
    REMAINING subgraph runs once per child over its own shallow copy of the step environment
    (independent name→result maps, shared values — the graph twin of ``Context.copy()``).
    Traversal is DEPTH-FIRST, so sibling order matches the nested-loop intuition and the flat
    engine's documented order. An empty expansion or a ``None`` child just drops that branch.

    NO steps is the IDENTITY graph — the seed comes straight back. That is what makes a bare
    ``Stream(source=..., ops=[])`` yield its source unchanged once the flat engine routes
    through this kernel.
    """
    if not steps:
        return [] if seed is None else [cast(Record, seed)]
    out: List[Record] = []
    if is_linear(steps, outputs):
        _run_linear(seed, steps, 0, out)
        return out
    if readers is None:
        readers = _result_readers(steps, outputs)
    base_remaining = {name: len(idx) for name, idx in readers.items()}
    _run_from(0, seed, steps, outputs, {}, base_remaining, None, out)
    return out


def is_linear(steps: Sequence[FlowStep], outputs: str) -> bool:
    """True when the graph is a straight chain — no named reference reaches back.

    Every step reads the one before it, nothing binds, nothing merges, and the yielded step
    is the last one. Such a graph needs no step ENVIRONMENT at all: the record can ride a
    local variable exactly as it did in the flat op loop, which is what keeps an ``ops:``
    list as cheap to run as before it became a graph (the env bookkeeping measured ~33%
    of engine overhead on a 23-step chain).
    """
    if not steps or outputs != steps[-1].name:
        return False
    return all(s.from_ is None and not s.bind and not s.merge_from for s in steps)


def _run_linear(
    record: Any,
    steps: Sequence[FlowStep],
    index: int,
    out: List[Record],
) -> None:
    """Run a straight chain from ``steps[index:]`` — the env-free path (see :func:`is_linear`).

    Same expansion contract as :func:`_run_from`: a 1→N step forks the remaining chain,
    depth-first, so sibling order matches the nested-loop intuition.
    """
    for i in range(index, len(steps)):
        op = steps[i].op
        if op is None:
            continue
        if _op_expands(op):
            for child in _expand(op, record):
                _run_linear(child, steps, i + 1, out)
            return
        result = _apply_op(record, op)
        if result is None:
            return
        record = result
    out.append(cast(Record, record))


def _run_from(
    index: int,
    seed: Any,
    steps: Sequence[FlowStep],
    outputs: str,
    env: Dict[str, Any],
    remaining: Dict[str, int],
    prev: Optional[str],
    out: List[Record],
) -> None:
    """Run ``steps[index:]`` over ``env``, appending every surviving result to ``out``.

    Recurses ONCE PER CHILD at an expanding step (recursion depth = the number of expanding
    steps on the path, not the record count), which is what gives depth-first sibling order
    for free.

    Each expansion branch gets its OWN shallow copy of the step environment (independent
    name→result maps, shared values), so siblings cannot see each other's results.
    """

    def read_result(name: str, *, copy: bool) -> Any:
        value = env[name]
        remaining[name] -= 1
        if remaining[name] <= 0:
            del env[name]
        elif copy:
            value = deepcopy(value)
        return value

    for i in range(index, len(steps)):
        step = steps[i]
        # 1. the input record (implicit stream reads move; explicit fan-out reads copy)
        if step.from_ is not None:
            record = read_result(step.from_, copy=True)
        elif prev is not None:
            record = read_result(prev, copy=False)
        else:
            record = seed

        # 2. fan-in: UNION the merge_from steps' entries (slot order, last wins)
        if step.merge_from:
            if not isinstance(record, dict):
                raise TypeError(
                    f"flow step {step.name!r}: merge_from is the record fan-in but the carrier is "
                    f"{type(record).__name__} — expected a record dict."
                )
            merged = dict(record)
            for ref in step.merge_from:
                value = read_result(ref, copy=True)
                if not isinstance(value, dict):
                    raise TypeError(
                        f"flow step {step.name!r}: merge_from step {ref!r} holds "
                        f"{type(value).__name__}, expected a record"
                    )
                merged.update(value)
            record = merged

        # 3. per-record parameter binds
        if step.op is not None:
            op = step.op
            for param, ref in step.bind.items():
                parsed = _split_bind_ref(ref)
                if parsed.attr is not None:
                    producer = next(s for s in steps if s.name == parsed.step)
                    value = _read_output(producer.op, parsed.attr)
                    if value is _MISSING:
                        raise AttributeError(
                            f"flow step {step.name!r}: bind {param}={ref!r} — "
                            f"step {parsed.step!r} op has no @output attribute {parsed.attr!r}"
                        )
                else:
                    value = read_result(parsed.step, copy=False)
                    if isinstance(value, dict) and parsed.key:
                        # "step[key]" = the named entry; bare "step" = the whole record.
                        value = value[parsed.key]
                setattr(op, param, value)

            # 4. a 1→N step forks the REMAINING subgraph, one branch per child
            if _op_expands(op):
                for child in _expand(op, record):
                    child_env = dict(env)
                    child_env[step.name] = child
                    _run_from(i + 1, seed, steps, outputs, child_env, dict(remaining), step.name, out)
                return

            result = _apply_op(record, op)
            if result is None:
                return
            record = result

        env[step.name] = record
        prev = step.name

    if outputs in env:
        out.append(cast(Record, env[outputs]))


def run_steps(
    seed: Any,
    steps: Sequence[FlowStep],
    outputs: str,
    readers: Optional[Dict[str, List[Tuple[int, str]]]] = None,
) -> Optional[Record]:
    """Strictly 1→1 twin of :func:`run_steps_multi` — one result back, or ``None``.

    For callers that need exactly one carrier (indexing, a single-record probe). An expanding
    step RAISES here rather than silently dropping its siblings; route those through
    :func:`run_steps_multi`.
    """
    for step in steps:
        if step.op is not None and _op_expands(step.op):
            raise TypeError(
                f"flow step {step.name!r}: {type(step.op).__name__!r} is a 1→N expanding op, which "
                "this strictly 1→1 route cannot carry — iterate the graph instead."
            )
    results = run_steps_multi(seed, steps, outputs, readers)
    return results[0] if results else None


def _graph_worker_task(
    seed: Any,
    steps: Sequence[FlowStep],
    outputs: str,
    families: Optional[List[Tuple[str, OpMatcher, OpInvoker]]] = None,
) -> List[Record]:
    """Spawn-worker entry point: re-register third-party op families, then run one record.

    Module-level for pickling (the same constraint :func:`recordstream.core._worker_task_multi`
    obeys). Returns a LIST because an expanding step makes one seed yield several records.
    ``readers`` is deliberately NOT passed across the boundary — it is cheap to derive once per
    worker call relative to the process hop, and shipping it would add a second pickled
    structure that must stay in sync with ``steps``.
    """
    _sync_op_families(families)
    return run_steps_multi(seed, steps, outputs)


@configurable(category="engine")
class FlowGraph:
    """Named-step graph engine — executes a ``flow:`` document natively.

    The named-step twin of :class:`~recordstream.core.Stream`, over the SAME kernel: steps run
    in document order against a per-record environment of named results, with fan-out isolation
    (copy-on-read, move on last read) and automatic result lifetimes. A LINEAR graph converts
    to a Stream (:meth:`to_stream`); a branchy one has no flat spelling by design.

    Args:
        source: Any iterable or indexable dataset (duck-typed) yielding record dicts; ``None`` = empty stream.
        flow: The flow mapping (step-name -> op / marker / step mapping) or a parsed list of FlowStep.
        outputs: Name of the step whose result is yielded. Blank (default) = the last step.
        chunk_size: Batch size for chunked iteration; ``0`` (the default) yields single records.
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
        self._readers: Optional[Dict[str, List[Tuple[int, str]]]] = None

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

    def _ensure_readers(self) -> Dict[str, List[Tuple[int, str]]]:
        """The reader accounting, computed ONCE per graph (see :func:`run_steps`)."""
        if self._readers is None:
            steps, outputs = self._ensure_parsed()
            self._readers = _result_readers(steps, outputs)
        return self._readers

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
        """Load a flat ``{ops: [...]}`` YAML document as a LINEAR step graph.

        No lifting is involved: a sequence IS a graph, so the op list becomes positional
        steps (``recordstream.core.linear_steps``) — the same compilation a ``Stream``'s
        ``ops`` list goes through, because they are the same thing spelled two ways.
        """
        from recordstream.core import Stream, linear_steps

        stream = Stream.from_ops_yaml(path, source=source)
        steps, outputs = linear_steps(stream.ops)
        return cls(source=source, flow=steps, outputs=outputs)

    # -- execution ---------------------------------------------------------

    def _run(self, seed: Any) -> Optional[Any]:
        """Run one record through the steps; ``None`` = filtered (an op returned None)."""
        steps, outputs = self._ensure_parsed()
        return run_steps(seed, steps, outputs, self._ensure_readers())

    def __iter__(self) -> Iterator[Any]:
        if self.source is None:
            return
        it = self._iter_records()
        if self._chunk_size > 0:
            batch: List[Record] = []
            for record in it:
                batch.append(record)
                if len(batch) == self._chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            yield from it

    def _iter_records(self) -> Iterator[Record]:
        if self._workers > 1:
            yield from self._iter_parallel()
            return
        assert self.source is not None
        steps, outputs = self._ensure_parsed()
        readers = self._ensure_readers()
        for item in self.source:
            yield from run_steps_multi(item, steps, outputs, readers)

    def _iter_parallel(self) -> Iterator[Record]:
        """Multiprocess execution — the graph's OWN spawn pool, one future per source record.

        Mirrors :meth:`recordstream.core.Stream._iter_parallel`: ``spawn`` (consistent with
        Loggair, no CI deadlocks), third-party op families shipped to the workers by
        reference. The steps pickle because their ops already must; the source never crosses
        the boundary (only the seed record does).
        """
        assert self.source is not None
        steps, outputs = self._ensure_parsed()
        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._workers, mp_context=ctx) as executor:
            extra_families = _extra_op_families()
            futures = [
                executor.submit(_graph_worker_task, item, steps, outputs, extra_families) for item in self.source
            ]
            for future in futures:
                yield from future.result()

    @property
    def _expands(self) -> bool:
        """True when any step op is 1→N — the length/index map is then unknowable."""
        return any(step.op is not None and _op_expands(step.op) for step in self._ensure_parsed()[0])

    def _guard_not_expanding(self, operation: str) -> None:
        if self._expands:
            raise TypeError(
                f"FlowGraph.{operation} is unavailable: a step op is 1→N EXPANDING, so the "
                "expanded length/index map is unknowable. Iterate the graph, wrap it in a torch "
                "IterableDataset, window at the SOURCE for random access, or call .collect()."
            )

    def __len__(self) -> int:
        from collections.abc import Sized

        self._guard_not_expanding("__len__")
        if isinstance(self.source, Sized):
            return len(self.source)
        return 0

    def __getitem__(self, index: int) -> Any:
        if self.source is None:
            raise TypeError("FlowGraph source is None — cannot index.")
        self._guard_not_expanding("__getitem__")
        if hasattr(self.source, "__getitem__"):
            raw = self.source[index]
        else:
            raise TypeError(
                f"FlowGraph source {type(self.source).__name__} does not support indexing; "
                "wrap it in a list or use iteration."
            )
        result = self._run(raw)
        if result is None:
            raise IndexError(f"Record {index} filtered out by the flow")
        return result

    def parallel(self, workers: int = 4) -> "FlowGraph":
        """Enable multiprocess execution on the graph's own spawn pool."""
        self._workers = workers
        return self

    def batch(self, chunk_size: int) -> "FlowGraph":
        """Group yielded records into lists of ``chunk_size``."""
        self._chunk_size = chunk_size
        return self

    def collect(self) -> List[Any]:
        """Materialize the full stream into a list."""
        return list(self)

    def to_stream(self) -> Any:
        """The ``Stream`` twin of a LINEAR graph — same source, same ops, same engine.

        Only a straight chain converts: a `Stream` carries an op LIST, which cannot express
        fan-out. A branchy graph has no flat spelling (that is what the deleted lowering pass
        manufactured, at the cost of destroying the structure), so it raises.
        """
        from recordstream.core import Stream

        steps, outputs = self._ensure_parsed()
        if not is_linear(steps, outputs):
            raise TypeError(
                "FlowGraph.to_stream: this graph is not a straight chain (it forks or merges), "
                "and a Stream's ops list cannot express that. Iterate the FlowGraph directly — "
                "it is the same engine."
            )
        return Stream(source=self.source, ops=[s.op for s in steps if s.op is not None])
