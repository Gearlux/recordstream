"""The per-record KERNEL — the one execution model, shared by both authoring forms.

``Stream``'s positional steps and a ``flow:`` document's author-named steps arrive here as
the same :class:`FlowStep` list (``docs/architecture.md`` §3). Two routes, and they MUST
agree record-for-record: :func:`_run_linear` for a straight chain (no step environment at
all — what keeps an ``ops:`` list cheap) and :func:`_run_from` for a graph that forks,
merges or binds. :func:`is_linear` is the gate between them.

Result lifetimes are automatic: :func:`_result_readers` counts each step's readers
slot-granularly and the kernel drops a result after its last one.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from loggair import get_logger

from recordstream.core.families import OpInvoker, OpMatcher, _apply_op, _expand, _op_expands, _sync_op_families
from recordstream.flow.steps import _MISSING, FlowStep, _read_output, _split_bind_ref
from recordstream.items import Record

logger = get_logger(__name__)


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

    Module-level for pickling (the same constraint
    :func:`recordstream.core.stream._worker_task` obeys). Returns a LIST because an expanding
    step makes one seed yield several records. ``readers`` is deliberately NOT passed across
    the boundary — it is cheap to derive once per worker call relative to the process hop, and
    shipping it would add a second pickled structure that must stay in sync with ``steps``.
    """
    _sync_op_families(families)
    return run_steps_multi(seed, steps, outputs)
