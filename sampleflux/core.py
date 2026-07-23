import concurrent.futures
import multiprocessing
from contextlib import nullcontext
from typing import Any, Callable, Collection, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union, cast

import torch.utils.data
from confluid import configurable
from confluid import load as _confluid_load
from confluid import materialize as _confluid_materialize
from confluid.fluid import Fluid as _ConfluidFluid
from loggair import get_logger

from sampleflux.bag.items import item_data, with_data
from sampleflux.bag.sample import Role, Sample, primary
from sampleflux.context import Context, activate
from sampleflux.projection import ProjectionField

logger = get_logger(__name__)

# Role a projection field maps onto in the typed bag (``metadata`` -> the ``aux`` role).
_PROJECTION_ROLES: Dict[str, str] = {"input": "input", "target": "target", "metadata": "aux"}


def _op_expands(op: Any) -> bool:
    """True when an op is a 1→N expanding op (explicit ``EXPANDS = True`` class attribute)."""
    return bool(getattr(op, "EXPANDS", False))


def _apply_op(sample: Sample, op: Any) -> Optional[Sample]:
    """Apply one op to the typed :class:`Sample` bag verbatim.

    The single op-application chokepoint shared by the sequential, parallel (via
    :func:`_worker_task`), streamed, and random-access (``__getitem__``) paths. A transform
    takes the whole bag and returns a new bag (or ``None`` to drop the sample); composing
    ops (``Parallel`` / ``Enable`` / ``TransformChain`` / ``RandomApply`` / the context ops)
    route their inner ops through here so every op is applied identically.
    """
    return cast(Optional[Sample], op(sample))


def _describe_deferred_source(source: Any) -> str:
    """Return a human-friendly description of a still-deferred Confluid source.

    Surfaces the tag/target so the error explains WHAT was deferred instead
    of just noting it isn't a live object.
    """
    target = getattr(source, "target", "<unknown>")
    target_name = target if isinstance(target, str) else getattr(target, "__qualname__", str(target))
    return f"{type(source).__name__}(target={target_name!r})"


def _fluid_source_guidance(source: Any) -> str:
    """Build an actionable message when Flux.source is still a Confluid Fluid."""
    return (
        f"Flux.source is still a deferred Confluid marker: {_describe_deferred_source(source)}. "
        "Confluid has not materialized it yet. Fixes: (a) in YAML, write the source as "
        "`!class:X()` (with parens) instead of `!class:X` so it becomes an Instance and is "
        "materialized at load time; (b) or call `flow(source)` on the source before handing "
        "it to Flux."
    )


def _fluid_op_guidance(op: Any, index: int) -> str:
    """Build an actionable message when a Flux op is still a Confluid Fluid."""
    return (
        f"Flux.ops[{index}] is still a deferred Confluid marker: {_describe_deferred_source(op)}. "
        "Ops must be live callables at iteration time. Fixes: (a) in YAML, write each op as "
        "`!class:X()` (with parens) so it becomes an Instance and is materialized at load "
        "time; (b) or call `flow(op)` on the op before handing it to Flux."
    )


def _check_ops_materialized(ops: List[Any]) -> None:
    """Raise a single actionable error if any op is still a Confluid Fluid marker."""
    for i, op in enumerate(ops):
        if isinstance(op, _ConfluidFluid):
            raise TypeError(_fluid_op_guidance(op, i))


@configurable
class FilterOp:
    """Configurable filter operation.

    The op form of :meth:`Flux.filter` — a predicate gate over the stream: the sample
    passes when the predicate returns ``True`` and is dropped otherwise (``__call__``
    returns ``None``, which every engine route treats as "skip this sample").

    Args:
        p: Predicate ``Sample -> bool``; the sample passes through when it returns ``True``, else is dropped.
            Defaults to ``None`` (zero-arg construction); a predicate must be set before the op runs.
    """

    def __init__(self, p: Optional[Callable[[Sample], bool]] = None):
        # Lazy / zero-arg: store config only; a missing predicate is validated lazily in __call__.
        self.p = p

    def __call__(self, s: Sample) -> Optional[Sample]:
        if self.p is None:
            raise ValueError("FilterOp.p (predicate) is not set — provide a Sample->bool callable before use.")
        return s if self.p(s) else None


@configurable
class WrappedOp:
    """Configurable transformation wrapper with smart mapping.

    The op form of :meth:`Flux.map` — lifts a plain function over one Sample field. The
    callable is ALWAYS stored as its importable ``module:function`` path (via
    :mod:`sampleflux.discovery`), so the op pickles across ``spawn`` workers and
    serializes into Confluid YAML verbatim; the live function resolves lazily on first
    call.

    Args:
        f: The wrapped callable, or its importable ``module:function`` path (stored as a string for serialization).
            Defaults to ``""`` (zero-arg construction); resolving an empty path fails lazily on first call.
        s: Which field to transform — ``"input"`` (default, the primary input field's payload),
            ``"target"`` (the primary target field's payload), or ``"all"`` (the whole ``Sample`` bag).
        kw: Extra keyword arguments forwarded to the wrapped callable on every call (defaults to none).
    """

    def __init__(self, f: Union[str, Callable] = "", s: str = "input", kw: Optional[Dict[str, Any]] = None):
        from sampleflux.discovery import get_callable_path

        # Lazy / zero-arg: store config only (the empty-path default resolves lazily via the `func`
        # property). EXPLICIT: always store the string path for serialization.
        self.f = get_callable_path(f) if callable(f) else f
        self.s = s
        self.kw = dict(kw) if kw else {}
        # Internal cache for the live callable
        self._func_cache: Optional[Callable] = None

    @property
    def func(self) -> Callable:
        if self._func_cache is None:
            from sampleflux.discovery import resolve_callable

            self._func_cache = resolve_callable(self.f)
        return self._func_cache

    def __call__(self, sample: Sample) -> Optional[Sample]:
        if self.s == "all":
            return cast(Sample, self.func(sample, **self.kw))
        role: Role = "input" if self.s == "input" else "target"
        key, item = primary(sample, role)
        new_data = self.func(item_data(item), **self.kw)
        return sample.replace_field(key, with_data(item, new_data))


class _Carried(NamedTuple):
    """A :class:`Sample` travelling the streamed route together with its per-sample Context."""

    sample: Any
    ctx: Context


def _expand(op: Any, sample: Any) -> List[Any]:
    """Run a 1→N EXPANDING op and return its flattened children."""
    raw = op(sample)
    if raw is None:
        return []
    return [child for child in raw if child is not None]


def _worker_task(sample: Any, ops: List[Any]) -> Optional[Any]:
    """Single-result worker for STRICTLY 1→1 op lists (the ``Parallel`` op's contract).

    Kept for callers that need exactly one carrier back; expanding ops raise here —
    route expanding pipelines through :func:`_worker_task_multi`.
    """
    results = _worker_task_multi(sample, ops, allow_expansion=False)
    return results[0] if results else None


def _worker_task_multi(sample: Any, ops: List[Any], allow_expansion: bool = True) -> List[Any]:
    """Top-level helper for multiprocess workers. Must be at top level for pickling.

    Runs one source :class:`Sample` through the op list and returns EVERY resulting sample —
    usually one, zero when filtered, several when a 1→N EXPANDING op fired; each expansion
    child continues through the REMAINING ops with a shallow copy of the per-sample Context,
    depth-first so sibling order matches the nested-loop intuition.

    Activates ONE fresh per-sample :class:`~sampleflux.context.Context` around the op loop so
    context ops (``Save``/``Use``/``Apply``/``Capture``/``MergeFields``) can move data between
    the linear stream and named cells — the executor itself stays a plain ``for op in ops``
    loop. Contexts are created inside the worker (spawn-safe: ops pickle, a Context never
    crosses a process boundary).
    """
    from collections import deque

    pending: "deque[Tuple[Any, Context, int]]" = deque([(sample, Context(), 0)])
    out: List[Any] = []
    while pending:
        current, ctx, start = pending.popleft()
        alive = True
        with activate(ctx):
            i = start
            while i < len(ops):
                op = ops[i]
                i += 1
                if _op_expands(op):
                    if not allow_expansion:
                        raise TypeError(
                            f"op {type(op).__name__!r} is a 1→N expanding op, which this strictly "
                            "1→1 route cannot carry — run it through the Flux iteration paths."
                        )
                    children = _expand(op, current)
                    if not children:
                        alive = False
                        break
                    # Depth-first: the first child continues inline; its siblings go to the
                    # FRONT of the queue (reversed, so sibling order is preserved).
                    for child in reversed(children[1:]):
                        pending.appendleft((child, ctx.copy(), i))
                    current = children[0]
                    continue
                result = _apply_op(current, op)
                if result is None:
                    alive = False
                    break
                current = result
        if alive and current is not None:
            out.append(current)
    return out


@configurable(category="engine")
class JointFlux:
    """
    Aggregates multiple Flux streams into a single joint stream.
    Each sub-flux maintains its own unique transformation chain.

    The iteration-only fan-in engine behind :meth:`Flux.joint`: each sub-flux applies
    its OWN op chain, so differently-processed streams concatenate lazily without
    materialization. For an indexable (random-access) concatenation of raw sources,
    use ``ConcatSource`` instead.

    Args:
        fluxes: The Flux streams to concatenate; iteration walks them in order and length is their sum.
            Defaults to ``None`` ⇒ an empty joint stream (zero-arg construction).
    """

    def __init__(self, fluxes: Optional[List["Flux"]] = None) -> None:
        # Lazy / zero-arg: store config only; no sub-fluxes ⇒ an empty stream.
        self.fluxes = fluxes if fluxes is not None else []

    def __iter__(self) -> Iterator[Sample]:
        """Iterate through all sub-fluxes sequentially."""
        for flux in self.fluxes:
            yield from flux

    def __len__(self) -> int:
        """Total length is the sum of all sub-fluxes."""
        return sum(len(f) for f in self.fluxes)


@configurable(category="engine")
class Flux(torch.utils.data.Dataset[Sample]):
    """
    The primary stream engine for SampleFlux.
    Wraps any iterable or indexed dataset and provides a functional API.

    Every carrier is a typed :class:`~sampleflux.bag.sample.Sample` bag, passed through the op
    chain verbatim (no coercion). ``source`` is duck-typed (any iterable; the Indexable
    protocol if ``__getitem__``/``__len__`` are present) and ``ops`` is a list of bare
    transforms ``Sample -> Optional[Sample]``.

    Args:
        source: Any iterable or indexable dataset (duck-typed) yielding ``Sample`` bags; ``None`` = empty stream.
        ops: Ordered transforms ``Sample -> Optional[Sample]`` applied lazily on access (``None`` = no ops).
        chunk_size: Parallel-processing chunk size; ``0`` (the default) processes sequentially.
    """

    def __init__(
        self,
        source: Optional[Iterable[Any]] = None,
        ops: Optional[List[Any]] = None,
        chunk_size: Optional[int] = 0,
    ) -> None:
        self.source = source
        self.ops: List[Any] = ops or []
        self._workers = 1
        self._chunk_size = chunk_size or 0
        # Populated on first random access when the source is iterable-only
        # (has ``__len__`` but not ``__getitem__``).
        self._indexable_cache: Optional[List[Any]] = None

    def _guard_live_source(self) -> Any:
        """Return the source, surfacing a clear error when it's still a Fluid marker."""
        if isinstance(self.source, _ConfluidFluid):
            raise TypeError(_fluid_source_guidance(self.source))
        return self.source

    @classmethod
    def from_source(cls, source: Any) -> "Flux":
        """Create a Flux from a DataSource."""
        return cls(source=source)

    @classmethod
    def joint(cls, fluxes: List["Flux"]) -> "Flux":
        """Create a new Flux that aggregates multiple other Flux streams."""
        return cls(source=JointFlux(fluxes))

    @classmethod
    def from_ops_yaml(cls, path: str, source: Optional[Iterable[Any]] = None) -> "Flux":
        """Attach an ops-only Confluid YAML (e.g. one exported by a pipeline-authoring tool) to ``source``.

        ``path`` is the ``{ops: [!class:...()]}`` document produced by an external graph
        exporter's ops-export. Op markers are materialized to live callables before being
        attached (``confluid.load`` leaves ``!class:`` markers nested under a mapping key
        deferred, so ``confluid.materialize`` flows them into live ops).
        """
        loaded = _confluid_load(path)
        raw_ops = loaded.get("ops", []) if isinstance(loaded, dict) else []
        ops = list(_confluid_materialize(raw_ops))
        return cls(source=source, ops=ops)

    @classmethod
    def from_flow_yaml(cls, path: str, source: Optional[Iterable[Any]] = None) -> "Flux":
        """Attach a ``{flow: {...}}`` graph document to ``source``, LOWERED to the serial form."""
        from sampleflux.flow import flow_yaml_to_flux

        return cast("Flux", flow_yaml_to_flux(path, source=source))

    @property
    def _expands(self) -> bool:
        """True when any (materialized) op is a 1→N expanding op — the pipeline is then iterable-only."""
        return any(not isinstance(op, _ConfluidFluid) and _op_expands(op) for op in self.ops)

    def _guard_not_expanding(self, operation: str) -> None:
        if self._expands:
            culprit = next(
                type(op).__name__ for op in self.ops if not isinstance(op, _ConfluidFluid) and _op_expands(op)
            )
            raise TypeError(
                f"Flux.{operation}: the pipeline contains the 1→N expanding op {culprit!r}, so the "
                "expanded length/index mapping is unknowable up front — the pipeline is ITERABLE-ONLY. "
                "Iterate it (or wrap in a torch IterableDataset); for random access, window/expand at "
                "the source instead, or materialize with list(flux) first."
            )

    def __len__(self) -> int:
        """Return the length of the underlying source if available."""
        from collections.abc import Sized

        source = self._guard_live_source()
        self._guard_not_expanding("__len__")
        if isinstance(source, Sized):
            return len(source)
        return 0

    def __getitem__(self, index: int) -> Any:
        """Random access: get the i-th sample with ops applied."""
        source = self._guard_live_source()
        if source is None:
            raise TypeError("Flux source is None — cannot index. Pass a DataSource / iterable to Flux(source=...).")
        self._guard_not_expanding("__getitem__")

        if hasattr(source, "__getitem__"):
            raw = source[index]
        elif hasattr(source, "__len__"):
            if self._indexable_cache is None:
                logger.debug(f"Flux: materializing iterable-only source {type(source).__name__} for random access.")
                self._indexable_cache = list(source)
            raw = self._indexable_cache[index]
        else:
            raise TypeError(
                f"Flux source {type(source).__name__} does not support indexing and has no __len__ "
                "(bare iterator). Map-style DataLoader random access is unsafe on a one-shot "
                "iterator; give the source a __len__ (then Flux caches on first access) or wrap "
                "it in ``list(...)`` before handing it to Flux."
            )
        _check_ops_materialized(self.ops)
        sample: Any = raw
        with activate(Context()):
            for op in self.ops:
                result = _apply_op(sample, op)
                if result is None:
                    raise IndexError(f"Sample {index} filtered out by {op}")
                sample = result
        return cast(Sample, sample)

    def to_sink(self, sink: Any) -> None:
        """Write the entire flux to a DataSink."""
        from sampleflux.storage.base import Storage

        target_sink: Any = sink if isinstance(sink, Storage) else nullcontext()

        with target_sink:
            for sample in self:
                sink.write(sample)
            sink.flush()

    def parallel(self, workers: int = 4) -> "Flux":
        """Enable multiprocess execution for the pipeline."""
        self._workers = workers
        return self

    def batch(self, chunk_size: int) -> "Flux":
        """Group samples into chunks (lists of N samples)."""
        self._chunk_size = chunk_size
        return self

    def map(self, func: Callable, select: str = "input", **kwargs: Any) -> "Flux":
        """Append a transformation to the flux."""
        op = WrappedOp(func, select, kwargs)
        self.ops.append(op)
        return self

    def filter(self, predicate: Callable[[Sample], bool]) -> "Flux":
        """Filter the flux based on a predicate."""
        self.ops.append(FilterOp(predicate))
        return self

    def __iter__(self) -> Iterator[Any]:
        """Execute the pipeline lazily."""
        if not self._guard_live_source():
            return

        if any(hasattr(op, "stream") and callable(op.stream) for op in self.ops):
            it = self._iter_streamed()
        elif self._workers > 1:
            it = self._iter_parallel()
        else:
            it = self._iter_sequential()

        if self._chunk_size > 0:
            batch = []
            for sample in it:
                batch.append(sample)
                if len(batch) == self._chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            yield from it

    def _iter_streamed(self) -> Iterator[Sample]:
        """Mixed per-sample / stream-level op chain (a stream-level op exposes ``.stream``)."""
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)

        def to_carried() -> Iterator[Optional[_Carried]]:
            for item in source:
                yield _Carried(item, Context())

        def per_sample(stream: Iterator[Optional[_Carried]], op: Any) -> Iterator[Optional[_Carried]]:
            expands = _op_expands(op)
            for c in stream:
                if c is None:
                    continue
                with activate(c.ctx):
                    if expands:
                        children = _expand(op, c.sample)
                    else:
                        s = _apply_op(c.sample, op)
                if expands:
                    for j, child in enumerate(children):
                        yield _Carried(child, c.ctx if j == 0 else c.ctx.copy())
                else:
                    yield None if s is None else _Carried(s, c.ctx)

        def strip(stream: Iterator[Optional[_Carried]], op: Any) -> Iterator[Optional[Sample]]:
            for c in stream:
                if c is None:
                    yield None
                    continue
                if c.ctx.live():
                    raise RuntimeError(
                        f"Flux: context cells {c.ctx.live()!r} are still live at the stream-level op "
                        f"{type(op).__name__!r}. Context cells cannot cross a stream-op boundary "
                        f"(e.g. Parallel) — drop them before it, or move the whole graph inside it."
                    )
                yield c.sample

        def wrap(stream: Iterator[Optional[Sample]]) -> Iterator[Optional[_Carried]]:
            for s in stream:
                yield None if s is None else _Carried(s, Context())

        carried: Iterator[Optional[_Carried]] = to_carried()
        for op in self.ops:
            if hasattr(op, "stream") and callable(op.stream):
                carried = wrap(op.stream(strip(carried, op)))
            else:
                carried = per_sample(carried, op)

        for c in carried:
            if c is not None:
                yield c.sample

    def _iter_sequential(self) -> Iterator[Sample]:
        """Standard single-threaded execution."""
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)
        for item in source:
            yield from _worker_task_multi(item, self.ops)

    def _iter_parallel(self) -> Iterator[Sample]:
        """Multiprocess execution engine."""
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)

        # We use 'spawn' to be consistent with Loggair and prevent CI deadlocks
        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._workers, mp_context=ctx) as executor:
            futures = []
            for item in source:
                futures.append(executor.submit(_worker_task_multi, item, self.ops))

            for future in futures:
                yield from future.result()

    def collect(self) -> List[Sample]:
        """Materialize the full flux into a list."""
        return list(self)

    def project(self, fields: Collection[ProjectionField]) -> Iterator[Sample]:
        """Yield pipeline-output Samples carrying only ``fields`` (the projection primitive).

        Implements :class:`sampleflux.projection.SupportsProjection`. Flux must run its op
        chain to produce each Sample (an op may consume the input), so this is the generic
        "iterate, then keep only fields of the requested roles" form. ``fields`` is a subset
        of ``{"input", "target", "metadata"}`` (mapped onto the ``input`` / ``target`` /
        ``aux`` roles). Lazy: a generator.
        """
        want_roles = {_PROJECTION_ROLES[f] for f in fields}
        for sample in self:
            keep = [k for k in sample.keys() if sample.role_of(k) in want_roles]
            yield Sample({k: sample[k] for k in keep}, {k: sample.role_of(k) for k in keep})
