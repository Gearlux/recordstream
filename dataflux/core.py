import concurrent.futures
import json
import multiprocessing
from contextlib import nullcontext
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch.utils.data
from confluid import configurable
from confluid import load as _confluid_load
from confluid import materialize as _confluid_materialize
from confluid.fluid import Fluid as _ConfluidFluid
from logflow import get_logger

from dataflux.projection import ProjectionField
from dataflux.sample import FEATURES_KEY, SPEC_KEY, TYPE_KEYS, Sample

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataflux.typespec import SampleType

logger = get_logger(__name__)


@lru_cache(maxsize=None)
def _serialized_type_keys(produces: "SampleType") -> Tuple[str, str]:
    """Serialize an op's ``PRODUCES`` to the two stored-type JSON strings, memoized per spec so the
    ``datasets.Features`` build happens once per distinct spec rather than once per sample."""
    features, extras = produces.to_hf_features()
    return json.dumps(features.to_dict()), json.dumps(extras)


def _refresh_type(sample: Sample, op: Any) -> Sample:
    """Keep a sample's stored type honest after an op — only when the sample already carries one.

    Default (untracked) pipelines never stamp a type, so this is a no-op and metadata is byte-identical
    to before. When a stored type IS present (set via :meth:`Sample.with_type` or loaded from a typed
    dataset), an op that declares ``PRODUCES`` refreshes it; an op that declares none drops it so
    :meth:`Sample.describe` falls back to inference rather than reporting a stale type.
    """
    if not any(key in sample.metadata for key in TYPE_KEYS):
        return sample
    produces = getattr(op, "PRODUCES", None)
    if produces is not None:
        features_key, spec_key = _serialized_type_keys(produces)
        return sample._replace(metadata={**sample.metadata, FEATURES_KEY: features_key, SPEC_KEY: spec_key})
    return sample._replace(metadata={k: v for k, v in sample.metadata.items() if k not in TYPE_KEYS})


def _apply_op(sample: Sample, op: Any) -> Optional[Sample]:
    """Apply one op and refresh the stored type. The single op-application chokepoint shared by the
    sequential, parallel (via :func:`_worker_task`), streamed, and random-access (``__getitem__``) paths."""
    result = op(sample)
    if result is None:
        return None
    return _refresh_type(result, op)


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

    Args:
        f: The wrapped callable, or its importable ``module:function`` path (stored as a string for serialization).
            Defaults to ``""`` (zero-arg construction); resolving an empty path fails lazily on first call.
        s: Which Sample slot to transform — ``"input"`` (default), ``"target"``, or ``"all"`` (the whole Sample).
        kw: Extra keyword arguments forwarded to the wrapped callable on every call (defaults to none).
    """

    def __init__(self, f: Union[str, Callable] = "", s: str = "input", kw: Optional[Dict[str, Any]] = None):
        from dataflux.discovery import get_callable_path

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
            from dataflux.discovery import resolve_callable

            self._func_cache = resolve_callable(self.f)
        return self._func_cache

    def __call__(self, sample: Sample) -> Optional[Sample]:
        try:
            if self.s == "input":
                new_input = self.func(sample.input, **self.kw)
                return sample._replace(input=new_input)
            elif self.s == "target":
                new_target = self.func(sample.target, **self.kw)
                return sample._replace(target=new_target)
            elif self.s == "all":
                return cast(Sample, self.func(sample, **self.kw))
            return sample
        except Exception as e:
            raise e


def _worker_task(sample: Sample, ops: List[Any]) -> Optional[Sample]:
    """Top-level helper for multiprocess workers. Must be at top level for pickling."""
    current_sample: Optional[Sample] = sample
    for op in ops:
        if current_sample is None:
            return None
        current_sample = _apply_op(current_sample, op)
    return current_sample


@configurable(category="engine")
class JointFlux:
    """
    Aggregates multiple Flux streams into a single joint stream.
    Each sub-flux maintains its own unique transformation chain.

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
    The primary stream engine for DataFlux.
    Wraps any iterable or indexed dataset and provides a functional API.

    Annotation design (kept intentionally ``Any``):
        Per the DataFlux mandate "Functional Purity: Transforms are plain
        Python callables. Never introduce base classes or complex inheritance
        for data operations.", ``source`` is duck-typed (any iterable; the
        Indexable protocol if ``__getitem__``/``__len__`` are present) and
        ``ops`` is a list of bare callables ``Sample -> Optional[Sample]``.
        No ``Source`` or ``Op`` ABC is introduced.

        Downstream auto-gen pydantic mirrors (``confluid.to_pydantic``)
        coerce abstract iterable types to ``Any`` so identity-tracked
        serialization (e.g. shared-source dataset-split patterns in
        navigaitor) works correctly — see
        ``confluid/pydantic_export.py:_ITER_TYPES_AS_ANY``.

    Args:
        source: Any iterable or indexable dataset (duck-typed) to wrap; ``None`` yields an empty stream.
        ops: Ordered callables ``Sample -> Optional[Sample]`` applied lazily on access (``None`` = no ops).
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
        """Return the source, surfacing a clear error when it's still a Fluid marker.

        Flux does not materialize deferred Confluid markers itself — that's
        Confluid's job — but if a user hands Flux a deferred Class/Instance
        marker we raise with an actionable message instead of letting the
        failure surface as ``num_samples=0`` or a generic ``TypeError`` deep
        inside torch's DataLoader.
        """
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
        """Attach an ops-only Confluid YAML (e.g. exported from FluxStudio) to ``source``.

        ``path`` is the ``{ops: [!class:...()]}`` document produced by
        :func:`fluxstudio.export.export_ops_yaml` (the ``fluxstudio export`` CLI or the
        canvas Export button). It also accepts an inline YAML string (``confluid.load``
        handles both).

        The op markers are **materialized to live callables** before being attached:
        ``confluid.load`` leaves ``!class:`` markers nested under a mapping key deferred
        (its final flow pass doesn't descend dict→list), so a plain ``load(path)["ops"]``
        would hand :class:`Flux` deferred ``Instance`` markers — which iteration rejects by
        design (see :meth:`_guard_live_source` / ``_check_ops_materialized``). Routing through
        :func:`confluid.materialize` flows the top-level list of markers into live ops.
        """
        loaded = _confluid_load(path)
        raw_ops = loaded.get("ops", []) if isinstance(loaded, dict) else []
        ops = list(_confluid_materialize(raw_ops))
        return cls(source=source, ops=ops)

    def __len__(self) -> int:
        """Return the length of the underlying source if available.

        Surfaces a clear error when the source is still a deferred Confluid
        marker so downstream callers (e.g. torch's DataLoader) don't end up
        reporting the opaque ``num_samples=0``.
        """
        from collections.abc import Sized

        source = self._guard_live_source()
        if isinstance(source, Sized):
            return len(source)
        return 0

    def __getitem__(self, index: int) -> Sample:
        """Random access: get the i-th sample with ops applied.

        Supports three source shapes:

        - **Indexable** (``__getitem__`` present) — delegates directly.
        - **Iterable with ``__len__``** (map-style-but-stream, like
          :class:`waivefront.regions_source.RegionsJsonSource`) — materializes
          the full source into a list on first access, caches it on the Flux
          instance, and indexes into the cache on every subsequent call.
          The list is built once per Flux lifetime, not once per epoch.
        - **Bare iterator** (no ``__len__``) — raises ``TypeError``. Caching
          a one-shot iterator silently would consume the user's source; if
          random access is genuinely needed, either give the source a
          ``__len__`` or wrap with ``list(source)`` explicitly at the call
          site.
        """
        source = self._guard_live_source()
        if source is None:
            raise TypeError("Flux source is None — cannot index. Pass a DataSource / iterable to Flux(source=...).")

        if hasattr(source, "__getitem__"):
            raw = source[index]
        elif hasattr(source, "__len__"):
            if self._indexable_cache is None:
                logger.debug(
                    f"Flux: materializing iterable-only source " f"{type(source).__name__} for map-style random access."
                )
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
        sample = Sample.from_any(raw)
        for op in self.ops:
            result = _apply_op(sample, op)
            if result is None:
                raise IndexError(f"Sample {index} filtered out by {op}")
            sample = result
        return sample

    def to_sink(self, sink: Any) -> None:
        """Write the entire flux to a DataSink."""
        from dataflux.storage.base import Storage

        # Open sink if it's a context-aware storage; otherwise no-op context.
        target_sink: Any = sink if isinstance(sink, Storage) else nullcontext()

        with target_sink:
            for sample in self:
                sink.write(sample)
            sink.flush()

    def parallel(self, workers: int = 4) -> "Flux":
        """
        Enable multiprocess execution for the pipeline.

        Args:
            workers: Number of worker processes to spawn.
        """
        self._workers = workers
        return self

    def batch(self, chunk_size: int) -> "Flux":
        """
        Group samples into chunks (lists of N samples).

        Args:
            chunk_size: Number of samples per chunk.
        """
        self._chunk_size = chunk_size
        return self

    def map(self, func: Callable, select: str = "input", **kwargs: Any) -> "Flux":
        """
        Append a transformation to the flux.
        """
        op = WrappedOp(func, select, kwargs)
        self.ops.append(op)
        return self

    def filter(self, predicate: Callable[[Sample], bool]) -> "Flux":
        """Filter the flux based on a predicate."""
        self.ops.append(FilterOp(predicate))
        return self

    def __iter__(self) -> Iterator[Any]:
        """Execute the pipeline lazily.

        Routing:
          * Any op exposes a callable ``stream`` attribute (e.g.
            :class:`dataflux.ops.parallel.Parallel`) → :meth:`_iter_streamed`,
            which composes the upstream iterator through stream-level ops.
          * Else ``self._workers > 1`` → legacy :meth:`_iter_parallel`.
          * Else :meth:`_iter_sequential`.
        """
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
        """Mixed per-sample / stream-level op chain.

        Per-sample ops are applied via ``op(sample)``. Ops that implement
        ``.stream(sample_iter)`` (e.g.
        :class:`dataflux.ops.parallel.Parallel`) are handed the upstream
        generator and yield transformed samples themselves. ``None`` results
        are filtered, matching :meth:`_iter_sequential`.
        """
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)

        def to_samples() -> Iterator[Sample]:
            for item in source:
                yield Sample.from_any(item)

        def per_sample(stream: Iterator[Optional[Sample]], op: Any) -> Iterator[Optional[Sample]]:
            for s in stream:
                if s is None:
                    continue
                yield _apply_op(s, op)

        stream: Iterator[Optional[Sample]] = to_samples()
        for op in self.ops:
            if hasattr(op, "stream") and callable(op.stream):
                stream = op.stream(stream)
            else:
                stream = per_sample(stream, op)

        for s in stream:
            if s is not None:
                yield s

    def _iter_sequential(self) -> Iterator[Sample]:
        """Standard single-threaded execution."""
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)
        for item in source:
            sample = Sample.from_any(item)
            result = _worker_task(sample, self.ops)
            if result is not None:
                yield result

    def _iter_parallel(self) -> Iterator[Sample]:
        """Multiprocess execution engine."""
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)

        # We use 'spawn' to be consistent with LogFlow and prevent CI deadlocks
        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._workers, mp_context=ctx) as executor:
            futures = []
            for item in source:
                sample = Sample.from_any(item)
                futures.append(executor.submit(_worker_task, sample, self.ops))

            for future in futures:
                result = future.result()
                if result is not None:
                    yield result

    def collect(self) -> List[Sample]:
        """Materialize the full flux into a list."""
        return list(self)

    def project(self, fields: Collection[ProjectionField]) -> Iterator[Sample]:
        """Yield pipeline-output Samples carrying only ``fields`` (the projection primitive).

        Implements :class:`dataflux.projection.SupportsProjection`. Flux must run
        its op chain to produce each Sample (an op may consume the input), so this
        is the generic "iterate, then drop unrequested fields" form — it cannot
        skip input construction the way a leaf source (e.g. an image dataset that
        reads only the label column) can. Lazy: a generator. ``fields`` is a
        subset of ``{"input", "target", "metadata"}``.
        """
        want = frozenset(fields)
        for sample in self:
            yield Sample(
                input=sample.input if "input" in want else None,
                target=sample.target if "target" in want else None,
                metadata=sample.metadata if "metadata" in want else {},
            )
