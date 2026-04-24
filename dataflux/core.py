import concurrent.futures
import multiprocessing
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union, cast

import torch.utils.data
from confluid import configurable
from confluid.fluid import Fluid as _ConfluidFluid
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)


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
    """Configurable filter operation."""

    def __init__(self, p: Callable[[Sample], bool]):
        self.p = p

    def __call__(self, s: Sample) -> Optional[Sample]:
        return s if self.p(s) else None


@configurable
class WrappedOp:
    """Configurable transformation wrapper with smart mapping."""

    def __init__(self, f: Union[str, Callable], s: str, kw: Dict[str, Any]):
        from dataflux.discovery import get_callable_path

        # EXPLICIT: Always store the string path for serialization
        self.f = get_callable_path(f) if callable(f) else f
        self.s = s
        self.kw = kw
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
        current_sample = op(current_sample)
    return current_sample


@configurable
class JointFlux:
    """
    Aggregates multiple Flux streams into a single joint stream.
    Each sub-flux maintains its own unique transformation chain.
    """

    def __init__(self, fluxes: List["Flux"]) -> None:
        self.fluxes = fluxes

    def __iter__(self) -> Iterator[Sample]:
        """Iterate through all sub-fluxes sequentially."""
        for flux in self.fluxes:
            yield from flux

    def __len__(self) -> int:
        """Total length is the sum of all sub-fluxes."""
        return sum(len(f) for f in self.fluxes)


@configurable
class Flux(torch.utils.data.Dataset[Sample]):
    """
    The primary stream engine for DataFlux.
    Wraps any iterable or indexed dataset and provides a functional API.
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
            raise TypeError(
                "Flux source is None — cannot index. Pass a DataSource / iterable to Flux(source=...)."
            )

        if hasattr(source, "__getitem__"):
            raw = source[index]
        elif hasattr(source, "__len__"):
            if self._indexable_cache is None:
                logger.debug(
                    f"Flux: materializing iterable-only source "
                    f"{type(source).__name__} for map-style random access."
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
            result = op(sample)
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
        """Execute the pipeline lazily (single or multi-process)."""
        if not self._guard_live_source():
            return

        it = self._iter_parallel() if self._workers > 1 else self._iter_sequential()

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
