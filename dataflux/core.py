import concurrent.futures
import multiprocessing
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

from confluid import configurable

from dataflux.sample import Sample


class OptionalContextManager:
    """Helper for 'with' statements where the object might not be a context manager."""

    def __enter__(self) -> "OptionalContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


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
class Flux:
    """
    The primary stream engine for DataFlux.
    Wraps any iterable or indexed dataset and provides a functional API.
    """

    def __init__(self, source: Optional[Iterable[Any]] = None, ops: Optional[List[Any]] = None) -> None:
        self.source = source
        self.ops: List[Any] = ops or []
        self._workers = 1

    @classmethod
    def from_source(cls, source: Any) -> "Flux":
        """Create a Flux from a DataSource."""
        return cls(source=source)

    @classmethod
    def joint(cls, fluxes: List["Flux"]) -> "Flux":
        """Create a new Flux that aggregates multiple other Flux streams."""
        return cls(source=JointFlux(fluxes))

    def __len__(self) -> int:
        """Return the length of the underlying source if available."""
        if self.source is not None and hasattr(self.source, "__len__"):
            return len(self.source)  # type: ignore
        return 0

    def to_sink(self, sink: Any) -> None:
        """Write the entire flux to a DataSink."""
        from dataflux.storage.base import Storage

        # 1. Open sink if it's a context-aware storage
        if isinstance(sink, Storage):
            target_sink: Any = sink
        else:
            target_sink = OptionalContextManager()

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

    def __iter__(self) -> Iterator[Sample]:
        """Execute the pipeline lazily (single or multi-process)."""
        if not self.source:
            return

        if self._workers > 1:
            yield from self._iter_parallel()
        else:
            yield from self._iter_sequential()

    def _iter_sequential(self) -> Iterator[Sample]:
        """Standard single-threaded execution."""
        if self.source is None:
            return
        for item in self.source:
            sample = Sample.from_any(item)
            result = _worker_task(sample, self.ops)
            if result is not None:
                yield result

    def _iter_parallel(self) -> Iterator[Sample]:
        """Multiprocess execution engine."""
        if self.source is None:
            return

        # We use 'spawn' to be consistent with LogFlow and prevent CI deadlocks
        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._workers, mp_context=ctx) as executor:
            futures = []
            for item in self.source:
                sample = Sample.from_any(item)
                futures.append(executor.submit(_worker_task, sample, self.ops))

            for future in futures:
                result = future.result()
                if result is not None:
                    yield result

    def collect(self) -> List[Sample]:
        """Materialize the full flux into a list."""
        return list(self)
