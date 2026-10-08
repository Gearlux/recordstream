"""Generic source→sink pipeline runner.

:class:`DatasetProcessor` orchestrates a :class:`~recordstream.core.stream.Stream` from source
to sink — a runnable that drives whole-dataset processing (windowing, format
conversion, data acquisition) with an optional console progress bar. It is the
generic, modality-neutral data-pipeline runner: it iterates the stream and writes
each item to the sink, carrier-agnostic (it never inspects item internals), so it
works for any ``Stream`` regardless of what flows through it.

Wired as the ``runnable:`` object of a config and run via ``recordstream run``, or
docked into a visual-editor canvas as a runnable node.

With ``workers`` above 1 the records are BUILT in spawn worker processes — record ``i`` is
``source[i]`` run through the ops in one of them — while the sink stays here and receives them in
index order (rationale: ``docs/architecture.md``, "A dataset run builds its records in workers").
"""

import multiprocessing
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Sized, Tuple, cast

from annotated_types import Interval
from confluid import configurable
from loggair import get_logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from typing_extensions import Annotated

from recordstream.core import Stream
from recordstream.core.families import OpInvoker, OpMatcher, _extra_op_families, _sync_op_families
from recordstream.core.stream import linear_steps
from recordstream.flow import _result_readers, run_steps_multi
from recordstream.items import Record
from recordstream.runnable import ProgressReporting
from recordstream.storage.base import Storage

logger = get_logger(__name__)

#: How many processes build records at once; 1 builds them in the running process, one at a time.
Workers = Annotated[int, Interval(ge=1)]

#: How many records per worker may be in flight — being built, or built and waiting for the sink in index order.
_WINDOW_PER_WORKER = 2

#: What a worker process holds, set once when it starts (:func:`_start_worker`): the stream's source and its ops
#: compiled to steps. A module global because a process pool's initializer can only leave state behind this way.
_WORKER: Dict[str, Any] = {}


def _start_worker(
    source: Any, steps: Sequence[Any], outputs: str, families: List[Tuple[str, OpMatcher, OpInvoker]]
) -> None:
    """A worker's start: the third-party op families re-registered (a spawn worker imports afresh), the source and
    the compiled ops kept for every record the worker builds."""
    _sync_op_families(families)
    _WORKER.update(source=source, steps=steps, outputs=outputs, readers=_result_readers(steps, outputs))


def _build(index: int) -> List[Record]:
    """Record ``index`` of the worker's source through its ops: a list, since a 1 -> N op gives several records and a
    filter none — exactly what the sequential run gives for that record."""
    return run_steps_multi(_WORKER["source"][index], _WORKER["steps"], _WORKER["outputs"], _WORKER["readers"])


def _stream_total(stream: Stream) -> Optional[int]:
    """``len(stream.source)`` when the source is sized, else ``None`` (a glob-based / streaming source)."""
    try:
        return len(stream.source)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


@configurable
class DatasetProcessor(ProgressReporting):
    """Orchestrate a RecordStream pipeline from source to sink.

    Args:
        stream: The :class:`~recordstream.core.stream.Stream` to execute. Required to run;
            defaulted to ``None`` for zero-arg construction (validated in
            :meth:`run`, the workspace lazy-construction rule).
        sink: Optional sink; when absent, records are materialized to a list.
        show_progress: If ``True``, wrap iteration with a ``rich.progress`` bar.
            The total is derived from ``len(stream.source)`` when available; an
            unsized source falls back to a count-only bar. Default ``False``.
        progress_desc: Optional label for the progress bar (defaults to
            ``"DatasetProcessor"``). Ignored when ``show_progress`` is ``False``.
        workers: How many spawn processes build the records at once — record i is source[i] through the ops,
            the sink gets them in order; 1 (default) builds them here, one at a time.
    """

    def __init__(
        self,
        stream: Optional[Stream] = None,
        sink: Optional[Any] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        workers: Workers = 1,
    ) -> None:
        self.stream = stream
        self.sink = sink
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        self.workers = workers

    def run(self) -> None:
        logger.info("Starting DatasetProcessor...")
        if self.stream is None:
            raise ValueError("DatasetProcessor.run() requires a 'stream' — none was configured.")
        workers = int(self.workers)
        # confluid refuses a value under 1 when the processor is BUILT; this catches one set later.
        if workers < 1:
            raise ValueError(f"DatasetProcessor: workers must be >= 1; got {self.workers}")
        # Confluid keeps Class kwargs deferred (post-construction paradigm) so
        # when the processor was loaded from YAML, ``self.stream``, its source,
        # its ops, and ``self.sink`` may all be Fluid stubs. Materialize them
        # here so callers don't need to know.
        from confluid import flow
        from confluid.fluid import Fluid

        stream = flow(self.stream) if isinstance(self.stream, Fluid) else self.stream
        if isinstance(stream.source, Fluid):
            stream.source = flow(stream.source)
        stream.ops = [flow(op) if isinstance(op, Fluid) else op for op in stream.ops]
        self.stream = stream
        sink = flow(self.sink) if isinstance(self.sink, Fluid) else self.sink

        # Drive an executor's progress bar (a GUI canvas) per item — independent of the console
        # ``show_progress`` rich bar; a no-op when no progress callback was injected.
        total = _stream_total(stream)
        desc = self.progress_desc or "DatasetProcessor"
        records: Iterable[Any] = stream if workers == 1 else _records_from_workers(stream, workers)
        iterator = self._wrap_progress(records, total)

        if sink:
            logger.info(f"Streaming data to sink: {sink.__class__.__name__}")
            # Replicates recordstream.core.stream.Stream.to_sink so we can iterate through
            # our progress wrapper while preserving the Storage context + flush.
            sink_ctx: Any = sink if isinstance(sink, Storage) else nullcontext()
            count = 0
            with sink_ctx:
                for record in iterator:
                    sink.write(record)
                    count += 1
                    self._report_progress(count, total, desc)
                sink.flush()
            logger.info(f"Streamed {count} record(s) to sink.")
        else:
            logger.info("No sink provided. Materializing data in-memory.")
            results = []
            for count, record in enumerate(iterator, start=1):
                results.append(record)
                self._report_progress(count, total, desc)
            logger.info(f"Processed {len(results)} records.")

        logger.info("Processing complete.")

    def _wrap_progress(self, records: Iterable[Any], total: Optional[int]) -> Iterable[Any]:
        """Wrap the records in rich.progress when ``show_progress`` is enabled.

        ``total`` is the source's length when it has one (:func:`_stream_total`) — not every
        DataSource implements ``__len__`` (e.g. glob-based streaming sources); a missing length
        degrades to a count-only bar instead of breaking the run.
        """
        if not self.show_progress:
            return records
        desc = self.progress_desc or "DatasetProcessor"
        return _ProgressIter(records, total=total, desc=desc)


def _records_from_workers(stream: Stream, workers: int) -> Iterator[Record]:
    """The stream's records, record ``i`` built as ``source[i]`` through the ops in one of ``workers`` spawn
    processes, yielded in index order — what iterating the stream gives, record for record.

    At most ``2 * workers`` records are being built or waiting at once: the next index is handed out only when the
    oldest record has been yielded, so a run of any length holds a bounded number of records. A record that fails
    stops the run with its own error; the indices not started yet are cancelled.
    """
    source = stream.source
    name = type(source).__name__
    if not (hasattr(source, "__len__") and hasattr(source, "__getitem__")):
        raise ValueError(
            f"DatasetProcessor: workers={workers} builds each record in a worker from its index, and the source "
            f"({name}) cannot be indexed — give it len() and [i], or set workers: 1"
        )
    for position, op in enumerate(stream.ops):
        if callable(getattr(op, "stream", None)):  # a stream-level op (Parallel) runs a pool of its own
            raise ValueError(
                f"DatasetProcessor: workers={workers}, and op {position} of the stream ({type(op).__name__}) starts "
                "workers of its own — a worker cannot start workers; set workers: 1 or take the Parallel out"
            )
    chunk_size = int(getattr(stream, "_chunk_size", 0) or 0)
    if chunk_size > 0:
        raise ValueError(
            f"DatasetProcessor: workers={workers} and a stream that batches its records (chunk_size {chunk_size}) "
            "— set workers: 1"
        )
    steps, outputs = linear_steps(stream.ops)
    count = len(cast(Sized, source))  # an indexable source, checked above
    logger.info(f"Building {count} record(s) on {workers} workers.")
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),  # the workspace's start method: no forked state
        initializer=_start_worker,
        initargs=(source, steps, outputs, _extra_op_families()),
    )
    waiting: Deque["Future[List[Record]]"] = deque()
    try:
        for index in range(count):
            waiting.append(pool.submit(_build, index))
            if len(waiting) >= _WINDOW_PER_WORKER * workers:
                yield from waiting.popleft().result()
        while waiting:
            yield from waiting.popleft().result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


class _ProgressIter:
    """Iterable wrapper that opens a ``rich.progress`` bar at ``__iter__`` time.

    Kept as a class (not a generator) so ``DatasetProcessor._wrap_progress``
    can return a value that is truthy to ``bool()`` even when empty, matching
    the contract of raw ``Stream`` which behaves like a ``Sized`` (``Stream``
    subclasses ``torch.utils.data.Dataset``).
    """

    def __init__(self, stream: Iterable[Any], total: Optional[int], desc: str) -> None:
        self._stream = stream
        self._total = total
        self._desc = desc

    def __iter__(self) -> Iterator[Any]:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(self._desc, total=self._total)
            for record in self._stream:
                yield record
                progress.update(task, advance=1)


__all__ = ["DatasetProcessor"]
