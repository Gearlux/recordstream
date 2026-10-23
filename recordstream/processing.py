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
``workers: auto`` builds one record to measure it, then as many at once as the machine's memory,
cores and GPU hold; with any count, no record starts while free memory is under the reserve the
run leaves the machine (:mod:`recordstream.resources`).

With ``start`` the run builds only ``source[start:]`` (a :class:`~recordstream.sources.range.RangeSource` over the
source): a run resumed where its sink stopped, the records before ``start`` never built again.
"""

import copy
import multiprocessing
import os
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import nullcontext
from typing import Any, Deque, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Sized, Tuple, Union, cast

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
from recordstream.resources import OutOfMemory, RecordCost, WorkerGovernor, gigabytes, measure, read_memory, reserve_for
from recordstream.runnable import ProgressReporting
from recordstream.sources.range import RangeSource
from recordstream.storage.base import Storage

logger = get_logger(__name__)

#: How many processes build records at once; 1 builds them in the running process, one at a time; ``auto`` measures a
#: record and builds as many at once as the machine holds.
Workers = Union[Annotated[int, Interval(ge=1)], Literal["auto"]]
#: The first source index a run builds; 0 builds every record.
Start = Annotated[int, Interval(ge=0)]

#: How many records per worker may be in flight — being built, or built and waiting for the sink in index order.
_WINDOW_PER_WORKER = 2

#: How often, while records build, the run reads the machine's free memory again.
_MONITOR_SECONDS = 0.5

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


def _build(index: int) -> Tuple[List[Record], RecordCost]:
    """Record ``index`` of the worker's source through its ops — a list, since a 1 -> N op gives several records and a
    filter none, exactly what the sequential run gives for that record — and what building it took."""
    return measure(
        lambda: run_steps_multi(_WORKER["source"][index], _WORKER["steps"], _WORKER["outputs"], _WORKER["readers"])
    )


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
            the sink gets them in order; 1 (default) builds them here, one at a time; ``auto`` builds one record to
            measure it, then as many as the machine's memory, cores and GPU hold. Above 1 no record starts while
            free memory is under the reserve the run leaves the machine.
        start: The first source index built — a run resumed where its sink stopped builds source[start:] and never
            the records before; 0 (default) builds every record. The source must be indexable.
    """

    def __init__(
        self,
        stream: Optional[Stream] = None,
        sink: Optional[Any] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        workers: Workers = 1,
        start: Start = 0,
    ) -> None:
        self.stream = stream
        self.sink = sink
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        self.workers = workers
        self.start = start

    def run(self) -> None:
        logger.info("Starting DatasetProcessor...")
        if self.stream is None:
            raise ValueError("DatasetProcessor.run() requires a 'stream' — none was configured.")
        workers: Union[int, Literal["auto"]] = "auto" if self.workers == "auto" else int(self.workers)
        # confluid refuses a value under 1 when the processor is BUILT; this catches one set later.
        if workers != "auto" and workers < 1:
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
        stream = _from_start(stream, int(self.start))
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


def _from_start(stream: Stream, start: int) -> Stream:
    """``stream`` building only ``source[start:]`` — a copy over a :class:`RangeSource`, the processor's own stream
    left as it was — or ``stream`` itself when ``start`` is 0."""
    if start == 0:
        return stream
    if start < 0:
        raise ValueError(f"DatasetProcessor: start must be >= 0; got {start}")
    source = stream.source
    if not (hasattr(source, "__len__") and hasattr(source, "__getitem__")):
        raise ValueError(
            f"DatasetProcessor: start={start} builds the records from source[{start}] on, and the source "
            f"({type(source).__name__}) cannot be indexed — give it len() and [i], or set start: 0"
        )
    resumed = copy.copy(stream)
    resumed.source = RangeSource(source, start=start)
    logger.info(f"Starting at record {start} of {len(cast(Sized, source))}.")
    return resumed


def _records_from_workers(stream: Stream, workers: Union[int, Literal["auto"]]) -> Iterator[Record]:
    """The stream's records, record ``i`` built as ``source[i]`` through the ops in a spawn worker, yielded in index
    order — what iterating the stream gives, record for record.

    A :class:`~recordstream.resources.WorkerGovernor` decides before each record whether it may start: ``workers``
    records at once at most (``auto``: one, measured, then as many as the machine holds), none while free memory is
    under the reserve. At most ``2 ×`` the planned count are being built or waiting at once: the next index is handed
    out only when there is room, so a run of any length holds a bounded number of records. A record that fails stops
    the run with its own error, in index order; the indices not started yet are cancelled. Free memory under the
    reserve with at most one record building stops the run (:class:`~recordstream.resources.OutOfMemory`), its
    workers ended.
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
    auto = workers == "auto"
    governor = WorkerGovernor(ceiling=(os.cpu_count() or 1) if workers == "auto" else workers, auto=auto)
    if auto:
        logger.info(f"Building {count} record(s), workers: auto — 1 worker builds record 0 to measure it.")
    else:
        logger.info(f"Building {count} record(s) on {workers} workers.")
    pool = ProcessPoolExecutor(
        max_workers=governor.ceiling,
        mp_context=multiprocessing.get_context("spawn"),  # the workspace's start method: no forked state
        initializer=_start_worker,
        initargs=(source, steps, outputs, _extra_op_families()),
    )
    waiting: Deque[Tuple[int, "Future[Tuple[List[Record], RecordCost]]"]] = deque()  # handed out, not yet yielded
    built: Dict[int, List[Record]] = {}  # finished records waiting for the ones before them
    told: Optional[int] = None if auto else governor.ceiling  # the count last logged
    logged = False  # whether a measured plan was logged: the first one is news, later moves follow the memory
    holding = False
    next_index = 0
    try:
        while next_index < count or waiting:
            for index, future in waiting:  # every finished record's cost, as soon as it is known
                if index not in built and future.done() and future.exception() is None:
                    built[index], cost = future.result()
                    governor.observe(cost)
            reading = read_memory(_worker_pids(pool))
            running = sum(not future.done() for _, future in waiting)
            if governor.starved(running, reading) and (running or next_index < count):
                raise OutOfMemory(_out_of_memory(governor, reading, waiting, next_index))
            plan = governor.plan(reading)
            reserve = reserve_for(reading.total)
            # Under the reserve nothing starts whatever the plan says: the hold warning below says so instead.
            if governor.measured and plan.workers != told and reading.available >= reserve:
                line = f"workers: {workers} → {plan.describe()}"
                if not plan.fits:
                    logger.warning(f"{line} — not even one record fits the free memory; building one at a time")
                elif not logged:
                    logger.info(line)
                else:
                    logger.debug(line)
                told, logged = plan.workers, True
            while (
                next_index < count
                and len(waiting) < _WINDOW_PER_WORKER * plan.workers
                and governor.admit(running, reading)
            ):
                waiting.append((next_index, pool.submit(_build, next_index)))
                next_index += 1
                running += 1
            if next_index < count and reading.available < reserve and not holding:
                logger.warning(
                    f"Memory: {gigabytes(reading.available)} free, under the {gigabytes(reserve)} reserve — holding "
                    f"record {next_index} until a worker finishes ({running} building)."
                )
            holding = next_index < count and reading.available < reserve
            while waiting and waiting[0][1].done():
                index, future = waiting.popleft()
                if index in built:
                    records = built.pop(index)
                else:  # finished since the costs were read — or failed, and result() raises its error
                    records, cost = future.result()
                    governor.observe(cost)
                yield from records
            if waiting and not waiting[0][1].done():
                wait([future for _, future in waiting], timeout=_MONITOR_SECONDS, return_when=FIRST_COMPLETED)
    except OutOfMemory:
        _end_workers(pool)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _out_of_memory(
    governor: WorkerGovernor, reading: Any, waiting: Deque[Tuple[int, "Future[Any]"]], next_index: int
) -> str:
    """Why the run stops: the free memory, the reserve, the record building alone or the one that cannot start."""
    building = [index for index, future in waiting if not future.done()]
    where = f"while record {building[0]} builds alone" if building else f"before record {next_index} starts"
    measured = (
        f"records measured so far peak at {gigabytes(governor.memory)}" if governor.measured else "no record measured"
    )
    return (
        f"DatasetProcessor: the machine has {gigabytes(reading.available)} free, under the "
        f"{gigabytes(reserve_for(reading.total))} reserve, {where} ({measured}) — stopped before the machine swaps; "
        "free memory, or build smaller records"
    )


def _worker_pids(pool: ProcessPoolExecutor) -> List[int]:
    """The pool's worker processes. Read from the pool's own map of them: the executor lists them nowhere public."""
    return list((getattr(pool, "_processes", None) or {}).keys())


def _end_workers(pool: ProcessPoolExecutor) -> None:
    """End the pool's workers mid-record — a record building alone under the reserve would only take more. Python
    3.14 gives the pool ``terminate_workers()``; until then its own map of its processes."""
    for process in list((getattr(pool, "_processes", None) or {}).values()):
        process.terminate()


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
