"""``DatasetProcessor(workers=N)``: record ``i`` is ``source[i]`` run through the stream's ops in one of N spawn
workers, and the sink receives the records in index order — the records a sequential run gives, in its order.

The source and the ops are module-level so a spawn worker can import them. A source here builds record ``i`` from
``i`` alone (a generator drawing record ``i`` from ``(seed, i)`` is the case this exists for), so a worker builds it
exactly as the main process would.
"""

import os
import re
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import psutil
import pytest
from pydantic import ValidationError

from recordstream import processing
from recordstream.core import Stream
from recordstream.core.wrappers import FilterOp
from recordstream.items import Record
from recordstream.ops.parallel import Parallel
from recordstream.processing import DatasetProcessor, Workers
from recordstream.resources import MemoryReading, OutOfMemory, RecordCost

GB = 2**30


def _roomy(pids: Iterable[int] = ()) -> MemoryReading:
    return MemoryReading(available=100 * GB, total=128 * GB, processes=len(list(pids)))


@pytest.fixture(autouse=True)
def _a_machine_with_room(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runs here read a machine with room to spare: what this machine has free is not what they test. The tests
    of the memory guard read a machine of their own."""
    monkeypatch.setattr(processing, "read_memory", _roomy)


class IndexedRecords:
    """Record ``i``: values drawn from a generator seeded by ``i``, the process that built it; ``fail_at`` refuses."""

    def __init__(self, count: int = 12, fail_at: Optional[int] = None) -> None:
        self.count = count
        self.fail_at = fail_at

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Record:
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index == self.fail_at:
            raise ValueError(f"record {index} cannot be built")
        return {"index": index, "values": np.random.default_rng(index).standard_normal(256), "pid": os.getpid()}

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]


class OnlyIterable:
    """A source that can be iterated but not indexed."""

    def __iter__(self) -> Iterator[Record]:
        yield {"index": 0}


class Halves:
    """A 1 -> N op: a record becomes its two halves."""

    EXPANDS = True

    def __call__(self, record: Record) -> List[Record]:
        values = record["values"]
        return [dict(record, values=values[:128], half=0), dict(record, values=values[128:], half=1)]


def keep_not_a_multiple_of_three(record: Record) -> bool:
    return bool(record["index"] % 3 != 0)


class ListSink:
    """Keeps what it is given, in order."""

    def __init__(self) -> None:
        self.records: List[Record] = []
        self.flushed = False

    def write(self, record: Record) -> None:
        self.records.append(record)

    def flush(self) -> None:
        self.flushed = True


def _run(workers: Workers, source: Any = None, ops: Optional[List[Any]] = None, **kwargs: Any) -> ListSink:
    sink = ListSink()
    SINKS.append(sink)
    stream = Stream(source=IndexedRecords() if source is None else source, ops=ops)
    DatasetProcessor(stream=stream, sink=sink, workers=workers, **kwargs).run()
    return sink


def _seen(records: List[Record]) -> List[Tuple[Any, ...]]:
    """Each record as (index, half, values) — everything but the process that built it."""
    return [(r["index"], r.get("half"), r["values"].tobytes()) for r in records]


class TestTheSameRecordsInTheSameOrder:
    def test_workers_give_the_records_a_sequential_run_gives(self) -> None:
        ops = [FilterOp(keep_not_a_multiple_of_three), Halves()]
        sequential, parallel = _run(1, ops=ops), _run(3, ops=ops)
        assert len(sequential.records) == 2 * 8  # 12 records, 4 dropped, each of the rest in two halves
        assert _seen(parallel.records) == _seen(sequential.records)
        assert parallel.flushed

    def test_each_record_is_built_in_a_worker(self) -> None:
        sequential, parallel = _run(1), _run(2)
        assert {r["pid"] for r in sequential.records} == {os.getpid()}
        assert os.getpid() not in {r["pid"] for r in parallel.records}

    def test_more_workers_than_records(self) -> None:
        assert _seen(_run(8, source=IndexedRecords(count=3)).records) == _seen(
            _run(1, source=IndexedRecords(3)).records
        )

    def test_an_empty_source(self) -> None:
        assert _run(2, source=IndexedRecords(count=0)).records == []

    def test_without_a_sink_the_records_are_still_built(self) -> None:
        DatasetProcessor(stream=Stream(source=IndexedRecords(count=4)), workers=2).run()


class TestProgress:
    def test_every_record_is_reported_as_it_reaches_the_sink(self) -> None:
        reports: List[Tuple[float, Optional[float], str]] = []
        processor = DatasetProcessor(stream=Stream(source=IndexedRecords(count=5)), sink=ListSink(), workers=2)
        processor.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
        processor.run()
        assert reports == [(float(i), 5.0, "DatasetProcessor") for i in range(1, 6)]

    def test_the_console_bar_wraps_the_workers_records(self) -> None:
        sink = _run(2, source=IndexedRecords(count=4), show_progress=True)
        assert [r["index"] for r in sink.records] == [0, 1, 2, 3]


class _StepPool:
    """Stands in for the process pool, in this process: a task submitted stays BUILDING until the run waits, and each
    wait finishes the oldest building one — except the tasks of ``FINISH_AT_ONCE``, which finish when submitted.

    It counts, at each submit, the records handed out and not yet in the sink (``most_held``) and the records
    building (``most_building``)."""

    def __init__(
        self,
        max_workers: int,
        mp_context: Any = None,
        initializer: Optional[Callable[..., None]] = None,
        initargs: Tuple[Any, ...] = (),
    ) -> None:
        self.max_workers = max_workers
        self.building: List[Tuple[Future, Callable[[], Any]]] = []  # type: ignore[type-arg]
        self.submitted = 0
        self.most_held = 0
        self.most_building = 0
        self.building_at_first_wait: Optional[int] = None
        if initializer is not None:
            initializer(*initargs)
        POOLS.append(self)

    def submit(self, fn: Callable[..., Any], index: int) -> "Future[Any]":
        future: "Future[Any]" = Future()
        self.building.append((future, lambda: fn(index)))
        if index in FINISH_AT_ONCE:
            self._finish(len(self.building) - 1)
        self.submitted += 1
        self.most_held = max(self.most_held, self.submitted - len(SINKS[-1].records))
        self.most_building = max(self.most_building, len(self.building))
        return future

    def finish_oldest(self) -> None:
        if self.building_at_first_wait is None:
            self.building_at_first_wait = len(self.building)
        if self.building:
            self._finish(0)

    def _finish(self, position: int) -> None:
        future, build = self.building.pop(position)
        try:
            future.set_result(build())
        except Exception as error:  # the record's own error, as a worker hands it back
            future.set_exception(error)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


POOLS: List[_StepPool] = []
SINKS: List[ListSink] = []
FINISH_AT_ONCE: set = set()


def _wait_finishes_the_oldest(futures: Any, timeout: Optional[float] = None, return_when: Any = None) -> None:
    POOLS[-1].finish_oldest()


@pytest.fixture
def step_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(processing, "ProcessPoolExecutor", _StepPool)
    monkeypatch.setattr(processing, "wait", _wait_finishes_the_oldest)
    POOLS.clear()
    FINISH_AT_ONCE.clear()


def _costs(memory: float = 0.01, footprint: float = 0.01, cores: float = 1.0) -> Callable[[Callable[[], Any]], Any]:
    """Stands in for the worker's measurement: every record costs the same."""
    cost = RecordCost(memory=int(memory * GB), footprint=int(footprint * GB), cores=cores)
    return lambda build: (build(), cost)


def _machine(*available: float, total: float = 18) -> Callable[..., MemoryReading]:
    """A machine whose free memory is ``available[k]`` GB at the k-th reading, the last one from then on."""
    readings = list(available)

    def read(pids: Iterable[int] = ()) -> MemoryReading:
        free = readings.pop(0) if len(readings) > 1 else readings[0]
        return MemoryReading(available=int(free * GB), total=int(total * GB))

    return read


@pytest.mark.usefixtures("step_pool")
def test_the_run_holds_at_most_twice_as_many_records_as_workers() -> None:
    """A run of thousands of records must not hold them all: with record 0 slow and the rest done at once, records
    are handed out until twice the workers wait for the sink, and no further."""
    FINISH_AT_ONCE.update(range(1, 50))
    sink = _run(3, source=IndexedRecords(count=50))
    assert [r["index"] for r in sink.records] == list(range(50))
    assert POOLS[0].most_held == 2 * 3


@contextmanager
def _logged(level: str = "INFO") -> Iterator[List[str]]:
    """The loggair messages logged inside the block (loguru: ``caplog`` sees nothing; ``complete()`` flushes)."""
    from loguru import logger

    seen: List[str] = []
    sink_id = logger.add(lambda message: seen.append(message.record["message"]), level=level)
    try:
        yield seen
        logger.complete()
    finally:
        logger.remove(sink_id)


class TestAuto:
    def test_auto_gives_the_records_a_sequential_run_gives(self) -> None:
        ops = [FilterOp(keep_not_a_multiple_of_three), Halves()]
        assert _seen(_run("auto", ops=ops).records) == _seen(_run(1, ops=ops).records)

    def test_auto_is_a_value_and_anything_else_is_refused(self) -> None:
        assert DatasetProcessor(workers="auto").workers == "auto"
        with pytest.raises(ValidationError, match="auto"):
            DatasetProcessor(workers="many")  # type: ignore[arg-type]  # the refusal under test

    @pytest.mark.usefixtures("step_pool")
    def test_one_record_is_measured_then_as_many_as_the_memory_holds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """13 GB free − 2 GB reserve: records of 4 GB × 1.25 and their 0.5 GB workers — two at once."""
        monkeypatch.setattr(processing, "measure", _costs(memory=4, footprint=0.5))
        monkeypatch.setattr(processing, "read_memory", _machine(13))
        with _logged() as seen:
            sink = _run("auto", source=IndexedRecords(count=8))
        assert [r["index"] for r in sink.records] == list(range(8))
        assert (POOLS[0].building_at_first_wait, POOLS[0].most_building) == (1, 2)
        assert POOLS[0].max_workers == os.cpu_count()
        assert "Building 8 record(s), workers: auto — 1 worker builds record 0 to measure it." in seen
        assert any(
            line.startswith(
                "workers: auto → 2 (memory: 13.0 GB free − 2.0 GB reserve, 5.0 GB a record + 0.5 GB a worker; "
            )
            for line in seen
        )

    @pytest.mark.usefixtures("step_pool")
    def test_the_count_follows_the_free_memory_and_only_the_first_plan_is_news(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """13 GB free: two records at once; then 9 GB: one. The first plan is logged as news, the move as detail."""
        monkeypatch.setattr(processing, "measure", _costs(memory=4, footprint=0.5))
        monkeypatch.setattr(processing, "read_memory", _machine(13, 13, 9))
        with _logged("DEBUG") as seen:
            sink = _run("auto", source=IndexedRecords(count=8))
        assert [r["index"] for r in sink.records] == list(range(8))
        plans = [line.split(" (")[0] for line in seen if line.startswith("workers:")]
        assert plans == ["workers: auto → 2", "workers: auto → 1"]

    @pytest.mark.usefixtures("step_pool")
    def test_a_record_too_large_for_the_free_memory_builds_alone_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(processing, "measure", _costs(memory=4, footprint=0.5))
        monkeypatch.setattr(processing, "read_memory", _machine(3.8))
        with _logged("WARNING") as seen:
            sink = _run("auto", source=IndexedRecords(count=4))
        assert len(sink.records) == 4
        assert POOLS[0].most_building == 1
        assert [line for line in seen if line.startswith("workers: auto → 1 (memory: 3.8 GB free")] != []
        assert all(
            line.endswith("— not even one record fits the free memory; building one at a time")
            for line in seen
            if line.startswith("workers:")
        )


class TestTheMemoryGuard:
    @pytest.mark.usefixtures("step_pool")
    def test_a_fixed_count_is_lowered_to_what_the_memory_holds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(processing, "measure", _costs(memory=4, footprint=0.5))
        monkeypatch.setattr(processing, "read_memory", _machine(13))
        with _logged() as seen:
            sink = _run(8, source=IndexedRecords(count=12))
        assert [r["index"] for r in sink.records] == list(range(12))
        assert POOLS[0].max_workers == 8
        assert POOLS[0].most_building == 8  # the first eight start before any record is measured
        assert [line for line in seen if line.startswith("workers: 8 → 2 (memory: 13.0 GB free")] != []

    @pytest.mark.usefixtures("step_pool")
    def test_no_record_starts_under_the_reserve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three records building; record 0 done, then 1.5 GB free, under the 2 GB reserve: record 3 waits until
        memory is free again."""
        monkeypatch.setattr(processing, "measure", _costs())
        monkeypatch.setattr(processing, "read_memory", _machine(13, 1.5, 13))
        with _logged("WARNING") as seen:
            sink = _run(3, source=IndexedRecords(count=6))
        assert [r["index"] for r in sink.records] == list(range(6))
        assert seen == [
            "Memory: 1.5 GB free, under the 2.0 GB reserve — holding record 3 until a worker finishes (2 building)."
        ]

    @pytest.mark.usefixtures("step_pool")
    def test_under_the_reserve_with_one_record_building_the_run_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(processing, "measure", _costs(memory=4))
        monkeypatch.setattr(processing, "read_memory", _machine(13, 1.5))
        message = (
            "DatasetProcessor: the machine has 1.5 GB free, under the 2.0 GB reserve, while record 1 builds alone "
            "(records measured so far peak at 4.0 GB) — stopped before the machine swaps; free memory, or build "
            "smaller records"
        )
        with pytest.raises(OutOfMemory, match=f"^{re.escape(message)}$"):
            _run(2, source=IndexedRecords(count=4))

    @pytest.mark.usefixtures("step_pool")
    def test_under_the_reserve_before_the_first_record_the_run_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(processing, "read_memory", _machine(1.5))
        with pytest.raises(OutOfMemory, match=r"before record 0 starts \(no record measured\)"):
            _run(2, source=IndexedRecords(count=4))

    @pytest.mark.usefixtures("step_pool")
    def test_a_record_that_fails_stops_the_run_in_index_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(processing, "measure", _costs())
        FINISH_AT_ONCE.update({2})
        with pytest.raises(ValueError, match="^record 2 cannot be built$"):
            _run(3, source=IndexedRecords(count=6, fail_at=2))
        assert [r["index"] for r in SINKS[-1].records] == [0, 1]

    def test_a_stopped_run_ends_the_worker_building_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A real worker, on a record that would take a minute: the run stops on the second reading and the worker
        is gone — within a deadline, so a regression fails here instead of hanging."""
        seen_pids: List[int] = []
        readings = iter([100.0])

        def read(pids: Iterable[int] = ()) -> MemoryReading:
            seen_pids.extend(pids)
            return MemoryReading(available=int(next(readings, 1.5) * GB), total=int(18 * GB))

        monkeypatch.setattr(processing, "read_memory", read)
        errors: List[BaseException] = []

        def run() -> None:
            try:
                _run("auto", source=SlowRecords())
            except BaseException as error:  # handed to the test thread
                errors.append(error)

        started = time.monotonic()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=60)
        assert not thread.is_alive(), "the run did not stop within 60 s"
        assert time.monotonic() - started < 30  # the record would have taken 60 s
        assert [type(error) for error in errors] == [OutOfMemory]
        assert "while record 0 builds alone" in str(errors[0])
        assert seen_pids and not any(psutil.pid_exists(pid) for pid in set(seen_pids))


class SlowRecords:
    """One record that takes a minute to build."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Record:
        time.sleep(60)
        return {"index": index}


class TestRefusals:
    def test_a_worker_error_stops_the_run_with_that_error(self) -> None:
        with pytest.raises(ValueError, match="^record 5 cannot be built$"):
            _run(2, source=IndexedRecords(count=12, fail_at=5))

    def test_a_source_that_cannot_be_indexed_is_refused(self) -> None:
        message = (
            "DatasetProcessor: workers=3 builds each record in a worker from its index, and the source (OnlyIterable) "
            "cannot be indexed — give it len() and [i], or set workers: 1"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _run(3, source=OnlyIterable())

    def test_a_stream_with_a_parallel_op_is_refused(self) -> None:
        message = (
            "DatasetProcessor: workers=2, and op 0 of the stream (Parallel) starts workers of its own — a worker "
            "cannot start workers; set workers: 1 or take the Parallel out"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _run(2, ops=[Parallel(ops=[Halves()], workers=2)])

    def test_a_stream_that_batches_records_is_refused(self) -> None:
        sink = ListSink()
        stream = Stream(source=IndexedRecords(), chunk_size=4)
        message = "DatasetProcessor: workers=2 and a stream that batches its records (chunk_size 4) — set workers: 1"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            DatasetProcessor(stream=stream, sink=sink, workers=2).run()

    def test_fewer_than_one_worker_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):  # the range mark, at construction
            DatasetProcessor(workers=0)
        processor = DatasetProcessor(stream=Stream(source=IndexedRecords()), sink=ListSink())
        processor.workers = 0  # set after construction: the run refuses it too
        with pytest.raises(ValueError, match="^DatasetProcessor: workers must be >= 1; got 0$"):
            processor.run()
