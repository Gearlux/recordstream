"""``DatasetProcessor(workers=N)``: record ``i`` is ``source[i]`` run through the stream's ops in one of N spawn
workers, and the sink receives the records in index order — the records a sequential run gives, in its order.

The source and the ops are module-level so a spawn worker can import them. A source here builds record ``i`` from
``i`` alone (a generator drawing record ``i`` from ``(seed, i)`` is the case this exists for), so a worker builds it
exactly as the main process would.
"""

import os
import re
from concurrent.futures import Future
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from recordstream import processing
from recordstream.core import Stream
from recordstream.core.wrappers import FilterOp
from recordstream.items import Record
from recordstream.ops.parallel import Parallel
from recordstream.processing import DatasetProcessor


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


def _run(workers: int, source: Any = None, ops: Optional[List[Any]] = None, **kwargs: Any) -> ListSink:
    sink = ListSink()
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


class _CountingPool:
    """Stands in for the process pool: runs each task in this process when it is submitted, and counts the tasks
    submitted whose result has not been taken yet — the records the run holds at once."""

    def __init__(
        self,
        max_workers: int,
        mp_context: Any = None,
        initializer: Optional[Callable[..., None]] = None,
        initargs: Tuple[Any, ...] = (),
    ) -> None:
        self.waiting = 0
        self.most = 0
        if initializer is not None:
            initializer(*initargs)
        POOLS.append(self)

    def submit(self, fn: Callable[..., Any], *args: Any) -> "Future[Any]":
        pool = self
        pool.waiting += 1
        pool.most = max(pool.most, pool.waiting)

        class Taken(Future):  # type: ignore[type-arg]
            def result(self, timeout: Optional[float] = None) -> Any:
                pool.waiting -= 1
                return super().result(timeout)

        future: "Future[Any]" = Taken()
        future.set_result(fn(*args))
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


POOLS: List[_CountingPool] = []


def test_the_run_holds_at_most_twice_as_many_records_as_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run of thousands of records must not hold them all: the next index is handed out only once the oldest
    record has gone to the sink."""
    monkeypatch.setattr(processing, "ProcessPoolExecutor", _CountingPool)
    POOLS.clear()
    sink = _run(3, source=IndexedRecords(count=50))
    assert [r["index"] for r in sink.records] == list(range(50))
    assert POOLS[0].most == 2 * 3


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
