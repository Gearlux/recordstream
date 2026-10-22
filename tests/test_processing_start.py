"""``DatasetProcessor(start=N)``: a run resumed where its sink stopped builds only ``source[N:]`` — the records a whole
run gives at those indices, in their order, one at a time or on workers — and never the records before ``N``.

The source is module-level so a spawn worker can import it, and builds record ``i`` from ``i`` alone.
"""

from typing import Any, Iterator, List

import numpy as np
import pytest

from recordstream.core import Stream
from recordstream.items import Record
from recordstream.processing import DatasetProcessor


class CountedRecords:
    """Record ``i``: values drawn from a generator seeded by ``i``; ``built`` lists the indices built here."""

    def __init__(self, count: int = 8) -> None:
        self.count = count
        self.built: List[int] = []

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Record:
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.built.append(index)
        return {"index": index, "values": np.random.default_rng(index).standard_normal(64)}

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]


class OnlyIterable:
    def __iter__(self) -> Iterator[Record]:
        yield {"index": 0}


class ListSink:
    def __init__(self) -> None:
        self.records: List[Record] = []

    def write(self, record: Record) -> None:
        self.records.append(record)

    def flush(self) -> None:
        return None


def _run(source: Any, **settings: Any) -> List[Record]:
    sink = ListSink()
    DatasetProcessor(stream=Stream(source), sink=sink, **settings).run()
    return sink.records


def test_start_builds_only_the_records_from_it_on() -> None:
    source = CountedRecords(8)
    records = _run(source, start=5)
    assert [record["index"] for record in records] == [5, 6, 7]
    assert source.built == [5, 6, 7]


def test_a_resumed_record_is_the_one_the_whole_run_gave() -> None:
    whole = _run(CountedRecords(8))
    resumed = _run(CountedRecords(8), start=5)
    for mine, theirs in zip(resumed, whole[5:]):
        assert mine["index"] == theirs["index"]
        assert np.array_equal(mine["values"], theirs["values"])


def test_start_on_workers_gives_the_same_records_in_order() -> None:
    sequential = _run(CountedRecords(8), start=3)
    parallel = _run(CountedRecords(8), start=3, workers=2)
    assert [record["index"] for record in parallel] == [3, 4, 5, 6, 7]
    for mine, theirs in zip(parallel, sequential):
        assert np.array_equal(mine["values"], theirs["values"])


def test_start_zero_is_the_whole_run_and_start_past_the_end_builds_nothing() -> None:
    assert [record["index"] for record in _run(CountedRecords(4), start=0)] == [0, 1, 2, 3]
    assert _run(CountedRecords(4), start=4) == []
    assert _run(CountedRecords(4), start=9) == []


def test_a_source_that_cannot_be_indexed_is_refused_when_start_is_set() -> None:
    with pytest.raises(ValueError, match="start=2 .*OnlyIterable.* cannot be indexed"):
        _run(OnlyIterable(), start=2)
    assert _run(OnlyIterable()) == [{"index": 0}]


def test_a_negative_start_is_refused_when_the_processor_is_built() -> None:
    with pytest.raises(ValueError, match="start"):
        DatasetProcessor(start=-1)


def test_the_processors_stream_keeps_its_source_after_a_run() -> None:
    source = CountedRecords(4)
    processor = DatasetProcessor(stream=Stream(source), sink=ListSink(), start=2)
    processor.run()
    assert processor.stream is not None and processor.stream.source is source
