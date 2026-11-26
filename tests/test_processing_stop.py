"""``DatasetProcessor(stop=N)``: a run builds only ``source[start:N]`` — the records a whole run gives at those indices,
in their order, one at a time or on workers — and never a record at ``N`` or after it. ``None`` (the default) builds to
the source's end, as before.

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


class ResumableSink:
    """A sink holding ``held`` records already: ``start: resume`` starts after them."""

    def __init__(self, held: int = 0) -> None:
        self.held = held
        self.records: List[Record] = []

    def write(self, record: Record) -> None:
        self.records.append(record)

    def flush(self) -> None:
        return None

    def resume_index(self) -> int:
        return self.held


def _run(source: Any, sink: Any = None, **settings: Any) -> List[Record]:
    sink = ResumableSink() if sink is None else sink
    DatasetProcessor(stream=Stream(source), sink=sink, **settings).run()
    return list(sink.records)


def _indices(records: List[Record]) -> List[int]:
    return [record["index"] for record in records]


def test_stop_builds_only_the_records_before_it() -> None:
    source = CountedRecords(8)
    assert _indices(_run(source, stop=3)) == [0, 1, 2]
    assert source.built == [0, 1, 2]


def test_start_and_stop_together_build_the_slice_between_them() -> None:
    source = CountedRecords(8)
    assert _indices(_run(source, start=2, stop=5)) == [2, 3, 4]
    assert source.built == [2, 3, 4]


def test_a_stopped_record_is_the_one_the_whole_run_gave() -> None:
    whole = _run(CountedRecords(8))
    stopped = _run(CountedRecords(8), stop=4)
    for mine, theirs in zip(stopped, whole[:4]):
        assert mine["index"] == theirs["index"] and np.array_equal(mine["values"], theirs["values"])


def test_stop_on_workers_gives_the_same_records_in_order() -> None:
    sequential = _run(CountedRecords(8), start=1, stop=6)
    parallel = _run(CountedRecords(8), start=1, stop=6, workers=2)
    assert _indices(parallel) == [1, 2, 3, 4, 5]
    for mine, theirs in zip(parallel, sequential):
        assert np.array_equal(mine["values"], theirs["values"])


def test_stop_past_the_end_builds_to_the_end_and_none_is_the_whole_run() -> None:
    assert _indices(_run(CountedRecords(4), stop=9)) == [0, 1, 2, 3]
    assert _indices(_run(CountedRecords(4), stop=None)) == [0, 1, 2, 3]


def test_stop_at_or_before_start_builds_nothing() -> None:
    assert _run(CountedRecords(8), start=5, stop=5) == []
    assert _run(CountedRecords(8), start=5, stop=2) == []
    assert _run(CountedRecords(8), stop=0) == []


def test_a_resumed_run_stops_at_the_same_index() -> None:
    assert _indices(_run(CountedRecords(8), sink=ResumableSink(held=3), start="resume", stop=5)) == [3, 4]
    assert _run(CountedRecords(8), sink=ResumableSink(held=6), start="resume", stop=5) == []


def test_a_source_that_cannot_be_indexed_is_refused_when_stop_is_set() -> None:
    with pytest.raises(ValueError, match="stop=2 .*OnlyIterable.* cannot be indexed"):
        _run(OnlyIterable(), stop=2)


def test_a_negative_stop_is_refused_when_the_processor_is_built() -> None:
    with pytest.raises(ValueError, match="stop"):
        DatasetProcessor(stop=-1)


def test_the_plan_counts_only_the_records_before_stop() -> None:
    processor = DatasetProcessor(stream=Stream(CountedRecords(8)), sink=ResumableSink(held=2), start="resume", stop=5)
    details = processor.plan().details
    assert (details["records"], details["start"], details["stop"], details["to_build"]) == (8, 2, 5, 3)
    whole = DatasetProcessor(stream=Stream(CountedRecords(8))).plan().details
    assert "stop" not in whole and whole["to_build"] == 8
