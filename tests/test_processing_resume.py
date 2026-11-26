"""``DatasetProcessor(start="resume")`` and ``DatasetProcessor.plan()``.

``start: resume`` asks the sink how many records it already holds — :meth:`resume_index`, counted from record 0 — and
builds only the rest, exactly as ``start: N`` would: a config run alone resumes, with no caller counting for it.
``plan()`` answers what a run would do without building a record: how many the source holds, where the run starts,
how many it builds, into which sink — and, when the source cannot be read, the reason, as a failed check.
"""

import json
from pathlib import Path
from typing import Any, Iterator, List

import pytest

from recordstream.core import Stream
from recordstream.items import Record
from recordstream.processing import DatasetProcessor
from recordstream.sources.files import FilesSource
from recordstream.workflow import Plan


class Numbered:
    """Record ``i`` is ``{"index": i}``; ``built`` lists the indices built."""

    def __init__(self, count: int = 6) -> None:
        self.count = count
        self.built: List[int] = []

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Record:
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.built.append(index)
        return {"index": index}

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]


class ResumableSink:
    """A sink that already holds ``held`` records and says so."""

    def __init__(self, held: int = 0) -> None:
        self.held = held
        self.records: List[Record] = []

    def resume_index(self) -> int:
        return self.held

    def write(self, record: Record) -> None:
        self.records.append(record)

    def flush(self) -> None:
        return None


class PlainSink:
    def __init__(self) -> None:
        self.records: List[Record] = []

    def write(self, record: Record) -> None:
        self.records.append(record)

    def flush(self) -> None:
        return None


def test_resume_builds_only_the_records_the_sink_does_not_hold() -> None:
    source, sink = Numbered(6), ResumableSink(held=4)
    DatasetProcessor(stream=Stream(source), sink=sink, start="resume").run()
    assert [record["index"] for record in sink.records] == [4, 5]
    assert source.built == [4, 5]


def test_resume_on_an_empty_sink_builds_every_record() -> None:
    source, sink = Numbered(3), ResumableSink(held=0)
    DatasetProcessor(stream=Stream(source), sink=sink, start="resume").run()
    assert [record["index"] for record in sink.records] == [0, 1, 2]


def test_resume_on_a_full_sink_builds_nothing() -> None:
    source, sink = Numbered(3), ResumableSink(held=3)
    DatasetProcessor(stream=Stream(source), sink=sink, start="resume").run()
    assert sink.records == [] and source.built == []


def test_resume_on_workers_gives_the_same_records_in_order() -> None:
    sink = ResumableSink(held=2)
    DatasetProcessor(stream=Stream(Numbered(5)), sink=sink, start="resume", workers=2).run()
    assert [record["index"] for record in sink.records] == [2, 3, 4]


def test_resume_with_a_sink_that_cannot_say_where_it_stopped_is_refused_before_a_record_is_built() -> None:
    source = Numbered(3)
    with pytest.raises(ValueError, match="start: resume .*PlainSink.* resume_index"):
        DatasetProcessor(stream=Stream(source), sink=PlainSink(), start="resume").run()
    assert source.built == []


def test_resume_without_a_sink_is_refused() -> None:
    with pytest.raises(ValueError, match="start: resume .*no sink"):
        DatasetProcessor(stream=Stream(Numbered(3)), start="resume").run()


def test_a_start_that_is_neither_a_number_nor_resume_is_refused_when_the_processor_is_built() -> None:
    with pytest.raises(ValueError, match="start"):
        DatasetProcessor(start="later")  # type: ignore[arg-type]


def test_the_plan_says_how_many_records_there_are_where_the_run_starts_and_how_many_it_builds() -> None:
    source = Numbered(6)
    plan = DatasetProcessor(stream=Stream(source), sink=ResumableSink(held=4), start="resume").plan()
    assert isinstance(plan, Plan)
    assert plan.details == {"records": 6, "start": 4, "to_build": 2, "sink": "ResumableSink"}
    assert plan.would_stop is False
    assert source.built == []  # planning builds nothing


def test_the_plan_of_a_fixed_start_and_of_no_sink() -> None:
    plan = DatasetProcessor(stream=Stream(Numbered(6)), start=5).plan()
    assert plan.details == {"records": 6, "start": 5, "to_build": 1, "sink": None}


def test_the_plan_of_a_folder_without_files_is_a_failed_check_naming_the_folder(tmp_path: Path) -> None:
    empty = tmp_path / "recordings"
    empty.mkdir()
    processor = DatasetProcessor(stream=Stream(FilesSource(root=str(empty), pattern="*.json")))
    plan = processor.plan()
    assert plan.would_stop is True
    (check,) = [check for check in plan.checks if not check.ok]
    assert str(empty) in check.detail
    assert check.detail.count("FilesSource") == 1  # the source's own message already names it


def test_the_plan_of_a_processor_without_a_stream_is_a_failed_check() -> None:
    plan = DatasetProcessor().plan()
    assert plan.would_stop is True
    assert "stream" in plan.checks[0].detail


def test_the_plan_is_json() -> None:
    plan = DatasetProcessor(stream=Stream(Numbered(2)), sink=ResumableSink(held=1), start="resume").plan()
    assert Plan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan


def test_a_resumed_record_reads_the_sink_when_the_run_starts_not_when_the_processor_is_built() -> None:
    """The workflow plans every step before the first runs, and an earlier step may empty the sink's folder."""
    sink = ResumableSink(held=5)
    processor: Any = DatasetProcessor(stream=Stream(Numbered(6)), sink=sink, start="resume")
    sink.held = 0
    processor.run()
    assert [record["index"] for record in sink.records] == list(range(6))
