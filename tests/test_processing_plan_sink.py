"""What a run would CHANGE on disk is part of its plan — a sink's ``planned_changes()``.

A sink that replaces what it holds when a run starts (a dataset folder's split) says so before the run: the
processor's plan asks its sink and lists the answers as the plan's actions, so ``recordstream run --dry_run`` — and
whoever shows that plan before starting a run — names every folder a run deletes. A sink with nothing to say changes
nothing of the plan.

Measured before: a sink deleted its split's pictures and labels when a run opened it, and the plan read ``records:
807, start: 0, to_build: 807, sink: YoloSink`` — no word of it.
"""

from typing import Iterator, List

from recordstream.core import Stream
from recordstream.items import Record
from recordstream.processing import DatasetProcessor
from recordstream.workflow import plan_lines


class Three:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> Record:
        return {"index": index}

    def __iter__(self) -> Iterator[Record]:
        return (self[index] for index in range(3))


class Plain:
    """A sink that says nothing about what it changes."""

    def write(self, record: Record) -> None:
        pass

    def flush(self) -> None:
        pass


class Replacing(Plain):
    """A sink that deletes two folders when a run opens it."""

    def __init__(self) -> None:
        self.asked = 0

    def planned_changes(self) -> List[str]:
        self.asked += 1
        return ["delete /data/train/images (807 files, 1.2G)", "delete /data/train/labels (807 files, 310.0K)"]


class Unsure(Plain):
    def planned_changes(self) -> List[str]:
        raise OSError("the folder cannot be listed")


def _plan(sink: Plain) -> DatasetProcessor:
    return DatasetProcessor(stream=Stream(source=Three()), sink=sink)


class TestWhatARunWouldChange:
    def test_the_sinks_changes_are_the_plans_actions(self) -> None:
        sink = Replacing()
        plan = _plan(sink).plan()
        assert plan.actions == [
            "delete /data/train/images (807 files, 1.2G)",
            "delete /data/train/labels (807 files, 310.0K)",
        ]
        assert not plan.would_stop
        assert sink.asked == 1

    def test_the_dry_run_prints_them_before_the_numbers(self) -> None:
        assert plan_lines(_plan(Replacing()).plan()) == [
            "DatasetProcessor",
            "      ok: the source can be read",
            "      delete /data/train/images (807 files, 1.2G)",
            "      delete /data/train/labels (807 files, 310.0K)",
            "      records: 3, start: 0, to_build: 3, sink: Replacing",
        ]

    def test_a_sink_that_says_nothing_plans_as_before(self) -> None:
        plan = _plan(Plain()).plan()
        assert plan.actions == []
        assert plan_lines(plan) == [
            "DatasetProcessor",
            "      ok: the source can be read",
            "      records: 3, start: 0, to_build: 3, sink: Plain",
        ]

    def test_a_run_without_a_sink_plans_no_change(self) -> None:
        assert DatasetProcessor(stream=Stream(source=Three())).plan().actions == []

    def test_a_sink_that_cannot_say_is_a_failed_check_never_a_crash(self) -> None:
        plan = _plan(Unsure()).plan()
        assert plan.would_stop
        assert plan.first_failure == "Unsure: the folder cannot be listed"

    def test_the_run_itself_never_asks(self) -> None:
        sink = Replacing()
        _plan(sink).run()
        assert sink.asked == 0
