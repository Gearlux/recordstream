"""A run that is ASKED to stop — ``ProgressReporting.set_stop_request``.

Whoever runs a runnable in the background (a job with a Stop button) hands it a question, "stop now?". A processor
asks it after each record; on yes it leaves its loop, flushes its sink as at a normal end and returns: what was written
is kept. A workflow hands the question down to the step it runs and starts no further step. A run handed no question —
a plain ``recordstream run`` — runs to its end, as before.

Measured before: the only hook was the progress callback, whose errors the processor swallows (a broken progress bar
must never abort a run), so a run could not be stopped between records at all.
"""

from typing import Any, Callable, Iterator, List

from recordstream.core import Stream
from recordstream.items import Record
from recordstream.processing import DatasetProcessor
from recordstream.workflow import Sequence


class Numbers:
    """Record ``i`` holds ``i``; module-level, so a spawn worker can import it."""

    def __init__(self, count: int = 6) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Record:
        if not 0 <= index < self.count:
            raise IndexError(index)
        return {"index": index}

    def __iter__(self) -> Iterator[Record]:
        return (self[index] for index in range(len(self)))


class Kept:
    """A sink that keeps what it is written and counts its flushes."""

    def __init__(self) -> None:
        self.written: List[int] = []
        self.flushes = 0

    def write(self, record: Record) -> None:
        self.written.append(int(record["index"]))

    def flush(self) -> None:
        self.flushes += 1


def _after(count: int, sink: Kept) -> Callable[[], bool]:
    """The question that says yes once ``sink`` holds ``count`` records."""
    return lambda: len(sink.written) >= count


class TestAProcessorAskedToStop:
    def test_it_stops_after_the_record_in_hand_and_flushes_the_sink(self) -> None:
        sink = Kept()
        processor = DatasetProcessor(stream=Stream(source=Numbers()), sink=sink)
        processor.set_stop_request(_after(2, sink))
        processor.run()
        assert sink.written == [0, 1]
        assert sink.flushes == 1  # ended as a run that reached its end: what was written is kept

    def test_it_reports_the_last_record_before_it_stops(self) -> None:
        sink, seen = Kept(), []
        processor = DatasetProcessor(stream=Stream(source=Numbers()), sink=sink)
        processor.set_progress_callback(lambda value, total, desc: seen.append((value, total)))
        processor.set_stop_request(_after(3, sink))
        processor.run()
        assert seen == [(1.0, 6.0), (2.0, 6.0), (3.0, 6.0)]

    def test_a_run_handed_no_question_runs_to_its_end(self) -> None:
        sink = Kept()
        DatasetProcessor(stream=Stream(source=Numbers()), sink=sink).run()
        assert sink.written == [0, 1, 2, 3, 4, 5]

    def test_a_question_that_says_no_changes_nothing(self) -> None:
        sink = Kept()
        processor = DatasetProcessor(stream=Stream(source=Numbers()), sink=sink)
        processor.set_stop_request(lambda: False)
        processor.run()
        assert sink.written == [0, 1, 2, 3, 4, 5]

    def test_a_question_that_raises_never_stops_or_breaks_the_run(self) -> None:
        def broken() -> bool:
            raise RuntimeError("the page went away")

        sink = Kept()
        processor = DatasetProcessor(stream=Stream(source=Numbers()), sink=sink)
        processor.set_stop_request(broken)
        processor.run()
        assert sink.written == [0, 1, 2, 3, 4, 5]

    def test_a_question_cleared_with_none_is_no_question(self) -> None:
        sink = Kept()
        processor = DatasetProcessor(stream=Stream(source=Numbers()), sink=sink)
        processor.set_stop_request(lambda: True)
        processor.set_stop_request(None)
        processor.run()
        assert sink.written == [0, 1, 2, 3, 4, 5]

    def test_it_stops_a_run_without_a_sink(self) -> None:
        asked: List[int] = []

        def third() -> bool:
            asked.append(1)
            return len(asked) >= 3

        processor = DatasetProcessor(stream=Stream(source=Numbers()))
        processor.set_stop_request(third)
        processor.run()
        assert len(asked) == 3  # asked once a record, and no fourth record was built

    def test_it_stops_a_run_on_workers_and_ends_them(self) -> None:
        sink = Kept()
        processor = DatasetProcessor(stream=Stream(source=Numbers(count=12)), sink=sink, workers=2)
        processor.set_stop_request(_after(2, sink))
        processor.run()
        assert sink.written == [0, 1]
        assert sink.flushes == 1


class TestAWorkflowAskedToStop:
    def test_the_question_reaches_the_step_and_no_further_step_starts(self) -> None:
        first, second = Kept(), Kept()
        steps: List[Any] = [
            DatasetProcessor(stream=Stream(source=Numbers()), sink=first),
            DatasetProcessor(stream=Stream(source=Numbers()), sink=second),
        ]
        workflow = Sequence(steps=steps)
        workflow.set_stop_request(_after(2, first))
        workflow.run()
        assert first.written == [0, 1]
        assert second.written == []

    def test_a_workflow_handed_no_question_runs_every_step(self) -> None:
        first, second = Kept(), Kept()
        Sequence(
            steps=[
                DatasetProcessor(stream=Stream(source=Numbers(count=2)), sink=first),
                DatasetProcessor(stream=Stream(source=Numbers(count=2)), sink=second),
            ]
        ).run()
        assert (first.written, second.written) == ([0, 1], [0, 1])
