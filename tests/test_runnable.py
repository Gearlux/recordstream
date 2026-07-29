"""Tests for the runnable protocol markers (TorchRunner + ProgressReporting)."""

from typing import List

from recordstream.runnable import ProgressReporting, TorchRunner


def test_torch_runner_sets_the_needs_autograd_flag() -> None:
    # The class is named for the framework; the FLAG is named for what it decides — a GUI
    # executor reads `__needs_autograd__` (duck-typed) to re-enable autograd around run().
    assert TorchRunner.__needs_autograd__ is True

    class Trainer(TorchRunner):
        pass

    assert Trainer().__needs_autograd__ is True  # inherited by subclasses


def test_progress_reporting_noop_without_callback() -> None:
    class R(ProgressReporting):
        pass

    r = R()
    # No callback set -> _report_progress is a silent no-op (never raises).
    r._report_progress(1, 10, "x")
    assert r._progress_callback is None


def test_progress_reporting_fires_when_set() -> None:
    reports: List[tuple] = []

    class R(ProgressReporting):
        pass

    r = R()
    r.set_progress_callback(lambda v, t, d: reports.append((v, t, d)))
    r._report_progress(3, 10, "step")
    assert reports == [(3.0, 10.0, "step")]


def test_progress_reporting_skips_nonpositive_total() -> None:
    reports: List[tuple] = []

    class R(ProgressReporting):
        pass

    r = R()
    r.set_progress_callback(lambda v, t, d: reports.append((v, t, d)))
    r._report_progress(1, None, "x")  # unknown total -> skipped
    r._report_progress(1, 0, "x")  # non-positive -> skipped
    assert reports == []


def test_progress_reporting_swallows_callback_errors() -> None:
    def boom(v: float, t: float, d: str) -> None:
        raise RuntimeError("sink broke")

    class R(ProgressReporting):
        pass

    r = R()
    r.set_progress_callback(boom)
    r._report_progress(1, 10, "x")  # must not raise — progress is cosmetic


def test_set_progress_callback_clears_with_none() -> None:
    class R(ProgressReporting):
        pass

    r = R()
    r.set_progress_callback(lambda v, t, d: None)
    r.set_progress_callback(None)
    assert r._progress_callback is None
