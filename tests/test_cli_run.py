"""Tests for the `recordstream run` CLI dispatch (recordstream.cli.run)."""

from typing import List

import pytest
from confluid import Target

from recordstream.cli import run
from recordstream.workflow import RequirementNotMet, StepFailed


def test_run_calls_runnable_run() -> None:
    calls: List[str] = []

    class R:
        def run(self) -> None:
            calls.append("ran")

    run(R())
    assert calls == ["ran"]


def test_run_flows_deferred_marker() -> None:
    from confluid import configurable

    log: List[str] = []

    @configurable
    class R:
        def __init__(self, tag: str = "") -> None:
            self.tag = tag

        def run(self) -> None:
            log.append(self.tag)

    # A deferred Confluid marker is flowed before run() is called.
    run(Target(R, tag="x"))
    assert log == ["x"]


def test_run_none_is_noop() -> None:
    run(None)  # must not raise


def test_run_non_runnable_is_handled() -> None:
    run(object())  # no .run() -> logged + returns, never raises


def test_a_failed_step_ends_the_command_with_the_steps_exit_status() -> None:
    """A workflow step whose process ended with status 3 ends ``recordstream run`` with 3, after one Error line."""

    class Failing:
        def run(self) -> None:
            raise StepFailed("convert: recordstream run convert.yaml ended with exit 3", 3)

    with pytest.raises(SystemExit) as ended:
        run(Failing())
    assert ended.value.code == 3


def test_a_refusal_ends_the_command_with_exit_1() -> None:
    class Refusing:
        def run(self) -> None:
            raise RequirementNotMet("source project.bashrc")

    with pytest.raises(SystemExit) as ended:
        run(Refusing())
    assert ended.value.code == 1
