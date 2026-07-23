"""Tests for the `sampleflux run` CLI dispatch (sampleflux.cli.run)."""

from typing import List

from confluid import Class

from sampleflux.cli import run


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
    run(Class(R, tag="x"))
    assert log == ["x"]


def test_run_none_is_noop() -> None:
    run(None)  # must not raise


def test_run_non_runnable_is_handled() -> None:
    run(object())  # no .run() -> logged + returns, never raises
