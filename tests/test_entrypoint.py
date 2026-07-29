"""Tests for the runnable entry-point marker (entrypoint / runnable_entrypoints / run_entrypoint)."""

import pytest

from recordstream.runnable import entrypoint, entrypoint_tasks, run_entrypoint, runnable_entrypoints


class _Runnable:
    @entrypoint("fit", role="trainer")
    def fit(self) -> None: ...

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None: ...

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None: ...

    @entrypoint("predict", role="predictor")
    def predict(self) -> None: ...

    def not_an_entrypoint(self) -> None: ...


def test_runnable_entrypoints_finds_all_marked_methods() -> None:
    eps = runnable_entrypoints(_Runnable)
    assert set(eps) == {"fit", "evaluate", "test", "predict"}
    assert eps["fit"] == {"task": "fit", "role": "trainer", "primary": False}
    assert eps["test"]["primary"] is True
    assert "not_an_entrypoint" not in eps


def test_entrypoint_tasks_by_role_primary_first() -> None:
    assert entrypoint_tasks(_Runnable, "trainer") == ["fit"]
    # test is primary → sorts first among evaluators.
    assert entrypoint_tasks(_Runnable, "evaluator") == ["test", "evaluate"]
    assert entrypoint_tasks(_Runnable, "predictor") == ["predict"]
    assert entrypoint_tasks(_Runnable, "nonexistent") == []


def test_entrypoints_are_inherited_via_mro() -> None:
    class _Sub(_Runnable):
        @entrypoint("fit", role="trainer")
        def fit(self) -> None: ...  # override keeps the marker

    eps = runnable_entrypoints(_Sub)
    assert set(eps) == {"fit", "evaluate", "test", "predict"}


def test_marker_does_not_break_calling_the_method() -> None:
    calls = []

    class _R:
        @entrypoint("fit", role="trainer")
        def fit(self) -> None:
            calls.append("fit")

    _R().fit()
    assert calls == ["fit"]


# ---- run_entrypoint: the markers ARE the dispatch table ---------------------- #


class _Dispatching:
    """The merged train+eval shape: one ``task`` knob, ``run()`` dispatching off the markers."""

    def __init__(self, task: str = "fit") -> None:
        self.task = task
        self.calls: list = []

    def run(self) -> object:
        return run_entrypoint(self, self.task)

    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> str:
        self.calls.append("fit")
        return "fitted"

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.calls.append("evaluate")

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        self.calls.append("test")

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        self.calls.append("predict")


@pytest.mark.parametrize("task", ["fit", "evaluate", "test", "predict"])
def test_run_entrypoint_calls_the_method_declaring_the_task(task: str) -> None:
    runnable = _Dispatching(task=task)
    runnable.run()
    assert runnable.calls == [task]


def test_run_entrypoint_returns_the_method_result() -> None:
    assert _Dispatching(task="fit").run() == "fitted"


def test_run_entrypoint_rejects_an_unknown_task_listing_the_declared_ones() -> None:
    runnable = _Dispatching(task="export")
    with pytest.raises(ValueError) as excinfo:
        runnable.run()
    message = str(excinfo.value)
    assert "Unknown task 'export'" in message
    # Declaration order, so the list reads as the class's capability list.
    assert "['fit', 'evaluate', 'test', 'predict']" in message
    assert runnable.calls == []


def test_a_new_entrypoint_dispatches_with_no_other_change() -> None:
    """The regression a hand-written ``{task: method}`` dict allowed: marker added, dict forgotten."""

    class _WithExport(_Dispatching):
        @entrypoint("export", role="exporter", primary=True)
        def export(self) -> None:
            self.calls.append("export")

    # The config generator pins `task:` from the markers — dispatch must agree with it.
    assert entrypoint_tasks(_WithExport, "exporter") == ["export"]
    runnable = _WithExport(task="export")
    runnable.run()
    assert runnable.calls == ["export"]


def test_run_entrypoint_dispatches_to_a_subclass_override() -> None:
    class _Override(_Dispatching):
        @entrypoint("fit", role="trainer", primary=True)
        def fit(self) -> str:
            self.calls.append("fit-override")
            return "overridden"

    runnable = _Override(task="fit")
    assert runnable.run() == "overridden"
    assert runnable.calls == ["fit-override"]


def test_run_entrypoint_never_fires_a_property_getter() -> None:
    """The merged runnables carry a dynamic ``__torch_runner__`` property; lookup must not touch it."""

    class _WithProperty(_Dispatching):
        @property
        def __torch_runner__(self) -> bool:
            raise AssertionError("property getter fired during dispatch")

    runnable = _WithProperty(task="test")
    runnable.run()
    assert runnable.calls == ["test"]
