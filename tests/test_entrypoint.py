"""Tests for the runnable entry-point marker (entrypoint / runnable_entrypoints)."""

from recordstream.runnable import entrypoint, entrypoint_tasks, runnable_entrypoints


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
