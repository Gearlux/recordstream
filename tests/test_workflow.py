"""Tests for recordstream.workflow — higher-order runnable combinators.

Covers Sequence / Conditional / Switch orchestration, the PathExists / Not /
AllOf / AnyOf predicates, zero-arg construction (the lazy-construction mandate),
the only-the-selected-branch-runs guarantee, error paths, and a Confluid
dump/load round-trip that proves a whole workflow serialises and runs (the
deferred branches are flowed inside run()).
"""

from typing import Any, Iterator, List

import confluid
import pytest
from confluid import configurable

from recordstream.workflow import AllOf, AnyOf, Conditional, Not, PathExists, Sequence, Switch

# A module-global run log so the @configurable runnables below survive a Confluid
# round-trip: their only config is a tag, and run() appends it here.
_RUN_LOG: List[str] = []


@configurable
class _RunStep:
    """A minimal runnable: ``run()`` records its tag in the module log."""

    def __init__(self, tag: str = "") -> None:
        self.tag = tag

    def run(self) -> None:
        _RUN_LOG.append(self.tag)


@configurable
class _FixedPredicate:
    """A predicate whose truth value is pinned in config (round-trippable)."""

    def __init__(self, value: bool = False) -> None:
        self.value = value

    def __call__(self) -> bool:
        return self.value


@pytest.fixture(autouse=True)
def _clear_log() -> Iterator[None]:
    _RUN_LOG.clear()
    yield
    _RUN_LOG.clear()


# --------------------------------------------------------------------------- #
# Sequence
# --------------------------------------------------------------------------- #
def test_sequence_runs_steps_in_order() -> None:
    Sequence([_RunStep("a"), _RunStep("b"), _RunStep("c")]).run()
    assert _RUN_LOG == ["a", "b", "c"]


def test_sequence_empty_is_noop() -> None:
    Sequence().run()  # zero-arg + empty: must not raise
    assert _RUN_LOG == []


def test_sequence_skips_none_entries() -> None:
    Sequence([_RunStep("a"), None, _RunStep("b")]).run()
    assert _RUN_LOG == ["a", "b"]


def test_sequence_caches_flowed_step() -> None:
    seq = Sequence([confluid.Target(_RunStep, tag="x")])
    seq.run()
    # After run, the deferred marker has been replaced by the live, flowed object.
    assert isinstance(seq.steps[0], _RunStep)
    assert _RUN_LOG == ["x"]


# --------------------------------------------------------------------------- #
# Conditional
# --------------------------------------------------------------------------- #
def test_conditional_true_runs_if_true_only() -> None:
    Conditional(condition=True, if_true=_RunStep("t"), if_false=_RunStep("f")).run()
    assert _RUN_LOG == ["t"]


def test_conditional_false_runs_if_false_only() -> None:
    Conditional(condition=False, if_true=_RunStep("t"), if_false=_RunStep("f")).run()
    assert _RUN_LOG == ["f"]


def test_conditional_none_branch_is_noop() -> None:
    Conditional(condition=True, if_true=None, if_false=_RunStep("f")).run()
    assert _RUN_LOG == []


def test_conditional_none_condition_is_falsy() -> None:
    Conditional(if_true=_RunStep("t"), if_false=_RunStep("f")).run()
    assert _RUN_LOG == ["f"]


def test_conditional_accepts_callable_condition() -> None:
    Conditional(condition=lambda: True, if_true=_RunStep("t")).run()
    assert _RUN_LOG == ["t"]


def test_conditional_accepts_predicate_condition() -> None:
    Conditional(condition=_FixedPredicate(True), if_true=_RunStep("t"), if_false=_RunStep("f")).run()
    assert _RUN_LOG == ["t"]


# --------------------------------------------------------------------------- #
# Switch
# --------------------------------------------------------------------------- #
def test_switch_selects_matching_case() -> None:
    Switch(select=lambda: "b", cases={"a": _RunStep("a"), "b": _RunStep("b")}).run()
    assert _RUN_LOG == ["b"]


def test_switch_falls_back_to_default_on_miss() -> None:
    Switch(select=lambda: "z", cases={"a": _RunStep("a")}, default=_RunStep("d")).run()
    assert _RUN_LOG == ["d"]


def test_switch_none_selector_uses_default() -> None:
    Switch(default=_RunStep("d")).run()
    assert _RUN_LOG == ["d"]


def test_switch_no_match_no_default_is_noop() -> None:
    Switch(select=lambda: "z", cases={"a": _RunStep("a")}).run()
    assert _RUN_LOG == []


def test_switch_coerces_non_string_key() -> None:
    Switch(select=lambda: 2, cases={"2": _RunStep("two")}).run()
    assert _RUN_LOG == ["two"]


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
def test_path_exists(tmp_path: Any) -> None:
    present = tmp_path / "model.ckpt"
    present.write_text("x")
    assert PathExists(str(present))() is True
    assert PathExists(str(tmp_path / "missing.ckpt"))() is False
    assert PathExists("")() is False  # empty path never True (no bogus Path('.') hit)


def test_not_negates() -> None:
    assert Not(True)() is False
    assert Not(False)() is True
    assert Not(None)() is True  # None is falsy -> Not(None) is True


def test_allof_is_and_with_vacuous_truth() -> None:
    assert AllOf([True, True])() is True
    assert AllOf([True, False])() is False
    assert AllOf()() is True  # empty AND is True


def test_anyof_is_or() -> None:
    assert AnyOf([False, True])() is True
    assert AnyOf([False, False])() is False
    assert AnyOf()() is False  # empty OR is False


def test_predicate_combinators_compose() -> None:
    # Not(AllOf(True, AnyOf(False, True))) == not (True and True) == False
    assert Not(AllOf([True, AnyOf([False, True])]))() is False


# --------------------------------------------------------------------------- #
# Error paths + zero-arg construction
# --------------------------------------------------------------------------- #
def test_run_non_runnable_branch_raises() -> None:
    with pytest.raises(TypeError, match="has no callable run"):
        Conditional(condition=True, if_true=object()).run()


@pytest.mark.parametrize("cls", [Sequence, Conditional, Switch])
def test_combinator_zero_arg_construct_and_run(cls: Any) -> None:
    cls().run()  # must construct AND run with no args (no-op), never raise
    assert _RUN_LOG == []


@pytest.mark.parametrize("cls", [PathExists, Not, AllOf, AnyOf])
def test_predicate_zero_arg_construct_and_call(cls: Any) -> None:
    assert cls()() in (True, False)  # zero-arg construct + call yields a bool


# --------------------------------------------------------------------------- #
# StreamStudio canvas integration — TorchRunner + ProgressReporting forwarding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [Sequence, Conditional, Switch])
def test_combinators_declare_needs_autograd(cls: Any) -> None:
    # A combinator may wrap a trainer, so it declares __needs_autograd__ — StreamStudio's executor
    # re-enables autograd for the whole run (otherwise the inner loss.backward() dies under
    # ComfyUI's inference_mode).
    assert cls().__needs_autograd__ is True


def test_progress_callback_forwarded_to_running_branch() -> None:
    from recordstream.runnable import ProgressReporting

    received: List[str] = []

    @configurable
    class _ProgressStep(ProgressReporting):
        def run(self) -> None:
            self._report_progress(1, 1, "step")  # fires through whatever callback was forwarded

    seq = Sequence([_ProgressStep()])
    seq.set_progress_callback(lambda v, t, d: received.append(d))  # executor injects on the combinator
    seq.run()
    assert received == ["step"]  # forwarded to the branch, which reported through it


def test_progress_forward_is_safe_when_branch_has_no_callback() -> None:
    seq = Sequence([_RunStep("a")])
    seq.set_progress_callback(lambda v, t, d: None)
    seq.run()  # _RunStep has no set_progress_callback -> _run must not raise
    assert _RUN_LOG == ["a"]


# --------------------------------------------------------------------------- #
# Confluid round-trip — a whole workflow serialises and runs
# --------------------------------------------------------------------------- #
def test_confluid_roundtrip_runs_nested_workflow() -> None:
    workflow = Sequence(
        [
            _RunStep("download"),
            Conditional(
                condition=_FixedPredicate(False),  # cache miss -> train branch
                if_true=_RunStep("skip"),
                if_false=_RunStep("train"),
            ),
            _RunStep("evaluate"),
        ]
    )
    restored: Any = confluid.load(confluid.dump(workflow))
    restored.run()
    assert _RUN_LOG == ["download", "train", "evaluate"]
