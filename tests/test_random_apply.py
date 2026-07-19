"""Tests for sampleflux.ops.random_apply.RandomApply."""

import pytest

from sampleflux.ops.random_apply import RandomApply
from sampleflux.sample import Sample


def _s(v: int = 0) -> Sample:
    return Sample(input=v, metadata={})


class _BumpOp:
    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input + 1)


def test_zero_arg_construction() -> None:
    assert RandomApply() is not None


def _run(op: RandomApply, sample: Sample) -> Sample:
    """Apply and narrow: these tests never exercise the drop (None) path."""
    out = op(sample)
    assert out is not None
    return out


def test_probability_zero_never_applies() -> None:
    op = RandomApply(op=_BumpOp(), probability=0.0)
    for _ in range(20):
        out = _run(op, _s(0))
        assert out.input == 0


def test_probability_one_always_applies() -> None:
    op = RandomApply(op=_BumpOp(), probability=1.0)
    for _ in range(20):
        out = _run(op, _s(0))
        assert out.input == 1


def test_raises_when_op_is_none() -> None:
    op = RandomApply(probability=1.0)
    with pytest.raises(ValueError, match="op"):
        op(_s())


def test_is_marked_random() -> None:
    assert getattr(RandomApply, "__confluid_random__", False) is True


def test_is_registered_configurable() -> None:
    from confluid.registry import resolve_class  # type: ignore[import-not-found]

    path = f"{RandomApply.__module__}.{RandomApply.__qualname__}"
    assert resolve_class(path) is RandomApply


def test_flows_confluid_fluid_op_lazily() -> None:
    from confluid import configurable
    from confluid.fluid import Class

    @configurable
    class _Inner:
        def __call__(self, sample: Sample) -> Sample:
            return sample._replace(input=sample.input + 10)

    fluid_op = Class(_Inner)
    op = RandomApply(op=fluid_op, probability=1.0)
    out = _run(op, _s(5))
    assert out.input == 15
    # second call reuses the cached flowed op
    out2 = _run(op, _s(5))
    assert out2.input == 15


def test_sample_passes_through_unchanged_when_skipped() -> None:
    s = _s(42)
    op = RandomApply(op=_BumpOp(), probability=0.0)
    assert op(s) is s


# ---------------------------------------------------------------------------
# random_state / reproducibility
# ---------------------------------------------------------------------------


def test_random_state_stored_on_instance() -> None:
    op = RandomApply(op=_BumpOp(), probability=0.5, random_state=42)
    assert op.random_state == 42


def test_random_state_none_is_default() -> None:
    op = RandomApply(op=_BumpOp())
    assert op.random_state is None


def test_gate_reproducible_with_seed() -> None:
    """Two RandomApply instances with the same seed must make identical gate decisions."""
    s = _s(0)
    op_a = RandomApply(op=_BumpOp(), probability=0.5, random_state=7)
    op_b = RandomApply(op=_BumpOp(), probability=0.5, random_state=7)
    results_a = [_run(op_a, s).input for _ in range(30)]
    results_b = [_run(op_b, s).input for _ in range(30)]
    assert results_a == results_b


def test_gate_different_seeds_produce_different_sequences() -> None:
    s = _s(0)
    op_a = RandomApply(op=_BumpOp(), probability=0.5, random_state=1)
    op_b = RandomApply(op=_BumpOp(), probability=0.5, random_state=2)
    results_a = [_run(op_a, s).input for _ in range(50)]
    results_b = [_run(op_b, s).input for _ in range(50)]
    assert results_a != results_b


def test_zero_arg_construction_with_random_state_none() -> None:
    op = RandomApply()
    assert op.random_state is None
    assert op._gate_rng is None  # lazily initialized on first call
