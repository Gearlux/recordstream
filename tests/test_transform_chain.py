"""Tests for :class:`sampleflux.ops.transform_chain.TransformChain`."""

from typing import List, Optional

from sampleflux.ops.transform_chain import TransformChain
from sampleflux.sample import Sample


def _s(v: int = 0) -> Sample:
    return Sample(input=v, target=None, metadata={})


class _AddOp:
    """Increment sample.input by a fixed delta."""

    def __init__(self, delta: int = 1) -> None:
        self.delta = delta

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input + self.delta)


class _TagOp:
    """Append a string tag to metadata['tags']."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __call__(self, sample: Sample) -> Sample:
        new_meta = dict(sample.meta)
        new_meta["tags"] = new_meta.get("tags", []) + [self.tag]
        return sample._replace(metadata=new_meta)


class _DropOp:
    """Always returns None — simulates a filter op."""

    def __call__(self, sample: Sample) -> Optional[Sample]:
        return None


class _ClosableOp:
    def __init__(self, name: str, log: List[str]) -> None:
        self.name = name
        self.log = log

    def __call__(self, sample: Sample) -> Sample:
        return sample

    def close(self) -> None:
        self.log.append(self.name)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_zero_arg_construction() -> None:
    """TransformChain() must construct with no arguments (lazy convention)."""
    chain = TransformChain()
    assert chain.ops == []


def test_empty_chain_is_identity() -> None:
    """An empty TransformChain passes the sample through unchanged."""
    chain = TransformChain()
    s = _s(42)
    out = chain(s)
    assert out is s


def test_happy_path_multiple_ops_applied_in_order() -> None:
    """Ops fire left-to-right; each op sees the output of the previous one."""
    chain = TransformChain(ops=[_AddOp(1), _AddOp(2), _AddOp(3)])
    out = chain(_s(0))
    assert out is not None
    assert out.input == 6  # 0 + 1 + 2 + 3


def test_ops_applied_in_declared_order_via_metadata_tags() -> None:
    """Ordering is visible: tags accumulate in declaration order."""
    chain = TransformChain(ops=[_TagOp("a"), _TagOp("b"), _TagOp("c")])
    out = chain(_s())
    assert out is not None
    assert out.meta["tags"] == ["a", "b", "c"]


def test_none_propagation_stops_chain_early() -> None:
    """If any op returns None the chain stops and propagates None."""
    called: List[str] = []

    class _RecordOp:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def __call__(self, sample: Sample) -> Sample:
            called.append(self.tag)
            return sample

    chain = TransformChain(ops=[_RecordOp("before"), _DropOp(), _RecordOp("after")])
    out = chain(_s())
    assert out is None
    assert called == ["before"]  # "after" must NOT have fired


def test_none_propagation_from_first_op() -> None:
    """None returned by the very first op also short-circuits the chain."""
    chain = TransformChain(ops=[_DropOp(), _AddOp(99)])
    out = chain(_s(0))
    assert out is None


def test_fluid_resolution_lazy_and_cached() -> None:
    """Confluid Fluid markers inside ops are resolved on first call and cached."""
    from confluid import configurable
    from confluid.fluid import Class, Fluid

    @configurable
    class _Inner:
        def __call__(self, sample: Sample) -> Sample:
            return sample._replace(input=sample.input + 10)

    fluid_op = Class(_Inner)
    chain = TransformChain(ops=[fluid_op])

    out1 = chain(_s(5))
    assert out1 is not None
    assert out1.input == 15

    # Slot must now hold the resolved instance, not a Fluid.
    assert not isinstance(chain.ops[0], Fluid)

    out2 = chain(_s(5))
    assert out2 is not None
    assert out2.input == 15


def test_close_propagates_to_all_inner_ops() -> None:
    """close() forwards to every inner op that implements it."""
    log: List[str] = []
    chain = TransformChain(ops=[_ClosableOp("x", log), _ClosableOp("y", log)])
    chain.close()
    assert log == ["x", "y"]


def test_close_on_empty_chain_is_safe() -> None:
    """close() on an empty chain must not raise."""
    TransformChain().close()  # must not raise


def test_close_skips_ops_without_close_method() -> None:
    """close() only calls close on ops that have it — no AttributeError on plain callables."""
    log: List[str] = []
    chain = TransformChain(ops=[_AddOp(1), _ClosableOp("z", log)])
    chain.close()
    assert log == ["z"]
