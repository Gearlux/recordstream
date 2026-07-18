"""Tests for 1→N expanding ops and iterable-only pipeline semantics."""

from typing import Iterator, List, Optional

import pytest
from confluid import configurable

from sampleflux.core import Flux, _worker_task, _worker_task_multi
from sampleflux.kinds import op_contract
from sampleflux.ops.context import Save, Use
from sampleflux.sample import Sample

# ---------------------------------------------------------------------------
# Fixture ops (module-level so they pickle for spawn parity)
# ---------------------------------------------------------------------------


@configurable
class SplitOp:
    """Expand one sample into ``count`` children (index appended to metadata).

    Args:
        count: Number of children yielded per incoming sample.
    """

    def __init__(self, count: int = 2) -> None:
        self.count = count

    def __call__(self, sample: Sample) -> Iterator[Sample]:
        for i in range(self.count):
            yield sample._replace(input=sample.input * 10 + i, metadata={**sample.meta, "child": i})


@configurable
class MarkedSplitOp:
    """An expansion op detected via the explicit EXPANDS marker (untyped __call__)."""

    EXPANDS = True

    def __call__(self, sample):  # type: ignore[no-untyped-def]
        return [sample, sample]


@configurable
class AddOp:
    """Add a constant to the input.

    Args:
        amount: Value added to ``sample.input``.
    """

    def __init__(self, amount: float = 1.0) -> None:
        self.amount = amount

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input + self.amount)


@configurable
class DropOddChildOp:
    """Filter inside an expansion: drop children with odd input."""

    def __call__(self, sample: Sample) -> Optional[Sample]:
        return None if int(sample.input) % 2 else sample


@configurable
class EmptySplitOp:
    """An expanding op that yields nothing (drops the sample entirely)."""

    def __call__(self, sample: Sample) -> Iterator[Sample]:
        return iter(())


def _samples(n: int = 2) -> List[Sample]:
    return [Sample(input=float(i), target=i, metadata={"idx": i}) for i in range(n)]


# ---------------------------------------------------------------------------
# Engine routes
# ---------------------------------------------------------------------------


class TestExpansion:
    def test_sequential_expansion_depth_first_order(self) -> None:
        out = list(Flux(source=_samples(2), ops=[SplitOp(count=2), AddOp(amount=0.5)]))
        # sample 0 -> children 0,1 -> +0.5 ; sample 1 -> 10,11 -> +0.5
        assert [s.input for s in out] == [0.5, 1.5, 10.5, 11.5]
        assert [s.meta["child"] for s in out] == [0, 1, 0, 1]

    def test_chained_expansions(self) -> None:
        out = list(Flux(source=_samples(1), ops=[SplitOp(count=2), SplitOp(count=2)]))
        # 0 -> [0, 1] -> [00,01,10,11] depth-first
        assert [s.input for s in out] == [0.0, 1.0, 10.0, 11.0]

    def test_none_drop_inside_expansion(self) -> None:
        out = list(Flux(source=_samples(1), ops=[SplitOp(count=4), DropOddChildOp()]))
        assert [s.input for s in out] == [0.0, 2.0]

    def test_empty_expansion_drops_the_sample(self) -> None:
        assert list(Flux(source=_samples(3), ops=[EmptySplitOp()])) == []

    def test_marked_expansion_via_class_attr(self) -> None:
        assert op_contract(MarkedSplitOp()).expands is True
        out = list(Flux(source=_samples(1), ops=[MarkedSplitOp()]))
        assert len(out) == 2

    def test_spawn_parallel_parity(self) -> None:
        seq = [s.input for s in Flux(source=_samples(3), ops=[SplitOp(count=2), AddOp()])]
        par = [s.input for s in Flux(source=_samples(3), ops=[SplitOp(count=2), AddOp()]).parallel(2)]
        assert seq == par

    def test_streamed_route_expansion(self) -> None:
        from sampleflux.ops.parallel import Parallel

        ops = [SplitOp(count=2), Parallel(ops=[AddOp(amount=0.5)], workers=1)]
        out = list(Flux(source=_samples(2), ops=ops))
        assert sorted(s.input for s in out) == [0.5, 1.5, 10.5, 11.5]

    def test_batch_over_expanded_stream(self) -> None:
        chunks = list(Flux(source=_samples(2), ops=[SplitOp(count=3)]).batch(4))
        assert [len(c) for c in chunks] == [4, 2]

    def test_expansion_with_context_ops(self) -> None:
        # A fork saved BEFORE the expansion is readable by each child (shallow ctx copy).
        ops = [Save(name="fork"), SplitOp(count=2), Use(name="fork")]
        out = list(Flux(source=_samples(2), ops=ops))
        # Use restores the pre-split fork for every child -> inputs are the originals.
        assert [s.input for s in out] == [0.0, 0.0, 1.0, 1.0]


class TestIterableOnly:
    def test_len_raises_with_actionable_message(self) -> None:
        flux = Flux(source=_samples(3), ops=[SplitOp()])
        with pytest.raises(TypeError, match="ITERABLE-ONLY.*SplitOp|SplitOp.*ITERABLE-ONLY"):
            len(flux)

    def test_getitem_raises(self) -> None:
        flux = Flux(source=_samples(3), ops=[SplitOp()])
        with pytest.raises(TypeError, match="iterable-only|ITERABLE-ONLY"):
            _ = flux[0]

    def test_non_expanding_pipeline_keeps_random_access(self) -> None:
        flux = Flux(source=_samples(3), ops=[AddOp()])
        assert len(flux) == 3 and flux[1].input == 2.0

    def test_strict_worker_task_rejects_expansion(self) -> None:
        with pytest.raises(TypeError, match="1→N expanding"):
            _worker_task(Sample(input=1.0), [SplitOp()])

    def test_worker_task_multi_returns_all(self) -> None:
        results = _worker_task_multi(Sample(input=1.0, metadata={}), [SplitOp(count=3)])
        assert [s.input for s in results] == [10.0, 11.0, 12.0]


class TestContextIsolationAcrossChildren:
    def test_children_have_independent_cell_sets(self) -> None:
        @configurable
        class SaveChildIdOp:
            def __call__(self, sample: Sample) -> Sample:
                from sampleflux.context import require

                require("SaveChildIdOp").put("mine", sample.meta["child"])
                return sample

        @configurable
        class ReadBackOp:
            def __call__(self, sample: Sample) -> Sample:
                from sampleflux.context import require

                return sample._replace(target=require("ReadBackOp").get("mine"))

        out = list(Flux(source=_samples(1), ops=[SplitOp(count=3), SaveChildIdOp(), ReadBackOp()]))
        assert [s.target for s in out] == [0, 1, 2]  # no cross-child leakage


def test_flowgraph_rejects_expanding_step() -> None:
    from sampleflux.flow import FlowGraph

    graph = FlowGraph(source=_samples(1), flow={"split": SplitOp()})
    with pytest.raises(NotImplementedError, match="expanding"):
        list(graph)
