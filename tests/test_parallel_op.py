"""Tests for :class:`dataflux.ops.parallel.Parallel`."""

from __future__ import annotations

import time
from typing import Iterable, Iterator, List, Optional

import numpy as np

from dataflux.core import Flux
from dataflux.ops.parallel import Parallel
from dataflux.sample import Sample

# Top-level functions/classes — workers must be able to pickle these.


def double_input(sample: Sample) -> Sample:
    return sample._replace(input=sample.input * 2)


def slow_double_input(sample: Sample) -> Sample:
    time.sleep(0.05)
    return sample._replace(input=sample.input * 2)


def drop_odd(sample: Sample) -> Optional[Sample]:
    return sample if int(sample.input.item()) % 2 == 0 else None


class _TrackingSource:
    """Iterable that records the maximum number of items pulled before any
    are consumed downstream — used to verify bounded prefetch."""

    def __init__(self, n: int, sleep_per_item: float = 0.0) -> None:
        self.n = n
        self.sleep_per_item = sleep_per_item
        self.pulled = 0
        self.consumed = 0
        self.max_outstanding = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.n):
            self.pulled += 1
            self.max_outstanding = max(
                self.max_outstanding, self.pulled - self.consumed
            )
            if self.sleep_per_item:
                time.sleep(self.sleep_per_item)
            yield np.array([i])

    def __len__(self) -> int:
        return self.n


def test_parallel_inline_call_applies_ops_sequentially() -> None:
    """``Parallel.__call__`` (used by ``Flux.__getitem__``) must apply its
    inner ops sequentially in the calling process — same result as if the
    ops were a plain list."""
    op = Parallel(ops=[double_input, double_input], workers=4)
    sample = Sample(input=np.array([3]))
    out = op(sample)
    assert out is not None
    assert out.input == np.array([12])  # 3 * 2 * 2


def test_parallel_stream_yields_in_source_order() -> None:
    source = [np.array([i]) for i in range(20)]
    flux = Flux(source, ops=[Parallel(ops=[slow_double_input], workers=4)])
    results = list(flux)
    assert [int(r.input.item()) for r in results] == [i * 2 for i in range(20)]


def test_parallel_stream_bounds_prefetch() -> None:
    """With ``workers=4`` and a slow inner op, the source iterator must not
    be drained more than ``2*workers + 1 = 9`` items ahead of consumption."""
    workers = 4
    expected_limit = 2 * workers + 1
    source = _TrackingSource(n=50)

    def consume(stream: Iterable[Sample]) -> List[Sample]:
        results: List[Sample] = []
        for sample in stream:
            source.consumed += 1
            results.append(sample)
        return results

    flux = Flux(source, ops=[Parallel(ops=[slow_double_input], workers=workers)])
    results = consume(flux)

    assert len(results) == 50
    assert (
        source.max_outstanding <= expected_limit
    ), f"prefetch unbounded: max_outstanding={source.max_outstanding} > {expected_limit}"


def test_parallel_filter_drops_nones_and_preserves_order() -> None:
    source = [np.array([i]) for i in range(10)]
    flux = Flux(source, ops=[Parallel(ops=[drop_odd, double_input], workers=3)])
    results = list(flux)
    assert [int(r.input.item()) for r in results] == [0, 4, 8, 12, 16]


def test_parallel_workers_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="workers="):
        Parallel(ops=[double_input], workers=0)


def test_parallel_close_propagates_to_inner_ops() -> None:
    closed: List[str] = []

    class ClosableOp:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, s: Sample) -> Sample:
            return s

        def close(self) -> None:
            closed.append(self.name)

    op = Parallel(ops=[ClosableOp("a"), ClosableOp("b")], workers=2)
    op.close()
    assert closed == ["a", "b"]
