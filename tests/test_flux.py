from typing import Any

import numpy as np
import pytest

from dataflux.core import Flux
from dataflux.sample import Sample


def test_basic_flux() -> None:
    source = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    pipeline = Flux(source)
    results = list(pipeline)
    assert len(results) == 2
    assert isinstance(results[0], Sample)
    assert np.array_equal(results[0].input, source[0])


def double_it(x: np.ndarray) -> np.ndarray:
    return x * 2


def is_greater_than_two(s: Sample) -> bool:
    # Use np.any() or similar to ensure a single boolean is returned for mypy
    return bool(np.any(s.input > 2))


def test_flux_map() -> None:
    source = [np.array([1, 2, 3])]
    pipeline = Flux(source).map(double_it)
    results = list(pipeline)
    assert np.array_equal(results[0].input, np.array([2, 4, 6]))


def test_flux_filter() -> None:
    source = [np.array([1]), np.array([2]), np.array([3]), np.array([4])]
    pipeline = Flux(source).filter(is_greater_than_two)
    results = list(pipeline)
    assert len(results) == 2
    assert results[0].input == 3


class MockSink:
    def __init__(self) -> None:
        self.written: list[Sample] = []

    def write(self, sample: Sample) -> None:
        self.written.append(sample)

    def flush(self) -> None:
        pass


def test_flux_to_sink() -> None:
    source = [np.array([1, 2]), np.array([3, 4])]
    sink = MockSink()
    Flux(source).to_sink(sink)
    assert len(sink.written) == 2
    assert np.array_equal(sink.written[0].input, source[0])


def full_transform(s: Sample) -> Sample:
    return s._replace(input=s.input * 2)


def test_wrapped_op_all() -> None:
    source = [np.array([10])]
    pipeline = Flux(source).map(full_transform, select="all")
    results = pipeline.collect()
    assert results[0].input == 20


def test_filter_op() -> None:
    from dataflux.core import FilterOp

    op = FilterOp(lambda s: bool(s.input > 5))
    s1 = Sample(input=10)
    s2 = Sample(input=2)
    assert op(s1) == s1
    assert op(s2) is None


def fail_op(x: Any) -> Any:
    raise ValueError("Intentional failure")


def test_wrapped_op_error() -> None:
    source = [np.array([1])]
    pipeline = Flux(source).map(fail_op)
    with pytest.raises(ValueError, match="Intentional failure"):
        pipeline.collect()


def target_transform(t: Any) -> Any:
    return t + 10


def test_wrapped_op_target() -> None:
    source = [Sample(input=1, target=5)]
    # select="target" hits lines 69-70
    pipeline = Flux(source).map(target_transform, select="target")
    results = pipeline.collect()
    assert results[0].target == 15


def test_wrapped_op_fallback() -> None:
    # select="unknown" hits line 73
    source = [Sample(input=1)]
    pipeline = Flux(source).map(lambda x: x, select="unknown")
    results = pipeline.collect()
    assert results[0].input == 1


def test_worker_task_none() -> None:
    from dataflux.core import _worker_task

    # hits line 83 by using two ops, first returning None
    assert _worker_task(Sample(input=1), [lambda s: None, lambda s: s]) is None


def test_flux_from_source() -> None:
    # hits from_source classmethod
    f = Flux.from_source([1, 2, 3])
    assert len(f) == 3


def test_flux_len_fallback() -> None:
    # hits line 134 (len 0 if no source or no len)
    f = Flux()
    assert len(f) == 0
    f2 = Flux(iter([1, 2]))  # iter doesn't have len
    assert len(f2) == 0


# --- Indexed access (__getitem__) ---


def test_getitem_basic() -> None:
    source = [np.array([10]), np.array([20]), np.array([30])]
    flux = Flux(source)
    sample = flux[1]
    assert isinstance(sample, Sample)
    assert np.array_equal(sample.input, np.array([20]))


def test_getitem_with_ops() -> None:
    source = [np.array([1]), np.array([2]), np.array([3])]
    flux = Flux(source).map(double_it)
    assert np.array_equal(flux[0].input, np.array([2]))
    assert np.array_equal(flux[2].input, np.array([6]))


def test_getitem_no_source() -> None:
    flux = Flux()
    with pytest.raises(TypeError):
        flux[0]


def test_getitem_non_indexable_source() -> None:
    flux = Flux(iter([1, 2, 3]))  # iterators don't support indexing
    with pytest.raises(TypeError):
        flux[0]


def test_getitem_out_of_range() -> None:
    source = [np.array([1])]
    flux = Flux(source)
    with pytest.raises(IndexError):
        flux[5]


class _IterableWithLen:
    """Iterable source with ``__iter__`` + ``__len__`` but NO ``__getitem__``.

    Mirrors the shape of ``waivefront.regions_source.RegionsJsonSource``.
    Tracks how many times ``__iter__`` is invoked so tests can prove Flux
    materializes only once per Flux lifetime.
    """

    def __init__(self, items: list) -> None:
        self._items = items
        self.iter_calls = 0

    def __iter__(self) -> Any:
        self.iter_calls += 1
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def test_getitem_iterable_with_len_materializes_once() -> None:
    """Flux caches iterable-only sources on first __getitem__ and reuses the cache."""
    source = _IterableWithLen([np.array([1]), np.array([2]), np.array([3])])
    flux = Flux(source)

    # Cache is empty until first random access.
    assert flux._indexable_cache is None
    assert source.iter_calls == 0

    s0 = flux[0]
    s1 = flux[1]
    s2 = flux[2]

    assert np.array_equal(s0.input, np.array([1]))
    assert np.array_equal(s1.input, np.array([2]))
    assert np.array_equal(s2.input, np.array([3]))
    # The source was iterated exactly once across the three random-access calls.
    assert source.iter_calls == 1
    assert flux._indexable_cache is not None
    assert len(flux._indexable_cache) == 3


def test_getitem_indexable_source_does_not_populate_cache() -> None:
    """List-backed (indexable) sources take the fast path and don't trigger the cache."""
    source = [np.array([1]), np.array([2])]
    flux = Flux(source)
    _ = flux[0]
    _ = flux[1]
    # Fast path: cache is never populated.
    assert flux._indexable_cache is None


def test_getitem_on_deferred_fluid_source_raises_actionable_error() -> None:
    """When the user hands Flux a still-deferred Confluid Class marker, indexing
    must raise with a human-readable hint — not the cryptic ``num_samples=0``
    or ``does not support indexing``.
    """
    from confluid import Class

    class Dummy:
        def __init__(self, x: int = 0) -> None:
            self.x = x

    flux = Flux(Class(Dummy, x=1))
    with pytest.raises(TypeError, match="deferred Confluid marker"):
        flux[0]
    with pytest.raises(TypeError, match="deferred Confluid marker"):
        len(flux)


def test_deferred_fluid_op_raises_actionable_error() -> None:
    """Same guard applies to ops: a still-deferred Class in ops[] surfaces a
    clear message pointing the YAML author at the `!class:X()` fix instead of
    letting DataLoader workers report `'Class' object is not callable`.
    """
    from confluid import Class

    class DummyOp:
        def __init__(self, factor: int = 1) -> None:
            self.factor = factor

        def __call__(self, sample: Sample) -> Sample:
            return sample

    source = [np.array([1]), np.array([2])]
    flux = Flux(source, ops=[Class(DummyOp, factor=2)])
    with pytest.raises(TypeError, match=r"Flux\.ops\[0\] is still a deferred Confluid marker"):
        flux[0]
    with pytest.raises(TypeError, match=r"Flux\.ops\[0\] is still a deferred Confluid marker"):
        list(flux)
