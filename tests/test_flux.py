import numpy as np
import pytest
from typing import Any
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


def test_optional_context_manager() -> None:
    from dataflux.core import OptionalContextManager

    with OptionalContextManager() as cm:
        assert cm is not None


def fail_op(x: Any) -> Any:
    raise ValueError("Intentional failure")


def test_wrapped_op_error() -> None:
    source = [np.array([1])]
    pipeline = Flux(source).map(fail_op)
    with pytest.raises(ValueError, match="Intentional failure"):
        pipeline.collect()


def test_flux_to_torch() -> None:
    # Just verify it exists and returns None for now
    pipeline = Flux([])
    assert pipeline.to_torch() is None
