import numpy as np

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
