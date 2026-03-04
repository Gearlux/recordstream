import numpy as np

from dataflux.core import Flux
from dataflux.sample import Sample


def test_basic_flux():
    source = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    pipeline = Flux(source)
    results = list(pipeline)
    assert len(results) == 2
    assert isinstance(results[0], Sample)
    assert np.array_equal(results[0].input, source[0])


def double_it(x: np.ndarray) -> np.ndarray:
    return x * 2


def is_greater_than_two(s: Sample) -> bool:
    return s.input > 2


def test_flux_map():
    source = [np.array([1, 2, 3])]
    pipeline = Flux(source).map(double_it)
    results = list(pipeline)
    assert np.array_equal(results[0].input, np.array([2, 4, 6]))


def test_flux_filter():
    source = [1, 2, 3, 4]
    pipeline = Flux(source).filter(is_greater_than_two)
    results = list(pipeline)
    assert len(results) == 2
    assert results[0].input == 3
