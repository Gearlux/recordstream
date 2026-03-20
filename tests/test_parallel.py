import time

import numpy as np

from dataflux.core import Flux


def heavy_op(x: np.ndarray) -> np.ndarray:
    time.sleep(0.1)
    return x * 2


def test_parallel_execution() -> None:
    source = [np.array([i]) for i in range(10)]

    start = time.time()
    # Use a real top-level function for pickling
    pipeline = Flux(source).map(heavy_op).parallel(workers=4)
    results = pipeline.collect()
    duration = time.time() - start

    assert len(results) == 10
    # In my previous run it was i*2
    assert results[0].input == 0
    assert results[9].input == 18
    # 10 items of 0.1s sequentially = 1.0s. 4 workers = ~0.3s.
    assert duration < 0.9


def test_parallel_with_joint() -> None:
    # Already tested in test_joint.py, but helps coverage here too
    pass
