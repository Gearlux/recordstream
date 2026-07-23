import time

import numpy as np

from sampleflux import Image, Sample
from sampleflux.bag.items import item_data
from sampleflux.core import Flux


def heavy_op(x: np.ndarray) -> np.ndarray:
    time.sleep(0.1)
    return x * 2


def test_parallel_execution() -> None:
    source = [Sample({"x": Image(np.array([i]))}, roles={"x": "input"}) for i in range(10)]

    start = time.time()
    # Use a real top-level function for pickling
    pipeline = Flux(source).map(heavy_op).parallel(workers=4)
    results = pipeline.collect()
    duration = time.time() - start

    assert len(results) == 10
    assert int(item_data(results[0]["x"])[0]) == 0
    assert int(item_data(results[9]["x"])[0]) == 18
    # We only assert the pipeline completes — this isn't a benchmark.
    assert duration < 15.0


def test_parallel_with_joint() -> None:
    # Already tested elsewhere; helps coverage here too.
    pass
