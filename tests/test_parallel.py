import time

import numpy as np

from sampleflux import Image, item_data
from sampleflux.core import Flux


def heavy_op(x: np.ndarray) -> np.ndarray:
    time.sleep(0.1)  # simulated per-record WORKLOAD (not a synchronization wait)
    return x * 2


def test_parallel_execution() -> None:
    source = [{"x": Image(np.array([i]))} for i in range(10)]

    start = time.time()
    # Use a real top-level function for pickling; key= targets the record entry's payload.
    pipeline = Flux(source).map(heavy_op, key="x").parallel(workers=4)
    results = pipeline.collect()
    duration = time.time() - start

    assert len(results) == 10
    assert isinstance(results[0]["x"], Image)  # item type survives the spawn round-trip
    assert int(item_data(results[0]["x"])[0]) == 0
    assert int(item_data(results[9]["x"])[0]) == 18
    # We only assert the pipeline completes — this isn't a benchmark.
    assert duration < 15.0
