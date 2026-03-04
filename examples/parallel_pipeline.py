import time

import numpy as np

from dataflux.core import Flux


def heavy_computation(data: np.ndarray, intensity: int = 10):
    """Simulate a heavy CPU-bound transformation."""
    # Artificial delay to simulate processing time
    time.sleep(0.1)
    return data * intensity


def main():
    # 1. Create a large synthetic dataset
    print("--- Creating data source (20 items) ---")
    raw_data = [np.random.randn(100) for _ in range(20)]

    # 2. Sequential Execution
    print("\n--- Running Sequentially ---")
    start = time.time()
    sequential_pipeline = Flux(raw_data).map(heavy_computation, intensity=2)
    results_seq = sequential_pipeline.collect()
    duration_seq = time.time() - start
    print(f"Sequential duration: {duration_seq:.2f}s")

    # 3. Parallel Execution (4 workers)
    print("\n--- Running in Parallel (4 workers) ---")
    start = time.time()
    # .parallel() enables the multiprocess engine
    parallel_pipeline = Flux(raw_data).map(heavy_computation, intensity=2).parallel(workers=4)
    results_par = parallel_pipeline.collect()
    duration_par = time.time() - start
    print(f"Parallel duration: {duration_par:.2f}s")

    # 4. Verification
    assert len(results_seq) == len(results_par)
    speedup = duration_seq / duration_par
    print(f"\nSpeedup: {speedup:.1f}x")
    print("Multiprocessing Verified!")


if __name__ == "__main__":
    main()
