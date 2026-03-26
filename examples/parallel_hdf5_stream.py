import time
from pathlib import Path

import numpy as np

from dataflux.core import Flux
from dataflux.storage.hdf5 import HDF5Sink, HDF5Source


def heavy_rescale(data: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """A simulated heavy transformation."""
    time.sleep(0.05)  # Simulate CPU work
    return data * factor


def main() -> None:
    h5_path = Path("examples/stream_results.h5")
    num_samples = 50

    print(f"--- Starting Parallel Stream to {h5_path} ---")
    print(f"Generating {num_samples} samples with 4 parallel workers...")

    # 1. Setup Source (Synthetic)
    raw_data = [np.random.randn(1000) for _ in range(num_samples)]

    # 2. Setup Sink
    sink = HDF5Sink(h5_path, overwrite=True)

    # 3. Build and Execute Pipeline
    # Processing happens in parallel, Writing happens sequentially in the main process
    start = time.time()
    Flux(raw_data).parallel(workers=4).map(heavy_rescale, factor=255.0).to_sink(sink)

    duration = time.time() - start
    print(f"Streaming completed in {duration:.2f}s")

    # 4. Verify the results
    print("\n--- Verifying HDF5 Sink Content ---")
    source = HDF5Source(h5_path)
    with source:
        print(f"Total samples in file: {len(source)}")
        sample = next(iter(source))
        print(f"First sample shape: {sample.input.shape}")
        print(f"First sample max value: {sample.input.max():.2f}")

    print("\nParallel HDF5 Streaming Verified!")


if __name__ == "__main__":
    main()
