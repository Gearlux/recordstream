from pathlib import Path

import confluid  # type: ignore[import-not-found]
import numpy as np

from dataflux.core import Flux
from dataflux.storage.hdf5 import HDF5Sink, HDF5Source


# 1. Define a simple transform
def rescale(data: np.ndarray, scale: float = 1.0):
    return data * scale


def main():
    # 2. Setup paths
    h5_path = Path("examples/test_data.h5")

    # 3. Create synthetic data and write to HDF5
    print(f"--- Writing synthetic data to {h5_path} ---")
    raw_data = [np.random.randn(10) for _ in range(5)]

    # We use overwrite=True to ensure a fresh file
    sink = HDF5Sink(h5_path, overwrite=True)
    Flux(raw_data).to_sink(sink)

    # Force close if not already closed by context
    sink.close()

    # 4. Serialize the Pipeline
    pipeline = Flux().map(rescale, scale=100.0)
    print("\n--- Serialized DataFlux Pipeline ---")
    yaml_state = confluid.dump(pipeline)
    print(yaml_state)

    # 5. Read back and process using reconstructed pipeline
    print("\n--- Reading back through Flux + HDF5Source ---")
    source = HDF5Source(h5_path)

    # Reconstruct the processing logic from YAML
    new_pipeline = confluid.load(yaml_state)
    new_pipeline.source = source  # Assign the live source

    for i, sample in enumerate(new_pipeline):
        print(f"Sample {i}: mean={sample.input.mean():.2f}")

    print("\nHDF5 end-to-end flow verified!")


if __name__ == "__main__":
    main()
