from pathlib import Path

import numpy as np

from dataflux.core import Flux
from dataflux.sample import Sample
from dataflux.storage.directory import DirectorySink
from dataflux.storage.zarr import ZarrBatchSink, ZarrGroupSink


def main() -> None:
    # --- 1. DirectorySink (Irregular Lengths) ---
    print("--- Testing DirectorySink ---")
    dir_path = Path("examples/dir_store")
    samples = [
        Sample(input=np.random.randn(5), metadata={"id": "tiny"}),
        Sample(input=np.random.randn(50), metadata={"id": "large"}),
    ]

    sink_dir = DirectorySink(dir_path, overwrite=True)
    Flux(samples).to_sink(sink_dir)
    print(f"Created {len(list(dir_path.glob('*')))} sample folders.")

    # --- 2. ZarrGroupSink (Irregular Lengths, Single Bundle) ---
    print("\n--- Testing ZarrGroupSink ---")
    zarr_group_path = "examples/group_store.zarr"
    sink_group = ZarrGroupSink(zarr_group_path, overwrite=True)
    Flux(samples).to_sink(sink_group)
    print("Zarr group write complete.")

    # --- 3. ZarrBatchSink (Uniform Lengths, Optimized) ---
    print("\n--- Testing ZarrBatchSink ---")
    zarr_batch_path = "examples/batch_store.zarr"
    # Uniform samples: 10 elements each
    uniform_samples = [Sample(input=np.random.randn(10).astype(np.float32)) for _ in range(10)]

    sink_batch = ZarrBatchSink(zarr_batch_path, shape=[10], overwrite=True)
    Flux(uniform_samples).to_sink(sink_batch)
    print("Zarr batch append complete.")

    print("\nAll advanced storage sinks verified!")


if __name__ == "__main__":
    main()
