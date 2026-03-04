import confluid  # type: ignore[import-not-found]
import numpy as np

from dataflux.core import Flux


# 1. Define simple functional transformations
def add_noise(data: np.ndarray, std: float = 0.1):
    return data + np.random.normal(0, std, data.shape)


def multiply(data: np.ndarray, factor: float = 2.0):
    return data * factor


# 2. Create raw data source
raw_data = [(np.array([1.0, 2.0]), 1), (np.array([3.0, 4.0]), 0)]

# 3. Build the Flux pipeline
# For robust serialization, we should use strings for function references
# or ensure the classes/functions are part of the registry.
pipeline = Flux(raw_data).map(multiply, factor=10.0).map(add_noise, std=0.01)

# 4. Serialize the Pipeline
print("\n--- Serialized DataFlux Pipeline ---")
try:
    yaml_state = confluid.dump(pipeline)
    print(yaml_state)
except Exception as e:
    print(f"Serialization failed: {e}")

# 5. Reconstruct and Execute
print("\n--- Reconstructing Pipeline from YAML ---")
try:
    # Explicitly pass as YAML string (containing \n ensures
    # load treats it as YAML) or ensure it's handled by path vs yaml logic.
    new_pipeline = confluid.load(yaml_state)

    new_pipeline.source = raw_data
    for sample in new_pipeline:
        print(f"Reconstructed Input: {sample.input}")
except Exception as e:
    print(f"Reconstruction failed: {e}")
    import traceback

    traceback.print_exc()
