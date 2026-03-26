import confluid  # type: ignore[import-not-found]
import numpy as np

from dataflux.core import Flux


# 1. Define simple functional transformations
def add_noise(data: np.ndarray, std: float = 0.1) -> np.ndarray:
    return data + np.random.normal(0, std, data.shape)


def multiply(data: np.ndarray, factor: float = 2.0) -> np.ndarray:
    return data * factor


def main() -> None:
    # 2. Create raw data source
    raw_data = [(np.array([1.0, 2.0]), 1), (np.array([3.0, 4.0]), 0)]

    # 3. Build the Flux pipeline
    # For robust serialization, we should use strings for function references
    # or ensure the classes/functions are part of the registry.
    pipeline = Flux(raw_data).map(multiply, factor=10.0).map(add_noise, std=0.01)

    # 4. Serialize the Pipeline
    # We set source=None before serialization to only serialize the "recipe"
    # and avoid serializing raw numpy data which is not YAML-safe.
    print("\n--- Serialized DataFlux Pipeline ---")
    yaml_state = ""
    try:
        pipeline.source = None
        yaml_state = confluid.dump(pipeline)
        print(yaml_state)
    except Exception as e:
        print(f"Serialization failed: {e}")

    # 5. Reconstruct and Execute
    print("\n--- Reconstructing Pipeline from YAML ---")
    try:
        if yaml_state:
            # Explicitly pass as YAML string (containing \n ensures
            # load treats it as YAML) or ensure it's handled by path vs yaml logic.
            new_pipeline = confluid.load(yaml_state)

            new_pipeline.source = raw_data
            for sample in new_pipeline:
                print(f"Reconstructed Output: {sample.input}")
        else:
            print("No YAML state to reconstruct.")
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
