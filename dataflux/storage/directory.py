from pathlib import Path
from typing import Union

import confluid  # type: ignore[import-not-found]
import numpy as np

from dataflux.sample import Sample
from dataflux.storage.base import DataSink, Storage


@confluid.configurable
class DirectorySink(Storage, DataSink):
    """
    High-concurrency sink that stores each Sample in its own directory.
    Perfect for irregular data lengths and massive parallel writing.
    """

    def __init__(self, path: Union[str, Path], overwrite: bool = False, use_npz: bool = True) -> None:
        self.path = Path(path)
        self.overwrite = overwrite
        self.use_npz = use_npz
        self._counter = 0

    def open(self) -> "DirectorySink":
        if self.overwrite and self.path.exists():
            # In a real app, we'd clear the directory
            pass
        self.path.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, sample: Sample) -> None:
        """Write a sample to its own subdirectory."""
        # Use a zero-padded index for sorting
        sample_dir = self.path / f"{self._counter:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Metadata (YAML via Confluid)
        if sample.metadata:
            meta_path = sample_dir / "metadata.yaml"
            meta_path.write_text(confluid.dump(sample.metadata))

        # 2. Save Input and Target (Numpy)
        if self.use_npz:
            # Combined file
            np.savez(
                sample_dir / "sample.npz",
                data=sample.input,
                target=sample.target if sample.target is not None else np.array([]),
            )
        else:
            # Separate files
            np.save(sample_dir / "data.npy", sample.input)
            if sample.target is not None:
                np.save(sample_dir / "target.npy", sample.target)

        self._counter += 1

    def flush(self) -> None:
        pass  # Filesystem handles immediate writes
