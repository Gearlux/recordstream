from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

import confluid  # type: ignore[import-not-found]
import h5py  # type: ignore[import-untyped]
import numpy as np

from dataflux.sample import Sample
from dataflux.storage.base import DataSink, DataSource, Storage


@confluid.configurable
class HDF5Source(Storage, DataSource):
    """Clean, high-performance HDF5 data source."""

    def __init__(self, path: Union[str, Path], sample_key: str = "data", target_key: Optional[str] = "target") -> None:
        self.path = Path(path)
        self.sample_key = sample_key
        self.target_key = target_key
        self._file: Optional[h5py.File] = None
        self._indices: List[str] = []

    def open(self) -> "HDF5Source":
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._build_index()
        return self

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def _build_index(self) -> None:
        """Simple index: look for datasets matching sample_key."""
        self._indices = []
        if self._file is None:
            return

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and name.endswith(self.sample_key):
                self._indices.append(name)

        self._file.visititems(visitor)

    def __len__(self) -> int:
        if not self._indices:
            self.open()
        return len(self._indices)

    def __iter__(self) -> Iterator[Sample]:
        self.open()
        if self._file is None:
            return

        for idx in self._indices:
            ds = self._file[idx]
            data = np.array(ds)

            # Resolve target
            target = None
            if self.target_key:
                target_path = idx.replace(self.sample_key, self.target_key)
                if target_path in self._file:
                    target = np.array(self._file[target_path])

            # Resolve metadata from attributes
            metadata = dict(ds.attrs)
            yield Sample(data, target, metadata)


@confluid.configurable
class HDF5Sink(Storage, DataSink):
    """Clean HDF5 data sink focused on Sample triplets."""

    def __init__(self, path: Union[str, Path], compression: Optional[str] = "gzip", overwrite: bool = False) -> None:
        self.path = Path(path)
        self.compression = compression
        self.overwrite = overwrite
        self._file: Optional[h5py.File] = None
        self._counter = 0

    def open(self) -> "HDF5Sink":
        if self._file is None:
            mode = "w" if self.overwrite else "a"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = h5py.File(self.path, mode)
        return self

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def write(self, sample: Sample) -> None:
        self.open()
        if self._file is None:
            return

        prefix = f"{self._counter:05d}"

        # Write data
        ds = self._file.create_dataset(f"{prefix}_data", data=sample.input, compression=self.compression)

        # Write attributes (metadata)
        for k, v in sample.metadata.items():
            ds.attrs[k] = v

        # Write target
        if sample.target is not None:
            self._file.create_dataset(f"{prefix}_target", data=sample.target, compression=self.compression)

        self._counter += 1

    def flush(self) -> None:
        if self._file:
            self._file.flush()
