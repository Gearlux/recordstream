from pathlib import Path
from typing import Iterator, Optional, Union

import h5py
import numpy as np
import torch
from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample
from dataflux.storage.base import DataSink, DataSource, Storage, to_numpy

logger = get_logger("dataflux.storage.hdf5")


@configurable
class HDF5Source(Storage, DataSource):
    """Clean, high-performance HDF5 data source."""

    def __init__(
        self,
        path: Union[str, Path] = "",
        sample_key: str = "data",
        target_key: Optional[str] = "target",
    ) -> None:
        # Lazy / zero-arg: store config only; the file is opened lazily in open() (an unset path
        # surfaces there, not in __init__).
        self.path = Path(path)
        self.sample_key = sample_key
        self.target_key = target_key
        self._file: Optional[h5py.File] = None

    def open(self) -> "HDF5Source":
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __iter__(self) -> Iterator[Sample]:
        self.open()
        if self._file is None:
            return

        prefixes = sorted([k.split("_data")[0] for k in self._file.keys() if k.endswith("_data")])

        for pref in prefixes:
            data = self._file[f"{pref}_data"][()]
            target = self._file[f"{pref}_target"][()] if f"{pref}_target" in self._file else None
            metadata = dict(self._file[f"{pref}_data"].attrs)
            # Merge array-valued metadata written as datasets under the per-sample meta group
            # (see HDF5Sink.write). Absent on files written before this layout — old files read unchanged.
            meta_grp = self._file.get(f"{pref}_meta")
            if isinstance(meta_grp, h5py.Group):
                for key, dset in meta_grp.items():
                    metadata[key] = dset[()]
            # Source returns Tensors to match schema
            yield Sample(input=torch.from_numpy(data), target=target, metadata=metadata)

    def __len__(self) -> int:
        self.open()
        if self._file is None:
            return 0
        return len([k for k in self._file.keys() if k.endswith("_data")])


@configurable
class HDF5Sink(Storage, DataSink):
    """High-performance HDF5 data sink focused on Sample triplets."""

    def __init__(
        self,
        path: Union[str, Path] = "",
        compression: Optional[str] = "gzip",
        overwrite: bool = False,
    ) -> None:
        # Lazy / zero-arg: store config only; the file is opened lazily in open().
        self.path = Path(path)
        self.compression = compression
        self.overwrite = overwrite
        self._file: Optional[h5py.File] = None
        self._counter = 0

    def open(self) -> "HDF5Sink":
        if self._file is None:
            mode = "w" if self.overwrite and self._counter == 0 else "a"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Opening HDF5 file for writing: {self.path} (mode={mode})")
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

        # Convert tensors to numpy for h5py
        input_data = to_numpy(sample.input)
        target_data = to_numpy(sample.target)

        # 1. Write Data
        kwargs = {}
        if self.compression and hasattr(input_data, "shape") and len(input_data.shape) > 0:
            kwargs["compression"] = self.compression

        ds = self._file.create_dataset(f"{prefix}_data", data=input_data, **kwargs)

        # 2. Write Metadata. Scalars/strings go on the data dataset's HDF5 attributes (compact,
        # round-trips for the common case). Array-valued metadata (e.g. a segmentation mask) CANNOT
        # be stored as an attribute — HDF5 caps attribute size ("object header message is too large")
        # and the str() fallback would silently truncate the array — so it is written as its own
        # dataset under a per-sample group ``{prefix}_meta/<key>`` (the "/" makes h5py auto-create the
        # group; arbitrary metadata keys are safe as dataset names). HDF5Source merges both back.
        for k, v in sample.metadata.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                arr = to_numpy(v)
                m_kwargs = {}
                if self.compression and getattr(arr, "ndim", 0) > 0:
                    m_kwargs["compression"] = self.compression
                self._file.create_dataset(f"{prefix}_meta/{k}", data=arr, **m_kwargs)
            else:
                try:
                    ds.attrs[k] = v
                except Exception:
                    ds.attrs[k] = str(v)

        # 3. Write Target
        if target_data is not None:
            t_kwargs = {}
            if self.compression and hasattr(target_data, "shape") and len(target_data.shape) > 0:
                t_kwargs["compression"] = self.compression

            self._file.create_dataset(f"{prefix}_target", data=target_data, **t_kwargs)

        self._counter += 1

    def flush(self) -> None:
        if self._file:
            self._file.flush()
