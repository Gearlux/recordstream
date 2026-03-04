from pathlib import Path
from typing import List, Optional, Union

import confluid  # type: ignore[import-not-found]
import zarr

from dataflux.sample import Sample
from dataflux.storage.base import DataSink, Storage


@confluid.configurable
class ZarrGroupSink(Storage, DataSink):
    """
    Stores each sample as a unique array within a Zarr group.
    Supports variable lengths while keeping data in a single bundle.
    """

    def __init__(self, path: Union[str, Path], overwrite: bool = False) -> None:
        self.path = str(path)
        self.overwrite = overwrite
        self._root: Optional[zarr.Group] = None
        self._counter = 0

    def open(self) -> "ZarrGroupSink":
        if self._root is None:
            self._root = zarr.open_group(self.path, mode="a")
            if self.overwrite:
                # In a real app, we'd clear the group
                pass
        return self

    def write(self, sample: Sample) -> None:
        self.open()
        # Use require_group to handle existing nodes safely
        name = f"sample_{self._counter:06d}"
        grp = self._root.require_group(name)  # type: ignore

        # 1. Save data and target (Explicit shape/dtype for Zarr v3)
        # Use require_dataset or delete if exists
        if "data" in grp:
            del grp["data"]
        grp.create_dataset("data", data=sample.input, shape=sample.input.shape, dtype=sample.input.dtype)

        if sample.target is not None:
            if "target" in grp:
                del grp["target"]
            grp.create_dataset("target", data=sample.target, shape=sample.target.shape, dtype=sample.target.dtype)

        # 2. Save metadata as Zarr attributes (.zattrs)
        if sample.metadata:
            grp.attrs.update(sample.metadata)

        self._counter += 1

    def flush(self) -> None:
        pass


@confluid.configurable
class ZarrBatchSink(Storage, DataSink):
    """
    Optimized for uniform data. Appends samples into a single large Zarr array.
    """

    def __init__(
        self,
        path: Union[str, Path],
        shape: List[int],
        dtype: str = "float32",
        chunks: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> None:
        self.path = str(path)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.chunks = tuple(chunks) if chunks else None
        self.overwrite = overwrite
        self._data_arr: Optional[zarr.Array] = None
        self._target_arr: Optional[zarr.Array] = None
        self._counter = 0

    def open(self) -> "ZarrBatchSink":
        if self._data_arr is None:
            # We create a resizable array (unlimited along first dimension)
            self._data_arr = zarr.open_array(
                store=f"{self.path}/data",
                mode="a" if not self.overwrite else "w",
                shape=(0,) + self.shape,
                chunks=(1,) + self.shape if not self.chunks else self.chunks,
                dtype=self.dtype,
            )
        return self

    def write(self, sample: Sample) -> None:
        self.open()
        # Append to the primary array
        # Zarr handles the resizing and chunking internally
        self._data_arr.append([sample.input], axis=0)  # type: ignore

        # Note: Handling metadata in a single-array sink requires
        # a separate attribute list or sidecar file.
        # For simplicity, we attach to the array attributes.
        self._counter += 1

    def flush(self) -> None:
        pass
