from pathlib import Path
from typing import Iterator, List, Optional, Union, cast

import confluid
import numpy as np
import torch
import zarr

from dataflux.sample import Sample
from dataflux.storage.base import DataSink, DataSource, Storage, to_numpy


@confluid.configurable
class ZarrGroupSink(Storage, DataSink):
    """
    Stores each sample as a unique array within a Zarr group.
    Supports variable lengths while keeping data in a single bundle.
    """

    def __init__(self, path: Union[str, Path] = "", overwrite: bool = False) -> None:
        # Lazy / zero-arg: store config only; the group is opened lazily in open().
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
        if self._root is None:
            raise RuntimeError("Zarr group not open")
        # Use require_group to handle existing nodes safely
        name = f"sample_{self._counter:06d}"
        grp = self._root.require_group(name)

        # 1. Save data and target. create_array needs a numpy array (it can't read a
        # torch tensor's dtype); to_numpy detaches/moves to CPU. overwrite=True replaces
        # an existing node, so no manual delete is needed on re-write.
        grp.create_array("data", data=to_numpy(sample.input), overwrite=True)

        if sample.target is not None:
            grp.create_array("target", data=to_numpy(sample.target), overwrite=True)

        # 2. Save metadata as Zarr attributes (.zattrs)
        if sample.metadata:
            grp.attrs.update(sample.metadata)

        self._counter += 1

    def flush(self) -> None:
        pass  # pragma: no cover


@confluid.configurable
class ZarrGroupSource(Storage, DataSource):
    """Read samples written by :class:`ZarrGroupSink` (one Zarr group per sample).

    Mirrors the group sink's layout: each ``sample_NNNNNN`` subgroup carries a
    ``data`` array, an optional ``target`` array, and the sample metadata as
    group attributes (``.zattrs``). Groups are iterated in sorted name order so
    the read order matches the write order.

    Args:
        path: Path to the Zarr group written by ZarrGroupSink.
        sample_key: Name of the per-sample array holding ``Sample.input``.
        target_key: Name of the per-sample array holding ``Sample.target`` (absent when the sample had no target).
    """

    def __init__(
        self,
        path: Union[str, Path] = "",
        sample_key: str = "data",
        target_key: str = "target",
    ) -> None:
        # Lazy / zero-arg: store config only; the group is opened lazily in open().
        self.path = str(path)
        self.sample_key = sample_key
        self.target_key = target_key
        self._root: Optional[zarr.Group] = None

    def open(self) -> "ZarrGroupSource":
        if self._root is None:
            self._root = zarr.open_group(self.path, mode="r")
        return self

    def close(self) -> None:
        self._root = None

    def __iter__(self) -> Iterator[Sample]:
        self.open()
        if self._root is None:
            return
        for name in sorted(self._root.group_keys()):
            grp = cast(zarr.Group, self._root[name])
            data = cast(zarr.Array, grp[self.sample_key])[:]
            target = cast(zarr.Array, grp[self.target_key])[:] if self.target_key in grp else None
            yield Sample(input=torch.from_numpy(np.asarray(data)), target=target, metadata=dict(grp.attrs))

    def __len__(self) -> int:
        self.open()
        if self._root is None:
            return 0
        return len(list(self._root.group_keys()))


@confluid.configurable
class ZarrBatchSink(Storage, DataSink):
    """
    Optimized for uniform data. Appends samples into a single large Zarr array.
    """

    def __init__(
        self,
        path: Union[str, Path] = "",
        shape: Optional[List[int]] = None,
        dtype: str = "float32",
        chunks: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> None:
        # Lazy / zero-arg: store config only; the array is created lazily in open() (an unset
        # path / shape surfaces there).
        self.path = str(path)
        self.shape = tuple(shape) if shape else ()
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
        if self._data_arr is None:
            raise RuntimeError("Zarr array not open")
        # Append to the primary array
        # Zarr handles the resizing and chunking internally
        self._data_arr.append([sample.input], axis=0)

        # Note: Handling metadata in a single-array sink requires
        # a separate attribute list or sidecar file.
        # For simplicity, we attach to the array attributes.
        self._counter += 1

    def flush(self) -> None:
        pass  # pragma: no cover


@confluid.configurable
class ZarrBatchSource(Storage, DataSource):
    """Read samples written by :class:`ZarrBatchSink` (one stacked array).

    The batch sink appends every sample's input along axis 0 of a single
    ``data`` array and stores no per-sample target or metadata, so this source
    yields input-only :class:`~dataflux.sample.Sample` objects — one per row of
    the leading axis.

    Args:
        path: Path to the Zarr store written by ZarrBatchSink (the directory holding the ``data`` array).
    """

    def __init__(self, path: Union[str, Path] = "") -> None:
        # Lazy / zero-arg: store config only; the array is opened lazily in open().
        self.path = str(path)
        self._data_arr: Optional[zarr.Array] = None

    def open(self) -> "ZarrBatchSource":
        if self._data_arr is None:
            self._data_arr = zarr.open_array(store=f"{self.path}/data", mode="r")
        return self

    def close(self) -> None:
        self._data_arr = None

    def __iter__(self) -> Iterator[Sample]:
        self.open()
        if self._data_arr is None:
            return
        for i in range(self._data_arr.shape[0]):
            yield Sample(input=torch.from_numpy(np.asarray(self._data_arr[i])))

    def __len__(self) -> int:
        self.open()
        if self._data_arr is None:
            return 0
        return int(self._data_arr.shape[0])
