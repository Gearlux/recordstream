from pathlib import Path
from typing import cast

import numpy as np
import torch

from dataflux.core import Flux
from dataflux.sample import Sample
from dataflux.storage.directory import DirectorySink
from dataflux.storage.hdf5 import HDF5Sink, HDF5Source
from dataflux.storage.zarr import ZarrBatchSink, ZarrGroupSink


def test_hdf5_storage(tmp_path: Path) -> None:
    h5_path = tmp_path / "test.h5"
    samples = [
        Sample(input=torch.randn(10), target=torch.tensor([1])),
        Sample(input=torch.randn(10), target=torch.tensor([0])),
    ]

    # Write
    sink = HDF5Sink(h5_path, overwrite=True)
    Flux(samples).to_sink(sink)
    sink.close()

    # Read
    source = HDF5Source(h5_path)
    loaded = list(source)
    assert len(loaded) == 2
    assert torch.allclose(loaded[0].input, samples[0].input)
    assert loaded[0].target == samples[0].target
    assert len(source) == 2
    source.close()


def test_zarr_group_storage(tmp_path: Path) -> None:
    zarr_path = tmp_path / "test.zarr"
    samples = [
        Sample(input=np.random.randn(5), metadata={"id": "a"}),
        Sample(input=np.random.randn(10), metadata={"id": "b"}),
    ]

    sink = ZarrGroupSink(zarr_path, overwrite=True)
    Flux(samples).to_sink(sink)

    # Verification (ZarrGroupSink doesn't have a Source yet, but we check files)
    assert zarr_path.exists()
    assert (zarr_path / "sample_000000").exists()
    assert (zarr_path / "sample_000001").exists()


def test_zarr_batch_storage(tmp_path: Path) -> None:
    zarr_path = tmp_path / "batch.zarr"
    samples = [Sample(input=np.ones((10, 10), dtype=np.float32)) for _ in range(5)]

    sink = ZarrBatchSink(zarr_path, shape=[10, 10], overwrite=True)
    Flux(samples).to_sink(sink)

    # Check if data was written
    import zarr

    z = zarr.open_array(store=f"{zarr_path}/data", mode="r")
    assert z.shape == (5, 10, 10)
    assert np.all(z[:] == 1.0)


def test_directory_storage(tmp_path: Path) -> None:
    dir_path = tmp_path / "out_dir"
    samples = [
        Sample(input=np.array([1, 2]), metadata={"name": "first"}),
        Sample(input=np.array([3, 4]), metadata={"name": "second"}),
    ]

    sink = DirectorySink(dir_path, overwrite=True)
    Flux(samples).to_sink(sink)


def test_directory_storage_separate(tmp_path: Path) -> None:
    dir_path = tmp_path / "out_dir_sep"
    samples = [
        Sample(input=np.array([1, 2]), target=np.array([0])),
    ]

    # use_npz=False hits lines 52-54
    sink = DirectorySink(dir_path, overwrite=True, use_npz=False)
    Flux(samples).to_sink(sink)

    assert (dir_path / "000000" / "data.npy").exists()
    assert (dir_path / "000000" / "target.npy").exists()


def test_hdf5_to_numpy_direct() -> None:
    from dataflux.storage.hdf5 import to_numpy

    # Hits line 19
    assert to_numpy(123) == 123


def test_hdf5_flush(tmp_path: Path) -> None:
    h5_path = tmp_path / "flush.h5"
    sink = HDF5Sink(h5_path)
    sink.open()
    sink.flush()  # Hits lines 107-110
    sink.close()


def test_hdf5_overwrite(tmp_path: Path) -> None:
    h5_path = tmp_path / "over.h5"
    s1 = [Sample(input=np.array([1]))]
    s2 = [Sample(input=np.array([2]))]

    # 1. Write first
    sink1 = HDF5Sink(h5_path, overwrite=True)
    Flux(s1).to_sink(sink1)
    sink1.close()

    # 2. Overwrite
    sink2 = HDF5Sink(h5_path, overwrite=True)
    Flux(s2).to_sink(sink2)
    sink2.close()

    # 3. Verify only s2 exists
    source = HDF5Source(h5_path)
    loaded = list(source)
    assert len(loaded) == 1
    assert loaded[0].input == 2


def test_zarr_group_with_target(tmp_path: Path) -> None:
    zarr_path = tmp_path / "target.zarr"
    samples = [Sample(input=np.array([1]), target=np.array([0]))]

    sink = ZarrGroupSink(zarr_path, overwrite=True)
    Flux(samples).to_sink(sink)

    import zarr

    z = zarr.open_group(str(zarr_path), mode="r")
    grp = cast(zarr.Group, z["sample_000000"])
    assert "target" in grp
