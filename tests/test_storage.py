from pathlib import Path
from typing import cast

import confluid
import numpy as np
import torch

from dataflux.core import Flux
from dataflux.sample import Sample
from dataflux.storage.directory import DirectorySink
from dataflux.storage.hdf5 import HDF5Sink, HDF5Source
from dataflux.storage.zarr import ZarrBatchSink, ZarrBatchSource, ZarrGroupSink, ZarrGroupSource


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


def test_hdf5_array_metadata_roundtrip(tmp_path: Path) -> None:
    """Array-valued metadata (e.g. a segmentation mask) survives the HDF5 round-trip.

    Regression test: writing such a value as an HDF5 *attribute* overflows the attribute
    size limit and the old str() fallback silently truncated it. It must now be stored as a
    dataset under the per-sample ``{prefix}_meta/`` group and read back byte-exact, while
    scalar metadata continues to round-trip via attributes.
    """
    h5_path = tmp_path / "meta.h5"
    mask = np.random.randint(0, 2, size=(128, 128), dtype=np.uint8)
    samples = [
        Sample(
            input=torch.randn(10),
            target=torch.tensor([1]),
            metadata={"id": "a", "samplerate": 100.0, "mask": mask},
        ),
        Sample(input=torch.randn(10), metadata={"id": "b"}),
    ]

    sink = HDF5Sink(h5_path, overwrite=True)
    Flux(samples).to_sink(sink)
    sink.close()

    source = HDF5Source(h5_path)
    loaded = list(source)
    source.close()

    assert len(loaded) == 2
    # Scalar metadata round-trips via attributes.
    assert loaded[0].metadata["id"] == "a"
    assert loaded[0].metadata["samplerate"] == 100.0
    assert loaded[1].metadata["id"] == "b"
    # Array metadata round-trips exactly (no truncation).
    assert np.array_equal(loaded[0].metadata["mask"], mask)
    # The sample without array metadata has no spurious mask key.
    assert "mask" not in loaded[1].metadata


def test_hdf5_array_metadata_no_compression(tmp_path: Path) -> None:
    """Array metadata is stored as a dataset even when compression is disabled."""
    h5_path = tmp_path / "meta_nc.h5"
    mask = np.arange(16, dtype=np.uint8).reshape(4, 4)

    sink = HDF5Sink(h5_path, compression=None, overwrite=True)
    Flux([Sample(input=np.array([1.0]), metadata={"mask": mask})]).to_sink(sink)
    sink.close()

    loaded = list(HDF5Source(h5_path))
    assert np.array_equal(loaded[0].metadata["mask"], mask)


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


def test_zarr_group_source_roundtrip(tmp_path: Path) -> None:
    zarr_path = tmp_path / "group_rt.zarr"
    samples = [
        Sample(input=np.arange(5, dtype="float32"), target=np.array([1]), metadata={"id": "a"}),
        Sample(input=np.arange(3, dtype="float32"), metadata={"id": "b"}),
    ]

    Flux(samples).to_sink(ZarrGroupSink(zarr_path, overwrite=True))

    source = ZarrGroupSource(zarr_path)
    loaded = list(source)
    assert len(loaded) == 2
    assert len(source) == 2
    # Input is returned as a tensor (matches HDF5Source); order matches write order.
    assert torch.equal(loaded[0].input, torch.arange(5, dtype=torch.float32))
    assert torch.equal(loaded[1].input, torch.arange(3, dtype=torch.float32))
    # Target round-trips; absent target stays None.
    assert np.array_equal(loaded[0].target, np.array([1]))
    assert loaded[1].target is None
    # Metadata round-trips via group attributes.
    assert loaded[0].metadata["id"] == "a"
    assert loaded[1].metadata["id"] == "b"
    source.close()


def test_zarr_group_sink_handles_torch_tensors(tmp_path: Path) -> None:
    """ZarrGroupSink writes torch-tensor input/target (e.g. streamed from HDF5Source).

    Regression: zarr's ``create_array`` can't read a torch tensor's dtype, so the sink
    must convert via ``to_numpy`` first — otherwise a torch-tensor sample raises
    ``TypeError: Cannot interpret 'torch.float32' as a data type``.
    """
    zarr_path = tmp_path / "torch.zarr"
    samples = [Sample(input=torch.arange(5, dtype=torch.float32), target=torch.tensor([1]))]
    Flux(samples).to_sink(ZarrGroupSink(zarr_path, overwrite=True))

    loaded = list(ZarrGroupSource(zarr_path))
    assert len(loaded) == 1
    assert torch.equal(loaded[0].input, torch.arange(5, dtype=torch.float32))
    assert np.array_equal(loaded[0].target, np.array([1]))


def test_zarr_batch_source_roundtrip(tmp_path: Path) -> None:
    zarr_path = tmp_path / "batch_rt.zarr"
    samples = [Sample(input=np.full((4,), i, dtype=np.float32)) for i in range(3)]

    Flux(samples).to_sink(ZarrBatchSink(zarr_path, shape=[4], overwrite=True))

    source = ZarrBatchSource(zarr_path)
    loaded = list(source)
    assert len(loaded) == 3
    assert len(source) == 3
    # Batch sink stores input only — one Sample per row of the leading axis.
    assert [int(s.input[0]) for s in loaded] == [0, 1, 2]
    assert all(s.target is None for s in loaded)
    source.close()


def test_zarr_sources_configurable_roundtrip(tmp_path: Path) -> None:
    group_src = ZarrGroupSource(tmp_path / "g.zarr", target_key="label")
    restored_group = confluid.load(confluid.dump(group_src))
    assert isinstance(restored_group, ZarrGroupSource)
    assert restored_group.path == group_src.path
    assert restored_group.target_key == "label"

    batch_src = ZarrBatchSource(tmp_path / "b.zarr")
    restored_batch = confluid.load(confluid.dump(batch_src))
    assert isinstance(restored_batch, ZarrBatchSource)
    assert restored_batch.path == batch_src.path
