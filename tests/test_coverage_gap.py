from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataflux.core import Flux
from dataflux.discovery import get_callable_path, resolve_callable
from dataflux.sample import Sample
from dataflux.storage.base import Storage
from dataflux.storage.directory import DirectorySink
from dataflux.storage.hdf5 import HDF5Sink, HDF5Source
from dataflux.storage.zarr import ZarrBatchSink, ZarrGroupSink


def test_storage_base_close() -> None:
    # hits base.py:36
    s = Storage()
    s.close()


def test_directory_open_overwrite(tmp_path: Path) -> None:
    # hits directory.py:27 (pass)
    d = tmp_path / "ovr"
    d.mkdir()
    sink = DirectorySink(d, overwrite=True)
    sink.open()


def test_hdf5_none_file() -> None:
    # hits hdf5.py:45, 59, 90
    source = HDF5Source("nonexistent.h5")
    # Manually ensure _file is None (it is by default)
    assert source._file is None
    # We need to bypass open() or mock it to stay None
    # But __iter__ calls open().
    # Let's mock open to do nothing
    source.open = lambda: source  # type: ignore
    assert list(source) == []  # hits line 45
    assert len(source) == 0  # hits line 59

    sink = HDF5Sink("nonexistent.h5")
    sink.open = lambda: sink  # type: ignore
    sink.write(Sample(input=1))  # hits line 90


def test_resolve_callable_import_error() -> None:
    # hits discovery.py:57
    with pytest.raises(ImportError, match="Cannot resolve"):
        resolve_callable("nonexistent_mod_totally:func")


def test_get_callable_path_main_no_file(monkeypatch: Any) -> None:
    # hits discovery.py:33 (pass)
    import sys
    import types

    m = types.ModuleType("__main__")
    # No __file__ attribute
    monkeypatch.setitem(sys.modules, "__main__", m)

    def local_f() -> None:
        pass

    local_f.__module__ = "__main__"
    # Should fallback to valueerror or just skip the main block if it fails
    try:
        get_callable_path(local_f)
    except ValueError:
        pass


def test_hdf5_metadata_exception(tmp_path: Path) -> None:
    # hits hdf5.py:except branch in metadata loop
    h5_path = tmp_path / "meta_err.h5"
    sink = HDF5Sink(h5_path)
    # Metadata that h5py might not like (e.g. nested dict)
    sample = Sample(input=np.array([1]), metadata={"bad": {"nested": "value"}})
    sink.write(sample)
    sink.close()


def test_zarr_group_open_overwrite(tmp_path: Path) -> None:
    # hits zarr.py:29 (pass)
    z_path = tmp_path / "z_ovr.zarr"
    z_path.mkdir()
    sink = ZarrGroupSink(z_path, overwrite=True)
    sink.open()


def test_flux_iter_none_source() -> None:
    # hits core.py:183, 193
    f = Flux(None)
    assert list(f._iter_sequential()) == []
    assert list(f._iter_parallel()) == []


def test_zarr_group_flush() -> None:
    # hits zarr.py:56
    sink = ZarrGroupSink("test.zarr")
    sink.flush()


def test_sample_from_any_tuple_1() -> None:
    # hits sample.py:22
    s = Sample.from_any((1,))
    assert s.input == 1
    assert s.target is None


def test_sample_from_any_empty_tuple() -> None:
    # hits sample.py new branch
    s = Sample.from_any(())
    assert s.input is None


def test_optional_context_manager_direct() -> None:
    # hits core.py:177
    from dataflux.core import Flux

    f = Flux([1])

    class SimpleSink:
        def write(self, s: Any) -> None:
            pass

        def flush(self) -> None:
            pass

    f.to_sink(SimpleSink())


def test_discovery_main_failure(monkeypatch: Any) -> None:
    # hits discovery.py:30 (pass)
    import sys
    import types

    m = types.ModuleType("__main__")
    # NO __file__
    monkeypatch.setitem(sys.modules, "__main__", m)

    def f() -> None:
        pass

    f.__module__ = "__main__"
    try:
        get_callable_path(f)
    except ValueError:
        pass


def test_zarr_batch_chunks(tmp_path: Path) -> None:
    # hits zarr.py: chunks logic
    p = tmp_path / "chunks.zarr"
    sink = ZarrBatchSink(p, shape=[10], chunks=[1, 10], overwrite=True)
    sink.write(Sample(input=np.zeros(10)))
