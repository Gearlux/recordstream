"""Tests for the metadata query layer (`sampleflux.storage.query`)."""

from pathlib import Path

import numpy as np
import pytest

from sampleflux.sample import Sample
from sampleflux.storage.hdf5 import HDF5Sink, HDF5Source
from sampleflux.storage.query import MetadataFilterSource, SupportsMetadataScan, scan_hdf5_metadata
from sampleflux.storage.zarr import ZarrGroupSink, ZarrGroupSource


class TestMetadataQuery:
    def _write_hdf5(self, path: Path) -> Path:
        sink = HDF5Sink(path=path)
        for i in range(4):
            sink.write(
                Sample(
                    input=np.ones(3) * i,
                    metadata={"snr_db": float(i * 5), "drone": "DJI" if i % 2 else "Parrot", "mask": np.ones((2, 2))},
                )
            )
        sink.flush()
        sink.close()
        return path

    def test_hdf5_scan_reads_no_arrays(self, tmp_path: Path) -> None:
        path = self._write_hdf5(tmp_path / "d.h5")
        scanned = list(scan_hdf5_metadata(path))
        assert len(scanned) == 4
        _key, meta = scanned[2]
        assert meta["snr_db"] == 10.0
        assert meta["mask"].startswith("<array shape=(2, 2)")  # stub, not the array

    def test_sources_implement_the_protocol(self, tmp_path: Path) -> None:
        path = self._write_hdf5(tmp_path / "d.h5")
        assert isinstance(HDF5Source(path=path), SupportsMetadataScan)
        assert isinstance(ZarrGroupSource(path=str(tmp_path / "z")), SupportsMetadataScan)

    def test_filter_source_where_expression_on_hdf5(self, tmp_path: Path) -> None:
        source = HDF5Source(path=self._write_hdf5(tmp_path / "d.h5"))
        view = MetadataFilterSource(source=source, where="snr_db >= 10")
        assert len(view) == 2
        assert [s.meta["snr_db"] for s in view] == [10.0, 15.0]
        assert view[0].meta["snr_db"] == 10.0  # random access into matches

    def test_filter_source_string_and_predicate_compose(self, tmp_path: Path) -> None:
        source = HDF5Source(path=self._write_hdf5(tmp_path / "d.h5"))
        view = MetadataFilterSource(source=source, where="drone == 'DJI'", predicate=lambda m: m["snr_db"] > 5)
        assert [s.meta["snr_db"] for s in view] == [15.0]

    def test_missing_key_is_non_matching_not_fatal(self, tmp_path: Path) -> None:
        source = HDF5Source(path=self._write_hdf5(tmp_path / "d.h5"))
        assert len(MetadataFilterSource(source=source, where="no_such_key > 1")) == 0

    def test_malformed_expression_fails_loudly(self, tmp_path: Path) -> None:
        source = HDF5Source(path=self._write_hdf5(tmp_path / "d.h5"))
        with pytest.raises(ValueError, match="failed"):
            len(MetadataFilterSource(source=source, where="snr_db +* 2"))

    def test_empty_filter_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty filter"):
            len(MetadataFilterSource(source=[Sample(1)]))

    def test_fallback_full_iteration_for_plain_sources(self) -> None:
        plain = [Sample(input=i, metadata={"v": i}) for i in range(5)]
        view = MetadataFilterSource(source=plain, where="v % 2 == 0")
        assert [s.input for s in view] == [0, 2, 4]

    def test_zarr_scan_and_filter(self, tmp_path: Path) -> None:
        sink = ZarrGroupSink(path=str(tmp_path / "z"))
        for i in range(3):
            sink.write(Sample(input=np.ones(2) * i, metadata={"v": i}))
        sink.flush()
        source = ZarrGroupSource(path=str(tmp_path / "z"))
        view = MetadataFilterSource(source=source, where="v == 1")
        assert len(view) == 1 and view[0].meta["v"] == 1
