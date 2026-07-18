"""Tests for the SigMF storage pair (`sampleflux.storage.sigmf`) and the metadata query layer."""

import json
from pathlib import Path

import numpy as np
import pytest

from sampleflux.sample import Sample
from sampleflux.storage.hdf5 import HDF5Sink, HDF5Source
from sampleflux.storage.query import MetadataFilterSource, SupportsMetadataScan, scan_hdf5_metadata
from sampleflux.storage.sigmf import SigMFSink, SigMFSource
from sampleflux.storage.zarr import ZarrGroupSink, ZarrGroupSource


def _iq(n: int = 16, seed: float = 1.0) -> np.ndarray:
    return (np.arange(n) * seed + 1j * np.arange(n)).astype(np.complex64)


class TestSigMFRoundTrip:
    def test_complex64_iq_round_trips(self, tmp_path: Path) -> None:
        sink = SigMFSink(path=tmp_path / "recs")
        sink.write(Sample(input=_iq(), target=None, metadata={"samplerate": 1e6, "drone": "DJI"}))
        sink.write(Sample(input=_iq(seed=2.0), target=3, metadata={"snr_db": 12.5}))
        sink.flush()

        source = SigMFSource(path=tmp_path / "recs")
        samples = list(source)
        assert len(samples) == len(source) == 2
        np.testing.assert_array_equal(samples[0].input, _iq())
        assert samples[0].input.dtype == np.complex64
        assert samples[0].meta["samplerate"] == 1e6 and samples[0].meta["drone"] == "DJI"
        assert samples[1].target == 3  # JSON-able target restored
        assert source[1].meta["snr_db"] == 12.5  # random access

    def test_float_and_int_dtypes(self, tmp_path: Path) -> None:
        for arr in (np.ones(4, dtype=np.float32), np.arange(4, dtype=np.int16)):
            sink = SigMFSink(path=tmp_path / str(arr.dtype))
            sink.write(Sample(input=arr, metadata={}))
            out = list(SigMFSource(path=tmp_path / str(arr.dtype)))[0]
            np.testing.assert_array_equal(out.input, arr)
            assert out.input.dtype == arr.dtype

    def test_unsupported_dtype_raises(self, tmp_path: Path) -> None:
        sink = SigMFSink(path=tmp_path / "bad")
        with pytest.raises(TypeError, match="core:datatype"):
            sink.write(Sample(input=np.ones(2, dtype=np.float16), metadata={}))

    def test_meta_file_shape_and_checksum(self, tmp_path: Path) -> None:
        sink = SigMFSink(path=tmp_path / "recs", checksum=True)
        sink.write(Sample(input=_iq(), metadata={"core:description": "capture"}))
        doc = json.loads(next((tmp_path / "recs").glob("*.sigmf-meta")).read_text())
        assert doc["global"]["core:datatype"] == "cf32_le"
        assert doc["global"]["core:version"] == "1.0.0"
        assert doc["global"]["core:description"] == "capture"  # core: keys ride verbatim
        assert len(doc["global"]["core:sha512"]) == 128
        assert doc["captures"] == [{"core:sample_start": 0}]

    def test_non_serializable_metadata_skipped_not_fatal(self, tmp_path: Path) -> None:
        sink = SigMFSink(path=tmp_path / "recs")
        sink.write(Sample(input=_iq(), metadata={"ok": 1, "bad": np.ones(3)}))
        out = list(SigMFSource(path=tmp_path / "recs"))[0]
        assert out.meta["ok"] == 1 and "bad" not in out.meta

    def test_sink_requires_path(self) -> None:
        with pytest.raises(ValueError, match="'path'"):
            SigMFSink().write(Sample(input=_iq()))

    def test_source_requires_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a directory"):
            len(SigMFSource(path=tmp_path / "missing"))

    def test_waivefront_vocab_hooks_round_trip(self, tmp_path: Path) -> None:
        meta = {
            "samplerate": 2e6,
            "center_freq": 2.4e9,
            "snr": "clean",
            "annotated_regions": [[100.0, 200.0, 0.5, 1.0]],
            "annotated_labels": ["wifi"],
            "drone": "DJI",
        }
        sink = SigMFSink(path=tmp_path / "wf", meta_encoder="waivefront.vocab.to_sigmf")
        sink.write(Sample(input=_iq(), metadata=meta))
        doc = json.loads(next((tmp_path / "wf").glob("*.sigmf-meta")).read_text())
        assert doc["global"]["core:sample_rate"] == 2e6
        assert doc["captures"][0]["core:frequency"] == 2.4e9
        assert doc["global"]["waivefront:snr_raw"] == "clean"  # unparseable snr kept verbatim
        annotation = doc["annotations"][0]
        assert annotation["core:freq_lower_edge"] == 100.0
        assert annotation["core:sample_start"] == int(0.5 * 2e6)
        assert annotation["core:label"] == "wifi" and annotation["waivefront:role"] == "annotated"

        out = list(SigMFSource(path=tmp_path / "wf", meta_decoder="waivefront.vocab.from_sigmf"))[0]
        assert out.meta["samplerate"] == 2e6 and out.meta["center_freq"] == 2.4e9
        assert out.meta["snr"] == "clean" and out.meta["drone"] == "DJI"
        region = out.meta["annotated_regions"][0]
        assert region[0] == 100.0 and region[2] == pytest.approx(0.5) and region[3] == pytest.approx(1.0)
        assert out.meta["annotated_labels"] == ["wifi"]


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
        assert isinstance(SigMFSource(path=tmp_path), SupportsMetadataScan)
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

    def test_sigmf_scan_and_filter(self, tmp_path: Path) -> None:
        sink = SigMFSink(path=tmp_path / "recs")
        for i in range(3):
            sink.write(Sample(input=_iq(seed=float(i + 1)), metadata={"snr_db": float(i * 10)}))
        source = SigMFSource(path=tmp_path / "recs")
        view = MetadataFilterSource(source=source, where="snr_db >= 10")
        assert len(view) == 2
        np.testing.assert_array_equal(view[0].input, _iq(seed=2.0))
