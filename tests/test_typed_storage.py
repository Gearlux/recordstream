"""Record key-group storage — HDF5/Zarr/Directory round-trips, format-tag guard, metadata queries."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from sampleflux import Image, Label, Regions, register_item
from sampleflux.storage.base import TYPED_FORMAT, require_record_format, restore_attrs, split_attrs
from sampleflux.storage.directory import DirectorySink, DirectorySource
from sampleflux.storage.hdf5 import HDF5Sink, HDF5Source
from sampleflux.storage.query import MetadataFilterSource, record_metadata, scan_hdf5_metadata, scan_zarr_metadata
from sampleflux.storage.zarr import ZarrBatchSink, ZarrBatchSource, ZarrGroupSink, ZarrGroupSource


@register_item
@dataclass
class _StoreSig:
    """An externally-registered data-bearing item (the shape a domain signal item takes)."""

    data: object = None
    samplerate: float = 1.0
    mask: object = None  # an ARRAY-valued attr — exercises the attrs/<key> dataset path


def _records() -> list:
    # Ragged across records: different box counts, one value with an array-valued attr,
    # plus PLAIN entries (scalar / string / bare array) riding the "plain" type tag.
    r0 = {
        "image": Image(np.arange(12, dtype=np.float32).reshape(2, 2, 3), layout="CHW"),
        "sig": _StoreSig(np.arange(8, dtype=np.float32), samplerate=20e6, mask=np.array([1, 0, 1], dtype=np.uint8)),
        "regions": Regions(boxes=[[0, 0, 1, 1], [1, 1, 2, 2]], labels=["a", "b"], canvas=(2, 2)),
        "label": Label("drone", classes=["x", "drone"]),
        "gain_db": -3.0,
        "source_file": "a.iq",
        "window": np.hanning(4),
    }
    r1 = {
        "image": Image(np.ones((2, 2, 3), dtype=np.float32)),
        "sig": _StoreSig(np.zeros(4, dtype=np.float32), samplerate=1e6, mask=np.array([0], dtype=np.uint8)),
        "regions": Regions(boxes=[[0, 0, 2, 2]], labels=["c"], canvas=(2, 2)),
        "label": Label("x", classes=["x", "drone"]),
        "gain_db": 1.5,
        "source_file": "b.iq",
        "window": np.hanning(4),
    }
    return [r0, r1]


def _assert_round_trip(back: list, expect: list) -> None:
    assert len(back) == len(expect)
    for got, want in zip(back, expect):
        assert list(got.keys()) == list(want.keys())  # insertion order preserved
        assert got["image"].layout == want["image"].layout
        assert np.array_equal(np.asarray(got["image"]), np.asarray(want["image"]))
        assert got["sig"].samplerate == want["sig"].samplerate
        assert np.array_equal(np.asarray(got["sig"].data), np.asarray(want["sig"].data))
        assert np.array_equal(np.asarray(got["sig"].mask), np.asarray(want["sig"].mask))  # array attr
        assert got["regions"].boxes == want["regions"].boxes  # ragged boxes
        assert got["regions"].canvas == want["regions"].canvas  # tuple preserved
        assert isinstance(got["regions"].canvas, tuple)
        assert got["label"].value == want["label"].value and got["label"].classes == want["label"].classes
        assert got["gain_db"] == want["gain_db"]  # plain scalar round-trips
        assert got["source_file"] == want["source_file"]  # plain string round-trips
        assert np.allclose(np.asarray(got["window"]), want["window"])  # plain bare-array round-trips


class TestAttrWireFormat:
    def test_split_restore_round_trip(self) -> None:
        attrs = {
            "s": "text",
            "i": 3,
            "f": 1.5,
            "b": True,
            "none": None,
            "tup": (2, 3),
            "nested": {"a": [1, (2, 3)]},
            "arr": np.arange(4),
        }
        plain, arrays = split_attrs(attrs)
        assert list(arrays) == ["arr"]
        back = restore_attrs(plain, arrays)
        assert back["s"] == "text" and back["i"] == 3 and back["f"] == 1.5 and back["b"] is True
        assert back["none"] is None
        assert back["tup"] == (2, 3) and isinstance(back["tup"], tuple)
        assert back["nested"] == {"a": [1, (2, 3)]}
        assert np.array_equal(back["arr"], np.arange(4))

    def test_numpy_scalars_become_python(self) -> None:
        plain, _ = split_attrs({"x": np.float32(2.5)})
        assert restore_attrs(plain, {})["x"] == 2.5


class TestFormatGuard:
    def test_require_record_format_accepts_current_tag(self) -> None:
        require_record_format(TYPED_FORMAT, "test")  # no raise

    @pytest.mark.parametrize("found", [None, "typedsample-v1", "bogus"])
    def test_require_record_format_rejects_everything_else(self, found: object) -> None:
        with pytest.raises(ValueError, match="typedrecord-v1"):
            require_record_format(found, "test")

    def test_hdf5_source_rejects_old_typedsample_tag(self, tmp_path: Path) -> None:
        # NO backward compat: a pre-record-model store must fail loudly with the clear error.
        path = tmp_path / "old.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["sampleflux_format"] = "typedsample-v1"
        with pytest.raises(ValueError, match="typedrecord-v1"):
            HDF5Source(path=path).open()

    def test_hdf5_sink_rejects_appending_to_old_tag(self, tmp_path: Path) -> None:
        path = tmp_path / "old.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["sampleflux_format"] = "typedsample-v1"
        sink = HDF5Sink(path=path)
        with sink:
            with pytest.raises(ValueError, match="typedrecord-v1"):
                sink.write(_records()[0])

    def test_zarr_group_source_rejects_old_tag(self, tmp_path: Path) -> None:
        import zarr

        path = str(tmp_path / "old.zarr")
        root = zarr.open_group(path, mode="a")
        root.attrs["sampleflux_format"] = "typedsample-v1"
        with pytest.raises(ValueError, match="typedrecord-v1"):
            ZarrGroupSource(path=path).open()


class TestHDF5:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "t.h5"
        sink = HDF5Sink(path=path, overwrite=True)
        with sink:
            for r in _records():
                sink.write(r)
            sink.flush()
        source = HDF5Source(path=path)
        with source:
            assert len(source) == 2
            _assert_round_trip(list(source), _records())

    def test_format_tag_stamped(self, tmp_path: Path) -> None:
        path = tmp_path / "t.h5"
        with HDF5Sink(path=path, overwrite=True) as sink:
            sink.write(_records()[0])
        with h5py.File(path, "r") as handle:
            assert handle.attrs["sampleflux_format"] == TYPED_FORMAT

    def test_non_dict_write_raises(self, tmp_path: Path) -> None:
        # The sink only accepts a record dict; a bare array is rejected loudly.
        sink = HDF5Sink(path=tmp_path / "typed.h5", overwrite=True)
        with sink:
            sink.write(_records()[0])
            with pytest.raises(TypeError, match="expected a record dict"):
                sink.write(np.zeros(3))


class TestZarr:
    def test_group_round_trip(self, tmp_path: Path) -> None:
        path = str(tmp_path / "g.zarr")
        sink = ZarrGroupSink(path=path)
        sink.open()
        for r in _records():
            sink.write(r)
        source = ZarrGroupSource(path=path)
        assert len(source) == 2
        _assert_round_trip(list(source), _records())

    def test_group_non_dict_write_raises(self, tmp_path: Path) -> None:
        path = str(tmp_path / "g.zarr")
        sink = ZarrGroupSink(path=path)
        sink.open()
        sink.write(_records()[0])
        with pytest.raises(TypeError, match="expected a record dict"):
            sink.write(np.zeros(3))

    def test_batch_typed_rows(self, tmp_path: Path) -> None:
        path = str(tmp_path / "b.zarr")
        sink = ZarrBatchSink(path=path, shape=[2, 2, 3], dtype="float32", overwrite=True)
        sink.open()
        for r in _records():
            sink.write(r)  # appends the FIRST record entry's payload
        source = ZarrBatchSource(path=path)
        rows = list(source)
        assert len(rows) == 2 and all(isinstance(r, dict) for r in rows)
        assert isinstance(rows[0]["image"], Image) and rows[0]["image"].layout == "CHW"  # uniform template
        assert np.asarray(rows[1]["image"]).shape == (2, 2, 3)


class TestDirectory:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "dir"
        sink = DirectorySink(path=path)
        sink.open()
        for r in _records():
            sink.write(r)
        source = DirectorySource(path=path)
        assert len(source) == 2
        _assert_round_trip(list(source), _records())

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            len(DirectorySource(path=tmp_path / "nope"))

    def test_non_dict_write_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="expected a record dict"):
            DirectorySink(path=tmp_path / "dir").write(np.zeros(3))


class TestMetadataQueries:
    def test_record_metadata_shape(self) -> None:
        meta = record_metadata(_records()[0])
        assert meta["image"] == {"layout": "CHW"}
        assert meta["sig"]["samplerate"] == 20e6
        assert meta["gain_db"] == {"value": -3.0}  # plain scalar -> its "value" attr
        assert meta["source_file"] == {"value": "a.iq"}
        assert meta["window"] == {}  # a plain ARRAY payload contributes no scalar metadata

    def test_hdf5_scan_is_nested_and_payload_free(self, tmp_path: Path) -> None:
        path = tmp_path / "q.h5"
        sink = HDF5Sink(path=path, overwrite=True)
        with sink:
            for r in _records():
                sink.write(r)
        scans = list(scan_hdf5_metadata(path))
        assert len(scans) == 2
        _, meta = scans[0]
        assert meta["sig"]["samplerate"] == 20e6  # plain attr decoded
        assert meta["image"]["layout"] == "CHW"
        assert "shape" in str(meta["sig"]["mask"])  # array attr is a STUB, not the array

    def test_zarr_scan_nested(self, tmp_path: Path) -> None:
        path = str(tmp_path / "q.zarr")
        sink = ZarrGroupSink(path=path)
        sink.open()
        for r in _records():
            sink.write(r)
        scans = list(scan_zarr_metadata(path))
        assert scans[1][1]["sig"]["samplerate"] == 1e6

    def test_where_key_attr_expression(self, tmp_path: Path) -> None:
        path = tmp_path / "w.h5"
        sink = HDF5Sink(path=path, overwrite=True)
        with sink:
            for r in _records():
                sink.write(r)
        source = HDF5Source(path=path)
        source.open()
        fast = MetadataFilterSource(source=source, where="sig.samplerate > 1e7")
        assert len(fast) == 1
        (match,) = list(fast)
        assert isinstance(match, dict) and match["sig"].samplerate == 20e6

    def test_full_iteration_fallback_on_records(self) -> None:
        # A plain list source (no iter_metadata protocol) of record dicts still filters,
        # via record_metadata — plain scalars addressable as <key>.value.
        filt = MetadataFilterSource(source=_records(), where="gain_db.value < 0")
        assert len(filt) == 1
        (match,) = list(filt)
        assert match["gain_db"] == -3.0

    def test_missing_attr_is_non_match(self) -> None:
        filt = MetadataFilterSource(source=_records(), where="sig.nonexistent > 0")
        assert len(filt) == 0
