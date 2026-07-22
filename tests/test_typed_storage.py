"""Typed field-group storage — HDF5/Zarr/Directory round-trips, carrier guards, typed queries."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sampleflux import Image, Label, Regions, Sample, TypedSample, register_item
from sampleflux.storage.base import restore_attrs, split_attrs
from sampleflux.storage.directory import DirectorySink, DirectorySource
from sampleflux.storage.hdf5 import HDF5Sink, HDF5Source
from sampleflux.storage.query import MetadataFilterSource, scan_hdf5_metadata, scan_zarr_metadata
from sampleflux.storage.zarr import ZarrBatchSink, ZarrBatchSource, ZarrGroupSink, ZarrGroupSource


@register_item
@dataclass
class _StoreSig:
    """An externally-registered data-bearing item (the shape a domain signal item takes)."""

    data: object = None
    samplerate: float = 1.0
    mask: object = None  # an ARRAY-valued attr — exercises the attrs/<key> dataset path


def _samples() -> list:
    # Ragged across samples: different box counts, one field with an array-valued attr.
    s0 = TypedSample(
        {
            "image": Image(np.arange(12, dtype=np.float32).reshape(2, 2, 3), layout="CHW"),
            "sig": _StoreSig(np.arange(8, dtype=np.float32), samplerate=20e6, mask=np.array([1, 0, 1], dtype=np.uint8)),
            "regions": Regions(boxes=[[0, 0, 1, 1], [1, 1, 2, 2]], labels=["a", "b"], canvas=(2, 2)),
            "label": Label("drone", classes=["x", "drone"]),
        },
        roles={"regions": "target", "label": "target", "sig": "aux"},
    )
    s1 = TypedSample(
        {
            "image": Image(np.ones((2, 2, 3), dtype=np.float32)),
            "sig": _StoreSig(np.zeros(4, dtype=np.float32), samplerate=1e6, mask=np.array([0], dtype=np.uint8)),
            "regions": Regions(boxes=[[0, 0, 2, 2]], labels=["c"], canvas=(2, 2)),
            "label": Label("x", classes=["x", "drone"]),
        },
        roles={"regions": "target", "label": "target", "sig": "aux"},
    )
    return [s0, s1]


def _assert_round_trip(back: list, expect: list) -> None:
    assert len(back) == len(expect)
    for got, want in zip(back, expect):
        assert list(got.keys()) == list(want.keys())  # insertion order preserved
        assert got.roles == want.roles
        assert got["image"].layout == want["image"].layout
        assert np.array_equal(np.asarray(got["image"]), np.asarray(want["image"]))
        assert got["sig"].samplerate == want["sig"].samplerate
        assert np.array_equal(np.asarray(got["sig"].data), np.asarray(want["sig"].data))
        assert np.array_equal(np.asarray(got["sig"].mask), np.asarray(want["sig"].mask))  # array attr
        assert got["regions"].boxes == want["regions"].boxes  # ragged boxes
        assert got["regions"].canvas == want["regions"].canvas  # tuple preserved
        assert isinstance(got["regions"].canvas, tuple)
        assert got["label"].value == want["label"].value and got["label"].classes == want["label"].classes


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


class TestHDF5Typed:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "t.h5"
        sink = HDF5Sink(path=path, overwrite=True)
        with sink:
            for s in _samples():
                sink.write(s)
            sink.flush()
        source = HDF5Source(path=path)
        with source:
            assert source.is_typed and len(source) == 2
            _assert_round_trip(list(source), _samples())

    def test_carrier_guards_both_directions(self, tmp_path: Path) -> None:
        typed_path = tmp_path / "typed.h5"
        sink = HDF5Sink(path=typed_path, overwrite=True)
        with sink:
            sink.write(_samples()[0])
        appender = HDF5Sink(path=typed_path)
        with appender:
            with pytest.raises(TypeError, match="typed field-group layout"):
                appender.write(Sample(input=np.zeros(3)))

        legacy_path = tmp_path / "legacy.h5"
        legacy = HDF5Sink(path=legacy_path, overwrite=True)
        with legacy:
            legacy.write(Sample(input=np.zeros(3), metadata={"k": 1}))
        appender2 = HDF5Sink(path=legacy_path)
        with appender2:
            with pytest.raises(TypeError, match="legacy Sample layout"):
                appender2.write(_samples()[0])

    def test_legacy_path_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.h5"
        sink = HDF5Sink(path=path, overwrite=True)
        with sink:
            sink.write(Sample(input=np.arange(4, dtype=np.float32), target=1, metadata={"snr_db": 12.0}))
            sink.flush()
        source = HDF5Source(path=path)
        with source:
            assert not source.is_typed
            (back,) = list(source)
        assert isinstance(back, Sample) and back.meta["snr_db"] == 12.0


class TestZarrTyped:
    def test_group_round_trip(self, tmp_path: Path) -> None:
        path = str(tmp_path / "g.zarr")
        sink = ZarrGroupSink(path=path)
        sink.open()
        for s in _samples():
            sink.write(s)
        source = ZarrGroupSource(path=path)
        assert source.is_typed and len(source) == 2
        _assert_round_trip(list(source), _samples())

    def test_group_carrier_guard(self, tmp_path: Path) -> None:
        path = str(tmp_path / "g.zarr")
        sink = ZarrGroupSink(path=path)
        sink.open()
        sink.write(_samples()[0])
        with pytest.raises(TypeError, match="typed field-group layout"):
            sink.write(Sample(input=np.zeros(3)))

    def test_batch_typed_rows(self, tmp_path: Path) -> None:
        path = str(tmp_path / "b.zarr")
        sink = ZarrBatchSink(path=path, shape=[2, 2, 3], dtype="float32", overwrite=True)
        sink.open()
        for s in _samples():
            sink.write(s)  # appends the PRIMARY input field's payload
        source = ZarrBatchSource(path=path)
        rows = list(source)
        assert len(rows) == 2 and all(isinstance(r, TypedSample) for r in rows)
        assert isinstance(rows[0]["image"], Image) and rows[0]["image"].layout == "CHW"  # uniform template
        assert np.asarray(rows[1]["image"]).shape == (2, 2, 3)


class TestDirectoryTyped:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "dir"
        sink = DirectorySink(path=path)
        sink.open()
        for s in _samples():
            sink.write(s)
        source = DirectorySource(path=path)
        assert len(source) == 2
        _assert_round_trip(list(source), _samples())

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            len(DirectorySource(path=tmp_path / "nope"))


class TestTypedQueries:
    def test_hdf5_scan_is_nested_and_payload_free(self, tmp_path: Path) -> None:
        path = tmp_path / "q.h5"
        sink = HDF5Sink(path=path, overwrite=True)
        with sink:
            for s in _samples():
                sink.write(s)
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
        for s in _samples():
            sink.write(s)
        scans = list(scan_zarr_metadata(path))
        assert scans[1][1]["sig"]["samplerate"] == 1e6

    def test_where_field_attr_expression(self, tmp_path: Path) -> None:
        path = tmp_path / "w.h5"
        sink = HDF5Sink(path=path, overwrite=True)
        with sink:
            for s in _samples():
                sink.write(s)
        source = HDF5Source(path=path)
        source.open()
        fast = MetadataFilterSource(source=source, where="sig.samplerate > 1e7")
        assert len(fast) == 1
        (match,) = list(fast)
        assert isinstance(match, TypedSample) and match["sig"].samplerate == 20e6

    def test_full_iteration_fallback_on_typed_samples(self) -> None:
        # A plain list source (no iter_metadata protocol) of TypedSamples still filters.
        filt = MetadataFilterSource(source=_samples(), where="image.layout == 'CHW'")
        assert len(filt) == 1

    def test_missing_attr_is_non_match(self) -> None:
        filt = MetadataFilterSource(source=_samples(), where="sig.nonexistent > 0")
        assert len(filt) == 0
