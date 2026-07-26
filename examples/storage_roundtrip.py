"""Storage round-trip: every sink has a matching source, one codec serializes everything.

The storage tour of the record model (`typedrecord-v1` — the key-group layout):

1. Write the SAME records through all three sink/source pairs — `HDF5Sink`↔`HDF5Source`,
   `ZarrGroupSink`↔`ZarrGroupSource`, `DirectorySink`↔`DirectorySource` — and read them back
   IDENTICAL: typed values keep their type AND their metadata attrs (`Image.layout`,
   `Label.classes`), and a plain scalar entry (`snr_db`) rides the same layout via the
   ``"plain"`` codec tag.
2. Query a store's metadata WITHOUT loading a single array: `MetadataFilterSource` over the
   HDF5 store filters on the scalar entry (`snr_db.value > 10`) through the
   `SupportsMetadataScan` protocol — attrs only, payloads untouched until iteration.

Everything is written to a temp directory — examples leave no artifacts behind.

Standalone, zero-arg, exit 0 (CI runs every ``examples/*.py``).
"""

import tempfile
from pathlib import Path

import numpy as np

from sampleflux import Image, Label, Record
from sampleflux.storage.directory import DirectorySink, DirectorySource
from sampleflux.storage.hdf5 import HDF5Sink, HDF5Source
from sampleflux.storage.query import MetadataFilterSource
from sampleflux.storage.zarr import ZarrGroupSink, ZarrGroupSource


def make_records(n: int = 4) -> list:
    rng = np.random.default_rng(0)
    return [
        {
            "image": Image(rng.random((8, 10, 3)).astype(np.float32)),  # typed: knows its layout
            "class": Label(i % 2, classes=["noise", "drone"]),  # typed: carries its vocabulary
            "snr_db": float(5 * i),  # plain scalar — just another key
        }
        for i in range(n)
    ]


def assert_roundtrip(original: Record, restored: Record) -> None:
    assert set(restored) == set(original)
    assert isinstance(restored["image"], Image) and restored["image"].layout == "HWC"
    assert np.array_equal(np.asarray(restored["image"]), np.asarray(original["image"]))
    assert isinstance(restored["class"], Label)
    assert restored["class"].value == original["class"].value
    assert restored["class"].classes == original["class"].classes
    assert restored["snr_db"] == original["snr_db"]  # the "plain" codec tag


def main() -> None:
    records = make_records()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        pairs = [
            (HDF5Sink(path=work / "store.h5"), HDF5Source(path=work / "store.h5")),
            (ZarrGroupSink(path=work / "store.zarr"), ZarrGroupSource(path=work / "store.zarr")),
            (DirectorySink(path=work / "store_dir"), DirectorySource(path=work / "store_dir")),
        ]

        # 1. Every sink has a matching source — write, read back, byte-identical typed records.
        for sink, source in pairs:
            with sink:
                for record in records:
                    sink.write(record)
                sink.flush()
            with source:
                restored = list(source)
            assert len(restored) == len(records)
            for original, back in zip(records, restored):
                assert_roundtrip(original, back)
            print(
                f"  {type(sink).__name__:14} -> {type(source).__name__:16} round-trip OK "
                f"({len(restored)} records; Image+attrs, Label+vocab, plain snr_db)"
            )

        # 2. Metadata-only querying: filter the HDF5 store on the plain scalar WITHOUT
        #    loading arrays (the SupportsMetadataScan protocol reads attrs only; a plain
        #    scalar is addressable as <key>.value).
        filtered = MetadataFilterSource(source=HDF5Source(path=work / "store.h5"), where="snr_db.value > 10")
        hits = list(filtered)
        assert [r["snr_db"] for r in hits] == [15.0]
        print(f"  MetadataFilterSource(where='snr_db.value > 10') -> {len(hits)} record (snr_db=15.0)")

    print("OK")


if __name__ == "__main__":
    main()
