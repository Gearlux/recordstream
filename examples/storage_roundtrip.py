"""Storage round-trips: HDF5 array-valued metadata + the Zarr sources.

Demonstrates two storage features end-to-end on synthetic data (no external
deps or services):

1. ``HDF5Sink`` / ``HDF5Source`` round-trips a ``Sample`` whose ``metadata``
   carries a 2-D array (a stand-in for a segmentation mask). Array metadata is
   stored as a dataset under a per-sample ``{prefix}_meta/`` group — it would
   otherwise overflow HDF5's attribute-size limit and be silently truncated.
2. ``ZarrGroupSink`` / ``ZarrGroupSource`` round-trips input + target + metadata,
   and ``ZarrBatchSink`` / ``ZarrBatchSource`` round-trips a stacked uniform array.
"""

import tempfile
from pathlib import Path

import numpy as np

from dataflux.core import Flux
from dataflux.sample import Sample
from dataflux.storage.hdf5 import HDF5Sink, HDF5Source
from dataflux.storage.zarr import ZarrBatchSink, ZarrBatchSource, ZarrGroupSink, ZarrGroupSource


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dataflux-storage-demo-") as tmp:
        root = Path(tmp)

        # 1. HDF5 with an array in metadata (the segmentation-mask case).
        print("--- HDF5: Sample with a 2-D mask in metadata ---")
        mask = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8)
        h5 = root / "ds.h5"
        Flux([Sample(input=np.random.randn(10), target=np.array([1]), metadata={"mask": mask, "snr": 12.0})]).to_sink(
            HDF5Sink(h5, overwrite=True)
        )
        loaded = next(iter(HDF5Source(h5)))
        print(f"  mask round-trips exact : {np.array_equal(loaded.metadata['mask'], mask)}")
        print(f"  scalar metadata kept   : snr={loaded.metadata['snr']}")

        # 2. Zarr group source — full input/target/metadata round-trip.
        print("\n--- Zarr group: ZarrGroupSink -> ZarrGroupSource ---")
        zg = root / "group.zarr"
        samples = [
            Sample(input=np.arange(5, dtype="float32"), target=np.array([1]), metadata={"id": "a"}),
            Sample(input=np.arange(3, dtype="float32"), metadata={"id": "b"}),
        ]
        Flux(samples).to_sink(ZarrGroupSink(zg, overwrite=True))
        for s in ZarrGroupSource(zg):
            tgt = None if s.target is None else s.target.tolist()
            print(f"  input={s.input.tolist()}  target={tgt}  id={s.metadata['id']!r}")

        # 3. Zarr batch source — stacked uniform array, input only.
        print("\n--- Zarr batch: ZarrBatchSink -> ZarrBatchSource ---")
        zb = root / "batch.zarr"
        Flux([Sample(input=np.full((4,), i, dtype=np.float32)) for i in range(3)]).to_sink(
            ZarrBatchSink(zb, shape=[4], overwrite=True)
        )
        print(f"  rows read back: {[int(s.input[0]) for s in ZarrBatchSource(zb)]}")

    print("\nStorage round-trips verified!")


if __name__ == "__main__":
    main()
