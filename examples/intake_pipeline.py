"""IntakeSource adapter demo.

Shows how the generic ``dataflux.storage.intake.IntakeSource`` adapter consumes
any intake DataSource — using a tiny in-process driver here so the example needs
no external data or services.

Demonstrates:
1. Wrapping an ``intake.source.base.DataSource`` that yields ``xarray.DataArray``
   partitions.
2. ``target_attr`` lifting the per-partition label from xarray attrs into
   ``Sample.target``.
3. Round-tripping a Flux pipeline through ``HDF5Sink`` → ``HDF5Source`` to show
   that the metadata flowing in via xarray attrs survives storage.
"""

import tempfile
from pathlib import Path
from typing import Any

import intake
import numpy as np
import xarray as xr

from dataflux.core import Flux
from dataflux.storage.hdf5 import HDF5Sink, HDF5Source
from dataflux.storage.intake import IntakeSource


class _DemoXArraySource(intake.source.base.DataSource):
    """Tiny in-process intake DataSource yielding 4 xarray.DataArray partitions."""

    container = "xarray"
    name = "_demo_xarray"
    version = "0.1"
    partition_access = True

    def __init__(self, n: int = 4, label: str = "demo", metadata: Any = None) -> None:
        super().__init__(metadata=metadata or {})
        self._n = n
        self._label = label

    def _get_schema(self) -> Any:
        return intake.source.base.Schema(
            datashape=None,
            dtype=str(np.dtype(np.float32)),
            shape=(8,),
            npartitions=self._n,
            extra_metadata={},
        )

    def _get_partition(self, i: int) -> xr.DataArray:
        return xr.DataArray(
            np.arange(8, dtype=np.float32) + (i * 100),
            dims=["x"],
            attrs={"label": self._label, "partition_index": i, "samplerate": 1000.0},
        )

    def _close(self) -> None:
        pass


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dataflux-intake-demo-") as tmp:
        # 1. Build an intake source and wrap it.
        intake_src = _DemoXArraySource(n=4, label="demo")
        df_src = IntakeSource(source=intake_src, target_attr="label")
        print(f"IntakeSource length: {len(df_src)} partitions")

        # 2. Iterate and inspect the first sample.
        first = next(iter(df_src))
        print(
            "First sample:",
            "input.shape=",
            tuple(first.input.shape),
            "input.dtype=",
            first.input.dtype,
            "target=",
            first.target,
            "metadata.keys=",
            sorted(first.metadata.keys()),
        )

        # 3. Pipe through Flux → HDF5Sink and read back.
        h5_path = Path(tmp) / "intake_roundtrip.h5"
        Flux.from_source(df_src).to_sink(HDF5Sink(h5_path, overwrite=True))

        loaded = list(HDF5Source(h5_path))
        print(f"Wrote {len(loaded)} samples to {h5_path}")
        print(
            "Roundtripped sample 2:",
            "shape=",
            tuple(loaded[2].input.shape),
            "samplerate(meta)=",
            loaded[2].metadata.get("samplerate"),
            "label(meta)=",
            loaded[2].metadata.get("label"),
        )

    print("OK — intake_pipeline.py finished.")


if __name__ == "__main__":
    main()
