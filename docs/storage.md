# Storage — sinks, sources and queryable metadata (`sampleflux.storage`)

SampleFlux makes it easy to move data between different formats:

```python
from sampleflux.storage.hdf5 import HDF5Source
from sampleflux.storage.zarr import ZarrGroupSink

# Stream from HDF5 to Zarr in parallel
Flux.from_source(HDF5Source("input.h5")) \
    .parallel(workers=8) \
    .map(heavy_op) \
    .to_sink(ZarrGroupSink("output.zarr"))
```

## Sinks and their matching sources

Every sink has a source that reads its layout back into `Sample` triplets:

| Backend | Sink | Source | Round-trips |
|---|---|---|---|
| HDF5 (sequential) | `HDF5Sink` | `HDF5Source` | input + target + metadata |
| Zarr group (one group / sample) | `ZarrGroupSink` | `ZarrGroupSource` | input + target + metadata |
| Zarr batch (one stacked array) | `ZarrBatchSink` | `ZarrBatchSource` | input only (uniform shape) |
| Directory (one dir / sample) | `DirectorySink` | — | — |

```python
from sampleflux.storage.zarr import ZarrGroupSink, ZarrGroupSource

Flux(samples).to_sink(ZarrGroupSink("ds.zarr", overwrite=True))
for sample in ZarrGroupSource("ds.zarr"):   # input/target as before, metadata from .zattrs
    ...
```

> Domain-specific storage formats implement the same `DataSink`/`DataSource` protocols in their own package — e.g. the SigMF waveform-recording pair (`SigMFSink`/`SigMFSource`) lives in `waivefront.sigmf`, not here. The engine never couples to a specific format.

## Array-valued metadata (e.g. segmentation masks)

`HDF5Sink` stores scalar/string metadata as HDF5 **attributes**, but HDF5 caps attribute size — a large array (a segmentation mask, a per-sample weight map) put in `Sample.metadata` would overflow that limit. So **array-valued metadata (`np.ndarray` / `torch.Tensor`) is written as its own dataset** under a per-sample group `{prefix}_meta/<key>`, and `HDF5Source` merges it back into `Sample.metadata` on read. This is fully backward-compatible: files written before this layout (no `{prefix}_meta` group) read exactly as before.

```python
sample = Sample(input=data, target=label, metadata={"mask": mask_2d, "snr": 12.0})
Flux([sample]).to_sink(HDF5Sink("ds.h5", overwrite=True))
loaded = next(iter(HDF5Source("ds.h5")))
loaded.metadata["mask"]   # the full array, byte-exact (not a truncated repr)
loaded.metadata["snr"]    # scalar, via attributes as before
```

## Queryable metadata (`sampleflux.storage.query`)

Filter stored samples by metadata predicates *without loading arrays*: sources implementing the `SupportsMetadataScan` protocol (`iter_metadata()`) scan only attrs / `.zattrs` / sidecar JSON — `HDF5Source` and `ZarrGroupSource` both do (and external storage sources can implement the structural protocol without importing this module), so **existing HDF5/Zarr files are queryable with no rewrite**:

```python
from sampleflux.storage.query import MetadataFilterSource

view = MetadataFilterSource(source=HDF5Source(path="d.h5"), where="snr_db > 10 and drone == 'DJI'")
len(view)          # matches counted from a metadata-only scan
flux = Flux(source=view, ops=[...])   # arrays load ONLY for matching samples
```

`where` uses the FormulaOp restricted namespace with metadata keys as variables (a missing key = non-matching, a malformed expression fails loudly); a programmatic `predicate=` composes with AND; sources without the protocol fall back to full-iteration filtering. Array-valued HDF5 metadata appears in the scan as shape/dtype stub strings (`"<array shape=(2, 2) dtype=float64>"`), so queries can test presence without a single array read.
