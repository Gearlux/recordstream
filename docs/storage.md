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

Every sink has a source that reads its layout back into typed `Sample` bags:

| Backend | Sink | Source | Round-trips |
|---|---|---|---|
| HDF5 (sequential) | `HDF5Sink` | `HDF5Source` | all fields + roles |
| Zarr group (one group / sample) | `ZarrGroupSink` | `ZarrGroupSource` | all fields + roles |
| Zarr batch (one stacked array) | `ZarrBatchSink` | `ZarrBatchSource` | primary input field only (uniform shape) |
| Directory (one dir / sample) | `DirectorySink` | `DirectorySource` | all fields + roles |

```python
from sampleflux.storage.zarr import ZarrGroupSink, ZarrGroupSource

Flux(samples).to_sink(ZarrGroupSink("ds.zarr", overwrite=True))
for sample in ZarrGroupSource("ds.zarr"):   # exact fields, roles and item attrs reconstructed
    ...
```

> Domain-specific storage formats implement the same `DataSink`/`DataSource` protocols in their own package — e.g. the SigMF waveform-recording pair (`SigMFSink`/`SigMFSource`) lives in `waivefront.sigmf`, not here. The engine never couples to a specific format.

## Array-valued item attributes

Each field is stored as its own group: the item's payload as a `data` dataset and its scalar attributes as HDF5 **attributes**. HDF5 caps attribute size, so any **array-valued attribute** (`np.ndarray` / `torch.Tensor`, e.g. a per-sample weight map) is written as its own sub-dataset under `attrs/` instead — a large array never overflows the attribute limit, and `HDF5Source` restores every attribute on read. A segmentation mask is not an attribute at all: it is a first-class `Mask` field with its own payload.

```python
from sampleflux import Sample, Image, Mask, item_data

sample = Sample({"image": Image(data), "mask": Mask(mask_2d)}, roles={"mask": "target"})
Flux([sample]).to_sink(HDF5Sink("ds.h5", overwrite=True))
loaded = next(iter(HDF5Source("ds.h5")))
item_data(loaded["mask"])   # the full mask array, byte-exact (not a truncated repr)
loaded.role_of("mask")      # "target" — roles round-trip too
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
