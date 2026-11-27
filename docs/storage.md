# Storage — sinks, sources and queryable metadata (`recordstream.storage`)

> Runnable tour: [`examples/storage_roundtrip.py`](../examples/storage_roundtrip.py) — the same
> records through all three sink/source pairs (typed values + a plain scalar, byte-identical
> round-trips) plus a `MetadataFilterSource` query that never loads an array.

RecordStream makes it easy to move data between different formats:

```python
from recordstream.storage.hdf5 import HDF5Source
from recordstream.storage.zarr import ZarrGroupSink

# Stream from HDF5 to Zarr in parallel
Stream.from_source(HDF5Source("input.h5")) \
    .parallel(workers=8) \
    .map(heavy_op, key="image") \
    .to_sink(ZarrGroupSink("output.zarr"))
```

## Sinks and their matching sources

Every sink has a source that reads its layout back into record dicts of typed values:

| Backend | Sink | Source | Round-trips |
|---|---|---|---|
| HDF5 (sequential) | `HDF5Sink` | `HDF5Source` | all keys + item types + attrs |
| Zarr group (one group / record) | `ZarrGroupSink` | `ZarrGroupSource` | all keys + item types + attrs |
| Zarr batch (one stacked array) | `ZarrBatchSink` | `ZarrBatchSource` | first record entry only (uniform shape) |
| Directory (one dir / record) | `DirectorySink` | `DirectorySource` | all keys + item types + attrs |

```python
from recordstream.storage.zarr import ZarrGroupSink, ZarrGroupSource

Stream(records).to_sink(ZarrGroupSink("ds.zarr", overwrite=True))
for record in ZarrGroupSource("ds.zarr"):   # exact keys, item types and attrs reconstructed
    ...
```

`GroupedFilesSink` is not one of these: it stores no record, it COPIES each record's file into a folder per group.

```yaml
sink: !class:recordstream.storage.grouped_files.GroupedFilesSink
  path: ${env:DATA_ROOT}/grouped     # where the copies go
  root: ${env:DATA_ROOT}/pictures    # keeps each file's place under it (class folders stay)
  file: file                         # the entry naming the file
  group: group                       # the entry naming its group; a record without it is in no group
```

| record | lands in |
|---|---|
| `{file: pictures/cat/1.png, group: 3}` | `grouped/cat/group_0003/1.png` |
| `{file: pictures/dog/9.png, group: 3}` | `grouped/dog/group_0003/9.png` |
| `{file: pictures/cat/2.png}` | `grouped/cat/2.png` |

The dataset read from is left as it is. A file that would land on an existing one is refused: two files of one name
need `root`, a second run needs an empty `path`. Only the named file is copied — a label file beside it is not.
[`GroupedFilesSource`](sources.md) reads the folder back, one record per group.

`GroupsFileSink` writes the groups down WITHOUT touching the dataset: one JSON file naming, by their place in the
run, the records that share a group entry.

```yaml
sink: !class:recordstream.storage.grouped_files.GroupsFileSink
  path: ${env:DATA_ROOT}/drones/train.groups.json
  group: group
```

| records written, in order | the file |
|---|---|
| `{group: 7}`, `{}`, `{group: 7}`, `{group: 2}`, `{group: 2}`, `{group: 9}` | `{"records": 6, "groups": [[0, 2], [3, 4]]}` |

A group of one is no group. The numbers are the source's indices only when the run reads the whole source in order
and drops no record. [`GroupedSource`](sources.md) reads the file over the same source.

All four backends share ONE logical schema — the **record key-group layout**, stamped
`recordstream_format = "typedrecord-v1"`: per record, one group per KEY carrying the value's
registered type name, the payload as a `data` dataset, and its attrs (scalars natively; structured
values JSON-tagged so tuples survive); a plain (non-item) value rides the `"plain"` type tag — an
array payload as `data`, a scalar under the `value` attr. Everything serializes through the item
codec (`recordstream/io.py`), so an externally-registered item type round-trips with zero storage
edits (see [record-model.md](record-model.md#storage--the-record-key-group-layout)).

> **No backward compatibility.** A store whose format tag is missing or pre-record
> (`typedsample-v1`) raises a `ValueError` telling you to re-generate it with a current sink —
> there is no legacy read path.

> Domain-specific storage formats implement the same `DataSink`/`DataSource` protocols in their own
> package — e.g. a waveform-recording format pair lives in the signal-domain package, not here. The
> engine never couples to a specific format.

## What a run deletes

A sink that replaces what it holds says so before a run, through `planned_changes()`: a line for each thing that is
gone once the run opens the sink. `recordstream run --dry_run` prints the lines as the plan's actions and changes
nothing (see [workflow](workflow.md)).

| Sink | With `overwrite: true` over existing data |
|---|---|
| `HDF5Sink` | `delete /data/train.h5 (1.2G)` |
| `ZarrBatchSink` | `delete /data/train.zarr/data (12 files, 310.0K)` |

A sink that appends, or whose file is not there yet, has no line. A sink of your own answers the same method, by
shape; `recordstream.storage.base.size_text` writes the size.

## Array-valued item attributes

Each key is stored as its own group: the item's payload as a `data` dataset and its scalar
attributes as HDF5 **attributes**. HDF5 caps attribute size, so any **array-valued attribute**
(`np.ndarray` / `torch.Tensor`, e.g. a per-record weight map riding an item's attrs) is written as
its own sub-dataset under `attrs/` instead — a large array never overflows the attribute limit, and
`HDF5Source` restores every attribute on read. A segmentation mask is not an attribute at all: it
is a first-class `Mask` value under its own key with its own payload.

```python
from recordstream import Image, Mask, item_data

record = {"image": Image(data), "mask": Mask(mask_2d)}
Stream([record]).to_sink(HDF5Sink("ds.h5", overwrite=True))
loaded = next(iter(HDF5Source("ds.h5")))
item_data(loaded["mask"])   # the full mask array, byte-exact (not a truncated repr)
loaded["image"].layout      # item attrs round-trip too
```

## Partial payload reads (`read_record_group`)

`recordstream.storage.hdf5.read_record_group(group, slices=...)` is the public form of the
HDF5 row decoder `HDF5Source` iterates through. `slices` maps a record KEY to a slice applied
to that field's `data` dataset **at read time** — an h5py partial read, so only the requested
span of the payload ever leaves the file. A consumer windowing large stored rows (e.g. a
whole-capture signal archived once, re-windowed onto many grids at read time) decodes each
window without materializing the row; the decoded item is identical to a full read followed by
an in-memory slice, and attrs are never sliced:

```python
import h5py
from recordstream.storage.hdf5 import read_record_group

with h5py.File("ds.h5", "r") as handle:
    window = read_record_group(handle["s000000"], slices={"signal": slice(1000, 2000)})
```

## Queryable metadata (`recordstream.storage.query`)

Filter stored records by metadata predicates *without loading arrays*: sources implementing the
`SupportsMetadataScan` protocol (`iter_metadata()`) scan only attrs / `.zattrs` / sidecar JSON —
`HDF5Source` and `ZarrGroupSource` both do (and external storage sources can implement the
structural protocol without importing this module), so **record-layout HDF5/Zarr files are
queryable with no extra index**:

```python
from recordstream.storage.query import MetadataFilterSource

view = MetadataFilterSource(source=HDF5Source(path="d.h5"), where="signal.samplerate > 1e6")
len(view)          # matches counted from a metadata-only scan
stream = Stream(source=view, ops=[...])   # arrays load ONLY for matching records
```

The scans yield the nested `{key: {attr: value}}` shape, and a `where` expression addresses it as
`<key>.<attr>` (a plain scalar entry appears under its `value` attr — `"snr_db.value > 10"`).
`where` uses the FormulaOp restricted namespace with metadata keys as variables (a missing key =
non-matching, a malformed expression fails loudly); a programmatic `predicate=` composes with AND;
sources without the protocol fall back to full-iteration filtering via `record_metadata(record)` —
the same nested shape derived from a live record. Array-valued attrs appear in the scan as
shape/dtype stub strings (`"<array shape=(2, 2) dtype=float64>"`), so queries can test presence
without a single array read. A key named like a Python keyword (e.g. `class`) can't be addressed
in an expression — use the programmatic `predicate` or a non-keyword key name.

## What was computed from records (`DerivedStore`)

A prediction, an embedding or a score is computed from a record by a chain of ops with its settings and weights.
`DerivedStore` keeps such values by three names: the **producer** (a hash of what computed them), the **output**'s
name, and the [record id](sources.md#identifying-a-record).

The usual way in is the `Derive` op. It wraps the ops whose outputs are worth keeping; for each record the ops run
once, and from then on the kept values are written onto the record without running them:

```yaml
stream: !class:recordstream.core.stream.Stream
  source: !class:recordstream.sources.identified.IdentifiedSource
    source: !class:recordstream.sources.files.FilesSource {root: $DATA_ROOT/captures, name: captures}
  ops:
    - !class:recordstream.ops.formats.ReadFile {}
    - !class:recordstream.ops.derive.Derive
      store: !class:recordstream.storage.derived.DerivedStore {root: $RUN_DIR/derived}
      outputs: [embedding]
      files: [weights/encoder.pt]
      ops:
        - !class:mypackage.Embed {weights: weights/encoder.pt}
```

The producer is `producer_hash(ops, files)`: the inner ops written as the YAML they build from, every setting
listed, plus the content of each file in `files`. Another op, another setting or another weights file is another
producer, kept beside the old one — so a kept value is never stale, and no file time is compared. A setting left at
its default and the same value written out are the same producer. An op whose settings cannot be written as a
document (a class that is not `@configurable` and was not built from a config) is refused: two of them with
different settings would be kept under one name.

Reading without the op:

```python
from recordstream.storage.derived import DerivedStore

store = DerivedStore(root="$RUN_DIR/derived")
table = store.table(op.producer, "embedding")
todo = table.missing(ids)                 # the ids with no value yet
vectors = table.array(ids)                # one row per id, in the order asked
store.producers()                         # [{producer, outputs: {name: records}, bytes, description}, …]
store.remove(producer)                    # delete one producer's values
```

On disk a producer is a folder and each output a folder inside it, holding an `index.json` and numbered `.npz`
shards that are written once and never changed; `producer.yaml` says what the producer is. Values go through the
item codec, so a `Boxes` or an `Image` comes back as its type. A value that holds arrays inside a mapping is
refused — keep each array as an output of its own. One process writes a table at a time; any number read it.

Where the store lives is the caller's choice: recordstream names no default folder. One root serves any number of
datasets and producers, because a record id names its dataset.
