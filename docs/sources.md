# Sources — HuggingFace, splits, ranges, concatenation (`recordstream.sources`)

`recordstream.sources` is a package with **one class per module**. Import from the package —
`from recordstream.sources import HuggingFaceSource, DatasetSplit, RangeSource, ConcatSource` —
but spell the **submodule** path in a config, because that is what `cls.__module__` says and
what a generated config emits:

| class | module | `!class:` path |
| --- | --- | --- |
| `HuggingFaceSource` | `huggingface.py` | `recordstream.sources.huggingface.HuggingFaceSource` |
| `DatasetSplit` | `split.py` | `recordstream.sources.split.DatasetSplit` |
| `RangeSource` | `range.py` | `recordstream.sources.range.RangeSource` |
| `ConcatSource` | `concat.py` | `recordstream.sources.concat.ConcatSource` |

The shorter `!class:recordstream.sources.HuggingFaceSource` still resolves (Confluid falls back to
a module-path import, and the package re-exports every name), so an older config keeps loading —
but a generated one will use the submodule spelling. Rationale:
[docs/architecture.md §11](architecture.md#11-sources-are-a-package-one-class-per-module--and-the-module-path-is-the-contract-2026-08-01).

## Hugging Face datasets

`HuggingFaceSource` turns any `datasets.Dataset` (a Hub repo id or a local imagefolder path) into plain record dicts of typed values: the `input_feature` column becomes an `Image` under the record key `"image"`, the `target_feature` column a `Label` under `"class"`, and each kept metadata column its own `Label` entry keyed by the column name (plus the source-provenance `hf_path` / `hf_split` entries) — traceability that often goes missing in bare dictionary loading.

- **`metadata_features` (which extra columns become record entries):** the sentinel **`"*"`** (or `["*"]`, the default) keeps **every column except `input_feature` / `target_feature`** — the full-traceability option, resolved against the dataset's real columns at load; an explicit list keeps exactly those columns; `None` / `[]` keep none.

```yaml
hf_train: !class:recordstream.sources.huggingface.HuggingFaceSource()
  path: ylecun/mnist
  input_feature: image
  target_feature: label
  metadata_features: ["*"]   # keep every other column as its own record entry (the default)
```

> **Use the namespaced repo id** (`ylecun/mnist`, never the legacy bare `mnist`): current `huggingface_hub` rejects namespace-less ids (`HfUriError: Repository id must be 'namespace/name'`, measured 2026-08-06). Worse than the hard failure is the soft one — with a stale local cache present, `datasets` logs "couldn't be found on the Hugging Face Hub" and silently loads the cached copy, so a bare id can appear to work on one machine and fail on a fresh one.

> **Lazy & zero-arg construction** — `HuggingFaceSource` follows the workspace lazy-init convention: the constructor does no work (no network), so `HuggingFaceSource()` is valid and building one is free. The dataset is downloaded only on first access to the read-only `.dataset` property (cached thereafter; reset `_dataset` to reload), and `.resolved_metadata_features` (the `"*"` expansion) is derived lazily from the loaded columns. `path` is therefore optional at construction and validated lazily — accessing `.dataset` with an empty `path` raises a clear `ValueError`.

## A plain list of files (`FilesSource`)

When the data is just files — someone handed them over, a tool dropped them in a folder —
`FilesSource` serves each path as one record, and an OP decodes it. The source knows nothing
about what a file means; the chain shows how a file becomes a record:

```yaml
pipeline: !class:recordstream.core.stream.Stream
  source: !class:recordstream.sources.files.FilesSource
    files: ["/data/incoming/dog3.png", "/data/incoming/IMG_2041.png"]
  ops:
    - !class:recordstream.ops.image.ReadImage {}   # {file} -> {file, image}
```

Each record starts as `{"file": "<path>"}`; `ReadImage` adds the decoded pixels (RGB, HWC
uint8) and leaves the path as provenance. There is deliberately **no label entry**: files
arrive unannotated, and whatever labels them adds that entry downstream. A file the imaging
library cannot open passes through UNCHANGED with a debug log — the record keeps its row and
one stray text file never costs the run around it. Other file kinds get their own read ops.

### Pointing it at a folder (`root` + `pattern`)

Instead of listing every path, point `FilesSource` at a folder. `pattern` is an ordinary
[`Path.glob`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) pattern
relative to `root`, and that one concept expresses every scope a review needs:

```yaml
source: !class:recordstream.sources.files.FilesSource
  root: $DATA_ROOT/captures      # ~ and $VAR are expanded
  pattern: "*/*"                 # every file one level down
```

| `pattern` | what it lists |
|---|---|
| `"*"` (default) | the folder's own files |
| `"*/*"` | every immediate subdirectory |
| `"**/*"` | everything below `root` |
| `"*/*.json"` | one file kind, across all subdirectories |
| `"bte_test_1/*"` | **one named subdirectory** — this is the "one directory at a time" scope |

Naming a subdirectory in the pattern *is* the per-directory filter, so there is no second
knob that means the same thing. Pass `files` **or** `root`, never both — two answers to
"which files" is a config bug, not a merge, and it is refused at construction.

Two listing rules keep the result usable. **Directories the glob matches are skipped** — only
files become records. **Dotfiles are never listed**, because a folder glob otherwise picks up
`.DS_Store` and friends, which are not data and fail to decode. And a pattern matching
**nothing raises**, naming the folder and the pattern, rather than handing the chain an empty
listing that surfaces as a confusing failure three ops later:

```
FilesSource: pattern '*' matched no files under '/data/captures' (a pattern is relative to
root: '*' is the folder's own files, '*/*' its subdirectories, '**/*' everything below it)
```

The scan is **lazy** — it happens on first use, never in the constructor — so a folder that
does not exist yet is reported when the source is read, not when the config is built.

### Asking a file what it says, without opening it

Some questions about a listing are not worth a decode. *Where was each of these recordings
made? How long is each one? Which of them carry annotations?* — for a format that pairs a small
metadata file with a large data file, all of that is in the sidecar, and opening the data file to
reach it is the expensive way round. Measured on one real capture library: **499 ms** to open a
capture against **0.03 ms** to read the sidecar beside it, so a survey of 1430 recordings is
twelve minutes one way and half a second the other.

A format may therefore answer from its metadata alone, and `scan_file` asks it to:

```python
from recordstream.formats import scan_file

for path in listing:
    record = scan_file(path)          # the sidecar or the header — never the payload
    if record is None:                # a companion half, or a format that cannot answer cheaply
        continue
    print(record["signal"].samplerate)
```

What comes back is an ordinary record with the payload left out, so the ops and graphs you
already have read a scan exactly like a decode — the same chain can survey a listing and analyse
one recording.

`None` means nobody could answer cheaply, and it is never a decode in disguise: a format that
cannot read its metadata separately (a raw sample file, where the samples *are* the file) simply
does not offer this, and you ask it for the real thing by name when you need the data.

To offer it from your own format, add one method beside `read`:

```python
class MyFormat:
    name = "mine"

    #: Entries `scan` fills with a PLACEHOLDER rather than the real thing. Optional; omit it
    #: when your cheap answer stands in for nothing.
    scan_stands_in = ("signal",)

    def scan(self, path: Path) -> Record:
        """What <path>'s sidecar says — no payload decode."""
```

`scan_stands_in` exists because a scan often *keeps* the payload's entry rather than dropping
it: the rate, the centre and the position live on that item, so a survey wants it there — with
an empty array behind it. Measured on two shipped formats, `scan` reports **0 samples** where
`read` reports **1,000,001** and **100,000,000**. That makes the entry present but not
*answered*, which a survey may ignore and a filter may not:

```python
from recordstream.formats import answers, scan_answered

record = scan_answered(path)          # the same record, placeholders MARKED
answers(record, ("regions",))         # True  — the sidecar really says this
answers(record, ("signal",))          # False — read the file if you need the samples
```

The marker rides the *value*, not the key, so a chain that renames the entry cannot lose it.

### A chain that reads files can answer cheaply too

A `Stream` with ops has to run its chain — an op may consume one entry to make another — so a
key-restricted walk through one used to decode every record. `ReadFile` now offers a cheap
variant of itself, and `Stream.project` runs that chain, checks whether the record carries what
was asked for, and re-runs the real chain per record when it does not:

```python
stream = Stream(source=FilesSource(root=captures, pattern="*/*.json"),
                ops=[ReadFile(), RenameField(src="signal", dst="input")])

for record in project(stream, ("regions",)):   # the sidecars answer; no capture is decoded
    ...
```

Measured on a 1430-recording library, one filter term: **3.4 min → 0.5 s**, with 41 records
falling back because they carry no annotations at all. A chain whose ops offer nothing cheap
behaves exactly as before, and so does one that would run with workers or a chunk size.

To offer it from your own op, return a cheaper version of yourself:

```python
class MyOp:
    def for_projection(self) -> "MyOp":
        """A variant producing the same record minus what it cannot make cheaply."""
```

## Train / val / test splitting (`DatasetSplit`)

`DatasetSplit` partitions any indexable source (implementing `__len__` and `__getitem__`) into reproducible **train / val / test** views. It is a `source` (`category="source"`) — it yields records and is wired into a trainer's `source:` slot — and it applies no ops, so it's a source, not an engine.

**Property API (preferred).** Configure **one** `DatasetSplit` with a `seed` and the held-out fraction(s), then read the three cached views off it — `split.train` / `split.val` / `split.test`:

```python
from recordstream import DatasetSplit
split = DatasetSplit(source=src, val_fraction=0.1, test_fraction=0.1, seed=42)
split.train   # ≈80% — the remainder      split.val   # ≈10%      split.test  # ≈10%
```

The views are disjoint and complementary, computed once over a single deterministic shuffle (cached). In a config each view is a `DatasetSplit` with its `split` selector set — write the recipe once (a YAML anchor on the first view) and merge it (`<<:`) into the others. Every view references the *same* `hf_train` (`!ref:` shares the instance), so the upstream source is loaded **exactly once**; a `DatasetSplit`'s own partition is one seeded shuffle over `len(source)`:

```yaml
hf_train: !class:recordstream.sources.huggingface.HuggingFaceSource()
  path: ylecun/mnist
  split: train

train_set: !class:recordstream.core.stream.Stream()
  source: &split_recipe !class:recordstream.sources.split.DatasetSplit()
    source: !ref:hf_train
    val_fraction: 0.1
    test_fraction: 0.1
    seed: 42
    split: train
val_set: !class:recordstream.core.stream.Stream()
  source: !class:recordstream.sources.split.DatasetSplit()
    <<: *split_recipe
    split: val
test_set: !class:recordstream.core.stream.Stream()
  source: !class:recordstream.sources.split.DatasetSplit()
    <<: *split_recipe
    split: test
```

(Reading a view by attribute reference — `!ref:my_split.train` — is no longer a config spelling; the config engine refuses it and names this rewrite.)

Omit `test_fraction` for a plain two-way train/val split; omit both fractions and `train` is the whole source (`val`/`test` empty).

**Select-one API.** Passing `split` makes the `DatasetSplit` *itself* iterate that one view (`split=None` ⇒ `train`), so it's directly usable as a single `source:`. `split` is the closed `Literal["train", "val", "test"]`, exported as `recordstream.SplitName`.

```yaml
val_set: !class:recordstream.sources.split.DatasetSplit()
  source: !ref:hf_train
  split: val
  val_fraction: 0.1
  seed: 42
```

## Range & concatenation sources

- **`RangeSource(source, start, stop)`** — a contiguous index slice `[start:stop)` over a source (negatives count from the end; clamped). The plain-slice counterpart to `DatasetSplit`.

    ```yaml
    first_half: !class:recordstream.sources.range.RangeSource()
      source: !ref:hf_train
      start: 0
      stop: 5000
    ```

- **`ConcatSource(sources)`** — joins multiple indexable sources into one longer indexable source (the indexable counterpart to `JointStream`, which is iteration-only). Because it's indexable, a `ConcatSource` can itself be wrapped by `DatasetSplit` / `RangeSource`.

    ```yaml
    combined: !class:recordstream.sources.concat.ConcatSource()
      sources:
        - !ref:train_main
        - !ref:extra_shard
    ```

**HuggingFace native slicing** (alternative, no RecordStream split needed): `split: "train[:90%]"` / `"train[90%:]"` on two `HuggingFaceSource`s.

> **Never mark a nested source `_partial_: true`.** A partial is a *deferred marker*, not an
> instance — a `source:` slot holding one fails at first use with an error naming the slot and
> the fix (the same guidance `Stream` gives), because nothing will flow it for you. A plain
> `_target_:` source is built at load time and is what the slot wants. The examples above use
> `${ref:…}` to a top-level source instead, which is also what lets several wrappers share one
> loaded source.

## A view forwards a key-restricted walk

A walk that names only some record keys (`recordstream.projection.project`) takes a source's own
cheap path — the one that skips building what nobody asked for — only when the object it is handed
implements `project(keys)`. Every view source here does, so a slice, a split or a concatenation
costs what the source underneath it costs and nothing more:

```python
from recordstream.projection import project
from recordstream.sources.range import RangeSource

# reads each record's labels; never decodes an image or a waveform
for record in project(RangeSource(source=big, start=0, stop=1000), ("class",)):
    ...
```

Measured on a source whose records carry ~15 MB of samples, for the same records and the same
keys: **0.006 s** per record straight from the source, **0.423 s** through a wrapper that did not
forward — a wrapper that only slices has no business making a walk more expensive than the one it
wraps.

Two things follow from a projected walk being an ITERATOR rather than an index:

- **A `RangeSource` still walks the records before `start`**, at the projected cost rather than a
  full read, and stops at `stop`.
- **A shuffled `DatasetSplit` view holds records until their turn comes** — projected records, the
  entries you asked for, never the payloads the projection exists to skip. An unshuffled view (a
  split with no fraction set) streams straight through.

`ConcatSource` and `JointStream` chain each part's own projection, so a part that projects cheaply
does, and one that does not keeps the ordinary fallback for itself alone.

## Identifying a dataset

A source can name the data it reads, so a run record, a report, or a log line can point at it.
Two handles, because they answer different questions:

| property | what it is | when it is `None` |
| --- | --- | --- |
| `dataset_uri` | the **canonical** identifier — machine-parseable and stable, the string two runs are compared on | the source has no dataset configured |
| `dataset_url` | a link a **person** can open | the data has no web page (anything local) |

```python
from recordstream import HuggingFaceSource, dataset_uri, dataset_url

source = HuggingFaceSource(path="ylecun/mnist", split="train")
source.dataset_uri   # 'hf://datasets/ylecun/mnist?split=train'
source.dataset_url   # 'https://huggingface.co/datasets/ylecun/mnist/viewer/default/train'
```

What `HuggingFaceSource` produces, for each way it can be configured:

| configuration | `dataset_uri` | `dataset_url` |
| --- | --- | --- |
| `path="ylecun/mnist", split="train"` | `hf://datasets/ylecun/mnist?split=train` | `…/ylecun/mnist/viewer/default/train` |
| `+ name="fashion", revision="abc123"` | `hf://datasets/ylecun/mnist?name=fashion&revision=abc123&split=train` | `…/ylecun/mnist/viewer/fashion/train` |
| `path="/data/imagefolder"` | `file:///data/imagefolder?split=train` | `None` |
| `path=""` | `None` | `None` |

Both read **stored configuration only** — asking never loads, downloads or opens anything, so a
source that is never iterated still names itself. Whether a `path` is a Hub repo id or a local
directory is decided by whether it exists on disk, the same question `load_dataset` answers. The
query parameters are sorted, so one configuration has exactly one URI.

**Use the free functions rather than the attributes when the source may be wrapped or deferred.**
`dataset_uri(x)` / `dataset_url(x)` materialize a `!class:` marker straight out of a config, and
follow a view's `.source` to the dataset underneath:

```python
from recordstream import RangeSource, Stream, dataset_uri

# a view reports the same DATASET — the slice it takes is its own configuration, not identity
dataset_uri(Stream(source=RangeSource(source=source, stop=100)))
# 'hf://datasets/ylecun/mnist?split=train'
```

That verbatim propagation is deliberate: one dataset reached two ways must compare equal. How
much of it a run consumed is recorded by the wrapper's own settings (`start` / `stop`, the split
fractions), not by mangling the handle.

A source holding **several** datasets declines to be one of them — `ConcatSource` answers `None`,
and `dataset_uris` is the plural form:

```python
from recordstream import ConcatSource, dataset_uri, dataset_uris

dataset_uri(ConcatSource(sources=[a, b]))    # None — a concatenation is not one dataset
dataset_uris(ConcatSource(sources=[a, b]))   # ['hf://datasets/…', 'file:///…']
```

**Adding identity to your own source** is defining the two properties — `SupportsDatasetIdentity`
is a `Protocol`, so there is nothing to register:

```python
@configurable(category="source")
class MyStoreSource:
    def __init__(self, bucket: str = "", prefix: str = "") -> None:
        self.bucket, self.prefix = bucket, prefix

    @property
    def dataset_uri(self) -> Optional[str]:
        return f"s3://{self.bucket}/{self.prefix}" if self.bucket else None

    @property
    def dataset_url(self) -> Optional[str]:
        return None   # no web page: honest, and never an error
```

Rationale: [docs/architecture.md §13](architecture.md#13-dataset-identity-is-a-protocol-and-a-view-propagates-it-verbatim-recordstreamuri-2026-08-02).

> **Note on `!ref:`** — Confluid `!ref:` resolves to the same live object as the referenced key, so a single `HuggingFaceSource` is loaded once and shared. Write the marker again when you want an independent instance instead.
