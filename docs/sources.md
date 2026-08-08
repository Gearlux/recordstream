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

## Train / val / test splitting (`DatasetSplit`)

`DatasetSplit` partitions any indexable source (implementing `__len__` and `__getitem__`) into reproducible **train / val / test** views. It is a `source` (`category="source"`) — it yields records and is wired into a trainer's `source:` slot — and it applies no ops, so it's a source, not an engine.

**Property API (preferred).** Configure **one** `DatasetSplit` with a `seed` and the held-out fraction(s), then read the three cached views off it — `split.train` / `split.val` / `split.test`:

```python
from recordstream import DatasetSplit
split = DatasetSplit(source=src, val_fraction=0.1, test_fraction=0.1, seed=42)
split.train   # ≈80% — the remainder      split.val   # ≈10%      split.test  # ≈10%
```

The views are disjoint and complementary, computed once over a single deterministic shuffle (cached), so the underlying source is consumed once. In Confluid YAML they're reachable by **attribute reference** — `!ref:my_split.train` / `.val` / `.test`. All three refs resolve to the *same* `DatasetSplit` instance, so the upstream source is loaded **exactly once**:

```yaml
hf_train: !class:recordstream.sources.huggingface.HuggingFaceSource()
  path: ylecun/mnist
  split: train

my_split: !class:recordstream.sources.split.DatasetSplit()
  source: !ref:hf_train
  val_fraction: 0.1
  test_fraction: 0.1
  seed: 42

train_set: !class:recordstream.core.stream.Stream() { source: !ref:my_split.train }
val_set:   !class:recordstream.core.stream.Stream() { source: !ref:my_split.val }
test_set:  !class:recordstream.core.stream.Stream() { source: !ref:my_split.test }
```

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

> **Note on `!ref:`** — Confluid `!ref:` resolves to the same live object as the referenced key (including attribute refs like `!ref:my_split.train`), so a single `HuggingFaceSource` is loaded once and shared. Use `!clone:` when you want an independent deep copy instead.
