# Sources — HuggingFace, splits, ranges, concatenation (`recordstream.sources`)

## Hugging Face datasets

`HuggingFaceSource` turns any `datasets.Dataset` (a Hub repo id or a local imagefolder path) into plain record dicts of typed values: the `input_feature` column becomes an `Image` under the record key `"image"`, the `target_feature` column a `Label` under `"class"`, and each kept metadata column its own `Label` entry keyed by the column name (plus the source-provenance `hf_path` / `hf_split` entries) — traceability that often goes missing in bare dictionary loading.

- **`metadata_features` (which extra columns become record entries):** the sentinel **`"*"`** (or `["*"]`, the default) keeps **every column except `input_feature` / `target_feature`** — the full-traceability option, resolved against the dataset's real columns at load; an explicit list keeps exactly those columns; `None` / `[]` keep none.

```yaml
hf_train: !class:recordstream.sources.HuggingFaceSource()
  path: mnist
  input_feature: image
  target_feature: label
  metadata_features: ["*"]   # keep every other column as its own record entry (the default)
```

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
hf_train: !class:recordstream.sources.HuggingFaceSource()
  path: mnist
  split: train

my_split: !class:recordstream.sources.DatasetSplit()
  source: !ref:hf_train
  val_fraction: 0.1
  test_fraction: 0.1
  seed: 42

train_set: !class:recordstream.core.Stream() { source: !ref:my_split.train }
val_set:   !class:recordstream.core.Stream() { source: !ref:my_split.val }
test_set:  !class:recordstream.core.Stream() { source: !ref:my_split.test }
```

Omit `test_fraction` for a plain two-way train/val split; omit both fractions and `train` is the whole source (`val`/`test` empty).

**Select-one API.** Passing `split` makes the `DatasetSplit` *itself* iterate that one view (`split=None` ⇒ `train`), so it's directly usable as a single `source:`. `split` is the closed `Literal["train", "val", "test"]`, exported as `recordstream.SplitName`.

```yaml
val_set: !class:recordstream.sources.DatasetSplit()
  source: !ref:hf_train
  split: val
  val_fraction: 0.1
  seed: 42
```

## Range & concatenation sources

- **`RangeSource(source, start, stop)`** — a contiguous index slice `[start:stop)` over a source (negatives count from the end; clamped). The plain-slice counterpart to `DatasetSplit`.

    ```yaml
    first_half: !class:recordstream.sources.RangeSource()
      source: !ref:hf_train
      start: 0
      stop: 5000
    ```

- **`ConcatSource(sources)`** — joins multiple indexable sources into one longer indexable source (the indexable counterpart to `JointStream`, which is iteration-only). Because it's indexable, a `ConcatSource` can itself be wrapped by `DatasetSplit` / `RangeSource`.

    ```yaml
    combined: !class:recordstream.sources.ConcatSource()
      sources:
        - !ref:train_main
        - !ref:extra_shard
    ```

**HuggingFace native slicing** (alternative, no RecordStream split needed): `split: "train[:90%]"` / `"train[90%:]"` on two `HuggingFaceSource`s.

> **Note on `!ref:`** — Confluid `!ref:` resolves to the same live object as the referenced key (including attribute refs like `!ref:my_split.train`), so a single `HuggingFaceSource` is loaded once and shared. Use `!clone:` when you want an independent deep copy instead.
