# RecordStream

**RecordStream** is a high-performance, functional data processing engine built for modern Machine Learning pipelines. It provides a clean, fluent API for streaming and transforming data from any source while maintaining strict compatibility with PyTorch and Hugging Face.

Part of the **Modular Quartet**: `Loggair`, `Confluid`, `Liquifai`, and `RecordStream`.

## 🚀 Key Features

-   **A record is a plain dict:** the [record model](docs/record-model.md) — a `dict` of typed values (`Image`, `Mask`, `Regions`, `Label`, `MultiLabel`, …), each owning its own metadata, with key names carrying meaning (`"image"`, `"mask"`, `"bboxes"`). No wrapper container, no role tags.
-   **Libraries run AS-IS:** bare [albumentations and torchvision `transforms.v2`](docs/augmentation.md) transforms drop straight into any ops list — the engine invokes each op family natively (one call = one joint draw across image/mask/boxes). No adapter classes anywhere.
-   **Type-dispatched native ops:** a `Transform` samples its parameters once per record and applies a per-type kernel to every value it handles — teach an existing op a new value type with one `@MyOp.kernel(NewType)` registration.
-   **Graph pipelines:** readable [`flow:` documents](docs/graph.md) of named steps — `from:` forks, `merge_from:` merges, `bind:` feeds one step's value into another's parameter. An `ops:` list is the same engine's linear spelling; both parse to one step graph.
-   **High Performance:** Native multiprocess support via `.parallel(workers=N)` using the safe `spawn` context; [1→N expanding ops](docs/kinds.md#1n-expanding-ops-iterable-only-pipelines) flatten in every route.
-   **Advanced Storage:** HDF5, Zarr and Directory backends with matching read-back sources and [metadata-only querying](docs/storage.md#queryable-metadata-recordstreamstoragequery) — filter stored datasets without loading a single array.
-   **Passive Introspection:** ops declare the value types they [handle / consume / produce](docs/record-model.md) and are discoverable by category for visual editors and schema generators.
-   **100% Reproducibility:** Entire pipelines are serializable via **Confluid** manifests.

## 🛠 Quick Start

One pipeline mixing a **bare albumentations Compose** (image + mask + boxes move together in one draw), a **bare torchvision v2 transform**, and a **native op** — no wrappers (mirrors [`examples/record_pipeline.py`](examples/record_pipeline.py)):

```python
import albumentations as A
import numpy as np
from recordstream import Stream, Image, Label, Mask, as_transform

records = [
    {
        "image": Image(rng.random((16, 20, 3)).astype(np.float32)),   # typed: knows its layout
        "mask": Mask((rng.random((16, 20)) > 0.5).astype(np.uint8)),
        "bboxes": [[2, 3, 6, 7]],                                     # albumentations vocabulary
        "labels": ["drone"],
        "class": Label("drone_x", classes=["noise", "drone_x"]),      # typed: knows its vocab
        "gain_db": -3.0,                                              # a scalar is just another key
    }
    for rng in (np.random.default_rng(i) for i in range(100))
]

stream = Stream(
    source=records,
    ops=[
        A.Compose(                                    # bare albumentations — as-is
            [A.HorizontalFlip(p=0.5)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        ),
        A.GaussNoise(p=1.0),                          # image only (its own kwarg vocabulary)
        as_transform(lambda d: d - 0.5, handles=(Image,)),   # native: a plain function op
    ],
).parallel(workers=4)

for record in stream:
    print(record["image"].shape, record["class"].value)   # image+mask+boxes flipped together
```

The same ops list in Confluid YAML — bare library transforms are ordinary `!class:` nodes:

```yaml
ops:
  - !class:albumentations.HorizontalFlip
    p: 0.5
  - !class:albumentations.GaussNoise
    p: 1.0
  - !class:recordstream.ops.numpy.Threshold
    low_level: 0.5
```

### Toggling a branch from the CLI (`Enable`)

Wrap any stretch of an ops list in `Enable` to switch the whole chain on or off from one flag.
The toggle is the declared `enabled` parameter; `name` identifies the wrapper so several of them
toggle independently:

```yaml
ops:
  - !class:recordstream.ops.numpy.Threshold {low_level: 0.5}
  - !class:recordstream.ops.enable.Enable
    name: visualize          # ← names THIS wrapper; scopes its CLI flag
    enabled: false           # ← off by default; the chain below is skipped
    ops:
      - !class:recordstream.ops.image.ConvertToImage {}
      - !class:recordstream.ops.debug.PrintRecordOp {}
```

```bash
recordstream run pipeline.yaml --visualize.enabled true   # this wrapper only
recordstream run pipeline.yaml --visualize.enabled+       # polarity shorthand → True
recordstream run pipeline.yaml --enabled false            # broadcast: every Enable off
```

Inner ops are not materialized until the wrapper first fires, so gating an expensive chain with
`enabled: false` costs nothing at startup. In Python the same wrapper is one call —
`Enable(ops=[...], name="visualize", enabled=False)` — which is what lets a visual editor or a
generated tool schema set the toggle too (see [docs/architecture.md](docs/architecture.md#6-every-knob-is-a-declared-parameter--the-enable-toggle-2026-07-27)).

## 📚 Documentation

| Page | Covers |
|---|---|
| [docs/record-model.md](docs/record-model.md) | The record data model: a plain dict of typed values, type-dispatched ops and kernels, mixing libraries as-is, custom item types, engines, storage layout |
| [docs/kinds.md](docs/kinds.md) | Writing ops (kernels, `field=`, type-changing ops), the collate registry (`collate_records`) + its read-back (`batch_values` / `batch_tensor` / `batch_metadata`), the Keras `RecordSequence` adapter, 1→N expanding ops |
| [docs/graph.md](docs/graph.md) | `flow:` documents + the `FlowGraph` engine, `ops:` as the linear spelling of the same step graph, expanding (1→N) steps, `Stream.from_ops_yaml` |
| [docs/sources.md](docs/sources.md) | `HuggingFaceSource`, `DatasetSplit` train/val/test views, `RangeSource`, `ConcatSource`, Confluid `!ref:` sharing, dataset identity (`dataset_uri` / `dataset_url`) |
| [docs/storage.md](docs/storage.md) | HDF5 / Zarr / Directory sinks & sources (`typedrecord-v1`), array-valued item attributes, the `SupportsMetadataScan` protocol + `MetadataFilterSource` querying |
| [docs/projection.md](docs/projection.md) | Key projection (`SupportsProjection`), lazy key walks (`iter_key`), one-peek `first_value`, `num_classes`, the fittable `LabelMap`, class-balance weights |
| [docs/predictions.md](docs/predictions.md) | The model boundary: prediction-output contracts (`ClassificationOutput` & co), `ensure_record_dataset`, the `PredictionsSink` protocol + the classification sink |
| [docs/image.md](docs/image.md) | Generic value→image conversion (`ConvertToImage`, `normalize_to_uint8`), mask→class-id conversion (`ConvertToMask`), array introspection helpers |
| [docs/configure.md](docs/configure.md) | Per-record op parameters (`ConfigureOp` and the `Capture`/`Apply` context ops) |
| [docs/runnable.md](docs/runnable.md) | Runnables (`run()` + `recordstream run`), the `@entrypoint` task/role markers + `run_entrypoint` dispatch with a worked example, `TorchRunner` / `ProgressReporting` |
| [docs/workflow.md](docs/workflow.md) | Workflow combinators (`Sequence`/`Conditional`/`Switch` + predicates): resume-safe multi-stage pipelines as ONE document |
| [docs/augmentation.md](docs/augmentation.md) | Augmentation via bare albumentations / torchvision `transforms.v2` — the op-family dispatch, key vocabulary, bbox recipes, seeding |
| [docs/architecture.md](docs/architecture.md) | Architecture decision records — the *why* behind non-obvious mechanisms (e.g. why collation is a pluggable registry) |

## 🧭 Scope: a modality-neutral engine

RecordStream deliberately contains **no domain-specific code** — every op, source and sink in this package is meaningful for any modality (arrays, tensors, images, generic metadata). Domain packages build on it and keep their own vocabulary:

- Signal/waveform items and ops (spectrograms, FFT windows, recording formats) live in the domain package, which registers its item types into the same registries.
- Task-specific trainers, collates and models live in their consuming projects.

## 🌐 Ecosystem Integration

RecordStream is designed to sit between your data catalog and your training loop, acting as the high-performance "glue" for ML pipelines:

- **Hugging Face** for community datasets and Arrow/Parquet loading — `HuggingFaceSource` turns a `datasets.Dataset` into record dicts of typed values with full metadata traceability, and [names the dataset it reads](docs/sources.md#identifying-a-dataset) so a run record can point at it (see [docs/sources.md](docs/sources.md)).
- **Confluid** for configuration: every pipeline is a YAML document, every op a `!class:` node — including bare library transforms — every run reproducible.
- **PyTorch**: `Stream` and `FlowGraph` implement the `Dataset` protocol (`__len__`/`__getitem__`/`.batch`/`.parallel`) and plug straight into a `DataLoader` with a [registry collate](docs/kinds.md#batching--collate_records--the-collate-registry-recordstreamcollate) (`collate_records` is the default).
- **Keras 3**: no `DataLoader` exists to do the batching, so [`RecordSequence`](docs/kinds.md#keras-recordsequence--the-batching-half-the-framework-leaves-to-you) is the `keras.utils.PyDataset` half — row order, slicing, per-epoch reshuffle, `collate_records` — and a `transform` callable supplies the batch shape, exactly as `collate_fn` does for torch.
- **Augmentation libraries**: [albumentations](https://albumentations.ai) and torchvision `transforms.v2` transforms run **as-is** in any ops list — the engine speaks each library's native convention (kwarg vocabulary vs dict walk), so there is nothing to wrap (see [docs/augmentation.md](docs/augmentation.md)).

## 🔧 Installation

```bash
pip install git+https://github.com/Gearlux/recordstream.git@main
```

The core engine is **numpy**, and installs no ML framework. A framework arrives only with the extra
that needs it:

| Extra | Provides |
|---|---|
| `torch` | The pieces that genuinely produce tensors — the `ToTensor` op, `batch_tensor`, and the `classification_output` / `segmentation_output` builders |
| `keras` | `recordstream.keras` — the `RecordSequence` `PyDataset` adapter and the `KERAS_BACKEND` ordering. Keras 3 is an API, so this names no compute engine; it runs on whichever of torch / TensorFlow / JAX you have |

```bash
pip install "recordstream[torch] @ git+https://github.com/Gearlux/recordstream.git@main"
```

Everything else works without either. A `Stream` is map-style (`__len__`/`__getitem__`), so a
`DataLoader` still accepts one directly on a torch install; `batch_values`, `multi_hot` and the
class-balance statistics return numpy, so a non-torch backend converts in one line. Reaching for
`recordstream.ops.ToTensor` without the extra raises an `ImportError` naming it.

## 📄 License

MIT
