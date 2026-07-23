# SampleFlux

**SampleFlux** is a high-performance, functional data processing engine built for modern Machine Learning pipelines. It provides a clean, fluent API for streaming and transforming data from any source while maintaining strict compatibility with PyTorch and Hugging Face.

Part of the **Modular Quartet**: `Loggair`, `Confluid`, `Liquifai`, and `SampleFlux`.

## 🚀 Key Features

-   **Functional Purity:** Transforms are simple Python callables. No complex base classes required.
-   **Typed Bag of Items:** A `Sample` is a named bag of typed items (`Image`, `Mask`, `Regions`, `Label`, …), each owning its own metadata — and [transforms dispatch on item type](docs/typed-model.md), so one sampled decision moves image + mask + boxes together.
-   **Graph pipelines, serial engine:** readable [`flow:` documents](docs/graph.md) with named steps, fan-out/fan-in and per-sample `bind:` parameters — executed natively by `FlowGraph` or lowered (bidirectionally, with pinned execution parity) to a flat context-ops list on the plain sequential `Flux` engine.
-   **High Performance:** Native multiprocess support via `.parallel(workers=N)` using the safe `spawn` context; [1→N expanding ops](docs/kinds.md#1n-expanding-ops-iterable-only-pipelines) flatten in every route.
-   **Advanced Storage:** HDF5, Zarr and Directory backends with matching read-back sources and [metadata-only querying](docs/storage.md#queryable-metadata-samplefluxstoragequery) — filter stored datasets without loading a single array.
-   **Passive Introspection:** transforms declare their [item-type contracts](docs/typed-model.md) (the item types they handle / consume / produce) and are discoverable by category for visual editors and schema generators.
-   **100% Reproducibility:** Entire pipelines are serializable via **Confluid** manifests.

## 🛠 Quick Start

```python
import numpy as np
from sampleflux import Sample, Image, Flux, as_transform, primary

# 1. A plain function becomes a transform, dispatched on item type
recenter = as_transform(lambda d: d - 0.5, handles=(Image,))

# 2. Build a pipeline over a source of typed samples
raw_data = [Sample({"input": Image(np.random.randn(10))}) for _ in range(100)]

flux = (
    Flux(source=raw_data, ops=[recenter])
    .filter(lambda s: primary(s, "input")[1].mean() > 0)
    .parallel(workers=4)
)

# 3. Collect or stream
for sample in flux:
    _, item = primary(sample, "input")   # (key, item)
    print(item.shape)
```

## 📚 Documentation

| Page | Covers |
|---|---|
| [docs/kinds.md](docs/kinds.md) | The transform taxonomy (field scope × call style), multi-type carriers (`Flux(native=True)`), the collate registry, 1→N expanding ops |
| [docs/graph.md](docs/graph.md) | `flow:` documents + the `FlowGraph` engine, the six Context ops on the serial engine, bidirectional flow⇄ops conversion, `Flux.from_ops_yaml` |
| [docs/sources.md](docs/sources.md) | `HuggingFaceSource`, `DatasetSplit` train/val/test views, `RangeSource`, `ConcatSource`, Confluid `!ref:` sharing |
| [docs/storage.md](docs/storage.md) | HDF5 / Zarr / Directory sinks & sources, array-valued metadata, the `SupportsMetadataScan` protocol + `MetadataFilterSource` querying |
| [docs/projection.md](docs/projection.md) | Field projection (`SupportsProjection`), lazy target walks, `num_classes`, the fittable `LabelMap` |
| [docs/image.md](docs/image.md) | Generic value→image conversion (`ConvertToImageOp`, `NormalizeToUint8Op`), array introspection helpers |
| [docs/configure.md](docs/configure.md) | Per-sample op parameters (`ConfigureOp` and the `Capture`/`Apply` context ops) |
| [docs/augmentation.md](docs/augmentation.md) | Augmentation via albumentations / torchvision `transforms.v2` — joint input+target (mask/boxes) adapters, the generated `Alb*`/`Tv*` per-transform ops, seeding, Confluid-native YAML |
| [docs/typed-model.md](docs/typed-model.md) | The typed-bag data model: a `Sample` is a named bag of typed items (each owning its metadata), type-dispatched transforms, torchvision/albumentations adapters, custom item types |
| [docs/architecture.md](docs/architecture.md) | Architecture decision records — the *why* behind non-obvious mechanisms (e.g. why collation is a pluggable registry) |

## 🧭 Scope: a modality-neutral engine

SampleFlux deliberately contains **no domain-specific code** — every op, source and sink in this package is meaningful for any modality (arrays, tensors, images, generic metadata). Domain packages build on it and keep their own vocabulary:

- Signal/waveform work (1-D FFT + windowing ops, SigMF recording storage, spectrograms, the annotation-join source) lives in the **waivefront** package.
- Task-specific trainers, collates and models live in their consuming projects.

## 🌐 Ecosystem Integration

SampleFlux is designed to sit between your data catalog and your training loop, acting as the high-performance "glue" for ML pipelines:

- **Hugging Face** for community datasets and Arrow/Parquet loading — `HuggingFaceSource` turns a `datasets.Dataset` into typed `Sample` bags with full metadata traceability (see [docs/sources.md](docs/sources.md)).
- **Confluid** for configuration: every pipeline is a YAML document, every op a `!class:` node, every run reproducible.
- **PyTorch**: `Flux` and `FlowGraph` implement the `Dataset` protocol (`__len__`/`__getitem__`/`.batch`/`.parallel`) and plug straight into a `DataLoader` with a [registry collate](docs/kinds.md#multi-type-carriers--the-collate-registry-samplefluxcollate).
- **Augmentation libraries**: `AlbumentationsOp` / `TorchvisionTransformOp` wrap [albumentations](https://albumentations.ai) and torchvision `transforms.v2` as ops that augment input AND target (mask / detection boxes) jointly — plus an auto-generated op per individual library transform (`AlbHorizontalFlip`, `TvColorJitter`, …), each a graph node and a Confluid `!class:` one-liner (see [docs/augmentation.md](docs/augmentation.md); torchvision via `pip install "sampleflux[vision]"`).

## 🔧 Installation

```bash
pip install git+https://github.com/Gearlux/sampleflux.git@main
```

## 📄 License

MIT
