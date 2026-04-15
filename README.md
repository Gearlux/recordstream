# DataFlux

**DataFlux** is a high-performance, functional data processing engine built for modern Machine Learning pipelines. It provides a clean, fluent API for streaming and transforming data from any source while maintaining strict compatibility with PyTorch and Hugging Face.

Part of the **Modular Quartet**: `LogFlow`, `Confluid`, `Liquify`, and `DataFlux`.

## 🚀 Key Features

-   **Functional Purity:** Transforms are simple Python callables. No complex base classes required.
-   **Standardized Sample Triplet:** Standardizes on `(input, target, metadata)` for full traceability.
-   **High Performance:** Native multiprocess support via `.parallel(workers=N)` using the safe `spawn` context.
-   **Advanced Storage:** Built-in support for high-performance backends:
    -   **HDF5**: Clean, efficient read/write.
    -   **Zarr**: Cloud-native, concurrent storage (Group and Batch modes).
    -   **Directory**: Robust concurrent writing for irregular data lengths.
-   **Passive Introspection:** Automatically generates JSON manifests for visual orchestration in **FluxStudio**.
-   **100% Reproducibility:** Entire pipelines are serializable via **Confluid** manifests.

## 🎯 Design Goals & Requirements

### Stream Engine
- **Functional API:** Provide a lazy, chainable pipeline API (`map`, `filter`, `batch`).
- **Standardized Samples:** Use the `Sample(input, target, metadata)` triplet as the primary data unit.
- **Parallel Execution:** Support high-performance multiprocess execution via `.parallel(workers=N)` using the `spawn` context.

### Storage
- **High-Performance Sinks:** Native support for HDF5 (sequential), Zarr (concurrent), and Directory (irregular) storage.
- **JointFlux Pattern:** Support aggregating multiple heterogeneous data sources into a single stream, preserving per-source transform chains.

### Metadata & Discovery
- **Passive Introspection:** Automatically discover available tools and ops for serialized manifests.
- **Serialization Symmetry:** Ensure full-pipeline states are serializable and reconstructible via Confluid.

## 🛠 Quick Start

```python
import numpy as np
from dataflux.core import Flux

# 1. Define a simple transformation
def normalize(data: np.ndarray, mean: float = 0.0):
    return data - mean

# 2. Build a pipeline
raw_data = [np.random.randn(10) for _ in range(100)]

flux = Flux(raw_data) \
    .map(normalize, mean=0.5) \
    .filter(lambda s: s.input.mean() > 0) \
    .parallel(workers=4)

# 3. Collect or stream
for sample in flux:
    print(sample.input.shape)
```

## 📦 Storage Integration

DataFlux makes it easy to move data between different formats:

```python
from dataflux.storage.hdf5 import HDF5Source
from dataflux.storage.zarr import ZarrGroupSink

# Stream from HDF5 to Zarr in parallel
Flux.from_source(HDF5Source("input.h5")) \
    .parallel(workers=8) \
    .map(heavy_op) \
    .to_sink(ZarrGroupSink("output.zarr"))
```

## ✂️ Train / Val Splitting

`DatasetSplit` carves a subset view out of any indexable source (implementing `__len__` and `__getitem__`). It supports three modes:

1. **Fraction mode** — pick a reproducible train/val split from a single source:

    ```yaml
    hf_train: !class:dataflux.sources.HuggingFaceSource()
      path: mnist
      split: train

    train_set: !class:dataflux.sources.DatasetSplit()
      source: !ref:hf_train
      split: train
      val_fraction: 0.1
      seed: 42

    val_set: !class:dataflux.sources.DatasetSplit()
      source: !ref:hf_train
      split: val
      val_fraction: 0.1
      seed: 42
    ```

    Same seed + same source length ⇒ deterministic, disjoint, complementary views.

2. **Range mode** — explicit slice:

    ```yaml
    first_half: !class:dataflux.sources.DatasetSplit()
      source: !ref:hf_train
      start: 0
      end: 5000
    ```

3. **HuggingFace native slicing** (alternative, no `DatasetSplit` needed):

    ```yaml
    train_src: !class:dataflux.sources.HuggingFaceSource()
      path: mnist
      split: "train[:90%]"
    val_src: !class:dataflux.sources.HuggingFaceSource()
      path: mnist
      split: "train[90%:]"
    ```

> **Note on `!ref:`** — Confluid `!ref:` resolves to the same live object as the referenced key, so a single `HuggingFaceSource` is loaded once and shared by both splits. Use `!clone:` when you want an independent deep copy instead.

## 🌐 Ecosystem Integration

DataFlux is designed to sit between your data catalog and your training loop, acting as the high-performance "glue" for ML pipelines.

### Intake (Data Discovery & Catalogs)
-   **Use Intake for:** Data discovery, remote storage abstraction (S3/GCS), and sharing "canned" datasets via YAML catalogs.
-   **Integration:** Wrap an Intake driver in a DataFlux `DataSource` to gain functional `.map()`, `.filter()`, and `.parallel()` capabilities on cataloged data.

### Hugging Face (Community & Standardized Datasets)
-   **Use Hugging Face for:** Accessing community datasets and leveraging the `datasets` library for efficient Arrow/Parquet loading.
-   **Integration:** Use DataFlux to transform `datasets.Dataset` objects into standardized `Sample` triplets, ensuring metadata traceability that often goes missing in simple dictionary-based records.

### DataFlux (The Functional Engine)
-   **Use DataFlux for:** The "inner loop" of your experiment. When you need high-performance multiprocess streaming, per-sample metadata preservation, and 100% reproducible pipelines via **Confluid** serialization.

## 🔧 Installation

```bash
pip install git+https://github.com/Gearlux/dataflux.git@main
```

## 📄 License

MIT
