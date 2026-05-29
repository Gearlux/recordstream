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
- **Discovery Categories:** `@configurable` classes are tagged with a confluid `category` (`Flux`/`JointFlux` → `dataset`, `FilterOp`/`WrappedOp` → `op`) so tools like navigaitor's `list_configurable_classes(category=...)` enumerate them by kind.
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

## 🏷 Type Specs

`dataflux.typespec` describes *what flows through a `Sample`* and lets ops declare what they accept/produce, so tools like FluxStudio can filter which nodes may connect. It is flexible by design — N-dimensional arrays across numpy/torch/tensorflow, **per-axis bounded ranges**, dtype families, images, and arbitrary Python types — and anything left unspecified defaults to `Any`.

```python
from dataflux.typespec import SampleType, ArrayType, Dim, PythonType, UnionType

# "a 2-D float array whose first axis is 1–10, second axis any size"
ArrayType(shape=(Dim.range(1, 10), Dim.any("N")), dtype="floating")
ArrayType.parse("3 h w", dtype="float32", framework="torch")  # jaxtyping-style shorthand
ArrayType.image("CHW", channels=3, dtype="float32", framework="torch")  # an image convenience
```

**Declare an op's contract** with the class attributes `ACCEPTS` / `PRODUCES` (each a `SampleType`; both default to `Any`, so annotating is optional and backward-compatible). No base class — transforms stay plain callables:

```python
@configurable
class StandardizeOp:
    ACCEPTS = SampleType(input=UnionType((ArrayType(dtype="numeric"), PythonType("PIL.Image.Image"))))
    PRODUCES = SampleType(input=ArrayType(dtype="floating", frameworks={"numpy"}))
    def __call__(self, sample): ...
```

Matching is asymmetric: `consumer.accepts(producer)` is strict (used at runtime against a concrete inferred type); `compatible(consumer, producer)` is permissive (used at edit time — `Any`/unknown on either side passes). A `Sample`'s own type comes from `sample.describe()` — it returns a type stored in the reserved metadata keys `__features__` (a `datasets.Features` dict) + `__spec__` (sidecar refinements), or infers one from the live data; attach a stored type with `sample.with_type(SampleType(...))`.

## 🔎 Field Projection & Class Counting

Walking a source for a single field (the classic case: counting classes from
*targets*) shouldn't pay to build the fields you don't need. `dataflux.projection`
adds an opt-in protocol plus lazy helpers:

```python
from dataflux import project, iter_targets, num_classes

# A source MAY implement SupportsProjection (`project(fields)`) to skip building
# unrequested fields — e.g. an image dataset reads only the label column for a
# target-only walk, never decoding an image.
for sample in project(my_source, ("target",)):
    ...                       # sample.input is None; sample.target populated

labels = list(iter_targets(my_source))     # lazy
n = num_classes(my_source)                 # max(class_id) + 1 — always walks
```

Sources that don't implement `SupportsProjection` still work via a correct
full-iteration fallback (just without the skip-decode speedup). `num_classes` is
a free function, not a `Flux` method: integer class-id semantics are
classification-specific, so the task-agnostic engine doesn't advertise it.

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

## 🔁 Reattach an ops-only YAML (`Flux.from_ops_yaml`)

A `{ops: [!class:…()]}` document — e.g. one exported from a FluxStudio canvas (`fluxstudio export …`) — can be attached to any source:

```python
from dataflux import Flux
from dataflux.sources import HuggingFaceSource

flux = Flux.from_ops_yaml("ops.yaml", source=HuggingFaceSource(path="mnist"))
```

The helper **materializes** the deferred `!class:` markers before attaching (via `confluid.materialize`) — necessary because `confluid.load` leaves markers nested under a mapping key deferred, and a `Flux` rejects deferred markers at iteration by design. The manual equivalent is `Flux(source=src, ops=confluid.materialize(confluid.load("ops.yaml")["ops"]))`.

## 🔗 Paired Join (Binary ↔ Annotations)

`PairedSource` joins a primary `DataSource` (e.g. raw binary samples) with a secondary mapping-shaped annotation store via a key function. It generalises the common "I have data, and I have a sidecar file of annotations that covers some of it" pattern. Three join policies cover the scenarios we actually see in ML research:

| Policy | Iterates | Use case |
|---|---|---|
| `left_outer` (default) | Every primary sample; attaches annotation when the key matches | Process everything, use labels where available |
| `inner` | Only primary samples whose key is in the store | Train/evaluate on the labeled subset |
| `right_driven` | Every key in the annotation store; resolves the primary sample via `primary_resolver(key, primary)` | Very sparse labels where full-primary enumeration is costly |

```yaml
primary: !class:waivefront.rfuav.data.source.RFUAVSource()
  root: /Volumes/Data/RFUAV
  window_samples: 1000000

labels: !class:annotaide.store.JSONFileAnnotationStore()
  path: /Volumes/Data/RFUAV-labels

paired: !class:dataflux.paired.PairedSource()
  primary: !ref:primary
  secondary: !ref:labels
  key_fn: "waivefront.rfuav.keys:sample_window_key"
  policy: left_outer
```

Annotation records are **flattened into `Sample.metadata`**, so a detection record `{bboxes, labels, scores}` shows up as three independent metadata keys. Two `metadata` keys are always populated: `annotated: bool` and `annotation_key: str`. Optional `prefix` and `store_full_under` parameters shape the layout.

### Coarser-granularity keys (broadcast and slicing)

`key_fn` is free to return a coarser key than the sample granularity. When multiple primary samples map to the same key, they all look up the same record:

- **Without `extract_fn`** — the record is broadcast identically into every matching sample's metadata (e.g. a scalar pack-level class label inherited by every window of that pack).
- **With `extract_fn`** — the record is projected per sample. The callable is invoked as `extract_fn(record, sample) -> dict | None`; returning `None` marks the sample unannotated (and filters it under `policy="inner"`). Use this when a pack-level annotation carries time-ranged content that must be trimmed to each window's bounds.

Multi-granularity joins (e.g. pack-level + window-level annotations merged together) compose by chaining `PairedSource` instances — the output of one is itself a `DataSource` that the next can consume.

### Callable resolution

`key_fn`, `extract_fn`, and `primary_resolver` all accept either a callable **or** a `"module:function"` string path resolved through `dataflux.discovery.resolve_callable`. The string form is what survives YAML round-trip via Confluid.

See [`examples/paired_annotations.py`](examples/paired_annotations.py) for a runnable end-to-end walkthrough of all four scenarios.

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
