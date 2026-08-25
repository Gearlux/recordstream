# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a1] — unreleased

First public pre-release. The surface it ships:

### The record model
- A record is a plain `dict` of typed values — `Image`, `Mask`, `Boxes`, `Label`,
  `MultiLabel` — each owning its own metadata. Key names carry meaning; there is no
  wrapper container and no role tags.
- Array-backed items subclass `NDArrayItem`, so numpy operations preserve their declared
  attributes. Structured items are dataclass wrappers. `register_item` opens the set to
  any package.
- `recordstream.io` is the serialization codec, so an externally registered item type
  round-trips through storage with no backend change.

### The engine
- One step-graph engine behind two facades: `Stream` / `JointStream` for the dataset
  surface (`__len__` / `__getitem__` / `.batch` / `.parallel` / `.project`), and
  `FlowGraph` for a `flow:` document of named steps with `from:` / `merge_from:` /
  `bind:` edges. An `ops:` list is the same engine's linear spelling.
- Ops dispatch on value type: a `Transform` samples its parameters once per record and
  applies a per-type kernel to every value it handles.
- Bare albumentations and torchvision `transforms.v2` transforms run as-is through the
  op-family dispatch — one call is one joint draw across image, mask and boxes. The
  family registry (`register_op_family`) is open to other libraries.
- 1→N expanding ops fork the remaining subgraph, in every route including spawn-parallel.
- Multiprocessing uses the `spawn` context; `ensure_materialized` and the OpenCV thread
  guard cover the two fork hazards a forked loader hits.

### Storage
- HDF5, Zarr and Directory sinks, each with a matching source, over the `typedrecord-v1`
  layout.
- Metadata is queryable without loading arrays: the `SupportsMetadataScan` protocol plus
  `MetadataFilterSource`.

### Sources and projection
- `HuggingFaceSource`, `DatasetSplit` train/val/test views, `RangeSource`, `ConcatSource`.
- Every source names the data it reads through `dataset_uri` / `dataset_url`, and a view
  propagates the handle verbatim.
- Key projection (`project`, `iter_key`, `first_value`, `num_classes`, `class_names`),
  the fittable `LabelMap`, and class-balance statistics.

### Running
- `recordstream run <config.yaml>` runs any Confluid-wired runnable — a trainer, an
  evaluator, a dataset processor, a workflow. The `@entrypoint` markers are the dispatch
  table for a runnable that drives several tasks off one `task` knob.
- Workflow combinators (`Sequence` / `Conditional` / `Switch`) plus predicates, so a
  resume-safe multi-stage pipeline is one document.

### Batching
- `collate_records` (the `"record"` default) and `collate_list`, differing in one
  decision — whether array payloads stack — plus the read-back half (`batch_values`,
  `batch_boxes`, `batch_tensor`, `batch_metadata`, `multi_hot`).
- `recordstream.keras.RecordSequence` is the Keras `PyDataset` half, since Keras 3 has no
  `DataLoader` to do the row scheduling.

### Install shape
- The core engine is numpy and installs no ML framework. `[torch]` adds the pieces that
  genuinely produce tensors; `[keras]` adds the `PyDataset` adapter and names no compute
  engine; `[vision]` enables the bare torchvision transform family.
