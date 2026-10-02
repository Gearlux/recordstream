# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`recordstream.sources.draw.DrawSource` and the draws `Choice` / `Uniform` / `Span` / `Repeat`
  (`recordstream.draws`)** — a generator's settings drawn anew for every record. The draws run top to
  bottom and each picks only among the values the generator accepts given the draws above it (its
  constructors and `check()` judge), so every record is a valid example; a `Choice` with no values
  takes every value the setting's type allows, a `Uniform` the setting's own range. Each record carries
  the drawn generator's settings file under `settings`, and depends only on the seed and its index.
  A list of draws that cannot be met raises `DrawRefused` with the generator's reason; a spec that
  cannot be read, `DrawSpecError`. Usage: `docs/sources.md`.
- **`GraphContract`'s class-vocabulary slot is `classes`** — an output slot named like a constructor
  parameter of any class the graph delivers is refused at declaration time (confluid broadcasts the
  slot's value into that parameter; measured on `class_names` beside a `Stream`), and the slot accepts
  a typed list or a wired source that declares its class names (`docs/graph-contract.md`).
- **`recordstream.flow.Tracer`** — one record through a `Stream` or a `FlowGraph` on the engine's
  own kernel, a snapshot per node (`ops[i]` for a Stream, the step keys for a flow). `check(seed)`
  refuses an unmet need before anything runs, located at the node, and carries flags across nodes;
  `run(seed, until=node)` pauses BEFORE a node; `step()` / `resume()` continue; `rerun_from(node,
  **params)` rebuilds ONE node through its constructor from its current values and recomputes only
  from there on (a refused value leaves the trace untouched); `value(node, entry)` hands out the
  real object; `to_dict()` is JSON with arrays summarised. Snapshots are references unless
  `copy_snapshots=True`; an op that edits its record in place is flagged `in_place`.
  Usage: `docs/trace.md`.
- **`recordstream.flow.subgraph.Subgraph`** (also `from recordstream.flow import Subgraph`) — a
  `flow:` mapping used as ONE op, written inline in the step that uses it (`steps:` the flow grammar,
  `result:` the inner step whose record it returns; blank = the last). The inside runs on the same
  kernel as any flow. `consumes` / `produces` / `flags` are derived from the inner ops (`produces` and
  `flags` from the steps on the lineage of `result` only — the record the subgraph returns). Refused
  before the first record, naming the outer step and the line of the subgraph that refuses (the inner
  marker of a nested one): a `result:` naming no inner step, a 1→N
  expanding inner op, an inner `from:` / `merge_from:` / `bind:` naming a step outside, a `from:` or
  `merge_from:` written inside an inner op's `!class:` marker, and an outer `bind:` / `from:` /
  `merge_from:` reading a step inside a subgraph. One subgraph is used twice by copying its block or by
  `steps: {include: prep.steps.yaml}` in each use. Usage: `docs/graph.md` → "Subgraphs".
- **`Tracer` looks inside a subgraph step** (in a `FlowGraph` or a `Stream`'s ops list): its inner
  nodes follow it in `names` as `prep/grey` (`ops[1]/grey`, nested `prep/inner/x`), and `run(until=…)`,
  `step()`, `resume()`, `rerun_from()`, `value()`, `statuses`, `generations`, `check()` and
  `to_dict()` take them (`to_dict` rows carry `parent`, a subgraph row `inner`).
- **`recordstream.flow.StepReferenceError`** — the `ValueError` a `from:` / `merge_from:` / `bind:`
  naming no earlier step raises; its `key`, `ref`, `target`, `step` and `param` say which reference.
- **`check_chain(..., raised=…)`** — the flags already raised when the chain starts (the steps ahead of
  a subgraph, when the chain is its inside). Default: none.

### Changed
- A `flow:` step name may not contain `/` (it names a node inside a subgraph, `prep/grey`):
  `flow: step name 'prep/grey' may not contain '/' …`.
- `parse_flow` no longer changes the steps mapping it is given: a bare marker's `from:` /
  `merge_from:` / `bind:` are read off a copy, so parsing the same steps twice wires the same graph.
- `Tracer` with `copy_snapshots=True` reruns a first node from a copy of the record as it arrived, so a
  first node that edits its record in place gives the same result on every rerun; a refusal raised while
  a tracer builds its plan starts with the tracer's `where`.
- A `Stream` holding a stream-level op (`Parallel`) opens every subgraph member before the first
  record and names it by its own index in `ops`.

## [0.1.0a2] — 2026-09-27

### Added
- **`recordstream.__version__`** reports the installed version, read from the package metadata
  (`importlib.metadata.version("recordstream")`), so it always matches `pyproject.toml`; an uninstalled
  source tree reports `0.0.0.dev0`.
- `recordstream.ops.contract.RecordContract` — a pass-through interface op asserting what
  each record carries at the boundary where it sits (`fields`: record key → registered
  item type name, `"*"` = present with any type). One class serves input and output
  boundaries, decided by position in the ops list; a violation raises `ContractError`
  naming the boundary, record ordinal, offending entry, and the entries present.
- **Chain contracts.** An op may declare `consumes` / `produces` (record key → item type),
  `reports`, `flags` and a per-instance `requires`. `check_chain(ops, provided=…, where=…)`
  walks a chain once and raises `ChainContractError` naming the op for an unmet `consumes`,
  a flag no earlier op raises, or a flag or report name declared twice. A chain that
  declares none of these passes unchanged. `flag_producers(ops)` maps each flag to the op
  that raises it.
- **Algorithms.** `Algorithm` with the slot declarations `Param` / `Input` / `Output`: an op
  states what it is tuned by, what it reads and what it computes, and writes `compute()`.
  Run as an op, it reads each input from the record entry of the same name and writes each
  output the same way (`keys` names other entries; `Output(replaces=...)` writes back where
  an input was read). The constructor, the settings schema, the output sockets, `consumes` /
  `produces` for `check_chain`, and `algorithm_spec()` are derived from the declarations.
  `run(**inputs)` computes without a record.
- **`ClassNamesOutput`** — the class vocabulary a pipeline delivers, as a graph output
  (`class_names`, `num_classes`); **`ClassNamesScan`** derives one by walking a source's
  label column, for a source that declares none.
- **File lists.** `FilesSource` delivers file paths as `{file}` records, from an explicit
  `files` list or from a folder (`root` + a `pattern` glob: `"*/*"`, `"**/*"`,
  `"one_dir/*"`). Directories and dotfiles are never listed, a pattern that matches nothing
  raises naming the folder and the pattern, and `files` together with `root` is refused.
  An `exclude` glob keeps a paired format's companion half out of the listing.
  `ReadImage` decodes the named file into an RGB `Image`; a file it cannot open passes
  through unchanged.
- **The file-format registry** (`recordstream.formats`). A package registers a
  `FileFormat` under the `recordstream.formats` entry-point group (`FORMAT_GROUP`);
  `file_formats()` lists what is installed. `ReadFile` decodes a `{file}` record through
  the format that matches it and refuses an unknown file naming the installed formats.
  `FilesSource` leaves out exactly the files an installed format consumes as a companion;
  `sibling()` / `bare_name()` pair a file with its companion. No formats ship in
  recordstream itself.
- **Answers from metadata alone.** A format may implement `scan` / `scan_file` to answer
  from its sidecar or header without decoding the payload. `Stream.project` runs that
  cheaper chain and re-runs the full one for any record it could not answer
  (`formats.answers`, `STAND_IN`). Every view source (`RangeSource`, `ConcatSource`,
  `DatasetSplit` views, `FilesSource`) forwards a key-restricted walk through
  `project_indices`, so projecting through a view keeps the source's fast path.
- **Detection review.** `ConvertToBoxes` / `ConvertFromBoxes` convert between canonical
  pixel xyxy and a source's own box layout; `box_iou`, `match_boxes` and `size_bucket`
  back review metrics. `Boxes` carries an optional `classes` vocabulary, like `Label`.
  `HuggingFaceSource.class_names` reads a detection dataset's nested vocabulary
  (`objects.category`).
- **`Normalize`** — per-channel standardization, `(x / max_value - mean) / std`, with
  ImageNet defaults; a uint8 `Image` comes out a float32 `Image`.
- **`Scale`** (`recordstream.ops.numpy`) maps a value range onto another,
  `[source_min, source_max]` → `[target_min, target_max]` (default `0..1`). A blank source bound
  is the integer type's full range, so a bare `Scale` takes `uint8` to `0..1`; a 12-bit sensor
  in `uint16` names `source_max: 4095`. A blank bound on a float is refused.
- **`ToType`** (`recordstream.ops.numpy`) casts to one of `float16` / `float32` / `float64` /
  `complex64` / `complex128` / `uint8` / `int16` / `int32` / `int64`, values unchanged. It
  refuses complex → real and values an integer type cannot hold.
- **`ConvertMode`** (`recordstream.ops.image`) forces a channel layout — `RGB`, `RGBA` or `L` —
  on `uint8` images; anything else is refused.
- `Scale`, `ToType` and `ConvertMode` change every `Image` when `field` is blank (never a
  `Mask`) and exactly the named entry when `field` is set.
- **`RandomNumber`** — a uniformly random number; wire the op itself for a fresh draw per
  record. **`PutField`** puts a value into the record under `key`; a callable value is
  called per record.
- **`ModelPredict`** gains `predict_batch` (one forward per chunk of records), `batch_size`,
  and `frame` (detection boxes scaled back to the size of a named entry).
- `HuggingFaceSource` reads only the dataset columns a projection asks for, and declares
  `class_names` as an output.

### Changed
- `ModelPredict(device=…)` defaults to `"auto"` (cuda, then mps, then cpu) and accepts only
  `auto` / `cpu` / `cuda` / `mps`; naming a device the machine lacks raises, listing the
  devices it has.
- A classification `ModelPredict` adds the confidence (the max probability) under `score`
  beside the predicted label; `score=""` turns it off.
- `ModelPredict` calls a model's `solidify()` only to build it and keeps calling the model
  it was given; the value `solidify()` returns no longer replaces it.
- **`ToTensor` converts and nothing else**: an array becomes a CHW tensor of the same element
  type, so `uint8` pixels arrive as a `uint8` tensor. It takes `field` and `output` only; the
  channel layout is `ConvertMode`, the value range `Scale`, the element type `ToType`, each
  placed before it. `to_tensor(img)` likewise takes the array only.

### Fixed
- **Array items keep their attributes through pickling**, so they survive a spawn worker
  (`Stream(...).parallel(n)`, `FlowGraph.parallel`, a DataLoader worker) in both directions.
  Before, every declared attribute came back as its class default: an `Image` with
  `layout="CHW"` arrived as `"HWC"` without an error, and a `db` spectrogram arrived as
  `scaling="none"`. Every `NDArrayItem` subclass inherits the fix. Storage was never
  affected, because it goes through the item codec.

## [0.1.0a1] — 2026-08-25

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
