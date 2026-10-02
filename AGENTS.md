# RecordStream Mandates

Rules only. The reasons are in [`docs/architecture.md`](docs/architecture.md) (cited as §N; two
records carry the number 8, cited as "§8 autograd" and "§8 model boundary"), usage is in the
[README](README.md) and `docs/*.md`, and most mechanisms explain themselves in their docstring.
Root `AGENTS.md` rules are not repeated here.

## Current state

The dataset engine, on PyPI as pre-release `0.1.0aN` from a public repo. A record is a plain
`dict` of typed values. Sources yield records; ONE per-record step-graph kernel runs ops over them
behind two facades, `Stream`/`JointStream` (the dataset surface) and `FlowGraph` (a `flow:`
document); storage sinks write them. On top sits the runnable layer: `recordstream run`,
`@entrypoint`, `Sequence`/`Conditional`/`Switch`, `DatasetProcessor`. The core is numpy; `torch`,
`keras`, `vision` and `notebook` are extras.

| area | modules |
|---|---|
| data model | `items.py`, `io.py` (codec), `dispatch.py`, `transform.py` (`Transform`, `Pipeline`), `algorithm.py` |
| engine | `core/` (`families` → `mapstyle` → `wrappers` → `stream`), `flow/` (`steps` → `parse` → `execute` → `graph`, plus `subgraph`, `trace`) |
| ops | `ops/`: `numpy`, `image`, `torch`, `target`, `structure`, `formula`, `formats` (`ReadFile`), `contract`, `predict`, `sink`, `debug`, the compose ops |
| sources | `sources/`: `huggingface`, `files`, `split`, `range`, `concat`, `draw`; `draws.py`; `formats.py` (file-format registry) |
| storage | `storage/`: `hdf5`, `zarr`, `directory`, `query`, `cache` |
| batching | `collate.py`, `batch.py` (read-back), `keras.py`, `loaders.py` |
| boundaries | `projection.py`, `uri.py`, `labels.py`, `outputs.py`, `predictions.py` |
| runnables | `runnable.py`, `workflow.py`, `processing.py`, `cli.py`; `discovery.py` |

Executed examples: `examples/*.py`; notebook: `notebooks/01 - cat_exploration.ipynb`.

*Sample* is not a record. It keeps its other meanings only: a signal sample (`samplerate`,
`window_samples`), a random draw (a `Transform` samples its params once per record), an external
key (LabelStudio's `sample_id`). Do not rename those.

Gotchas:
- `ToTensor` emits a live CHW `torch.Tensor` as a PLAIN record value; an `NDArrayItem` cannot hold one.
- `FormulaOp`'s sandbox adds `amax`/`amin`/`mean`/`std`/`median`, function style. `Switch`'s knob is `select`.
- `HuggingFaceSource` yields `image`/`class` plus every other column (`metadata_features="*"`, the
  default and the ONE sentinel; add no other magic string).
- A `flow:` step carrying `bind:` uses the plain-mapping `op:` form; confluid consumes a mapping
  under a `!class:` marker as addressed config.

## Scope

- Modality-neutral: an op, source or sink here must make sense for an image, a waveform AND a
  tabular dataset. Otherwise it belongs in a domain package, which registers its items, formats
  and op families into the same registries. Ship only generic items and no native augmentation.
- Framework-neutral: `import recordstream` imports no ML framework (`tests/test_optional_torch.py`).
  Code that needs torch or keras sits behind its extra with a lazy import. Recognise a tensor with
  `_compat.is_torch_tensor` (`sys.modules`), never by importing. Helpers return numpy; only
  `ops.torch`, `batch_tensor`, the `outputs` builders and `loaders` are torch. Annotate with the
  real `MapStyle` class, never the string. §9.

## Record model

- No container, no roles, no type field: key names carry meaning, a value's type is its class,
  metadata lives on the value (`_item_attrs` / dataclass fields) or as a plain key. §1.
- Inside a kernel read a payload with `item_data`; at a boundary that reads a configured key use
  `item_value` (a source may have wrapped the value in a `Label`). Resolve a `field=` entry with
  `resolve_item` / `resolve_entry`, never a hand-written find loop.
- An `NDArrayItem` subclass overriding `__reduce__` / `__setstate__` extends the base ones
  (`tests/test_items.py::TestPickle`). §1 amendment.
- `Boxes` is pixel-only: half-open `[x0, y0, x1, y1]` on the `(H, W)` `canvas`. Another coordinate
  system is a new item in its domain package. §14.
- Every op that makes or re-frames a `Boxes` fills `canvas`, also for an empty target. The frame
  lookup stays narrow: `"image"`, then the first `Image`, else `None`
  (`tests/test_convert_to_mask.py::TestBoxesKnowTheirFrame`).
- Serialize only through `io.py` (`register_io` per exact type).

## Ops

- Three shapes. `Transform` + `@Op.kernel(ItemType)` for "every value of a type" (params drawn
  once per record). `Algorithm` for a NEW op that reads and writes named entries (§21). A bare
  library transform, as-is. Existing hand-written type-changing ops keep working and migrate one
  at a time.
- `Algorithm` (`tests/test_algorithm.py`): never write `__init__`; outputs are read-only and
  `compute()` returns them; inputs are found by NAME only (user decision 2026-09-27), so a pipeline
  with another name sets `keys` or adds a `RenameField`; `replaces` is declared in the class (user
  decision 2026-09-27); `run()` works on a shallow copy; slots are field specifiers, not
  annotation markers.
- Libraries run as-is through `core.families._apply_op`. Never write an adapter class. A new
  library is one `register_op_family(name, matcher, invoker)` with module-level functions (spawn
  workers re-register them) that match by MRO name without importing (`tests/test_op_families.py`).
- Every composing op (`Pipeline`, `RandomApply`, `Enable`, `Parallel`, `ConfigureOp`) applies its
  inner ops through `_apply_op`, never `op(record)`, and returns `Optional[Record]`
  (`tests/test_pipeline.py`).
- A geometry-changing albumentations or torchvision-v2 transform warns once per type when a
  `Boxes` sat out the call. Classify by the library's own taxonomy, never by applying the
  transform to probe boxes (that draws from the RNG). The v2 check reads a private module and
  fails open (`::test_the_geometry_signal_still_matches_this_torchvision`).
- `cv2.setNumThreads(0)` fires on the first albumentations USE, never at import
  (`tests/test_fork_safety.py`). It and `ensure_materialized` guard different fork crashes; a
  forking consumer needs both.
- `ToTensor` converts only (user instruction 2026-09-27). Range, layout and type are `Scale`,
  `ConvertMode`, `ToType` placed before it; `ops.numpy.entries_to_change` picks their entries
  (blank `field` = every `Image`, never a `Mask`). `Scale` gets no data-driven default. §20.
  (`tests/test_scale_and_to_type.py`, `tests/test_convert_mode.py`,
  `tests/test_typed_target_ops.py::TestToTensor`)
- `ConvertToMask` converts only, to an `int64 [H, W] Mask` under `output="mask"` (the albumentations
  key, so a joint transform moves it with the image). No `offset`/`mapping`/`dtype` knobs:
  `FormulaOp`, `EncodeTarget` and `batch_tensor(dtype=)` do those. An RGB mask raises
  (`tests/test_convert_to_mask.py`).
- The `ops.image` helpers (`value_to_image`, `normalize_to_uint8`, `array_histogram`,
  `draw_text`, …) are functions, not ops. `normalize_to_uint8` is the one quantizer.
  `array_histogram` passes explicit `np.linspace` edges, never `bins=<int>` (numpy 2.2 fails above
  65 536 elements). matplotlib stays a lazy import. Domain rendering stays in its domain package.
- `ConfigureOp` is the per-record-parameter op. Wiring between steps is step grammar (below), never
  an op.

## Configuration surface

- A knob a front end must set is a declared, defaulted, `Args:`-documented constructor parameter,
  never an undeclared setattr attribute; a derived value is declared too (`None` derives). `Enable`
  is the reference. §6. (`tests/test_enable.py::TestIntrospectionContract`)
- Every node-facing class documents each `__init__` param on one line of an `Args:` block; an
  `Algorithm` gets it from `doc=` (`tests/test_node_docs.py`).
- Stricter than the root rule: every `@configurable` here must build as `Cls()`
  (`tests/test_lazy_construction.py`). An op validates in `__call__`, a view source in a cached
  property, storage in `.open()`.
- A `source:` slot holding a deferred marker (`_partial_: true`, a hand-built `PartialClass`)
  raises with guidance and is never flowed (user decision 2026-08-10). The free functions
  (`project`, `dataset_uri`, `LabelMap.encode`) and the `ops:` list do flow one
  (`tests/test_view_sources_deferred.py`).
- A pipeline config round-trips through confluid YAML and gives identical output after loading.

## Discovery

- Categories name a role (§25, `tests/test_categories.py`): `engine`, `source` (views included),
  `op` (every canvas op; the palette is a positive allowlist, so an untagged op vanishes), `sink`
  (storage sinks only), `value`, `contract`. A YAML-only class gets none (`FilterOp`, `WrappedOp`,
  `LabelMap`, the storage sources, the draws, `RecordSequence`). Every op also gets a `group=`. A
  predictions sink is never `category="sink"`.
- `__all__` is load-bearing in `sources`, `core` and `flow`: `scan_module` finds nothing in a
  package `__init__`, so a missing name drops from the palette. A new source = one module + a
  re-export + an `__all__` entry. One entry point per package; a new `ops.*` or `storage.*`
  module gets its own, then `aisland setup`.
- Generators write the SUBMODULE `!class:` path (`recordstream.core.stream.Stream`). Never pin
  `__module__` back; it breaks `confluid.registry.key_for` and `inspect.getsource`. §11, §12.

## Execution

- One kernel, `flow.execute.run_steps_multi`; an `ops:` list compiles to positional steps.
  Fan-out, fan-in and cross-step values are step grammar (`from:` / `merge_from:` / `bind:`),
  never ops. `from:` names an earlier step. The linear fast path must match the general path. A
  branchy graph has no flat spelling (`to_stream()` raises). Never reintroduce a lowering pass or
  a context/cell plane. §3. (`tests/test_typed_flow.py`)
- An `EXPANDS = True` op forks the remaining steps per child, depth-first; the pipeline is then
  iterable-only (`__len__`/`__getitem__` raise) and `run_steps` raises rather than drop siblings
  (`::TestExpandingSteps`).
- Layering (§12, `tests/test_module_layout.py`): imports run one way inside `core/` and `flow/`;
  `flow.execute` imports `core.families` at module level, `core.stream` imports `flow` only inside
  function bodies. Private names re-exported by `core/__init__.py` stay out of `__all__`.
  Monkeypatch the module that USES a symbol, not the one defining it.
- Parallel routes use the `spawn` context; every op must pickle.
- `Tracer` (§22, `tests/test_trace.py`) is a probe around the SAME kernel: when the kernel touches
  an op in a new way, extend the probe, never special-case the tracer in the kernel. Its
  constructor stores only; snapshots are references (`copy_snapshots=True` is opt-in);
  `rerun_from` rebuilds through the constructor from the op's current values; `check()` checks
  each node over its record lineage and raises a located `ChainContractError` before any node runs.
- `Subgraph` (§23, `tests/test_subgraph.py`, `tests/test_trace_subgraph.py`): the inside runs on
  the same kernel, never flattened, never on its own executor.
  - The parameter is `result`, never `outputs` (confluid broadcasts a document's `outputs:`).
  - `consumes` / `produces` / `flags` are derived, on the lineage of `result`.
  - Every structural mistake is refused before the first record, naming the outer step and
    `file:line:col`: an inner reference to an outer step, an outer `bind:` into an inner step
    (user decision), an expanding inner op, a `from:` inside an inner `!class:` marker.
  - Inner nodes are `<step>/<inner>`, so a step name may not contain `/`.
  - Never import `subgraph.py` from `parse.py` or `core`; duck-type on `FLOW_SUBGRAPH`.
  - Reuse is by copy, never YAML anchors or `!ref:`.
- A pipeline's interface is the pass-through op `RecordContract`; its position is its role, so
  it gets no `role` knob. §15, §16.

## Projection and file formats

- `project` / `iter_key` / `first_value` take a source's cheap path only through
  `SupportsProjection`, and materialize a deferred source first. `num_classes`, `class_names` and
  `first_value` stay free functions, never `Stream` methods (`tests/test_projection.py`).
- `Stream.project` forwards to its source only when `ops` is empty. With ops it runs each op's
  `for_projection()` variant and re-runs the real chain for any record `formats.answers` says fell
  short (§19): a scan placeholder is MARKED (`STAND_IN`) on the value, never removed; a cheap
  chain that raises counts as falling short; a dropped record is not re-run; no cheap path with
  workers, a `chunk_size` or a stream-level op (`::TestAChainWithOpsCanAnswerCheaply`).
- Every view source forwards a key-restricted walk through `project_indices`; never "simplify"
  it to a `list()` of the view. §18. (`::TestTheViewSourcesForwardProjection`)
- `HuggingFaceSource.project` narrows with `select_columns`; never `select_columns([])` (zero
  rows); any narrowing failure falls back to the full walk
  (`tests/test_huggingface_source.py::TestProjectionNarrowsTheDataset`).
- `scan_file` never falls back to `read`, and a format that cannot answer cheaply does not
  implement `scan`. §17. (`tests/test_formats.py::TestScanningAFileWithoutDecodingIt`)

## Sources

- `FilesSource` takes a `files` list OR `root` + `pattern` (both is refused). The pattern is the
  scope control, so add no `<group>_filter` knob. Directories and dotfiles are never listed; an
  empty match raises; listing is lazy (`tests/test_files_source_root.py`). The served listing is
  memoised on the listing it filtered plus exclude and formats, never on a flag or on `files` alone
  (`tests/test_files_source_formats.py::TestTheServedListingIsResolvedOncePerFileList`).
- Identity (§13, `tests/test_dataset_uri.py`): `dataset_uri` is the canonical handle,
  `dataset_url` a link for a person or `None`. Both read stored config only. The free functions
  follow `.source`; a wrapper never decorates the URI; `ConcatSource` answers `None`.
- Draws (§24, `tests/test_draws.py`, `tests/test_draw_source.py`): the generator is the only
  judge. Never restate its rules in a draw or a sampler, and never filter whole draws. A `Repeat`'s
  elements are not `Repeat`s. A `DrawSpecError` is never caught as "no room". A rule a generator
  checks only in `compute()` needs a `check()`. Never reorder draws silently: order decides what
  a later default allows.
- Fork safety: `ensure_materialized` reads one whole record in the parent (`len()` is not
  enough); `prepare_record_dataset` composes it with `ensure_record_dataset`
  (`tests/test_record_source.py`).

## Batching and the model boundary

- Two collates differ only in stacking: `"record"` (default) and `"list"`; every read-back helper
  accepts both. A stack failure names the key, the shapes and the `"list"` way out. Only
  shape-generic, parameter-free collates register (decided 2026-08-06): a task's batch shape is a
  callable passed to `collate_fn=` / `transform=`, never registered. (`tests/test_batch.py`)
- The read-back helpers are the collate read backwards, with no task shaping; only
  `batch_tensor` is torch, and `dtype`/`device` are the caller's parameters.
- Import keras THROUGH `recordstream.keras` (it sets `KERAS_BACKEND`). `RecordSequence` stays out
  of the package root, is not `@configurable`, and computes its row order lazily. Its test file
  stays task-free. §10. (`tests/test_keras_sequence.py`)
- `recordstream.loaders` is torch-only and not root-exported; `loader_slots` refuses `shuffle` as
  a shared kwarg and `persistent_workers=True` with zero workers (`tests/test_loaders.py`).
- Model boundary (§8 model boundary): the `outputs` contracts are generic in the array type;
  detection has no builder; `RestorationOutput` is the one key `image`, with no `residual` key and
  no clamping in the builder. `DataSink.write(record)` and `PredictionsSink.write(prediction,
  metadata)` stay two protocols; collapsing them is a TASKS.md decision.
- Labels (`tests/test_labels.py`): `is_class_id` is the one "id or name?" rule and `bool` is not an
  id; `LabelMap.to_ids` passes ids through; a `LabelMap` is fitted once at train time, then pinned
  and saved as `class_names.json`. `Stream.class_names` is a declared slot, read with
  `class_names(*sources)`. Say `class_names`, never `label_names` (in HF transformers that means
  the input keys holding labels). No scikit-learn.
- `class_counts` / `inverse_frequency_weights` take already-walked targets, return numpy (`None`
  when nothing was counted) and ignore out-of-range ids. How a loss takes weights stays in the
  consuming runnable.

## Runnables

- A merged runnable's `run()` dispatches through `run_entrypoint(self, self.task)`, never a
  hand-written dict. A new capability = decorate the method + extend the task Literal; the
  standard set is `RunnableTask`. §7. (`tests/test_entrypoint.py`)
- The autograd flag is `__needs_autograd__`; its one reader fails open, so the flag and its reader
  change together or not at all. §8 autograd.
- A runner builds the bound node with `cli.materialize_runnable()` (`flow_mode="manual"`), never a
  bare `flow()`, and a consumer's own CLI calls the same helper (`tests/test_cli_materialize.py`).

## Storage

- Every backend implements `DataSource`/`DataSink` (`storage/base.py`), and a new sink ships its
  matching source in the same change.
- The layout is `typedrecord-v1` with NO legacy read path (user decision 2026-07-25): an old store
  raises via `require_record_format`. `docs/storage.md`.
- Array sinks convert through `to_numpy`; zarr uses `create_array(..., overwrite=True)`, never
  `create_dataset`. `read_record_group(group, slices=…)` is the one HDF5 row decoder
  (`tests/test_typed_storage.py::TestReadRecordGroupSlices`).
- A metadata scan never loads an array; `SupportsMetadataScan` is structural.

## Testing

- To assert on a warning: loggair is loguru, so `caplog` stays empty and `capfd` races the
  enqueued sink. Add a sink and call `logger.complete()`.

## Packaging and release

- This project's own mypy config must be self-sufficient: CI runs `mypy .` here with only
  `[[tool.mypy.overrides]]`, so every optional or lazily imported third party needs an entry there
  even when the root `mypy.ini` has one. Routine runs stay at the root.
- The notebook runs with no workspace `.env`: an unset `DATA_ROOT` falls back to the HF cache, a
  set-but-missing one raises. Reproduce by executing a copy outside the workspace.
- README links to repo files are absolute GitHub URLs; `docs/*.md` keep relative links
  (`tests/test_docs_links.py`).
- Floors state what is tested; a dependent writes `recordstream>=0.1.0a1`. The release metadata,
  the `Development Status` classifier matching the version, `CHANGELOG.md` in the sdist and no
  direct-URL requirement are pinned by `tests/test_packaging.py`.
