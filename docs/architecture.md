# RecordStream architecture

The *why* behind recordstream's module boundaries and mechanisms. The user-facing documentation
([README](../README.md), the per-topic `docs/*.md`) shows **how to use** each surface; this
document records **why the surface is shaped the way it is** — so a reader who asks "why does this
module exist?" finds the answer here instead of reverse-engineering it from git history.

Maintenance rules:

- Every record keeps the five elements **Context → Decision → Consequences → Example → What you
  may change**, dated (see the workspace `AGENTS.md` → "Architecture Decisions Are Documented").
- A change that alters a mechanism updates its record **in the same change**.
- A superseded decision is **deleted**, not archived: whatever it still binds is folded into its
  successor record. History lives in git, not here.

## The system at a glance

| Layer | Modules | What it is | Where the *why* lives |
|---|---|---|---|
| Data model | `items.py`, `io.py` | A record is a plain `dict` of typed values; one codec serializes any value | [§1](#1-the-record-data-model-and-the-type-dispatched-op-engine-2026-07-25) |
| Native ops | `transform.py`, `dispatch.py`, `ops/*` | Type-dispatched `Transform`s (kernels, `field=`) + structural/compose/context ops | [§1](#1-the-record-data-model-and-the-type-dispatched-op-engine-2026-07-25) |
| Library interop | `core._apply_op`, `register_op_family` | External libraries run as-is via the op-family dispatch — no adapters | [§1](#1-the-record-data-model-and-the-type-dispatched-op-engine-2026-07-25) |
| Engines | `core.py` (`Stream`/`JointStream`), `flow.py` (`FlowGraph`) | One op-application chokepoint, four routes; a named-step graph engine with pinned lowering parity | [§1](#1-the-record-data-model-and-the-type-dispatched-op-engine-2026-07-25), [§3](#3-the-per-record-context-is-an-ambient-wiring-plane-recordstreamcontext-2026-07-17), [§5](#5-the-engines-own-callable-wrappers-live-in-corepy-2026-07-20) |
| Graph wiring | `context.py`, `ops/context.py` | Fan-out/fan-in/cross-branch values on the plain sequential engine | [§3](#3-the-per-record-context-is-an-ambient-wiring-plane-recordstreamcontext-2026-07-17) |
| Batching | `collate.py` | Grouping is the engine's; stacking is a pluggable registry | [§2](#2-batching-is-two-stage-collation-is-a-pluggable-registry-recordstreamcollate-2026-07-17) |
| Storage & query | `storage/*` | The `typedrecord-v1` key-group layout over the codec; metadata scans without array loads | [§1](#1-the-record-data-model-and-the-type-dispatched-op-engine-2026-07-25) (contracts) + [storage.md](storage.md) |
| Introspection & serialization | `discovery.py` | Callable↔string identity + registration-free module scans | [§4](#4-callablestring-serialization--passive-introspection-recordstreamdiscovery-2026-07-20) |
| Runnables & workflows | `runnable.py`, `workflow.py`, `processing.py`, `cli.py` | `run()` objects, entry-point markers, combinators, the one `recordstream run` runner | [§7](#7-the-entrypoint-markers-are-the-dispatch-table-run_entrypoint-2026-07-29) + [runnable.md](runnable.md), [workflow.md](workflow.md) |
| Model boundary | `outputs.py`, `predictions.py`, `core.ensure_record_dataset`, `labels.class_counts` | Dataset normalization in, prediction contracts + sinks out, class-balance statistics | [§8](#8-the-model-boundary-belongs-to-the-package-that-reads-it-2026-07-29) + [predictions.md](predictions.md) |

---

## 1. The record data model and the type-dispatched op engine (2026-07-25)

### Context

The rejected alternative was a bespoke record container: typed items (that part was right)
wrapped in a `Record` class with per-key role tags, plus an adapter registry that wrapped every
external library transform in an adapter object before it could touch a record (two adapter
classes, a coercion registry, and ~170 generated per-transform wrapper ops — all maintenance
surface). The container was the friction point: role tags duplicated what key names already say
(`"mask"` *is* the mask), and dict-native libraries — torchvision `transforms.v2` walks dicts,
albumentations takes named kwargs — were kept at arm's length from a carrier they could have
consumed directly. With two op-authoring surfaces (native transforms vs the adapter/generated
families), "where does augmentation come from?" had three answers.

### Decision

Collapse to ONE carrier and ONE op engine:

- **A record is a plain `dict`** — `recordstream.items.Record = Dict[str, Any]` — of **typed
  values** (`Image`/`Mask`/`Regions`/`Label`, base `NDArrayItem`; open registry `register_item`;
  uniform payload accessors `item_data`/`with_data`). No container class, no roles, no
  `primary()`: **key names carry meaning** (`"image"`, `"mask"`, `"bboxes"`, `"class"`), and a
  scalar side value is just another key. Metadata is attrs on the typed value (`Image.layout`,
  `Label.classes`) or more dict keys (`"samplerate": 30.72e6`). Items are deliberately NOT
  confluid-`@configurable`: an ndarray subclass builds through `__new__`, which fights the
  `__init__` validation wrap — they live in their own registry.
- **Native ops are type-dispatched `Transform`s** (`recordstream/transform.py`):
  `get_params(record)` draws shared parameters ONCE per record, per-type kernels
  (`@MyOp.kernel(ItemType)`, MRO-aware registry in `recordstream/dispatch.py`) apply to every
  handled value, `field=` pins one key. The second sanctioned shape — type-CHANGING ops
  (`Threshold`: array→`Mask`, `ConvertToImage`: array→`Image`, `ConnectedComponents`:
  `Mask`→`Regions`, the target ops) — overrides `__call__`, resolves its source by an explicit
  `field=` or the first value of the natural type, and raises a `ValueError` naming the record's
  keys on every miss.
- **External libraries run AS-IS through the engine's op-family dispatch**
  (`recordstream.core._apply_op`): an albumentations op receives exactly its own kwarg vocabulary
  (`image`/`mask`/`masks`/`bboxes`/`keypoints`/`labels` keys present in the record; one call =
  one joint draw; array outputs re-wrapped in the incoming `NDArrayItem` type so `Image`/`Mask`
  survive); a torchvision-v2 op is called on the dict as-is; everything else is `op(record)` with
  `None` = drop. **The families are an open registry** — `register_op_family(name, matcher,
  invoker)`; the built-ins register through the same API, dispatch checks last-registered first,
  and matchers/invokers are module-level functions so spawn workers rebuild the registry. Family
  detection is by MRO module name — no eager imports, no adapters, no generated wrappers.
  Box-carrying augmentation is the library's own `A.Compose(..., bbox_params=...)`; seeding is
  the libraries' own mechanisms.
- **`Pipeline(transforms=[...])`** (`recordstream/transform.py`) is THE sequential composer; every
  composing op routes inner ops through `_apply_op`, so bare library transforms nest anywhere a
  native op does.
- **Tensors are plain values.** `ToTensor` writes a LIVE CHW-float `torch.Tensor` under its key —
  a record value can be anything (`collate_records` stacks tensors natively, storage converts via
  `to_numpy` on write, a downstream tv2 op transforms them as-is). An `Image` itself cannot hold
  a tensor (`NDArrayItem.__new__` runs `np.asarray`); a typed tensor ITEM base is a tracked
  follow-up (root `TASKS.md`).
- **Storage is the record key-group layout** (`typedrecord-v1`): everything serializes through
  the `recordstream/io.py` codec; plain values ride the `"plain"` tag; NO backward compatibility
  with the pre-record layout (an old/untagged store raises via
  `storage/base.py::require_record_format` — an explicit decision: re-generate, never accrete
  legacy readers).
- **Projection and collation are key-addressed**: `project(source, keys)` / `iter_key` /
  `num_classes(key="class")`; the collate registry's default is `"record"` = `collate_records`.

### Consequences

- Zero adapter surface: a new library version's transforms are available the moment the library
  is — nothing to regenerate; a NEW library family is one `register_op_family` call from any
  package.
- Cross-key consistency is the LIBRARY's own joint draw (albumentations Compose / tv2's dict
  walk) for augmentation, and `get_params`-once for native ops — one mechanism per world, both
  automatic.
- YAML needs no special forms: a bare `!class:albumentations.HorizontalFlip {p: 0.5}` sits in an
  `ops:` list like any native op (deferred markers flow at route entry).
- The albumentations vocabulary is load-bearing: a value augments only if it rides one of the
  library's key names — routing is an explicit `RenameField`, never engine magic.
- **Contracts that outlive refactors:** the `(row_min, row_max, col_min, col_max)` inclusive
  integer bin-box order of `connected_component_bboxes` (a downstream back-projection reads
  exactly that order); the `typedrecord-v1` tag + no-back-compat rule; the albumentations key
  vocabulary; the `"module:qualname"` callable-path format (§4).
- Anything that used the old container API must migrate — there are deliberately no aliases and
  no legacy read path.

### Example

One `Stream` ops list mixing both worlds, no wrappers:

```python
import albumentations as A
from recordstream import Stream, Image, as_transform

stream = Stream(source=records, ops=[
    A.Compose([A.HorizontalFlip(p=0.5)],
              bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"])),
    A.GaussNoise(p=1.0),                                 # bare library op — as-is
    as_transform(lambda d: d - 0.5, handles=(Image,)),   # native type-dispatched op
])
```

The same shape in YAML:

```yaml
ops:
  - !class:albumentations.HorizontalFlip
    p: 0.5
  - !class:recordstream.ops.numpy.Threshold
    low_level: 0.5
```

### What you may change (and where it's documented)

- **A new item type** — one class + `@register_item` (array-backed: subclass `NDArrayItem`,
  declare `_item_attrs`); usage in [record-model.md](record-model.md).
- **A new per-type behaviour for an existing op** — `@Op.kernel(ItemType)`, no core edit.
- **A new library family** — one `register_op_family(name, matcher, invoker)` call from any
  package (MRO module-name matcher + the library's native calling convention; module-level
  functions so spawn workers rebuild the registry). Never an adapter/wrapper class. The
  built-ins register through the same API; dispatch is last-registered-first, so
  forks/extensions shadow their base library by registering later. Usage:
  [record-model.md](record-model.md) → "A new library family".
- **The `typedrecord-v1` tag and the no-back-compat rule are contracts** — changing the on-disk
  layout means a NEW tag and a re-generation story, never a silent dual-read path.

---

## 2. Batching is two-stage; collation is a pluggable registry (`recordstream.collate`, 2026-07-17)

### Context

Turning N pipeline items into one batched carrier has two distinct halves:

1. **Grouping** — the engine yields groups of N items (`Stream.batch` / `FlowGraph.batch` yield
   `list`s, and a torch `DataLoader` hands its `collate_fn` a list).
2. **Stacking** — a *collate function* turns one group into one batched carrier.

The engine owns grouping; it must NOT own stacking, because stacking is task-shaped: historically
every consuming project shipped its own task collate (classification, segmentation, detection),
and divergent batched-metadata conventions emerged between them.

### Decision

`recordstream/collate.py` is a **pluggable registry of collate functions keyed by representation**:
`register_collate(key)` / `get_collate(key)` / `collate(items, key=None)`, where an omitted key
uses the default **`"record"`** collate (`collate_records`) — N plain record dicts into ONE
batched record: per key, typed values encode through the `recordstream/io.py` codec, payloads stack
(torch → stacked tensor, numpy → stacked array, else a list), each declared item attr becomes a
LIST of per-record values (decoded back into one batched item of the same type), and a
`"plain"`-tagged value batches as the plain list. Batches must be key-homogeneous — a mismatch
raises. Consuming projects register task aliases (`"detection"`, `"yolo"`, …) **additively**;
re-registering a key deliberately overwrites so a consumer can replace a default. The divergent
consumer conventions were deliberately NOT unified here — the registry is an addressable home
consumers opt into, not a forced migration.

The open, string-keyed half of the registry exists first and foremost for **AI-callable tools**
(the workspace converges on an MCP tool surface — see the root `AGENTS.md` end-goal): a JSON tool
argument can carry `"collate": "yolo"` but never a Python function object, and a tool schema can
offer the legal values only if the set is discoverable at runtime (`registered_collates()`). In
ordinary Python (and in YAML via a dotted `!ref:` to the function), passing the collate function
directly remains the normal path.

### Consequences

- The engine stays task-agnostic: recordstream stacks by key + item type, never
  classification/detection/….
- Item metadata batches deterministically: per-record attrs become lists on the ONE batched item
  (`batch["image"].layout == ["HWC", "HWC", ...]`), plain values become plain lists — there is
  no second batched-metadata convention in this package.
- A task whose batch shape the generic rules cannot express (detection's ragged per-record
  boxes) opts OUT entirely and emits its model family's native contract — see the worked
  example in [record-model.md](record-model.md) → "Batching".
- Registration happens at module import, so a key exists only after its defining module has been
  imported.

### Example

```python
from torch.utils.data import DataLoader

from recordstream import Stream, collate, collate_records, get_collate, register_collate

stream = Stream(source=my_source, ops=[...])

batch = collate([stream[0], stream[1]])                  # the "record" default
batch["image"].shape                                 # stacked payloads, one batched Image
batch["image"].layout                                # per-record attrs -> a list

loader = DataLoader(stream, batch_size=8, collate_fn=collate_records)


# A task alias registers additively (runs when the defining module is imported).
@register_collate("yolo")
def yolo_collate(items):
    ...  # stack to the task's own batch layout


loader = DataLoader(stream, batch_size=8, collate_fn=get_collate("yolo"))
```

### Addendum: the READ-BACK lives here too (`recordstream.batch`, 2026-07-29)

**Context.** Every model boundary has to undo the three rules above: get past a wrapper item,
turn a per-record list into one tensor, transpose the leftover columns into per-record dicts for
a predictions sink. That is not task knowledge — it is the collate's own convention read
backwards. Two consumer packages had independently written it: one for classification, one for
segmentation, with two near-identical private `_batch_metadata` implementations and two separate
test files pinning them. A third consumer would have written a third.

**Decision.** `recordstream.batch` ships the inverse beside the collate — `batch_values`,
`batch_tensor`, `batch_metadata` — and it carries **no dtype or shape opinion**. Rule 2 above
says turning class names into an `[N]` int64 tensor is "the model boundary's one explicit step,
not a generic-engine guess"; that still holds. What moved is *reading*, not *shaping*.

**Consequences.** The three shapes a consumer actually wants — a classifier's `[N]` int64 ids, a
multi-label trainer's `[N, C]` float multi-hot, a segmenter's `[N, H, W]` int64 mask — all start
from `batch_values` and are shaped by a small task-specific function the consumer keeps. Folding
those three into one shared helper would produce a function whose body is a task switch, which
is the thing the collate registry exists to avoid.

**Example.**

```python
from recordstream import batch_tensor, batch_values, batch_metadata

x = batch_tensor(batch, "image", device=self.device)      # generic: one [N, 3, H, W] tensor
meta = batch_metadata(batch, exclude=("image", "class"))  # generic: N per-record dicts

# task-specific, stays in the consumer:
ids = torch.as_tensor(batch_values(batch, "class"))       # a classifier's [N] class ids
mask = batch_tensor(batch, "target").long()               # a segmenter's [N, H, W] int64 mask
```

### What you may change (and where it's documented)

- **Plugging in your own batch layout** is the supported extension point — decorate a function
  with `@register_collate("your-key")` and select it via `get_collate`/`collate`. Usage:
  [kinds.md](kinds.md); the detection walkthrough: [record-model.md](record-model.md).
- **Changing the default collate's semantics** (how `"record"` stacks, the attrs-become-lists
  convention) is an architectural change: every batch consumer depends on it. Update this record
  and the recordstream `AGENTS.md` metadata mandate together.
- **Adding a reader** to `recordstream.batch` is fine when it is the collate read backwards.
  Adding one that shapes for a task (promotes a dtype, builds a multi-hot) is not — that belongs
  to the consumer, or the helper becomes a task switch.

---

## 3. The graph IS the execution model — the lowering pass was deleted (2026-07-30)

*(Supersedes "The per-record Context is an ambient wiring plane", 2026-07-17.)*

### Context

Between 2026-07-17 and 2026-07-30 this package had two ways to run a pipeline. A `flow:`
document (named steps, explicit `from:`/`merge_from:`/`bind:` edges) was the readable authoring
form; a flat `ops:` list was the execution form. A **lowering pass** (`to_ops`) compiled the
first into the second by inserting six *context ops* — `Save`/`Use`/`Drop`/`Apply`/`Capture`/
`MergeFields` — that moved records through an ambient per-record cell store, and a **lifting
pass** (`from_ops`) reconstructed a flow document from such a list. Execution parity in both
directions was a pinned contract with its own suite.

The arrangement was coherent but it cost a second executor (`FlowGraph` duplicating `Stream`'s
iteration, length, indexing and batching while being strictly less capable — no `to_sink`, no
`project`, and a `NotImplementedError` on 1→N expanding steps), a permanent parity tax on every
change to an op's semantics, and an ambient `contextvars` plane that nothing in the workspace's
15 real configs ever used. Adoption told the story plainly: every config on disk was a linear
`ops:` list; zero were `flow:` documents; the only producer of branchy pipelines — the visual
editor — compiled its canvas graph *down* to context ops and then lifted it *back* to a flow
document purely for readability.

The decisive argument was about the consumer nobody had built yet. A lowered list re-encodes
dataflow as imperative mutation of named cells, which is exactly the information a compiler
needs and cannot recover: reverse-dependency analysis walks `node.inputs` backwards from the
outputs, and a flat list has no inputs. Handing a compiler the lowered form means asking it to
run the lifting pass first to rebuild what was just destroyed.

### Decision

**One execution model: the step graph.** Both spellings parse to the same `FlowStep` list and run
through the same per-record kernel (`recordstream.flow.run_steps_multi`).

- An `ops:` list compiles to positional steps (`core.linear_steps` — `s0`, `s1`, …) whose names
  never surface. A sequence IS a graph; no lifting is involved.
- A `flow:` document parses to the same steps with author-chosen names and explicit edges.
- The kernel takes an **env-free fast path** for a straight chain (`is_linear`), so the linear
  case carries none of the graph bookkeeping.
- `to_ops`, `from_ops`, `Stream.from_flow_yaml`, `recordstream.context` and
  `recordstream.ops.context` are **deleted**, with no back-compat shims.

Fan-out, fan-in and cross-step values are expressed as step GRAMMAR rather than as ops: `from:`
is the fork, `merge_from:` the union, `bind:` the cross-step value (including a producer's live
`@output` via `step.attr`). Branch isolation, which the cell store provided by deep-copying on
read, is now a property of the environment: each expansion branch gets its own shallow copy of
the step env, and a fan-out read copies.

### Consequences

- **One executor.** `Stream` and `FlowGraph` are two facades over one kernel; the parity suite is
  gone because there is nothing left to keep in parity.
- **Expanding ops work everywhere.** The graph gained 1→N support (the remaining subgraph runs per
  child, depth-first) that the old `FlowGraph` refused outright.
- **A branchy pipeline has no flat spelling — deliberately.** `FlowGraph.to_stream()` raises for
  one, and a visual editor's ops-export raises pointing at its flow export. This is the honest
  consequence of deleting the pass that manufactured such a spelling.
- **Compilation becomes possible.** A backend reads `FlowGraph.steps` and maps each step to an IR
  node with real `inputs`; reverse-dependency pruning runs on the result.
- **Measured cost:** on a 23-step pipeline of trivial ops the graph engine was 1.41x the old flat
  loop; hoisting a per-record analysis pass and adding the linear fast path brought it to 1.02x,
  and with real ops in the chain the difference is not measurable.
- **Lost with the cell store:** a hand-written wiring op that stashed a value under its own cell
  name. Anything that must persist belongs in the record; anything that wires belongs in the
  grammar.

### Example

```yaml
# Fan-out -> two branches -> fan-in, entirely in step grammar. No cells, no snapshots.
flow:
  spec:   !class:mypkg.MakeSpectrogram {}
  masked: !class:recordstream.ops.numpy.Threshold {low_level: 0.5, from: spec}
  boost:  !class:mypkg.Boost {from: spec}          # second reader of `spec` = the fork
  out:    {from: boost, merge_from: [masked]}      # union, last-write-wins
outputs: out
```

```python
# The same graph, and what a compiler front end reads off it.
from recordstream.flow import parse_flow

steps, outputs = parse_flow(doc["flow"], doc["outputs"])
for step in steps:
    print(step.name, "<-", step.from_, step.merge_from)   # every edge, explicit
# out <- boost ('masked',)
```

### What you may change (and where it's documented)

- **Adding a step-grammar key** is an architectural change: it widens the contract every consumer
  (the engine, a compiler front end, a visual editor's compiler) reads. Update this record, the
  `AGENTS.md` flow mandate, and [graph.md](graph.md) together.
- **The linear fast path** (`is_linear`) is an optimization, not a semantic: it must produce
  results identical to the general path, and the suite pins that both spellings agree.
- **Do NOT reintroduce a lowering pass.** A flat list that encodes branches as cell mutations is
  a second execution model wearing the first one's clothes; the reason it was removed is written
  above. If a future runtime genuinely needs a flattened schedule, it owns that pass — over its
  own IR, downstream of the graph.

---

## 4. Callable↔string serialization + passive introspection (`recordstream.discovery`, 2026-07-20)

### Context

Two workspace mandates — *Serialization Symmetry* (every pipeline round-trips through Confluid
YAML) and *Passive Introspection* (tools discover pipeline pieces without hand-written
definitions) — need a bridge the Confluid registry deliberately does not provide. The registry is
a **curated, opt-in catalog**: classes *and* builder functions participate, but only after an
explicit `@configurable`/`register()`, keyed by name/category/task/role, resolving *strings →
callables* for config materialization. What it does NOT do: produce a string **from** a live
callable (the dump direction a bare-function value like a mapped transform needs), resolve a
callable out of a plain `.py` script or `__main__`, or walk a module to introspect every callable
*defined in it* — registered or not.

### Decision

`recordstream/discovery.py` is one small stdlib-only module with **two halves**:

- **Serialization** — `get_callable_path(fn)` → an importable `"module:qualname"` string
  (resolving `__main__` to the script filename so the path survives process boundaries) and
  `resolve_callable(path)` back to the live object (module import, `.py`-file load, or an
  already-callable passthrough).
- **Introspection** — `introspect_callable(fn)` → a JSON-serializable schema (path, name, doc,
  per-parameter type/default/required), and `scan_module(module_or_py)` applying it to every
  callable *defined in* a module (`__module__`-filtered, so imports don't leak in).

Curated discovery (MCP form-specs, task/category option pickers) deliberately does **not** use
this module — it builds on the Confluid registry. The two surfaces answer different questions:
`scan_module` reflects over *a module, no curation required*; the registry resolves *a curated
name/category*.

### Consequences

- `WrappedOp` stores its callable as the string path and resolves it lazily — which is exactly
  what makes it pickle across `spawn` workers and serialize into YAML verbatim.
- The dotted-path idiom became the workspace's generic **string-callable hook pattern**:
  consuming packages resolve their own hook knobs (metadata encoders, exporter callables) through
  `resolve_callable` instead of hand-rolling import dances.
- A visual editor's node bridge scans op/source modules and auto-generates one node (plus its
  property-panel widgets) per callable — no manual node definitions anywhere.
- The `__module__ == module` filter in `scan_module` is a real contract: a class defined
  elsewhere and merely *imported* into a module is invisible to it (registry-based passes exist
  for that case).
- **One acknowledged overlap**: `resolve_callable`'s plain module-import branch resolves the same
  importable-function targets confluid's `resolve_class` module-path branch / `!ref:` grammar can
  — two spellings of one job. The non-overlapping remainder (path *production*, `.py`-file and
  `__main__` handling, module scans) is why the module exists; whether the resolution half should
  delegate to confluid is a tracked follow-up in the root `TASKS.md`.

### Example

```python
import numpy as np

from recordstream.discovery import get_callable_path, resolve_callable, scan_module

path = get_callable_path(np.sqrt)     # "numpy:sqrt" — YAML/pickle-safe identity
fn = resolve_callable(path)           # back to the live callable
fn is resolve_callable(fn)            # an already-callable argument passes through

schemas = scan_module("recordstream.ops.numpy")   # one JSON schema per op defined there
```

### What you may change (and where it's documented)

- **Adding a string-callable knob to your own class**: reuse `resolve_callable` (the
  `WrappedOp.f` pattern) — never write a bespoke import dance.
- **The `"module:qualname"` format and the module-local scan filter are contracts** — serialized
  pipelines and node bridges depend on both; changing either is an architectural change that must
  update this record.

---

## 5. The engine's own callable wrappers live in `core.py` (2026-07-20)

### Context

Three classes sit in `core.py` next to the `Stream` engine that look, at first glance, like they
belong elsewhere: `FilterOp` and `WrappedOp` (op-shaped, so why not `ops/`?) and `JointStream`
(a second engine in the engine module).

### Decision

They stay in `core.py` because of **who constructs them and which way imports flow**. All three
are the construction targets of `Stream`'s own fluent API — `.filter(pred)` appends a `FilterOp`,
`.map(fn)` appends a `WrappedOp`, `Stream.joint([...])` wraps a `JointStream` — so the engine itself
instantiates them. And `core.py` is the *bottom* of the op-facing layer: every composing op in
`ops/` imports `core._apply_op` (the op-family dispatch chokepoint); moving `FilterOp`/`WrappedOp`
into `ops/` would make `core` import from `ops` and close an import cycle. `JointStream` is
`Stream`'s iteration-only fan-in sibling (`category="engine"`), 20 lines that exist to be
`Stream.joint`'s return value — a module of its own would be structure for structure's sake
(`FlowGraph` earns its separate module by size and its own document grammar).

`FilterOp`/`WrappedOp` carry **no discovery category** on purpose: they wrap a *raw Python
callable*, which no GUI can wire, so they are neither canvas ops nor sources — bare
`@configurable` keeps them YAML-round-trippable while the positive category allowlist keeps them
off visual canvases.

### Consequences

- `ops/` stays a pure consumer of `core` — the layering is one-directional.
- `WrappedOp` is a package-root export (the public "lift a plain function" surface, and its
  stored-string `f` is the reference use of the discovery serialization half); `FilterOp` is not
  root-exported (normally reached via `Stream.filter`; importable as `recordstream.core.FilterOp`).
- `JointStream` is YAML-addressable (`!class:recordstream.core.JointStream()`) and canvas-composable as
  an engine node; its indexable counterpart for raw sources is `ConcatSource`.

### Example

```python
stream = (
    Stream(source=src)
    .map(np.sqrt, key="image")                      # appends WrappedOp(f="numpy:sqrt", key="image")
    .filter(lambda r: float(r["image"].max()) > 0)  # appends FilterOp(p=...)
)
both = Stream.joint([stream_a, stream_b])                 # Stream(source=JointStream([stream_a, stream_b]))
```

### What you may change (and where it's documented)

- **A new engine-constructed helper** (another fluent-API target) belongs in `core.py` for the
  same import-direction reason; an op users wire *directly* (YAML/canvas) belongs in `ops/` with
  a category and group.
- **Do not add a discovery category to `FilterOp`/`WrappedOp`** — surfacing a raw-callable
  parameter on a canvas is a dead widget; the taxonomy is pinned in `tests/test_categories.py`.

## 6. Every knob is a DECLARED parameter — the `Enable` toggle (2026-07-27)

### Context

`Enable` gates an inner ops list behind one boolean. Its original design leaned on Confluid's
post-construction paradigm: **any** boolean attribute set on the instance was the toggle, and that
attribute's NAME became the CLI flag — `visualize: false` in YAML produced `--visualize`, and a
`name:` was only needed to disambiguate two wrappers. Nothing about the toggle was declared; it
existed purely as a runtime attribute Confluid setattr'd from an unrecognised YAML key.

That works for exactly one front-end — hand-written YAML — because only the YAML loader has a
channel for undeclared keys. Every other caller reads the *signature*:

- `to_pydantic(Enable).model_fields` returned `['ops']`, so a schema/form/canvas generator built a
  node with no toggle at all — the wrapper rendered as a pass-through and then raised at runtime.
- `Enable(ops=[...], visualize=True)` raised `ValidationError: Extra inputs are not permitted`
  (the generated config model forbids extras), so neither Python nor a generated tool call could
  construct a toggled wrapper — the only spelling was construct-then-setattr.
- `confluid.accepts_key(Enable, "visualize")` was `False`, so liquifai *silently dropped* the bare
  broadcast the docstring advertised (`--visualize true`). Only `--<name>.<toggle>` landed, and
  only via the addressed branch's "the key is already in the YAML kwargs" escape hatch.

A knob that only YAML can reach is a knob three of the four front-ends cannot offer.

### Decision

The toggle is a **declared, defaulted constructor parameter** — `enabled: bool = True` — exposed as
a **settable property**, and instance identity is the **declared `name`**, which scopes the flag to
`--<name>.enabled`. Dynamic toggle naming is retired.

A property rather than a plain attribute for two reasons: `confluid.accepts_key` admits "public
settable class attributes", so the property keeps `enabled` overridable independently of the
signature; and it gives ONE funnel to reject a non-bool, so a quoted YAML `enabled: "true"` fails
at its `file:line` instead of being silently truthy.

The retired form is not silently ignored: a stray public boolean attribute (what `visualize: false`
now lands as) raises on first record with the replacement spelling in the message.

### Consequences

- One declaration serves every front-end: YAML key, `--enabled` / `--<name>.enabled` override,
  Python kwarg, generated tool/form schema, canvas widget. No front-end-specific glue.
- The generalisable rule: **if a front-end must set it, declare it.** A value that only ever
  arrives via post-construction setattr is reachable from YAML alone.
- Breaking change: `visualize: false` (and any other dynamic toggle name) must become
  `name: visualize` + `enabled: false`; the CLI flag becomes `--visualize.enabled`.
- `flag_name` is gone — with a fixed toggle name there is nothing to introspect.
- Strictness is deliberate: `enabled` accepts only `bool`. Every CLI form already delivers a real
  bool (`--enabled true`, `--enabled=false`, `--enabled+`, `enabled=true`), so the rejection only
  catches genuinely ambiguous config.

### Example

```yaml
- !class:recordstream.ops.enable.Enable
  name: visualize
  enabled: false
  ops: [ !class:recordstream.ops.image.ConvertToImage {} ]
```

```bash
recordstream run pipeline.yaml --visualize.enabled true   # addressed: this wrapper
recordstream run pipeline.yaml --enabled false            # broadcast: every wrapper
```

```python
op = Enable(ops=[convert], name="visualize", enabled=False)   # one call — no setattr step
op.enabled = True                                             # property setter; non-bool raises

to_pydantic(Enable).model_fields           # {'ops', 'enabled', 'name'} — the schema surface
accepts_broadcast(Enable, "enabled")       # True — the bare --enabled form now lands
```

### What you may change (and where it's documented)

- **Adding a knob to any op**: declare it in `__init__` with a default and an `Args:` line. Reach
  for post-construction setattr only for values a *config layer* injects, never for a user-facing
  switch. Usage lives in the project README (`Toggling a branch from the CLI`).
- **Distinguishing instances**: use `name:` — Confluid reads it for hierarchy labelling and
  liquifai for `--<name>.<key>` addressing. Do not invent a per-class flag vocabulary; that was
  the retired design.
- **More than one switch in a chain**: use several `Enable` wrappers with distinct names rather
  than teaching one wrapper several toggles — each name is independently addressable, and the
  broadcast form still flips them all.

## 7. The `@entrypoint` markers ARE the dispatch table (`run_entrypoint`, 2026-07-29)

### Context

A merged train+eval runnable exposes several capabilities from ONE class and selects between them
with a single `task` knob. Two readers need to know the task→capability mapping: the runnable's own
`run()`, which must call the right method, and a discovery consumer (a config generator, a visual
editor), which must know that one class both trains and evaluates and which `task` value means
"evaluate". The `@entrypoint(task, role, primary)` marker was introduced for the second reader only;
`run()` carried its own copy:

```python
dispatch = {"fit": self.fit, "evaluate": self.evaluate, "test": self.test, "predict": self.predict}
```

So every merged runnable stated the same mapping twice — once in the decorators, once in the dict —
and three consumer packages carried that same five-line block verbatim. The two copies drift in a
direction that bites: a config generator pins `task:` from `entrypoint_tasks` (the markers), so a
capability added to the markers and forgotten in the dict yields a *generated* config that dies at
dispatch with "unknown task" while discovery advertises it as supported. Nothing could catch that —
the dict is not derived from anything, so no test can compare it to a source of truth.

### Decision

The markers are the ONE table, and `run_entrypoint(runnable, task)` is their runtime half: it builds
`{declared task: method name}` from `runnable_entrypoints(type(runnable))`, calls the match, and
raises `ValueError` on an unknown task listing the declared ones in declaration order. A merged
runnable's `run()` is then `run_entrypoint(self, self.task)` — the decorators are the only place the
mapping exists.

The lookup reads markers off raw function objects via `vars()` (as `runnable_entrypoints` already
did), so a dynamic `__needs_autograd__` property never fires during dispatch.

### Consequences

- Adding a capability is ONE edit: decorate a method. Discovery and dispatch cannot disagree,
  because they read the same annotations.
- The error message doubles as the class's capability list, in declaration order rather than the
  sorted order a set would give.
- What is lost: the dict form let a type checker verify `self.fit` exists; `getattr(self, name)()`
  is `Any`. Cheap here — the methods are decorated in the same file, and a wrong name would have to
  survive its own `@entrypoint` line.
- Cost is one MRO walk per `run()` — once per training run.
- The markers are now load-bearing at RUNTIME, not just for discovery: dropping an `@entrypoint`
  breaks the run, where before it only emptied a picker. That is the intended direction (a silent
  discovery gap becomes a loud dispatch failure), but it means the decorators are no longer
  optional metadata for a class that dispatches this way.

### Example

```python
from recordstream import TorchRunner, entrypoint, run_entrypoint

class Classifier(TorchRunner):
    def __init__(self, task: str = "fit") -> None:
        self.task = task

    def run(self) -> None:
        run_entrypoint(self, self.task)          # no second copy of the mapping

    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None: ...

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None: ...
```

```python
>>> Classifier(task="test").run()          # calls Classifier.test()
>>> Classifier(task="export").run()
ValueError: Unknown task 'export'; expected one of ['fit', 'test'].
```

### What you may change (and where it's documented)

- **Adding a capability**: decorate the method with `@entrypoint("<task>", role=..., primary=...)`
  and extend the runnable's own `task` Literal. Nothing else — usage lives in `docs/runnable.md`.
- **A capability that is NOT config-selectable**: leave it undecorated and call it directly; the
  marker means "reachable through `task:`", so decorating a helper would advertise it to config
  generators as a runnable capability.
- **A different dispatch policy** (aliases, a default task, a per-role default): build it on top of
  `runnable_entrypoints` rather than beside it — the invariant to preserve is that the markers stay
  the only place the mapping is written down.

## 8. The autograd marker is named for the framework; its FLAG for what it decides (2026-07-29)

### Context

`TorchRunner` exists so a GUI executor — which evaluates graph nodes under
`torch.inference_mode()` for cheap, grad-free runs — can tell "this run does gradient descent"
from "this run is inference-only" and re-enable autograd around the former. The mixin set a flag
named after ITSELF, `__torch_runner__`, and that name answers a question nobody asks at the one
place it is read:

```python
torch_runner = bool(getattr(runnable, "__torch_runner__", False))   # "is this a torch runner?"
```

Every runnable in this workspace is a torch runnable, so read literally the flag is always true —
yet it is deliberately false for an evaluator, and the merged train+eval runnables override it as
a per-task property whose body (`return self.task == "fit"`) contradicts its own name: predicting
with a torch model does not stop the object from being "a torch runner". The name described the
declaring class instead of the decision the reader makes with it.

### Decision

Keep the CLASS name (`TorchRunner` — autograd is a torch concept, and a non-torch backend would
not inherit this mixin at all), rename the FLAG to **`__needs_autograd__`**. The two names then
answer different questions on purpose: which framework's execution mode is at stake, and whether
this particular run needs gradients.

No compatibility alias. The flag is a duck-typed contract with exactly one reader, so the rename
lands in both packages at once — consistent with the workspace's no-back-compat precedent.

### Consequences

- The dynamic per-task override reads as what it means, which is where the old name hurt most.
- **The read fails OPEN** (`getattr(runnable, "__needs_autograd__", False)`): a reader left on the
  old name sees `False` for every runnable and silently executes training under `inference_mode`
  until `loss.backward()` raises *"element 0 of tensors does not require grad"*. That is why the
  rename is all-or-nothing across the reader and the declarer — never a partial rollout.
- An external duck-typed implementer (an object that sets the flag without inheriting the mixin)
  must be updated by hand; there is no import to break and therefore no compile-time signal.

### Example

```python
class TorchRunner:
    __needs_autograd__: bool = True          # inherited by trainers and workflow combinators


class Classifier(TorchRunner, L.LightningModule):
    @property
    def __needs_autograd__(self) -> bool:    # type: ignore[override]
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"
```

```python
# the executor side (one reader, no import of this package)
if getattr(runnable, "__needs_autograd__", False):
    with torch.inference_mode(False), torch.enable_grad():
        runnable.run()
else:
    runnable.run()
```

### What you may change (and where it's documented)

- **A runnable that never trains**: do not inherit `TorchRunner` at all — the absent flag is the
  statement. Usage lives in `docs/runnable.md`.
- **A runnable that sometimes trains**: override `__needs_autograd__` as a property, as above.
- **Another execution-mode marker** (a "needs a GPU", "must run single-process" flag): follow the
  same rule — name the class for the concern, the flag for the decision the executor makes, and
  remember that a duck-typed read of a missing flag is silent.

## 8. The model boundary belongs to the package that reads it (2026-07-29)

### Context

Four surfaces used to live in the workspace's experiment-**tracking** library: a dataset
normalizer (`ensure_record_dataset`), the prediction-output contracts (`ClassificationOutput` &
co) with their torch builders, a predictions sink (`PredictionsSink` +
`ClassificationPredictionsSink`), and class-imbalance weighting (`apply_class_weights` and its
inverse-frequency arithmetic).

None of them tracked anything. The normalizer's whole body was "already a `Stream`? else wrap in
one". The sink's own module docstring justified its placement circularly — it lived there
*because the contract and the output type lived there* — while its body was record plumbing plus a
numpy `argsort`, threading its result through recordstream ops. And the sink advertised itself as
modality-neutral while building its diagnostics from `pack_id` / `iq_file` /
`window_start_sample`: signal-domain keys, in a class a tabular classifier was supposed to reuse.

The pattern underneath: each of these describes a boundary whose only READER is elsewhere, and a
contract that outlives its reader accumulates justifications instead of users.

### Decision

A package owns a contract when it owns the reader. So:

- `ensure_record_dataset` / `RecordSource` land beside `Stream` — the only type they know.
- `recordstream.outputs` holds the contracts *and* their torch builders, because
  `recordstream.predictions` — the sink that reads `probs` by name — is right next to it.
- `recordstream.predictions` holds the sink and the `PredictionsSink` protocol.
- `class_counts` / `inverse_frequency_weights` land beside `LabelMap`, because how often each
  class occurs is a statistic over the labels.

The line is drawn at the *framework convention*, not at "does this import torch" (this package
already hard-depends on torch — a `Stream` IS a `torch.utils.data.Dataset`). What did NOT move:
whether a loss accepts a `weight` argument and how to inject it. That is `torch.nn`'s constructor
convention — Keras takes `class_weight` on `fit()` — so it lives in the consuming runnable as an
overridable method, and this package never learns what a loss is.

### Consequences

- The weights come back as **numpy**, matching `recordstream.batch` (only `batch_tensor` is
  torch). A torch caller writes `torch.as_tensor(w)`; a Keras backend feeds the same array to
  `fit(class_weight=…)`. One statistic, no framework baked in.
- `inverse_frequency_weights` absorbed the `LabelMap.to_ids` flattening consumers used to write by
  hand, so a multi-label target counts for every class it names with no call-site branch.
- The engine's own rules bit immediately and usefully: the package-wide zero-arg-construction
  sweep failed on the imported sink (`ops` was a required constructor argument), so the check
  moved to `write()` where it belongs. Stricter host, better tenant.
- Two sink protocols now coexist (`DataSink.write(record)` vs
  `PredictionsSink.write(prediction, metadata)`). That is deliberate — a model emits a batch while
  the sink contract is per-record, so the halves arrive separately — and load-bearing downstream,
  where a visual editor's node palette keys off the distinction. Collapsing them is filed in
  `TASKS.md` rather than left to drift.
- No back-compat aliases: a stale import from the old location fails loudly.

### Example

```python
# a trainer, walking its targets exactly once and reusing that pass three ways
targets = self._walk_targets(self.train_set)          # ONE pass
self.label_map = LabelMap.fit(targets)                # (1) the encoding
num_classes = self.label_map.num_classes              # (2) the head size
weights = inverse_frequency_weights(targets, num_classes, self.label_map)   # (3) the balance

if weights is not None:
    self.apply_class_weights(weights)                 # framework hook — torch: loss.weight = ...
```

```python
# the boundary on the way out
def predict_step(self, batch, batch_idx):
    out = classification_output(self(x))              # recordstream.outputs
    self.predictions_sink.write(out, metadata)        # recordstream.predictions
    return out
```

### What you may change (and where it's documented)

- **A new prediction contract**: add it to `recordstream.outputs`, generic in the array type, and
  give it a builder only if the payload is DERIVED (logits → probs). A payload the model hands you
  directly (boxes) gets no builder. Usage lives in `docs/predictions.md`.
- **Another task's predictions sink**: implement `PredictionsSink` beside the classification one,
  or in the domain package when it needs domain geometry (a detector's back-projection to
  time/frequency does).
- **A different balancing policy** (effective-number, sqrt-inverse): add it beside
  `inverse_frequency_weights` in `recordstream.labels` as another statistic returning numpy. Do
  NOT add the injection here — that stays a per-backend method on the runnable.

---

## 9. torch is an extra; the engine is numpy (2026-07-30)

### Context

`recordstream` declared `torch` as a hard dependency, so `import recordstream` imported ~2GB of
PyTorch — and marainer inherited it transitively, declaring no torch of its own. That was fine
while every consumer was a Lightning trainer. It stopped being fine when a second training engine
landed: a Keras-on-TensorFlow install, or a plain-numpy dataset-conversion job, paid for a
framework it never called.

Auditing what actually needed torch found the coupling was almost entirely nominal:

- **`Stream` and `FlowGraph` subclassed `torch.utils.data.Dataset`.** This was the expensive
  line, and it bought nothing. `Dataset` is an empty base — `DataLoader` duck-types its argument,
  needing only `__len__` and `__getitem__` (verified against a plain class with those two
  methods and no base). Nothing in the workspace does `isinstance(x, Dataset)`, and nothing
  subclasses `Stream`.
- **`storage/base.py` and `ops/image.py` imported torch for `isinstance(x, torch.Tensor)` alone** —
  to decide whether a payload needed `.detach().cpu().numpy()` before being written or rendered.
- Only `ops/torch.py` (`ToTensor`) and `outputs.py`'s `softmax`/`argmax` builders genuinely
  compute with it.

The two isinstance sites are the interesting case, because the naive fix — a lazy in-function
`import torch` — still *imports torch* the first time a record is written.

### Decision

**`torch` moved from `dependencies` to `[project.optional-dependencies] torch`, and the core
imports no framework.** Four mechanisms, one per coupling:

1. **The `Dataset` base is dropped** in favour of a `MapStyle` Protocol (`__len__` +
   `__getitem__`) — the engine still *says* "map-style dataset" in its own vocabulary, and
   `RecordSource = Union[MapStyle, Iterable[Record]]` stays the contract `ensure_record_dataset`
   enforces.
2. **Type identity without an import**: `recordstream._compat.is_torch_tensor` consults
   `sys.modules` rather than importing. This is exact, not a heuristic — *a torch tensor cannot
   exist in a process that has not imported torch*, so the absence of the module proves the
   negative. It is the same instinct as the op-family matchers, which identify an albumentations
   or torchvision transform by its MRO module name.
3. **`ToTensor` is a lazy export** — `recordstream.ops` maps it in `_OPTIONAL_OPS` and resolves it
   in a PEP 562 module `__getattr__`, raising an `ImportError` that names the extra instead of a
   traceback from three libraries down. `__dir__` still advertises it so completion works.
4. **`outputs.py` splits by what needs a runtime**: the `TypedDict` contracts stay module-level
   (they are typing-only, and generic in the array type), while `classification_output` /
   `segmentation_output` import torch in the function body — they are the only part that computes.

### Consequences

- **`DataLoader(stream)` now needs `cast(Any, stream)` in type-checked code.** torch's *stub*
  declares `Dataset[T]`; the runtime accepts any map-style object. This is a stub's stricter view
  of a contract that works, and the bridge belongs at the four call sites (all in tests) rather
  than in the engine — re-adding the base to satisfy a stub would restore the 2GB dependency to
  silence a type checker.
- **`MapStyle` must be referenced as the real class in any annotation a consumer introspects, never
  a string forward-ref.** confluid evaluates annotations in the *consumer's* namespace, so
  `RecordSource = Union["MapStyle", ...]` raised `NameError: name 'MapStyle' is not defined` from
  a consumer's `__init__` scan, three packages away.
- **The numpy-return rule elsewhere is now load-bearing, not stylistic.** `batch_values`,
  `multi_hot`, `batch_metadata` and the class-balance statistics return numpy precisely so this
  boundary holds; only `batch_tensor` is torch.
- **`recordstream.ops.torch` cannot be eagerly imported by anything in the package** — a new
  convenience re-export there would silently undo all of the above.

### Example

```python
# storage/base.py — recognise a tensor without importing torch
from recordstream._compat import is_torch_tensor

def to_numpy(data):
    return data.detach().cpu().numpy() if is_torch_tensor(data) else np.asarray(data)
```

```python
# recordstream/ops/__init__.py — the op is reachable, the import is not eager
_OPTIONAL_OPS = {"ToTensor": ("recordstream.ops.torch", "torch")}

def __getattr__(name):
    entry = _OPTIONAL_OPS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, extra = entry
    try:
        return getattr(importlib.import_module(module_path), name)
    except ImportError as exc:
        raise ImportError(f"recordstream.ops.{name} needs the {extra!r} extra: "
                          f"pip install 'recordstream[{extra}]'") from exc
```

### What you may change (and where it's documented)

- **Add another optional-framework op**: put it in its own module, add one `_OPTIONAL_OPS` entry
  and one `__all__` entry, and declare the extra in `pyproject.toml`. No other edit — the error
  message and `dir()` follow from the mapping.
- **Recognise another framework's tensor type** (a TF tensor, a jax array): add a sibling to
  `_compat.py` using the same `sys.modules` rule. Do not add a module-level import of that
  framework anywhere in the core.
- **Need a torch-typed return from an existing helper**: add a `dtype=`/`device=` parameter and
  keep the numpy default, as `batch_tensor` does — do not change an existing numpy return, or the
  next backend has to reimplement it.
- **Installation** is documented in the README's Installation section; what each extra provides
  is the `pyproject.toml` comment beside it.

## 10. The framework's batching half lives beside the collate (`recordstream.keras`, 2026-07-30)

### Context

Batching a record source has two halves: **what one batch contains** (recordstream's
`collate_records`) and **which rows go in which batch** (the row order, the slicing, the short
final batch, the per-epoch reshuffle). For torch, the second half is free — a `DataLoader` does it,
duck-typing any `MapStyle` source (§9) and taking `collate_fn=collate_records` for the first half.
So recordstream shipped only half of the pair, and nobody noticed the other half was missing.

Keras 3 has no `DataLoader`. `keras.utils.PyDataset.__getitem__` must return a whole BATCH, so a
consumer has to write that loop itself. The first one did, in a training project — a
`RecordSequence(keras.utils.PyDataset)` inside an image classifier — and the result read as if the
adapter were part of the task. It was not: sixty percent of it (row order, `np.arange`, the rng,
`on_epoch_end`, `collate_records([source[i] for i in rows])`) mentioned nothing about
classification, while its torch twin in the same project was a single
`LazyClass(DataLoader, shuffle=True, collate_fn=collate_records)` line. A second Keras consumer
would have copied the file.

### Decision

**`recordstream.keras.RecordSequence` owns the DataLoader half; the consumer passes the batch
shape in.** The split is drawn exactly where torch draws it: `transform` is the `collate_fn`
equivalent, a callable mapping one collated record to what the model consumes. With no
`transform`, `__getitem__` hands over the batched record — the identity, which is also what
`batches()` yields for pairing per-record predictions with per-record metadata.

The module also owns the **`KERAS_BACKEND` ordering**, which is why it exists as one module rather
than a class dropped somewhere. Keras 3 reads that variable at import time and defaults to
`tensorflow`, which `recordstream[keras]` does not install (Keras is an API; the compute engine is
the operator's choice), so a bare `import keras` dies inside `keras.src.tree.optree_impl` with
`ModuleNotFoundError: No module named 'tensorflow'`. A `setdefault` to the first backend actually
present — probed with `find_spec`, so nothing is imported just to look — has to run in the LOWEST
layer that imports keras: import sorters put a library import above a first-party one, so a
consumer's own shim sorts BELOW `from recordstream.keras import RecordSequence` and would lose the
race.

### Consequences

- **`RecordSequence` is absent from the package root, deliberately.** `inspect.getmembers` — what
  `discovery.scan_module` and the GUI bridges call — getattrs every name a module advertises, so a
  PEP 562 lazy export at the root (the `recordstream.ops.ToTensor` pattern) would import keras on
  every discovery scan of a torch-only install. The import path is the boundary marker:
  `from recordstream.keras import RecordSequence`.
- **It is not `@configurable` and carries no discovery `category`.** It is engine plumbing a
  runnable builds in code, like `collate_records`; tagging it would put a keras import in the
  registry scan for a class no YAML wires.
- **The row order is a lazy `@property`, not constructor state.** `len(source)` is real work for a
  deferred source (a `HuggingFaceSource` LOADS its dataset to answer it), and recordstream
  constructors do none — so `RecordSequence()` builds zero-arg and a missing `source` is reported
  by `indices` with a clear message.
- **A consumer's keras imports now route through recordstream.** A training project keeps its own
  one-line shim for spelling, but the ordering rule has one home; a project that imports keras
  ahead of `recordstream.keras` reintroduces the TensorFlow failure.
- **The extra names no compute engine.** `keras = ["keras>=3.0"]` only; torch/TF/jax come from
  whichever consumer extra selected one, and `_first_installed_backend` adapts to what is there.

### Example

```python
# The consumer supplies the SHAPE; the engine supplies the batching.
from recordstream.keras import RecordSequence

def to_xy(batch):                                    # the classification decision, 3 lines
    x = np.asarray(batch_values(batch, "image"), dtype="float32")
    return x, np.asarray(batch_values(batch, "class"), dtype="int64")

seq = RecordSequence(stream, batch_size=32, shuffle=True, transform=to_xy)
model.fit(seq, epochs=3)

# ...and the torch twin, for the symmetry this restores:
loader = DataLoader(cast(Any, stream), batch_size=32, shuffle=True, collate_fn=collate_records)
```

### What you may change (and where it's documented)

- **Add another framework's batching adapter** (a JAX/`grain` sampler, a TF `tf.data` generator):
  a sibling module behind its own extra, same split — the engine owns row order + collate, the
  caller owns the batch shape via a `transform`-shaped parameter. Do not grow `RecordSequence` a
  framework switch.
- **`PyDataset`'s prefetch knobs are already declared** (`workers` / `use_multiprocessing` /
  `max_queue_size`, forwarded to `super().__init__()` at Keras's own defaults). Any further
  passthrough follows the same rule — a named, defaulted, `Args:`-documented parameter, never a
  `**kwargs` escape hatch, per the declared-parameter mandate (§6). Pinned by
  `test_every_knob_is_a_declared_parameter`.
- **Usage** is [docs/kinds.md](kinds.md#keras-recordsequence--the-batching-half-the-framework-leaves-to-you);
  what the extra provides is the `pyproject.toml` comment beside it.
