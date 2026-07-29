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
| Runnables & workflows | `runnable.py`, `workflow.py`, `processing.py`, `cli.py` | `run()` objects, entry-point markers, combinators, the one `recordstream run` runner | no record yet — [runnable.md](runnable.md), [workflow.md](workflow.md) |

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

## 3. The per-record Context is an ambient wiring plane (`recordstream.context`, 2026-07-17)

### Context

Graph-shaped pipelines — fan-out, fan-in, cross-branch values — need somewhere to hold a value
between the op that produces it and the op that consumes it. The obvious candidate, extra keys on
the record itself, was rejected: the record is the carrier that **persists** — it flows into
sinks, crosses process boundaries, and is the record's serialized identity — while wiring data is
transient scaffolding that should be gone by the end of a well-formed graph. Three constraints
shaped the mechanism: ops keep the plain `__call__(record)` signature (no threading a context
parameter through every op), the executor stays a bare `for op in ops` loop (graphs run on the
*plain sequential engine*), and a linear pipeline's records must stay byte-for-byte untouched.

### Decision

`recordstream/context.py` is a **per-record named-cell store activated ambiently**: the engine
creates one fresh `Context` per source item and activates it around the op loop via a
`contextvars.ContextVar`; the six wiring ops (`Save`/`Use`/`Drop`/`Apply`/`Capture`/`MergeFields`
in `recordstream.ops.context`) reach it inside `__call__` through `require(op_name)` — no signature
change anywhere. Deliberate semantics: cells are stored **by reference** and copy-on-read is the
*reading* op's decision (`Use` deep-copies unless `drop` frees the cell = move); a missing cell
on read or delete **raises loudly** with the live-cell list (a liveness bug must never pass
silently); 1→N expansion children get `Context.copy()` (shallow — independent cell *sets*, shared
values); cells may NOT cross a stream-level op boundary (`Parallel` raises on live cells — each
inner chain gets its own contexts). A `Context` is never `@configurable` and never appears in
YAML — it is pure runtime plumbing. The public surface is two-tier by design: the `Context` class
is a package-root export, while `activate`/`current`/`require` stay module-qualified — reachable,
but visibly plumbing. `FlowGraph` deliberately does NOT use this module: its named-step documents
give the compiler full knowledge of cell lifetimes, so it manages its own per-record env
directly, held to the context-op semantics by the pinned flow⇄ops execution-parity contract.

### Consequences

- A plain sequential `ops:` list executes a real fan-out/fan-in graph — which is exactly what
  graph exporters (a visual canvas, the `flow:` compiler) lower to, so ONE executor serves both
  linear and graph pipelines.
- Linear pipelines are provably untouched: no context op ⇒ the Context is created and never
  used; the record-byte-identical invariant is pinned in the record-model suite under `tests/`.
- Spawn-parallelism is safe by construction: contexts are created *inside* the worker and never
  pickled or shared across processes.
- Ambient state cuts both ways: running an op list containing context ops *outside* an engine
  needs an explicit `with activate(Context()):` — forgetting it is a loud, actionable
  `RuntimeError`, not silent misbehavior.
- Custom ops can join the wiring plane through the same `require()` seam the built-in six use —
  the module being public is what keeps the wiring plane open rather than a closed set of six.

### Example

```python
from recordstream import Stream
from recordstream.ops.context import MergeFields, Save

# Fan-out/fan-in on the PLAIN sequential engine: snapshot → mutate the stream → merge back.
stream = Stream(
    source=my_source,
    ops=[
        Save(name="clean"),                                             # snapshot into a cell
        my_augment_op,                                                  # the stream mutates freely
        MergeFields(sources=["clean"], keys=["mask"], drop=["clean"]),  # fan-in, cell freed
    ],
)

# The same op list outside an engine needs the Context an engine would have created:
from recordstream.context import Context, activate, require

with activate(Context()):
    for op in ops:
        record = op(record)

# A custom op joins the wiring plane through the same seam the built-in six use:
#     require("MyOp").get("clean")   /   require("MyOp").put("my_cell", value)
```

### What you may change (and where it's documented)

- **Writing a custom wiring op** is the supported extension point: call `require("YourOpName")`
  inside `__call__`, follow the by-reference/copy-on-read discipline, and free cells you consume.
  Usage of the six built-in ops lives in [graph.md](graph.md).
- **Keep the surface narrow.** Don't root-export `activate`/`current`/`require`, and don't grow
  `Context` into a general blackboard — anything that should *persist with the record* belongs in
  the record itself, not in a cell.
- **Changing cell semantics** (by-reference storage, loud missing-cell errors, the `Parallel`
  boundary rule, `copy()` shallowness) is an architectural change: the flow⇄ops parity suite and
  the pinned context invariants define the contract. Update this record and the recordstream
  `AGENTS.md` context mandate together.

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
