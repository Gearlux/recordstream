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

### What you may change (and where it's documented)

- **Plugging in your own batch layout** is the supported extension point — decorate a function
  with `@register_collate("your-key")` and select it via `get_collate`/`collate`. Usage:
  [kinds.md](kinds.md); the detection walkthrough: [record-model.md](record-model.md).
- **Changing the default collate's semantics** (how `"record"` stacks, the attrs-become-lists
  convention) is an architectural change: every batch consumer depends on it. Update this record
  and the recordstream `AGENTS.md` metadata mandate together.

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
