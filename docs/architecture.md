# Architecture decisions

The *why* behind sampleflux's non-obvious module boundaries and mechanisms. The user-facing
documentation ([README](../README.md), the per-topic `docs/*.md`) shows **how to use** each surface;
this document records **why the surface is shaped the way it is** — the context, the decision, and
the consequences — so a reader who asks "why does this module exist?" finds the answer here instead
of reverse-engineering it from git history.

Each entry is a short decision record: **Context → Decision → Consequences → Example → What you may
change**. When a change alters one of these mechanisms, update its record in the same change (see
the workspace `AGENTS.md` → "Architecture Decisions Are Documented"). Superseded records are kept
as history, banner-marked with a pointer to their successor.

---

## One type-dispatched op engine — plain-dict records, libraries as-is (2026-07-25)

### Context

The previous data model (the typed-bag `Sample`, recorded below and now superseded) got the item
half right — typed values owning their metadata — but wrapped them in a bespoke container with
per-key role tags. That container was the friction point: every external library needed an adapter
before it could touch a sample (`coerce_transform` + a matcher/factory registry + two adapter
classes + ~170 GENERATED per-transform op wrappers, all maintenance surface), the role tags
duplicated what key names already say (`"mask"` *is* the mask), and dict-native libraries —
torchvision `transforms.v2` walks dicts, albumentations takes named kwargs — were kept at arm's
length from a carrier they could have consumed directly. Meanwhile a second op-authoring surface
(the adapter/generated families) competed with the native type-dispatched `Transform`, so "where
does augmentation come from?" had three answers.

### Decision

Collapse to ONE carrier and ONE op engine:

- **A sample is a plain `dict`** — `sampleflux.items.Record = Dict[str, Any]` — of **typed values**
  (`Image`/`Mask`/`Regions`/`Label`, base `NDArrayItem`; open registry `register_item`; uniform
  payload accessors `item_data`/`with_data`). No container class, no roles, no `primary()`:
  **key names carry meaning** (`"image"`, `"mask"`, `"bboxes"`, `"class"`), and a scalar side value
  is just another key. Metadata is attrs on the typed value (`Image.layout`, `Label.classes`) or
  more dict keys (`"samplerate": 30.72e6`).
- **Native ops are type-dispatched `Transform`s** (`sampleflux/transform.py`): `get_params(record)`
  draws shared parameters ONCE per record, per-type kernels (`@MyOp.kernel(ItemType)`, MRO-aware
  registry in `sampleflux/dispatch.py`) apply to every handled value, `field=` pins one key. The
  second sanctioned shape — type-CHANGING ops (`Threshold`, `ConvertToImage`, the target ops) —
  overrides `__call__`.
- **External libraries run AS-IS through the engine's op-family dispatch**
  (`sampleflux.core._apply_op`, three branches): an albumentations op receives exactly its own kwarg
  vocabulary (`image`/`mask`/`masks`/`bboxes`/`keypoints`/`labels` keys present in the record; one
  call = one joint draw; array outputs re-wrapped in the incoming `NDArrayItem` type so
  `Image`/`Mask` survive); a torchvision-v2 op is called on the dict as-is; everything else is
  `op(record)` with `None` = drop. Family detection is by MRO module name — no eager imports, no
  adapters, no generated wrappers. Box-carrying augmentation is the library's own
  `A.Compose(..., bbox_params=...)`; seeding is the libraries' own mechanisms.
- **`Pipeline(transforms=[...])`** (`sampleflux/transform.py`) is THE sequential composer —
  `TransformChain` was deleted; every composing op routes inner ops through `_apply_op`.
- **Storage is the record key-group layout** (`typedrecord-v1`): everything serializes through the
  `sampleflux/io.py` codec; plain values ride the `"plain"` tag; NO backward compatibility with the
  pre-record layout (an old/untagged store raises via `storage/base.py::require_record_format` —
  an explicit decision: re-generate, don't accrete legacy readers).
- **Projection and collation are key-addressed**: `project(source, keys)` / `iter_key` /
  `num_classes(key="class")`; the collate registry's default is `"record"` = `collate_records`.

### Consequences

- Zero adapter surface: the two adapter classes, the coercion registry, and both generated op
  families are gone; a new library version's transforms are available the moment the library is —
  nothing to regenerate.
- Cross-key consistency is the LIBRARY's own joint draw (albumentations Compose / tv2's dict walk)
  for augmentation, and `get_params`-once for native ops — one mechanism per world, both automatic.
- YAML needs no special forms: a bare `!class:albumentations.HorizontalFlip {p: 0.5}` sits in an
  `ops:` list like any native op (deferred markers flow at route entry).
- The albumentations vocabulary is load-bearing: a value augments only if it rides one of the
  library's key names — routing is an explicit `RenameField`, never engine magic.
- Anything that used `Sample`, roles, `primary()`, `typed_collate`, `ProjectionField`, or a
  `typedsample-v1` store must migrate — there are deliberately no aliases and no legacy read path.

### Example

One `Flux` ops list mixing both worlds, no wrappers:

```python
import albumentations as A
from sampleflux import Flux, Image, as_transform

flux = Flux(source=records, ops=[
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
  - !class:sampleflux.ops.numpy.Threshold
    low_level: 0.5
```

### What you may change (and where it's documented)

- **A new item type** — one class + `@register_item` (array-backed: subclass `NDArrayItem`,
  declare `_item_attrs`); usage in [record-model.md](record-model.md).
- **A new per-type behaviour for an existing op** — `@Op.kernel(ItemType)`, no core edit.
- **A new library family** — a new branch in `core._apply_op` (MRO module-name matcher + the
  library's native calling convention). Never an adapter/wrapper class; update this record when a
  branch is added.
- **The `typedrecord-v1` tag and the no-back-compat rule are contracts** — changing the on-disk
  layout means a NEW tag and a re-generation story, never a silent dual-read path.

---

## Batching is two-stage; collation is a pluggable registry (`sampleflux.collate`, 2026-07-17, updated 2026-07-25)

### Context

Turning N pipeline items into one batched carrier has two distinct halves:

1. **Grouping** — the engine yields groups of N items (`Flux.batch` / `FlowGraph.batch` yield
   `list`s, and a torch `DataLoader` hands its `collate_fn` a list).
2. **Stacking** — a *collate function* turns one group into one batched carrier (stacked tensors +
   batched metadata).

The engine owns grouping; it must NOT own stacking, because stacking is task-shaped: historically
every consuming project shipped its own task collate (classification, segmentation, detection),
and divergent batched-metadata conventions emerged between them.

### Decision

`sampleflux/collate.py` is a **pluggable registry of collate functions keyed by representation**:
`register_collate(key)` / `get_collate(key)` / `collate(items, key=None)`, where an omitted key
uses the default **`"record"`** collate (`collate_records`) — N plain record dicts into ONE batched
record: per key, typed values encode through the `sampleflux/io.py` codec, payloads stack
(torch → stacked tensor, numpy → stacked array, else a list), each declared item attr becomes a
LIST of per-record values (decoded back into one batched item of the same type), and a
`"plain"`-tagged value batches as the plain list. Batches must be key-homogeneous — a mismatch
raises. Consuming projects may register task aliases (`"yolo"`, `"segmentation"`, …)
**additively**; re-registering a key deliberately overwrites (logged at debug) so a consumer can
replace a default. The divergent consumer conventions were deliberately NOT unified here — the
registry is an addressable home consumers opt into, not a forced migration.

### Primary intended consumer: the MCP tool surface

The open, string-keyed half of the registry exists first and foremost for **AI-callable tools**
(the workspace converges on an MCP tool surface — see the root `AGENTS.md` end-goal): a JSON tool
argument can carry `"collate": "yolo"` but never a Python function object, and a tool schema can
offer the legal values only if the set is discoverable at runtime (`registered_collates()`). The
registry is the collate layer's MCP-readiness — a stable, JSON-serializable, enumerable name per
batch layout. In ordinary Python (and in YAML via a dotted `!ref:` to the function), passing the
collate function directly remains the normal path; the registry never replaces it.

### Consequences

- The engine stays task-agnostic: sampleflux stacks by key + item type, never
  classification/detection/…
- Item metadata batches deterministically: per-record attrs become lists on the ONE batched item
  (`batch["image"].layout == ["HWC", "HWC", ...]`), plain values become plain lists — there is no
  second batched-metadata convention in this package.
- One addressable lookup (`get_collate("yolo")`) replaces scattered cross-package imports — once a
  consumer registers. Registration happens at module import, so a key exists only after its
  defining module has been imported.
- **Current usage:** only the `"record"` default is registered here; the open registration surface
  is capacity held for the MCP tool surface above.

### Example

```python
from torch.utils.data import DataLoader

from sampleflux import Flux, collate, collate_records, get_collate, register_collate

flux = Flux(source=my_source, ops=[...])

batch = collate([flux[0], flux[1]])                  # the "record" default
batch["image"].shape                                 # stacked payloads, one batched Image
batch["image"].layout                                # per-record attrs -> a list

loader = DataLoader(flux, batch_size=8, collate_fn=collate_records)


# A task alias registers additively (runs when the defining module is imported).
@register_collate("yolo")
def yolo_collate(items):
    ...  # stack to the task's own batch layout


loader = DataLoader(flux, batch_size=8, collate_fn=get_collate("yolo"))
```

### What you may change (and where it's documented)

- **Plugging in your own batch layout** is the supported extension point — decorate a function with
  `@register_collate("your-key")` and select it via `get_collate`/`collate`. Usage lives in
  [kinds.md](kinds.md).
- **Changing the default collate's semantics** (how `"record"` stacks, the attrs-become-lists
  convention) is an architectural change: every batch consumer depends on it. Update this record
  and the sampleflux `AGENTS.md` metadata mandate together.

---

## The per-record Context is an ambient wiring plane (`sampleflux.context`, 2026-07-17)

### Context

Graph-shaped pipelines — fan-out, fan-in, cross-branch values — need somewhere to hold a value
between the op that produces it and the op that consumes it. The obvious candidate, extra keys on
the record itself, was rejected: the record is the carrier that **persists** — it flows into sinks,
crosses process boundaries, and is the sample's serialized identity — while wiring data is
transient scaffolding that should be gone by the end of a well-formed graph. Three constraints
shaped the mechanism: ops keep the plain `__call__(record)` signature (no threading a context
parameter through every op), the executor stays a bare `for op in ops` loop (graphs run on the
*plain sequential engine*), and a linear pipeline's behavior — its records, byte-for-byte — must be
completely untouched.

### Decision

`sampleflux/context.py` is a **per-record named-cell store activated ambiently**: the engine
creates one fresh `Context` per source item and activates it around the op loop via a
`contextvars.ContextVar`; the six wiring ops (`Save`/`Use`/`Drop`/`Apply`/`Capture`/`MergeFields`
in `sampleflux.ops.context`) reach it inside `__call__` through `require(op_name)` — no signature
change anywhere. Deliberate semantics: cells are stored **by reference** and copy-on-read is the
*reading* op's decision (`Use` deep-copies unless `drop` frees the cell = move); a missing cell on
read or delete **raises loudly** with the live-cell list (a liveness bug must never pass
silently); 1→N expansion children get `Context.copy()` (shallow — independent cell *sets*, shared
values); cells may NOT cross a stream-level op boundary (`Parallel` raises on live cells — each
inner chain gets its own contexts). A `Context` is never `@configurable` and never appears in
YAML — it is pure runtime plumbing. The public surface is two-tier by design: the `Context` class
is a package-root export, while `activate`/`current`/`require` stay module-qualified
(`sampleflux.context.…`) — reachable, but visibly plumbing. `FlowGraph` deliberately does NOT use
this module: its named-step documents give the compiler full knowledge of cell lifetimes, so it
manages its own per-record env directly, held to the context-op semantics by the pinned
flow⇄ops execution-parity contract.

### Consequences

- A plain sequential `ops:` list executes a real fan-out/fan-in graph — which is exactly what
  graph exporters (a visual canvas, the `flow:` compiler) lower to, so ONE executor serves both
  linear and graph pipelines.
- Linear pipelines are provably untouched: no context op ⇒ the Context is created and never used;
  the record-byte-identical invariant is pinned in the record-model suite under `tests/`.
- Spawn-parallelism is safe by construction: contexts are created *inside* the worker and never
  pickled or shared across processes.
- Ambient state cuts both ways: running an op list containing context ops *outside* an engine
  needs an explicit `with activate(Context()):` — forgetting it is a loud, actionable
  `RuntimeError`, not silent misbehavior.
- Custom ops can join the wiring plane through the same `require()` seam the built-in six use —
  the module being public is what keeps the wiring plane open rather than a closed set of six.

### Example

```python
from sampleflux import Flux
from sampleflux.ops.context import MergeFields, Save

# Fan-out/fan-in on the PLAIN sequential engine: snapshot → mutate the stream → merge back.
flux = Flux(
    source=my_source,
    ops=[
        Save(name="clean"),                                             # snapshot into a cell
        my_augment_op,                                                  # the stream mutates freely
        MergeFields(sources=["clean"], keys=["mask"], drop=["clean"]),  # fan-in, cell freed
    ],
)

# The same op list outside an engine needs the Context an engine would have created:
from sampleflux.context import Context, activate, require

with activate(Context()):
    for op in ops:
        record = op(record)

# A custom op joins the wiring plane through the same seam the built-in six use:
#     require("MyOp").get("clean")   /   require("MyOp").put("my_cell", value)
```

### What you may change (and where it's documented)

- **Writing a custom wiring op** is the supported extension point: call
  `require("YourOpName")` inside `__call__`, follow the by-reference/copy-on-read discipline, and
  free cells you consume. Usage of the six built-in ops lives in [graph.md](graph.md).
- **Keep the surface narrow.** Don't root-export `activate`/`current`/`require`, and don't grow
  `Context` into a general blackboard — anything that should *persist with the record* belongs in
  the record itself, not in a cell.
- **Changing cell semantics** (by-reference storage, loud missing-cell errors, the `Parallel`
  boundary rule, `copy()` shallowness) is an architectural change: the flow⇄ops parity suite and
  the pinned context invariants define the contract. Update this record and the sampleflux
  `AGENTS.md` context mandate together.

---

## Callable↔string serialization + passive introspection (`sampleflux.discovery`, recorded 2026-07-20)

### Context

Two workspace mandates — *Serialization Symmetry* (every pipeline round-trips through Confluid
YAML) and *Passive Introspection* (tools discover pipeline pieces without hand-written
definitions) — need a bridge the Confluid registry deliberately does not provide. The registry is
a **curated, opt-in catalog**: classes *and* builder functions participate, but only after an
explicit `@configurable`/`register()` (the Registry Discipline mandate), keyed by
name/category/task/role, resolving *strings → callables* for config materialization. What it does
NOT do: produce a string **from** a live callable (the dump direction a bare-function value like a
mapped transform needs), resolve a callable out of a plain `.py` script or `__main__`, or walk a
module to introspect every callable *defined in it* — registered or not.

### Decision

`sampleflux/discovery.py` is one small stdlib-only module with **two halves**:

- **Serialization** — `get_callable_path(fn)` → an importable `"module:qualname"` string
  (resolving `__main__` to the script filename so the path survives process boundaries) and
  `resolve_callable(path)` back to the live object (module import, `.py`-file load, or an
  already-callable passthrough).
- **Introspection** — `introspect_callable(fn)` → a JSON-serializable schema (path, name, doc,
  per-parameter type/default/required), and
  `scan_module(module_or_py)` applying it to every callable *defined in* a module
  (`__module__`-filtered, so imports don't leak in).

Curated discovery (MCP form-specs, task/category option pickers) deliberately does **not** use
this module — it builds on the Confluid registry, which registers classes AND builder functions,
opt-in by name. This module is the **registration-free complement**: the two surfaces answer
different questions — `scan_module` reflects over *a module, no curation required*; the registry
resolves *a curated name/category*.

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
  — two spellings of one job (`"module:qualname"` here vs `"module.attr"` there). The
  non-overlapping remainder (path *production* via `get_callable_path`, `.py`-file and `__main__`
  handling, module scans) is why the module exists; whether the resolution half should delegate
  to confluid is a tracked follow-up in the root `TASKS.md`.

### Example

```python
import numpy as np

from sampleflux.discovery import get_callable_path, resolve_callable, scan_module

path = get_callable_path(np.sqrt)     # "numpy:sqrt" — YAML/pickle-safe identity
fn = resolve_callable(path)           # back to the live callable
fn is resolve_callable(fn)            # an already-callable argument passes through

schemas = scan_module("sampleflux.ops.numpy")   # one JSON schema per op defined there
```

### What you may change (and where it's documented)

- **Adding a string-callable knob to your own class**: reuse `resolve_callable` (the
  `WrappedOp.f` pattern) — never write a bespoke import dance.
- **The `"module:qualname"` format and the module-local scan filter are contracts** — serialized
  pipelines and node bridges depend on both; changing either is an architectural change that
  must update this record.

---

## The engine's own callable wrappers live in `core.py` (`FilterOp`/`WrappedOp`/`JointFlux`, recorded 2026-07-20)

### Context

Three classes sit in `core.py` next to the `Flux` engine that look, at first glance, like they
belong elsewhere: `FilterOp` and `WrappedOp` (op-shaped, so why not `ops/`?) and `JointFlux`
(a second engine in the engine module).

### Decision

They stay in `core.py` because of **who constructs them and which way imports flow**. All three
are the construction targets of `Flux`'s own fluent API — `.filter(pred)` appends a `FilterOp`,
`.map(fn)` appends a `WrappedOp`, `Flux.joint([...])` wraps a `JointFlux` — so the engine itself
instantiates them. And `core.py` is the *bottom* of the op-facing layer: every composing op in
`ops/` imports `core._apply_op` (the op-family dispatch chokepoint); moving `FilterOp`/`WrappedOp`
into `ops/` would make `core` import from `ops` and close an import cycle. `JointFlux` is
`Flux`'s iteration-only fan-in sibling (`category="engine"`), 20 lines that exist to be
`Flux.joint`'s return value — a module of its own would be structure for structure's sake
(`FlowGraph` earns its separate module by size and its own document grammar).

`FilterOp`/`WrappedOp` carry **no discovery category** on purpose: they wrap a *raw Python
callable*, which no GUI can wire, so they are neither canvas ops nor sources — bare
`@configurable` keeps them YAML-round-trippable while the positive category allowlist keeps them
off visual canvases.

### Consequences

- `ops/` stays a pure consumer of `core` — the layering is one-directional.
- `WrappedOp` is a package-root export (the public "lift a plain function" surface, and its
  stored-string `f` is the reference use of the discovery serialization half); `FilterOp` is not
  root-exported (normally reached via `Flux.filter`; importable as `sampleflux.core.FilterOp`).
- `JointFlux` is YAML-addressable (`!class:sampleflux.core.JointFlux()`) and canvas-composable
  as an engine node; its indexable counterpart for raw sources is `ConcatSource`.

### Example

```python
flux = (
    Flux(source=src)
    .map(np.sqrt, key="image")                      # appends WrappedOp(f="numpy:sqrt", key="image")
    .filter(lambda r: float(r["image"].max()) > 0)  # appends FilterOp(p=...)
)
both = Flux.joint([flux_a, flux_b])                 # Flux(source=JointFlux([flux_a, flux_b]))
```

### What you may change (and where it's documented)

- **A new engine-constructed helper** (another fluent-API target) belongs in `core.py` for the
  same import-direction reason; an op users wire *directly* (YAML/canvas) belongs in `ops/` with
  a category and group.
- **Do not add a discovery category to `FilterOp`/`WrappedOp`** — surfacing a raw-callable
  parameter on a canvas is a dead widget; the taxonomy is pinned in `tests/test_categories.py`.

---

## ~~The typed-bag model: a named bag of typed items (`sampleflux.bag`, 2026-07-21)~~ — SUPERSEDED

> **Superseded (2026-07-25)** by
> [One type-dispatched op engine — plain-dict records, libraries as-is](#one-type-dispatched-op-engine--plain-dict-records-libraries-as-is-2026-07-25).
> The `Sample` container, role tags, `primary()`, the adapter coercion registry, and the
> `sampleflux.bag` package were removed; the typed items, the kernel-dispatch idea, and the item
> codec carried forward into the record model. Kept as history — do not follow.

### Context (historical)

Before the typed model, the carrier was a fixed `(input, target, metadata)` 3-tuple where `metadata`
was one flat `dict` shared by the whole sample. Everything that is not literally the model input or
target rode that dict by string key: segmentation masks, `[f0,f1,t0,t1]` region lists, window locators,
`spectrogram_params`, power stats, `snr_db`, a signal's samplerate, an image's canvas size, a
label's class names. Two structural costs follow. First, **metadata has no owner** — `samplerate`
belongs to *the signal*, `canvas` to *the image*, but the flat dict severs that link. Second, **a
transform cannot move several fields together** — flipping an image and its mask and its boxes with
one shared decision is inexpressible when the fields are `input`, `target`, and `metadata["regions"]`
respectively, so the era's augmentation adapters hard-coded a `TargetMode = Literal["none","mask","boxes"]`
knob per op instead. `target` was also overloaded — sometimes a bare string (`"drone_x"`), sometimes a
`{boxes, labels}` dict.

### Decision (historical)

`sampleflux.bag` modeled a sample as a **named bag of typed items with per-field role tags**
(`Sample`, roles `input`/`target`/`aux`/`pred`, immutable copy-on-write mutators), dispatched
transforms on item TYPE via a kernel registry, batched via `typed_collate` (a batched `Sample`),
and plugged external libraries in through a **coercion registry of adapters**
(`register_adapter`/`coerce_transform` — a `Pipeline` wrapped each bare torchvision-v2 /
albumentations transform in an adapter object at composition time).

### What survived, and what was undone (2026-07-25)

- **Survived into the record model:** typed items owning their metadata (the HYBRID
  ndarray-subclass / dataclass-wrapper realization, `item_data`/`with_data`, `register_item`), the
  once-per-record kernel dispatch (`sampleflux.dispatch`), and the item codec idea
  (`sampleflux/io.py` — storage backends never inspect item internals).
- **Undone:** the `Sample` container (a plain dict now), role tags (key names carry meaning),
  `primary()` (key addressing), `typed_collate` (→ `collate_records`), and the ENTIRE adapter plane
  — coercion registry, adapter classes, `only=` per-key filters (→ `field=`) — replaced by the
  engine-level op-family dispatch (`core._apply_op`), which calls each library natively instead of
  wrapping it.

### Example (historical shape — no longer runs)

```python
sample = Sample({"image": Image(rgb), "regions": Regions(boxes)}, roles={"regions": "target"})
out = Pipeline([v2.RandomHorizontalFlip(p=1.0), A.GaussNoise(p=1.0)])(sample)   # adapter-coerced
```

### What you may change

Nothing — superseded. Extension points live in the successor record above.

---

## ~~Native typed transforms that change a field's TYPE (`ConvertToImage`/`Threshold`/`ConnectedComponents`, 2026-07-22)~~ — SUPERSEDED

> **Superseded (2026-07-25)** by
> [One type-dispatched op engine — plain-dict records, libraries as-is](#one-type-dispatched-op-engine--plain-dict-records-libraries-as-is-2026-07-25),
> which promotes this record's core insight — the type-changing `__call__`-override op as the
> second sanctioned shape — to a rule of the data model itself. The ops survive (`ConvertToImage`:
> array → `Image`, `Threshold`: array → `Mask`, `ConnectedComponents`: `Mask` → `Regions`,
> plus the target ops) but now read/write plain record KEYS (`field=` in, `output=` out) — the
> role tags, the `Sample` shims, and the legacy-op delegation described below are gone.
> Kept as history — do not follow the role/shim details.

### Context (historical)

Two shapes of typed transform exist. The first is the augmentation shape the base `Transform`
was built for: it `handles` an item type and, per handled field, applies a registered kernel that
returns *the same type* (a flip returns a flipped `Image`), so `image`, `mask`, and `boxes` move
together. But a running detection/segmentation front-end needs a different shape: **read one
field, write a field of a DIFFERENT type**. Turning a numeric array into a displayable image,
thresholding an array into a boolean mask, and labelling that mask into a set of bin boxes are
each a *type change* (`array → Image`, `array → Mask`, `Mask → Regions`), not an in-place
per-type edit. No library provides them.

### Decision (historical)

Add native typed **twins** that subclass `Transform` and OVERRIDE `__call__` (rather than register
a kernel), reading one field and writing a different-typed item; resolve the source field by an
explicit `field=` name or the first item of the natural type, with every miss raising a
`ValueError` naming the sample's fields; write the output with role tags chosen semantically; and
delegate each twin to its legacy op's math verbatim for byte-parity.

### What survived, and what was undone (2026-07-25)

- **Survived:** the two-shapes rule; the `field=`-or-first-natural-type source resolution with loud
  `ValueError` misses; the `(row_min, row_max, col_min, col_max)` inclusive integer bin-box
  contract of `connected_component_bboxes` (**still load-bearing** — a downstream back-projection
  reads exactly that order); truthful `consumes`/`produces` graph metadata.
- **Undone:** role tags on outputs (an op now writes a named `output` key — `Threshold`'s default
  `output="mask"`, `ConvertToImage`'s `output="image"`); the legacy `(input, target, metadata)` ops
  and the shim-`Sample` delegation (the legacy ops are deleted; the math lives in the shared free
  functions `threshold_array` / `connected_component_bboxes` / `value_to_image`).

### Example (current successor shape)

```python
from sampleflux.ops.numpy import ConnectedComponents, Threshold

record = Threshold(field="spec", low_level=-30.0)(record)      # + record["mask"]  (a Mask)
record = ConnectedComponents(field="mask")(record)             # + record["regions"] (a Regions)
```

### What you may change

The bin-box tuple order remains a contract (see the successor record); everything else here is
history.

---

## ~~A typed field cannot hold a live torch tensor — `ToTensor` stores CHW-float numpy (2026-07-22)~~ — SUPERSEDED (decision REVERSED 2026-07-25)

> **Superseded (2026-07-25, user decision)**: the constraint below was a TYPED-BAG artifact —
> every field had to be a typed item, and an `NDArrayItem` coerces its payload through
> `np.asarray`, so a live tensor could not ride a field. In the RECORD model a value can be
> ANYTHING (the `"plain"` codec tag covers storage, `collate_records._stack` stacks torch
> tensors natively, a bare torchvision-v2 op transforms them as-is), so **`ToTensor` now writes
> the LIVE CHW-float `torch.Tensor` under the key** (in place by default, `output=` for a new
> key) — no numpy round-trip, and the op's name is again the truth. `Image` itself still cannot
> hold a tensor (it IS an ndarray subclass); the torch-`Tensor`-subclass ITEM base (a typed
> tensor value with attrs) remains the documented follow-up (root `TASKS.md`).

### Context (historical)

A typed classification front-end needs to turn the working image into the model's input tensor and
the class-name label into the encoded target id. Two facts shape the ops: (1) there is no shared
metadata dict — the label already rides a `Label` value that owns its metadata; (2) an array item
is an `np.ndarray` SUBCLASS whose `__new__` runs `np.asarray(data)`, so **a payload is coerced to
numpy** — an `Image` cannot hold a live `torch.Tensor`, and a bare tensor stored directly has no
registered item type for the collate / storage codec.

### Decision (historical, largely still in force)

`ToTensor` resolves an array-bearing key, runs the HWC→CHW + `normalize` conversion, and writes an
`Image(layout="CHW")` whose payload is CHW `float32` numpy — NOT a live tensor; in place by
default so the working key keeps its name. `EncodeTarget` / `DecodeTarget` map a `Label`'s value
through a config-pinned `mapping` and write the encoded `Label` back (carrying the source label's
`classes`). `MetadataToTarget` stays as the escape hatch for a label that rode as another value's
attribute — largely redundant when a source emits the label as a `Label` under its own key.

### Example (current successor shape)

```python
from sampleflux.ops.target import EncodeTarget
from sampleflux.ops.torch import ToTensor

record = {"image": Image(hwc_uint8), "class": Label("cat")}
record = ToTensor(field="image")(record)                                     # record["image"] is now a LIVE CHW float32 torch.Tensor
record = EncodeTarget(mapping={"cat": 0, "dog": 1}, field="class")(record)   # Label(0), classes kept
```

### What you may change

- **The Tensor-subclass item follow-up** — the tensor currently rides as a PLAIN value (no item
  attrs); a torch-`Tensor`-subclass item base would make it a typed value with metadata again.
  Update this record and the `sampleflux/items.py` note together when it lands.
