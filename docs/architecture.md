# Architecture decisions

The *why* behind sampleflux's non-obvious module boundaries and mechanisms. The user-facing
documentation ([README](../README.md), the per-topic `docs/*.md`) shows **how to use** each surface;
this document records **why the surface is shaped the way it is** — the context, the decision, and
the consequences — so a reader who asks "why does this module exist?" finds the answer here instead
of reverse-engineering it from git history.

Each entry is a short decision record: **Context → Decision → Consequences → Example → What you may
change**. When a change alters one of these mechanisms, update its record in the same change (see
the workspace `AGENTS.md` → "Architecture Decisions Are Documented").

---

## Batching is two-stage; collation is a pluggable registry (`sampleflux.collate`, 2026-07-17)

### Context

Turning N pipeline items into one batched carrier has two distinct halves:

1. **Grouping** — the engine yields groups of N items (`Flux.batch` / `FlowGraph.batch` yield
   `list`s, and a torch `DataLoader` hands its `collate_fn` a list).
2. **Stacking** — a *collate function* turns one group into one batched carrier (stacked tensors +
   batched metadata).

The engine owns grouping; it must NOT own stacking, because stacking is task-shaped: historically
every consuming project shipped its own task collate (classification, segmentation, detection),
and **two divergent batched-metadata conventions** emerged — the list-form
`Sample(metadata=[...])` batch (`Sample.is_batched` True) versus a dict-nested
`metadata={"per_sample": [...]}` form. In addition, the multi-type carrier engine
(`Flux(native=True)`, see [kinds.md](kinds.md)) meant sampleflux itself needed stacking behavior
*keyed by carrier kind* — a `Sample`, a metadata-free pair, a bare value, and the
`InputMeta`/`TargetMeta` views each batch differently.

### Decision

`sampleflux/collate.py` is a **pluggable registry of collate functions keyed by representation**:
`register_collate(key)` / `get_collate(key)` / `collate(items, key=None)`, where an omitted key
uses the default `"typed"` collate (`typed_collate`) — batching a list of typed-bag `Sample`s into
one batched `Sample` (per-item type dispatch itself is the sibling kernel registry
`sampleflux.bag.dispatch`, which walks each item's MRO). sampleflux registers the `"typed"`
default; consuming projects may register task aliases (`"yolo"`, `"segmentation"`, …)
**additively**. Re-registering a key deliberately overwrites (logged at debug) so a consumer can
replace a default.

Two things were deliberately **not** done:

- **Existing task collates were not moved here.** The registry is an addressable home consumers
  can opt into, not a forced migration — consuming projects keep shipping and wiring their own
  collate functions directly (e.g. via a Confluid `!ref:` to the function's dotted path).
- **The divergent metadata conventions were not unified.** The dict-nested
  `{"per_sample": [...]}` convention stays with the project that owns it; unification is a
  tracked follow-up in the root `TASKS.md`, not a side effect of introducing the registry.

### Primary intended consumer: the MCP tool surface

The open, string-keyed half of the registry exists first and foremost for **AI-callable tools**
(the workspace converges on an MCP tool surface — see the root `AGENTS.md` end-goal): a JSON tool
argument can carry `"collate": "yolo"` but never a Python function object, and a tool schema can
offer the legal values only if the set is discoverable at runtime (`registered_collates()`). The
registry is the collate layer's MCP-readiness — a stable, JSON-serializable, enumerable name per
batch layout. In ordinary Python (and in YAML via a dotted `!ref:` to the function), passing the
collate function directly remains the normal path; the registry never replaces it.

### Consequences

- The engine stays task-agnostic: sampleflux knows *kinds*, never classification/detection/…
- `Flux(native=True)` pipelines and the examples get correct batching per carrier kind with zero
  configuration (`DataLoader(flux, collate_fn=get_collate("sample"))`).
- One addressable lookup (`get_collate("yolo")`) replaces scattered cross-package imports — once a
  consumer registers. Registration happens at module import, so a key exists only after its
  defining module has been imported.
- Batched metadata's list form (`Sample.is_batched`) is produced here, which is why the
  `Sample.metadata` `dict | list[dict]` duality exists (see the sampleflux `AGENTS.md` metadata
  mandate).
- **Current usage (as of 2026-07-20):** only the five kind defaults are registered; the live call
  sites are one training example (`get_collate("sample")` as a `DataLoader` collate) and the test
  pins. No consuming project registers or looks up yet — the open registration surface is capacity
  held for the MCP tool surface above, and is provisional until that consumer lands.

### Example

```python
from torch.utils.data import DataLoader

from sampleflux import Flux, collate, get_collate, register_collate

flux = Flux(source=my_source, ops=[...])

# Kind-dispatched: Samples stack via the "sample" default (list-form batched metadata).
batch = collate([flux[0], flux[1]])
assert batch.is_batched

# Explicit key — the DataLoader glue.
loader = DataLoader(flux, batch_size=8, collate_fn=get_collate("sample"))


# A task alias registers additively (runs when the defining module is imported).
@register_collate("yolo")
def yolo_collate(items):
    ...  # stack to the task's own batch layout


loader = DataLoader(flux, batch_size=8, collate_fn=get_collate("yolo"))
```

### What you may change (and where it's documented)

- **Plugging in your own batch layout** is the supported extension point — decorate a function with
  `@register_collate("your-key")` and select it via `get_collate`/`collate`. Usage lives in
  [kinds.md → the collate registry](kinds.md#multi-type-carriers--the-collate-registry-samplefluxcollate).
- **Changing a default collate's semantics** (e.g. how `"sample"` stacks, or the list-form metadata
  convention) is an architectural change: every batch consumer (losses, predictions sinks,
  `batch_meta` readers) depends on it. Update this record and the metadata mandate together.

---

## The per-sample Context is an ambient wiring plane (`sampleflux.context`, 2026-07-17)

### Context

Graph-shaped pipelines — fan-out, fan-in, cross-branch values — need somewhere to hold a value
between the op that produces it and the op that consumes it. The obvious candidate,
`sample.metadata`, was rejected: metadata is the **accumulating bus that rides inside each
sample** — it persists into sinks, crosses process boundaries, and is part of the sample's
serialized identity, while wiring data is transient scaffolding that should be gone by the end of
a well-formed graph. Three constraints shaped the mechanism: ops keep the plain
`__call__(sample)` signature (no threading a context parameter through every op), the executor
stays a bare `for op in ops` loop (graphs run on the *plain sequential engine*), and a linear
pipeline's behavior — including its metadata, byte-for-byte — must be completely untouched.

### Decision

`sampleflux/context.py` is a **per-sample named-cell store activated ambiently**: the engine
creates one fresh `Context` per source item and activates it around the op loop via a
`contextvars.ContextVar`; the six wiring ops (`Save`/`Use`/`Drop`/`Apply`/`Capture`/`Mix` in
`sampleflux.ops.context`) reach it inside `__call__` through `require(op_name)` — no signature
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
manages its own per-sample env directly, held to the context-op semantics by the pinned
flow⇄ops execution-parity contract.

### Consequences

- A plain sequential `ops:` list executes a real fan-out/fan-in graph — which is exactly what
  graph exporters (a visual canvas, the `flow:` compiler) lower to, so ONE executor serves both
  linear and graph pipelines.
- Linear pipelines are provably untouched: no context op ⇒ the Context is created and never used;
  the metadata-byte-identical invariant is pinned in `tests/test_context.py`.
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
from sampleflux.ops.context import Mix, Save

# Fan-out/fan-in on the PLAIN sequential engine: snapshot → mutate the stream → merge back.
flux = Flux(
    source=my_source,
    ops=[
        Save(name="clean"),                        # snapshot the pristine sample into a cell
        my_augment_op,                             # the stream mutates freely
        Mix(target_from="clean", drop=["clean"]),  # fan-in: target from the snapshot, cell freed
    ],
)

# The same op list outside an engine needs the Context an engine would have created:
from sampleflux.context import Context, activate, require

with activate(Context()):
    for op in ops:
        sample = op(sample)

# A custom op joins the wiring plane through the same seam the built-in six use:
#     require("MyOp").get("clean")   /   require("MyOp").put("my_cell", value)
```

### What you may change (and where it's documented)

- **Writing a custom wiring op** is the supported extension point: call
  `require("YourOpName")` inside `__call__`, follow the by-reference/copy-on-read discipline, and
  free cells you consume. Usage of the six built-in ops lives in [graph.md](graph.md).
- **Keep the surface narrow.** Don't root-export `activate`/`current`/`require`, and don't grow
  `Context` into a general blackboard — anything that should *persist with the sample* belongs on
  the metadata bus, not in a cell.
- **Changing cell semantics** (by-reference storage, loud missing-cell errors, the `Parallel`
  boundary rule, `copy()` shallowness) is an architectural change: the flow⇄ops parity suite and
  the pinned context invariants (`tests/test_context.py`, `tests/test_flow.py`) define the
  contract. Update this record and the sampleflux `AGENTS.md` context mandate together.

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
  handling, module scans, `ACCEPTS`/`PRODUCES` schemas) is why the module exists; whether the
  resolution half should delegate to confluid is a tracked follow-up in the root `TASKS.md`.

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
`ops/` imports `core._apply_op` (the contract-aware chokepoint); moving `FilterOp`/`WrappedOp`
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
    .map(np.sqrt)                                # appends WrappedOp(f="numpy:sqrt")
    .filter(lambda s: float(s.input.max()) > 0)  # appends FilterOp(p=...)
)
both = Flux.joint([flux_a, flux_b])              # Flux(source=JointFlux([flux_a, flux_b]))
```

### What you may change (and where it's documented)

- **A new engine-constructed helper** (another fluent-API target) belongs in `core.py` for the
  same import-direction reason; an op users wire *directly* (YAML/canvas) belongs in `ops/` with
  a category and group.
- **Do not add a discovery category to `FilterOp`/`WrappedOp`** — surfacing a raw-callable
  parameter on a canvas is a dead widget; the taxonomy is pinned in `tests/test_categories.py`.

---

## The typed-bag model: a named bag of typed items (`sampleflux.bag`, 2026-07-21)

### Context

Before the typed model, the carrier was a fixed `(input, target, metadata)` 3-tuple where `metadata`
was one flat `dict` shared by the whole sample. Everything that is not literally the model input or
target rode that dict by string key: segmentation masks, `[f0,f1,t0,t1]` region lists, window locators,
`spectrogram_params`, power stats, `snr_db`, a signal's samplerate, an image's canvas size, a
label's class names. Two structural costs follow. First, **metadata has no owner** — `samplerate`
belongs to *the signal*, `canvas` to *the image*, but the flat dict severs that link. Second, **a
transform cannot move several fields together** — flipping an image and its mask and its boxes with
one shared decision is inexpressible when the fields are `input`, `target`, and `metadata["regions"]`
respectively, so today's augmentation adapters hard-code a `TargetMode = Literal["none","mask","boxes"]`
knob per op instead. `target` is also overloaded — sometimes a bare string (`"drone_x"`), sometimes a
`{boxes, labels}` dict.

### Decision

`sampleflux.bag` models a sample as a **named bag of typed items with per-field role tags**, and
dispatches transforms on item TYPE via a kernel registry:

- **Items own their metadata.** An item is a typed value plus the metadata that describes *it*
  (`Image(arr, layout)`, `Regions(boxes, labels, canvas)`, `Label(value, classes)`). The
  realization is HYBRID: array-backed items (`Image`/`Mask`) subclass `np.ndarray` with
  attribute-preserving `__array_finalize__`, so a type-agnostic op touches them as an array;
  structured items (`Regions`/`Label`) are dataclass wrappers. A uniform `item_data` / `with_data`
  pair hides the difference from kernels. sampleflux ships only MODALITY-NEUTRAL items; signal-domain
  items (`Signal`, `Spectrogram`) live in the domain package and register into the same registry (see
  "Consequences").
- **`Sample` is a named bag; `input`/`target` are role TAGS, not positions.** A field carries a
  role (`input`/`target`/`aux`/`pred`); `inputs()`/`targets()`/`aux()` read them at the
  train/collate/sink boundary. A field changes role without moving keys. The sample is immutable —
  every mutator returns a new sample (copy-on-write).
- **Transforms sample params ONCE, then dispatch a kernel per item type** (the torchvision-v2
  `_KERNEL_REGISTRY` pattern, structurally the same registry idea as `sampleflux.collate`). Kernels
  are registered per `(transform, item type)` and resolved by MRO. Targeting is by type, with an
  optional `only=[keys]` filter.
- **External libraries plug in through adapters, dropped in BARE.** A `Pipeline` COERCES each element
  (`coerce_transform`): a `Transform` is used as-is; a foreign object is wrapped by whichever adapter
  a matcher/factory pair claims it (`register_adapter`). The built-in torchvision-v2 and albumentations
  adapters register a matcher (by MRO module name — no eager library import) at package load, so
  `v2.Normalize(...)` / `A.GaussNoise(...)` go straight into a `Pipeline` with no explicit wrapper. A
  plain function becomes a transform via `as_transform`; a new item type is taught to an existing
  transform with one `@Transform.kernel(NewType)` registration. This keeps consumer-dialect knowledge
  (how to recognise/adapt a library) OUT of the core and open for any user library.

This is **THE sampleflux data model** — the one carrier every source, op, engine and sink handles.
It deliberately introduces a `Transform` base and typed item classes; the "Functional Purity" mandate
(see `AGENTS.md`) holds because that base is a thin type-dispatch shell and the per-type kernels stay
plain callables.

### Consequences

- **Cross-field consistency is free** — one sampled decision flips image + mask + boxes together,
  the thing a flat-metadata triple could not do.
- **Names and types work together**, so the "torchvision uses types / albumentations uses names"
  split is resolved by one container: the key is the name, the item is the type.
- **The subpackage is `bag`, an internal module home** — the whole typed surface is imported from
  the package top level (`from sampleflux import Sample, ...`), so the module layout is never in a
  consumer's import path and can move without touching consumers.
- **Batching is `typed_collate`** — it returns a batched `Sample` (payloads stacked per field,
  per-item attrs collected as lists, roles preserved); there is no `list[dict]` batch-in-metadata
  form.
- **Deliberately deferred** (see root `TASKS.md`): a torch-`Tensor`-subclass item base (torch
  payloads ride wrapper items for now), confluid-native item-type discovery, the generated
  `Tv*`/`Alb*` families in this namespace, FluxStudio typed side sockets, and the `decode` path.

### Example

```python
from sampleflux import Sample, Image, Mask, Regions, Label, Pipeline
from torchvision.transforms import v2
import albumentations as A

sample = Sample(
    {"image": Image(rgb), "mask": Mask(seg), "regions": Regions(boxes, canvas=(H, W)), "class": Label("drone_x")},
    roles={"mask": "target", "regions": "target", "class": "target"},
)
out = Pipeline([
    v2.RandomHorizontalFlip(p=1.0),  # Image + Mask + Regions together (one library draw)
    v2.Normalize(m, s),              # Image (torchvision v2, by type) — wrapped by a registered adapter
    A.GaussNoise(p=1.0),             # Image (albumentations, by name) — wrapped by a registered adapter
])(sample)
# image flipped+normalized+noised; mask+regions flipped consistently; out["class"] untouched.
# sampleflux ships NO native augmentation transforms — the libraries cover that via coercion.
# Signal-domain items + the Fourier transform live in the domain package and register into the
# same registries — a bare Fourier() drops into this Pipeline with no core edit.
```

### What you may change (and where it's documented)

- **A new item type** — add a class + `@register_item` (usage: [typed-model.md](typed-model.md)); if
  it is array-backed, subclass `NDArrayItem` and declare `_item_attrs`.
- **A new per-type behaviour for an existing transform** — register a kernel
  (`@Transform.kernel(ItemType)`), no core edit.
- **The typed surface is imported from the package top level** — `bag/*` is the internal module
  home; never teach a `sampleflux.bag.*` import path, so the module layout can change without
  touching consumers.

---

## Native typed transforms that change a field's TYPE (`ConvertToImage`/`Threshold`/`ConnectedComponents`, 2026-07-22)

### Context

Two shapes of typed transform exist. The first is the augmentation shape the base `Transform`
was built for: it `handles` an item type and, per handled field, applies a registered kernel that
returns *the same type* (a flip returns a flipped `Image`), so `image`, `mask`, and `boxes` move
together and library transforms (torchvision v2 / albumentations) drop in through the adapter
coercion registry. The workspace deliberately ships **no** native transforms of that shape —
libraries cover it.

But a running detection/segmentation front-end needs a different shape: **read one field, write a
field of a DIFFERENT type**. Turning a numeric array into a displayable image, thresholding an
array into a boolean mask, and labelling that mask into a set of bin boxes are each a *type
change* (`array → Image`, `array → Mask`, `Mask → Regions`), not an in-place per-type edit. No
library provides them, and the earlier ops that did (`ConvertToImageOp`, `ThresholdOp`,
`ConnectedComponentsOp`) operated on a flat `(input, target, metadata)` triple, which the typed
model does not carry. Without typed equivalents a `Sample` pipeline could not reach `Regions` from a
raw array — the critical path for typed detection was blocked.

### Decision

Add native typed **twins** that subclass `Transform` and OVERRIDE `__call__` (rather than register
a kernel), reading one field and writing a different-typed item — the same shape the domain
package's `Spectrogram` twin (`Signal → Spectrogram`) already established:

- A twin declares `handles` / `consumes` / `produces` **truthfully** as graph metadata (e.g.
  `ConnectedComponents`: `consumes=(Mask,)`, `produces=(Regions,)`), but does its work in
  `__call__`, not through the kernel-dispatch loop — kernel dispatch is for same-type per-field
  edits, and a type change has one input field and one output field.
- The source field is resolved by a small `_find_*` helper: an explicit `field=` name, else the
  first item of the natural type (a `Mask` for `ConnectedComponents`) or the first array-bearing
  item — every miss raises a `ValueError` naming the sample's fields.
- The output is written with `sample.replace_field(output, item)` + `sample.set_role(output, role)`
  (copy-on-write), and the role is chosen semantically: the working image is `input`, a threshold
  mask and raw connected-component boxes are `aux` (intermediates, and specifically NOT `pred` —
  that role is reserved for a detector's output).
- Each twin **reuses its legacy op's math verbatim** so the numbers are pinned identical:
  `ConvertToImage` calls the shared `_render_rgb`/`_bound_longest_side` render core;
  `ConnectedComponents` calls the shared `connected_component_bboxes` helper; `Threshold`
  delegates to a legacy `ThresholdOp` instance run on a shim `Sample`. The twins are STRICTLY
  ADDITIVE — the legacy ops are untouched, because many consumers still use them via the `Sample`
  path.

The generic connected-components output format is a hard contract: `Regions.boxes` is a list of
`(row_min, row_max, col_min, col_max)` inclusive integer tuples (**row bounds first, then column
bounds**). A downstream back-projection reads exactly that order to map bins to a world / signal
coordinate frame, so the tuple order is load-bearing, not incidental.

### Consequences

- A `Sample` carrying a raw 2-D array runs `ConvertToImage → Threshold → ConnectedComponents`
  end-to-end and arrives at a `Regions` field with no legacy `Sample` anywhere — the typed
  detection/segmentation front-end is unblocked.
- Parity is free and provable: because each twin reuses the legacy math, a twin's output is
  byte-identical to a legacy run on the equivalent `Sample` (pinned in
  `tests/test_typed_generic_ops.py`).
- `ConvertToImage` does NOT republish `image_width_px` / `image_height_px` (the legacy op wrote
  them into the shared metadata dict). The `Image` item's array SHAPE carries the pixel
  dimensions, and the typed model has no shared dict to write into — a consumer reads the dims off
  the payload.
- `Threshold`'s `{meta_key}` expression grammar has no typed home (an item owns its own metadata;
  there is no shared sample dict), so only numeric literals and `$ENV` bounds resolve in the twin;
  a `{key}` bound raises loudly. Literal dB thresholds — the critical path — are unaffected.
- The twins carry `category="op"` + `group="image"`/`"numpy"`, so they are discoverable exactly
  like the legacy ops (their modules were already entry-pointed; a class added to a registered
  module needs no new entry point).
- The two DETECTION-TARGET twins `CocoToTorchVisionDetection` / `MasksToDetectionBoxes`
  (`sampleflux/ops/target.py`, `group="structure"`) are the SAME shape reaching one step further:
  they read one source field (a `Label` carrying a COCO `objects` mapping, or a `Mask`) and write
  the torchvision detection target as a `Regions` item — `boxes` = the `[N,4]` xyxy tensor,
  `labels` = the class-id tensor — tagged **`target`** (not `aux`: this IS the supervised target a
  loss consumes, whereas `ConnectedComponents`'s raw blobs are an intermediate). `Regions` is the
  natural typed home for a bounding-box set and the batch-friendly one — `typed_collate` gathers
  per-sample `Regions` into a list of targets (the variable-N detection batch convention, since
  boxes can't be stacked), exactly as it gathers a classification target `Label`. Byte-parity is
  again free (each delegates to its legacy `*Op` on a shim `Sample`). Pinned in
  `tests/test_typed_detection_target_ops.py`.

### Example

```python
from sampleflux import Sample, Mask
from sampleflux.ops.image import ConvertToImage
from sampleflux.ops.numpy import Threshold, ConnectedComponents

sample = Sample({"spec": Mask(db_spectrogram)})            # a raw 2-D array item
sample = ConvertToImage()(sample)                               # + Image field (role "input")
sample = Threshold(field="spec", low_level=-30.0)(sample)       # + Mask field (role "aux")
sample = ConnectedComponents(field="mask")(sample)              # + Regions field (role "aux")

sample["boxes"].boxes  # [(row_min, row_max, col_min, col_max), ...] — the pinned bin-box contract
```

### What you may change (and where it's documented)

- **A twin's source-field resolution or output role** — keep the `_find_*` → `replace_field` →
  `set_role` shape and a loud `ValueError` on a miss; `aux` vs `pred` is a semantic choice
  (raw detections are `aux`).
- **The `(row_min, row_max, col_min, col_max)` bin-box order is a contract** — a back-projection
  depends on it; changing it is an architectural change that must update this record and every
  consumer.
- **Do not modify the legacy ops or reimplement their math in a twin** — a twin reuses the legacy
  math so parity is guaranteed; the twins are additive and the legacy `Sample`-path consumers must
  keep working.

## A typed field cannot hold a live torch tensor — `ToTensor` stores CHW-float numpy (`ToTensor`/`EncodeTarget`/`DecodeTarget`/`MetadataToTarget`, 2026-07-22)

### Context

The typed detection twins above reach `Regions`; a typed CLASSIFICATION front-end needs the other
two shapes: turn the working image into the model's **input tensor**, and turn the class-name label
into the encoded **target id**. The earlier ops that did this (`ToTensorOp`, `MetadataToTargetOp`,
`EncodeTargetOp` / `DecodeTargetOp`) operated on a flat `(input, target, metadata)` triple. Two
facts of the typed model shape the twins: (1) there is NO shared metadata dict — the label already
rides a `Label` field that owns its metadata; (2) an array item is an `np.ndarray` SUBCLASS whose
`__new__` runs `np.asarray(data)`, so **a field payload is coerced to numpy** — an `Image` cannot
hold a live `torch.Tensor` (verified: `item_data(Image(tensor))` is an `ndarray`), and a bare tensor
stored directly as a field value has no registered item type, so `typed_collate` / the storage codec
(`bag.io.encode_item`) cannot serialize it.

### Decision

Add native typed twins subclassing `Transform` and overriding `__call__` (the same shape as the
detection twins), each reusing its legacy op VERBATIM on a shim `Sample` for byte-parity:

- **`ToTensor`** (`ops/torch.py`, `group="torch"`) resolves an array-bearing field (explicit `field`
  or the first array/PIL item), runs `ToTensorOp` (HWC→CHW + `normalize`), and writes an `Image`
  with `layout="CHW"`. Because `NDArrayItem` coerces the payload, the stored value is a CHW `float32`
  **numpy** array whose values equal `ToTensorOp(...).input.numpy()` — NOT a live tensor. By default
  it REPLACES the source field in place so the field's `input` role is preserved (`output` writes a
  new field tagged `input` instead). `typed_collate` stacks these payloads with `np.stack`; the
  numpy→tensor conversion is the collate / model boundary's job, exactly as for any numpy dataset. A
  Tensor-subclass item that would let a field carry a live tensor is the documented follow-up
  (`bag/items.py` note + root TASKS.md).
- **`EncodeTarget` / `DecodeTarget`** (`ops/target.py`, `group="structure"`) resolve a `Label` field,
  map its `.value` through the config-pinned `mapping` by delegating to `EncodeTargetOp` /
  `DecodeTargetOp` (so the non-empty-mapping validation AND the shared `_lookup` are byte-identical),
  and write a new `Label` (carrying the source label's `classes`) tagged `target`. In place by
  default (`output` blank).
- **`MetadataToTarget`** is provided for PARITY / config-compat but is largely REDUNDANT in the typed
  model: a source emits the label directly as a `Label` field already tagged `target`, so no
  metadata→target move is needed. The twin reads a field's natural value (a `Label`'s `.value`, else
  its array payload) or a named attribute (`key=`) and writes a target `Label` — the escape hatch for
  a label that rode as another item's attribute.

### Consequences

- A `Sample` carrying an HWC `Image` (role input) + a name `Label` (role target) runs
  `ToTensor → EncodeTarget` into a CHW-float input field + an int-id target field, with no legacy
  `Sample` anywhere — the typed classification front-end is unblocked.
- The model-input payload is CHW-float **numpy**, not a live `torch.Tensor`; a consumer / trainer
  tensorizes at the collate or forward boundary. This is a deliberate current limitation, not a bug —
  it disappears when the Tensor-subclass item lands.
- The twins carry `category="op"` + the legacy `group`, so they are discoverable like the legacy ops
  (their modules — `sampleflux-ops-torch` / `sampleflux-ops-target` — are already entry-pointed; a
  class added to a registered module needs no new entry point).

### Example

```python
from sampleflux import Sample, Image, Label
from sampleflux.ops.torch import ToTensor
from sampleflux.ops.target import EncodeTarget

sample = Sample(
    {"image": Image(hwc_uint8), "class": Label("cat")},
    roles={"image": "input", "class": "target"},
)
sample = ToTensor(field="image")(sample)                 # image -> CHW float32 Image (role input, in place)
sample = EncodeTarget(mapping={"cat": 0, "dog": 1}, field="class")(sample)  # class -> Label(0) (role target)
```

### What you may change (and where it's documented)

- **The Tensor-subclass item follow-up** — once a field can carry a live tensor, `ToTensor` should
  store it directly; update this record and the `bag/items.py` note together.
- **`ToTensor`'s replace-in-place default vs a new output field** — keep role preservation (in place)
  as the default; a new `output` field is tagged `input`.
- **Do not modify the legacy ops or reimplement their math in a twin** — the twins delegate to the
  legacy ops for byte-parity and are strictly additive.
