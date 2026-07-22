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
dispatches on the *detected* kind of the first item (`sampleflux.kinds.classify_carrier`).
sampleflux registers the five kind defaults (`"sample"`, `"pair"`, `"value"`, `"input_meta"`,
`"target_meta"`); consuming projects may register task aliases (`"yolo"`, `"segmentation"`, …)
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
  per-parameter type/default/required, the declared `ACCEPTS`/`PRODUCES` typespec contract), and
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

## The typed-bag model: a named bag of typed items (`sampleflux.bag`, PoC, 2026-07-21)

### Context

The classic carrier is `Sample(input, target, metadata)` — a fixed 3-tuple where `metadata` is one
flat `dict` shared by the whole sample. Everything that is not literally the model input or target
rides that dict by string key: segmentation masks, `[f0,f1,t0,t1]` region lists, window locators,
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
- **`TypedSample` is a named bag; `input`/`target` are role TAGS, not positions.** A field carries a
  role (`input`/`target`/`aux`/`pred`); `inputs()`/`targets()`/`aux()` read them at the
  train/collate/sink boundary. A field changes role without moving keys. The sample is immutable
  (copy-on-write), mirroring `Sample._replace`.
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

This is a **coexisting proof of concept**, not a replacement: it lives beside the classic engine and
changes none of it. The existing "Functional Purity", "Sample Triplet", and "Stored Type Is Derived"
mandates are scoped to the classic engine (see `AGENTS.md`), because the typed model deliberately
introduces a `Transform` base and typed item classes.

### Consequences

- **Cross-field consistency is free** — one sampled decision flips image + mask + boxes together,
  the thing the flat-metadata model could not do.
- **Names and types coexist**, so the "torchvision uses types / albumentations uses names" split is
  resolved by one container: the key is the name, the item is the type.
- **The subpackage is `bag`, not `typed`** — `sampleflux.typespec.typed` (the `@typed(...)` contract
  decorator) is re-exported at the package root as `sampleflux.typed`, so a `sampleflux/typed/`
  submodule would shadow it. `bag` is collision-free; `TypedSample` / dispatch / docstrings carry the
  "typed" concept.
- **Batching stays in `sampleflux.collate`** (transforms are per-sample); the classic model's
  `list[dict]` batch-in-metadata form is not carried into the bag model.
- **Deliberately deferred** (see root `TASKS.md`): a torch-`Tensor`-subclass item base (torch
  payloads ride wrapper items for now), confluid-native item-type discovery, the generated
  `Tv*`/`Alb*` families in this namespace, FluxStudio typed side sockets, and the `decode` path.

### Example

```python
from sampleflux import TypedSample, Image, Mask, Regions, Label, Pipeline
from torchvision.transforms import v2
import albumentations as A

sample = TypedSample(
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
- **Do not name the `bag` subpackage `typed`** — it shadows `sampleflux.typed` (the `@typed`
  decorator). The rename rationale is pinned here and in the module docstrings.
- **Promoting this from PoC to the default model** is a workspace-wide decision that would re-scope
  the classic-engine mandates and port every consumer — out of scope for the proof of concept.
