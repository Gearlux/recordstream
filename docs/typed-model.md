# The typed-bag model — THE sampleflux data model

> **Status: the data model (migration in progress).** The typed bag replaces the classic
> `Sample(input, target, metadata)` triple; the legacy engine survives only until every consumer
> has migrated (staged in the root `TASKS.md`), after which it is deleted and `TypedSample` is
> renamed `Sample`. Import the typed surface from the PACKAGE TOP LEVEL
> (`from sampleflux import TypedSample, Image, Transform, ...`) — internal module paths are
> transitional. The design rationale is recorded in
> [architecture.md](architecture.md#the-typed-bag-model-a-named-bag-of-typed-items-sampleflux-bag-poc-2026-07-21).

## Why

In the classic model everything that is not literally the model input or target — a segmentation
mask, `[f0,f1,t0,t1]` regions, a signal's samplerate, an image's canvas size, a label's class
names — is jammed into one flat `metadata` dict keyed by string, disconnected from the value it
describes. That makes two things hard: metadata has no natural home, and a transform cannot move
several fields together consistently (flip an image → flip its mask → flip its boxes).

The typed-bag model fixes both: **a sample is a named bag of typed items, and metadata lives on the
item it describes.** Transforms dispatch on item *type*.

## The pieces

### Items — typed values that own their metadata

sampleflux is **modality-neutral**, so its core ships only generic items — images, masks, boxes,
labels. (Signal-domain items live in the domain package; see below.)

```python
from sampleflux import Image, Mask, Regions, Label

Image(rgb_hwc, layout="HWC")                        # an image knows its layout
Mask(seg_hw)                                         # a mask shares its image's frame
Regions(boxes=[[1,1,4,4]], labels=["drone"], canvas=(8, 10))
Label("drone_x", classes=["noise", "drone_x"])
```

Items are **hybrid**: array-backed items (`Image`, `Mask`) subclass `np.ndarray`, so a
type-agnostic operation touches them as an array and their extra attributes survive numpy ops;
structured items (`Regions`, `Label`) are dataclass wrappers. A uniform payload accessor hides the
difference from kernels:

```python
from sampleflux import item_data, with_data
item_data(Image(arr))                  # -> the plain ndarray
with_data(Image(a, layout="CHW"), b)   # a copy carrying b, layout preserved
```

### `TypedSample` — a named bag with role tags

```python
from sampleflux import TypedSample

sample = TypedSample(
    {"image": Image(rgb), "regions": Regions(boxes), "class": Label("drone_x")},
    roles={"regions": "target", "class": "target"},   # default role is "input"
)
sample.inputs()    # {"image": Image(...)}
sample.targets()   # {"regions": Regions(...), "class": Label(...)}
sample.set_role("regions", "aux")   # copy-on-write; a field's role changes without moving keys
```

`input` / `target` / `aux` / `pred` are **tags read at the train/collate/sink boundary**, not tuple
positions. `TypedSample` is immutable — every mutator returns a new sample.

### Transforms — type dispatch with once-per-sample parameters

A transform samples its parameters once, then applies a per-type kernel to each handled field.
Fields it does not handle pass through. Because the parameters are sampled **once** and shared,
image / mask / boxes move consistently — the thing the flat-metadata model could not express. A
transform may also CHANGE an item's type under the same key (e.g. the domain `Fourier` turns a
`Signal` field into a `Spectrogram` in place).

**sampleflux ships no native augmentation transforms** — geometric/photometric augmentation comes
from torchvision `transforms.v2` / albumentations through the coercion registry below; native
transforms exist only where no library covers them (domain packages register their own).

### Mixing libraries — one pipeline, many worlds

torchvision `transforms.v2` dispatches by type, albumentations by keyword name. **Bare library
transforms drop straight into a `Pipeline`** — a registered adapter wraps each one automatically, and
each transform hits only the field(s) it handles:

```python
from torchvision.transforms import v2
import albumentations as A

Pipeline([
    v2.RandomHorizontalFlip(p=0.5),           # torchvision v2: Image + Mask + Regions together (one draw)
    v2.Normalize(mean, std),                  # torchvision v2: Image  (wrapped automatically)
    A.GaussNoise(p=1.0),                      # albumentations: Image  (wrapped automatically)
])(sample)
```

The coercion is a small **registry** (`sampleflux.bag.register_adapter` / `coerce_transform`): the
built-in torchvision-v2 and albumentations adapters register a matcher (by MRO module name, no eager
import) at package load. Teach a `Pipeline` about your own library's transforms with one call:

```python
from sampleflux import register_adapter
register_adapter(lambda o: type(o).__module__.startswith("mylib"), lambda o: MyLibAdapter(o))
```

For surgical control — target one field key with a library transform — construct the adapter
explicitly: `TorchvisionV2Adapter(v2.Normalize(...), only=["image"])`.

Runnable end-to-end: [`examples/typed_pipeline.py`](../examples/typed_pipeline.py).

## Extending it

### A custom transform from a plain function

```python
from sampleflux import as_transform, Image
brighten = as_transform(lambda d: d + 0.1, handles=(Image,), only=["image"])
```

### A custom item type + a kernel for an existing transform — no core edit

```python
from sampleflux import register_item
from mypkg.transforms import MyGeoTransform   # any Transform subclass

@register_item
class Keypoints:
    def __init__(self, points): self.points = points

@MyGeoTransform.kernel(Keypoints)
def _(item, params):
    return move_points(item, params)
```

Dispatch is MRO-aware: a kernel registered for a base item type also serves its subclasses, and a
subclass transform inherits its base's kernels until it overrides them.

### Signal-domain items live in the domain package (`waivefront.bag`)

This is the same mechanism, applied across packages: because sampleflux is modality-neutral, the
signal-domain `Signal` / `Spectrogram` items and the `Fourier` transform (`Signal` → `Spectrogram`)
live in `waivefront.bag` and register into the SAME registries on import — so a bare `Fourier()`
drops into a `sampleflux.bag.Pipeline` alongside the generic transforms with no core edit. See
`waivefront/examples/05_typed_bag_signal.py`.

## Engines — Flux and FlowGraph carry the typed bag

A `TypedSample` is **never coerced**: on every `Flux` route (sequential / parallel / streamed /
`__getitem__`) and in `FlowGraph`, a typed source item passes through verbatim and each op receives
the whole bag (`Pipeline` transforms, structure ops, and the compose plane — `TransformChain`,
`RandomApply`, `Enable`, `Apply`, `Capture` — all route typed carriers correctly).

```python
Flux(source=typed_source, ops=[v2.RandomHorizontalFlip(p=0.5), Fourier()]).to_sink(HDF5Sink(...))
```

### Typed fan-in (`merge_from`) and field binds (`step[key]`)

In a `flow:` document, the typed fan-in is **`merge_from`** — the UNION of the named steps' fields
and roles, in slot order, last-write-wins on a key collision (the typed replacement for the legacy
metadata dict-merge). The idiom for a derived-field branch: produce, `SelectFields` the new
field(s), merge:

```yaml
flow:
  start:     {}
  masked:    {op: !class:mypkg.MakeMask(), from: start}
  mask_only: {op: !class:sampleflux.ops.structure.SelectFields(keys: [mask]), from: masked}
  boosted:   {op: !class:mypkg.Boost(), from: start}
  out:       {from: boosted, merge_from: [mask_only]}
```

`bind:` references gain a field form: `step[key]` binds the named ITEM of that step's bag as an op
parameter; a bare `step` reference binds the step's PRIMARY input-role item
(`sampleflux.primary`). Lowering (`to_ops`) compiles `merge_from` to the `MergeFields` context op
and key-binds to `Apply(key=...)`; lifting (`from_ops`) round-trips both. `target_from` /
`metadata_from` stay legacy-`Sample`-only (a typed step using them raises; `merge_from` on a legacy
carrier likewise).

## Storage — the typed field-group layout

All three backends (`HDF5Sink`↔`HDF5Source`, `ZarrGroupSink`↔`ZarrGroupSource`,
`DirectorySink`↔`DirectorySource`) write a `TypedSample` in ONE logical schema: per sample, one
group per FIELD carrying the item's registered type name, its role, the payload as a dataset, and
its attrs (scalars natively — queryable; arrays as sub-datasets; structured values JSON-tagged so
tuples survive). The store is stamped `sampleflux_format = "typedsample-v1"`; a store holds ONE
carrier — appending a legacy `Sample` to a typed store (or vice versa) raises. Backends never
inspect item internals — everything serializes through the `sampleflux.bag.io` codec
(`encode_item`/`decode_item`), so an externally-registered item type round-trips with zero storage
edits; `register_io(MyItem, encode=..., decode=...)` overrides the default structural codec when
needed.

```python
sink = HDF5Sink(path="out.h5", overwrite=True)
with sink:
    for sample in flux:           # TypedSamples
        sink.write(sample)
back = list(HDF5Source(path="out.h5"))   # exact TypedSamples: fields, roles, order, tuple attrs
```

`ZarrBatchSink` (the uniform single-array sink) appends the PRIMARY input field's payload per row
and stores a one-time item template — per-sample attr variation needs `ZarrGroupSink`.

### Querying typed stores without loading arrays

The metadata scans yield the nested `{field: {attr: value}}` shape, and a `where` expression
addresses it as `<field>.<attr>`:

```python
fast = MetadataFilterSource(source=HDF5Source(path="out.h5"), where="signal.samplerate > 1e6")
```

Array-valued attrs appear as shape/dtype stubs (presence/shape testable, never loaded). A field
named like a Python keyword (e.g. `class`) can't be addressed in an expression — use the
programmatic `predicate` or a non-keyword field name.

## Interop with the classic `Sample`

Run a typed pipeline against the existing sources/sinks by bridging both ways. The lowering is
lossless — the whole bag is encoded in the legacy metadata while `input` / `target` still expose the
primary payloads for a legacy consumer:

```python
from sampleflux.bag.interop import to_legacy, to_typed
legacy = to_legacy(sample)        # a classic Sample; to_typed(legacy) == sample
typed = to_typed(legacy)          # exact reconstruction
# adopting an arbitrary legacy dataset needs a per-dataset builder:
to_typed(raw_sample, builder=lambda s: TypedSample({"image": Image(s.input), "class": Label(s.target)}))
```

## What is NOT here yet (follow-ups)

A torch-`Tensor`-subclass item base (torch payloads currently ride in wrapper items), confluid-native
item-type discovery, the generated per-transform families (`Tv*` / `Alb*`) in this namespace,
FluxStudio typed side sockets, and the `decode` (inverse) path. See the root `TASKS.md`.
