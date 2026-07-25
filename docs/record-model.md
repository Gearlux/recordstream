# The record model — THE sampleflux data model

A sample is a **plain `dict`** of **typed values**. Import the whole surface from the PACKAGE TOP
LEVEL (`from sampleflux import Record, Image, Mask, Regions, Label, Transform, Pipeline,
as_transform, item_data, with_data, register_item, register_kernel, register_io, collate_records, ...`).
The design rationale is recorded in
[architecture.md](architecture.md#one-type-dispatched-op-engine--plain-dict-records-libraries-as-is-2026-07-25).

## Why

If everything that is not literally the model input or target — a segmentation mask,
region boxes, a signal's samplerate, an image's layout, a label's class names — is jammed into one
flat `metadata` dict keyed by string, it is disconnected from the value it describes. And if the
carrier is a bespoke container class, every external library needs an adapter before it can touch it.

The record model fixes both. **A sample is a plain dict, values are typed, and metadata lives on the
value it describes** — an `Image` carries its `layout`, a `Label` its `classes`. **Key names carry
meaning** (`"image"`, `"mask"`, `"bboxes"`, `"labels"`, `"class"` — the same convention as every torch
batch dict and albumentations' keyword vocabulary), so libraries that already understand dicts or
named kwargs run **as-is**, with no wrapper anywhere. A scalar side value is just another key:

```python
record = {
    "image": Image(rgb_hwc),                              # typed: knows its layout
    "mask": Mask(seg_hw),                                 # shares the image's frame
    "bboxes": [[2, 3, 6, 7]],                             # albumentations vocabulary
    "labels": ["drone"],
    "class": Label("drone_x", classes=["noise", "drone_x"]),
    "samplerate": 30.72e6,                                # a plain value is just another key
}
```

There is deliberately **no container class** — `Record` is a type alias (`Dict[str, Any]` in
`sampleflux.items`), ops receive and return ordinary dicts, and `None` means "drop this record"
(filter semantics).

## The pieces

### Items — typed values that own their metadata

sampleflux is **modality-neutral**, so its core ships only generic items — images, masks, boxes,
labels. (Domain items — a signal, a spectrogram — live in the domain package; see below.)

```python
from sampleflux import Image, Mask, Regions, Label

Image(rgb_hwc, layout="HWC")                        # an image knows its layout ("HWC" default / "CHW")
Mask(seg_hw)                                         # a mask shares its image's frame
Regions(boxes=[[1,1,4,4]], labels=["drone"], canvas=(8, 10), extras={"snr_db": [12.5]})
Label("drone_x", classes=["noise", "drone_x"])
```

Items are **hybrid**: array-backed items (`Image`, `Mask`) subclass `NDArrayItem` — an `np.ndarray`
subclass whose declared `_item_attrs` survive numpy operations via `__array_finalize__` — so a
type-agnostic operation touches them as an array; structured items (`Regions`, `Label`) are dataclass
wrappers (a bounding-box set is not an array). A uniform payload accessor hides the difference from
kernels:

```python
from sampleflux import item_data, with_data
item_data(Image(arr))                  # -> the plain ndarray
with_data(Image(a, layout="CHW"), b)   # a copy carrying b, layout preserved
```

`register_item` / `is_item` / `item_types` / `get_item_type` / `item_type_names` are the open item
registry — the extensibility surface a domain package or user type plugs into (one class + one
decorator, no core edit).

### Ops — type dispatch with once-per-record parameters

A `Transform` (`sampleflux.transform`) samples its parameters ONCE per record
(`get_params(record)`), then applies a per-type **kernel** to every value whose type it handles
(`@MyOp.kernel(ItemType)`, resolved MRO-aware by `sampleflux.dispatch`). Values it does not handle
pass through. Because the parameters are sampled once and shared, one op moves every handled value
with the SAME decision — the torchvision-v2 model. Targeting is by TYPE; the `field=` constructor
parameter pins an op to one named key when a record holds several values of a handled type.

```python
import numpy as np
from sampleflux import Image, Record, Transform

class Brighten(Transform):
    handles = (Image,)

    def __init__(self, strength: float = 0.1, field: str | None = None) -> None:
        super().__init__(field=field)
        self.strength = strength
        self._rng = np.random.default_rng(7)

    def get_params(self, record: Record) -> dict:
        return {"offset": self._rng.uniform(0.0, self.strength)}   # drawn ONCE per record

@Brighten.kernel(Image)
def _brighten_image(value: Image, params: dict) -> Image:
    return Image(np.asarray(value) + params["offset"], layout=value.layout)
```

The second sanctioned op shape is the **type-changing op** — read one key, write a differently-typed
item (`Threshold`: array → `Mask`, `ConvertToImage`: array → `Image`, `ConnectedComponents`:
`Mask` → `Regions`, the target ops). It subclasses `Transform` and overrides `__call__` instead of
registering a same-type kernel, declaring `handles` / `consumes` / `produces` truthfully as graph
metadata (next section).

### Declaring an op's type interface — `handles` / `consumes` / `optional` / `produces`

Every `Transform` carries four class-level tuples of item types. They are the op's **type
interface**: what a reader (or a machine — a visual editor's typed sockets, a pipeline linter)
learns about the op without executing it or loading the kernel registry.

| Attribute | Meaning | Enforced at runtime? |
|---|---|---|
| `handles` | The value types this op processes — every record value of one of these types is touched, everything else passes through. | Only by `FunctionTransform` / `as_transform` (`isinstance(value, self.handles)` is its application gate). For a kernel op, actual dispatch is the kernel registry (`dispatch(type(self), type(value))`) — `handles` must MIRROR the registered kernels. |
| `consumes` | The input types the op NEEDS to do useful work (its required inputs). Convention: an empty `consumes` means "same as `handles`". | No — declarative. |
| `optional` | Input types the op uses when present but works without (e.g. a geometric op that also moves a `Mask` if the record has one). | No — declarative. |
| `produces` | The types the op ADDS or CHANGES — its output contract (what a downstream op can rely on finding). | No — declarative. |

Concretely, `ToTensor` declares:

```python
class ToTensor(Transform):
    handles = (NDArrayItem,)     # touches array-backed values
    consumes = (NDArrayItem,)    # needs at least one array-bearing key to act on
    produces = (torch.Tensor,)   # writes a LIVE CHW float tensor under `output` (or in place)
```

(A `produces` entry need not be a registered item type — `ToTensor`'s output is a plain
record value, which is exactly what the declaration should say.)

**When `handles` and `consumes` differ.** They coincide for a simple one-type op (`Threshold`,
the FFT ops), and diverge in two directions:

- **Optional riders — `handles` ⊃ `consumes`.** A joint geometric op MAY move several types with
  one draw but only REQUIRES one of them:

  ```python
  class JointFlip(Transform):
      handles  = (Image, Mask, Regions)   # everything ONE draw may move
      consumes = (Image,)                 # the only input it needs to be useful
      optional = (Mask, Regions)          # moved together with the image when present
  ```

  A record with just an `Image` is fine; a record that also carries a `Mask`/`Regions` gets them
  moved consistently. Declaring `consumes = handles` here would wrongly tell a reader (or a
  pipeline linter) that a mask is required.

- **Read-only reference inputs — `consumes` ⊃ `handles`.** An op may NEED a value it never
  changes. A denoiser that estimates the noise floor from the signal but excludes the
  ground-truth ON regions when a mask is available follows the same logic with `optional`
  (a real op: `handles = consumes = (Signal,)`, `optional = (Mask, GridMask)`, `produces =
  (Signal,)` — the mask is read, never written). The required-reference variant looks like:

  ```python
  class ScaleBoxesToImage(Transform):
      handles  = (Regions,)          # the only type it CHANGES
      consumes = (Regions, Image)    # ...but it cannot run without the reference Image (its shape)
  ```

In short: `handles` = "what I write", `consumes` = "what must be present", `optional` = "what I
use when present" — the three answer different questions, and only collapse into one tuple for
the simplest ops.

**Limiting a multi-input op to NAMED keys.** The type interface says *what kinds* of values an
op works with; *which record entry* each input comes from is CONFIG. A single-input op uses the
base `field=` param (one key, still type-gated). A multi-input op declares **one `<input>_field`
constructor param per input slot** — defaulting to the conventional key name, resolved and
validated lazily in `__call__`:

```python
class KeepRegionsOnMask(Transform):
    """Drop regions whose center pixel is OFF in the activity mask.

    Args:
        mask_field: Record key of the activity Mask to test against. Defaults to "mask".
        regions_field: Record key of the Regions to filter. Defaults to "regions".
        output: Key the filtered Regions are written to; blank (default) replaces regions_field in place.
    """

    handles = (Regions,)             # the only type it CHANGES
    consumes = (Mask, Regions)       # both inputs must be present
    produces = (Regions,)

    def __init__(self, mask_field: str = "mask", regions_field: str = "regions", output: str = "") -> None:
        super().__init__()
        self.mask_field = mask_field
        self.regions_field = regions_field
        self.output = output

    def __call__(self, record: Record) -> Record:
        for name, want in ((self.mask_field, Mask), (self.regions_field, Regions)):
            if name not in record:
                raise ValueError(f"{type(self).__name__}: no {name!r} key in record (keys: {list(record)})")
            if not isinstance(record[name], want):
                raise TypeError(f"{type(self).__name__}: {name!r} is {type(record[name]).__name__}, expected {want.__name__}")
        mask, regions = record[self.mask_field], record[self.regions_field]
        keep = [b for b in regions.boxes if mask[int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2)]]
        out = Regions(boxes=keep, labels=regions.labels, scores=regions.scores, canvas=regions.canvas)
        return {**record, (self.output or self.regions_field): out}
```

So a record carrying several masks and several region sets is disambiguated entirely in config —
the op looks ONLY at the named entries:

```yaml
- !class:mypkg.KeepRegionsOnMask
  mask_field: activity_mask      # not the segmentation mask under "mask"
  regions_field: predictions     # not the ground truth under "regions"
```

This is the established pattern for every shipped multi-input op (e.g. the region→target ops
take `image_field="image"` + `regions_field="regions"` + `output="target"`). Two rules keep it
predictable: the defaults are the CONVENTIONAL key names (so the common record shape needs zero
config), and a wrong/missing key fails lazily in `__call__` with the key list in the message —
never silently falls back to a different entry when an explicit name was given.

Rules of use:

- **Declare truthfully or not at all.** Nothing validates these tuples against the op's behavior,
  so wrong metadata is worse than missing metadata — it misleads both readers and any tool that
  consumes it. A kernel op's `handles` changes when its kernel registrations change; keep them in
  sync (an externally-registered kernel widens the REAL dispatch without widening `handles` — that
  is fine, `handles` documents the op author's contract, the registry documents the deployment).
- **Kernel ops rarely need more than `handles`** — dispatch and pass-through already follow from
  the registry; `consumes`/`produces` earn their keep on type-CHANGING ops, where the `__call__`
  override hides the type flow that kernels would have made explicit.
- **These tuples never gate execution** (except the `FunctionTransform` case above). If an op must
  refuse to run without an input, validate lazily in `__call__` with a clear error — the same
  lazy-validation convention every op follows.

**sampleflux ships no native augmentation ops** — geometric/photometric augmentation comes from
torchvision `transforms.v2` / albumentations run as-is (next section); native ops exist only where
no library covers them.

### Mixing libraries — as-is, no adapters

The engine's single op-application chokepoint, `sampleflux.core._apply_op(record, op)`, dispatches
on the op's FAMILY (by MRO module name, no eager import) and invokes each family the way its own
library expects:

- **albumentations** — the op receives exactly its own kwarg vocabulary: the
  `image`/`mask`/`masks`/`bboxes`/`keypoints`/`labels` keys present in the record, nothing else. One
  call = one joint draw across them; array outputs are re-wrapped in the incoming value's item type,
  so an `Image`/`Mask` keeps its type and metadata through the library.
- **torchvision `transforms.v2`** — called on the record dict as-is (tv2 walks dicts natively).
  Layout conversions are the library's own transforms (`v2.ToImage()`) — the engine never converts
  silently.
- **everything else** — `op(record)`; `None` drops the record.

So bare library transforms sit in one list with native ops — in `Flux(ops=[...])`, in a `Pipeline`,
in a `flow:` step:

```python
import albumentations as A
from sampleflux import Pipeline

Pipeline([
    A.Compose(                                   # box-carrying augmentation: the library's own Compose
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    ),
    A.GaussNoise(p=1.0),                         # image only — its own kwarg vocabulary
    Brighten(strength=0.2),                      # native type-dispatched op
])(record)
# image + mask + bboxes flipped together (one joint draw); record["class"] untouched.
```

The same holds in YAML — a bare library transform is an ordinary `!class:` node in an `ops:` list
(the engine flows deferred markers at route entry):

```yaml
ops:
  - !class:albumentations.HorizontalFlip
    p: 0.5
  - !class:albumentations.GaussNoise
    p: 1.0
```

See [augmentation.md](augmentation.md) for the full key-vocabulary / bbox / seeding recipes.
Runnable end-to-end: [`examples/record_pipeline.py`](../examples/record_pipeline.py).

### `Pipeline` — the sequential composer

`Pipeline(transforms=[...])` (`sampleflux.transform`, `@configurable(category="op",
group="compose")`) wraps an ordered op list so it appears as one named block in a config and one
node on a visual canvas: zero-arg/lazy (config-deferred markers flow on first call), entries applied
through `_apply_op` (so bare library transforms nest exactly as in a bare ops list), `None`
propagation (a filter-drop stops the chain), and `close()` propagation to inner ops that own
resources.

## Extending it

### A custom op from a plain function

```python
from sampleflux import as_transform, Image
brighten = as_transform(lambda d: d + 0.1, handles=(Image,), field="image")
```

### A custom item type + a kernel for an existing op — no core edit

```python
from dataclasses import dataclass, field
from sampleflux import register_item
from mypkg.transforms import MyGeoTransform   # any Transform subclass

@register_item
@dataclass
class Keypoints:
    data: list = field(default_factory=list)   # a `data` field = the payload slot

@MyGeoTransform.kernel(Keypoints)
def _(value, params):
    return move_points(value, params)
```

Dispatch is MRO-aware: a kernel registered for a base item type also serves its subclasses, and a
subclass transform inherits its base's kernels until it overrides them.

### Domain items live in the domain package

The same mechanism, applied across packages: because sampleflux is modality-neutral, a signal-domain
package defines its own items (a signal, a spectrogram) and its own type-changing ops, registers
them with `register_item`, and they become first-class record values — dispatchable, collatable,
storable — with no core edit.

### A new library family

Supporting a new external transform library is NOT an adapter class — it is one new branch in
`core._apply_op` (an MRO module-name matcher plus the library's native calling convention), so every
engine route and composing op picks it up at once.

## Engines — Flux and FlowGraph carry the record

Every carrier is a plain record dict, and every route applies ops through `_apply_op` — sequential,
spawn-parallel, streamed, and random-access (`__getitem__`) alike, in `Flux` and in `FlowGraph`.
Composing ops (`Pipeline`, `RandomApply`, `Enable`, `Parallel`, `ConfigureOp`, the context ops
`Apply`/`Capture`) route their inner ops through the same chokepoint, so a bare library transform
nests anywhere a native op does.

```python
Flux(source=my_source, ops=[A.GaussNoise(p=1.0), Brighten()]).to_sink(HDF5Sink(path="out.h5"))
```

`Flux.map(func, key=None)` lifts a plain function over one record entry (`key=None` hands it the
whole dict — internally a `WrappedOp`, which stores the callable as its importable path so it
pickles across `spawn` workers); `Flux.project(keys)` yields partial records restricted to the
requested keys (see [projection.md](projection.md)).

### Graph fan-in (`merge_from`) and entry binds (`step[key]`)

In a `flow:` document, the fan-in is **`merge_from`** — the UNION of the named steps' record
entries, in slot order, last-write-wins on a key collision. The idiom for a derived-entry branch:
produce, `SelectFields` the new key(s), merge:

```yaml
flow:
  start:     {}
  masked:    {op: !class:sampleflux.ops.numpy.Threshold(low_level=0.5), from: start}
  mask_only: {op: !class:sampleflux.ops.structure.SelectFields(keys: [mask]), from: masked}
  boosted:   {op: !class:mypkg.Boost(), from: start}
  out:       {from: boosted, merge_from: [mask_only]}
```

`bind:` references have three shapes: a bare `step` binds the step's WHOLE result record,
`step[key]` binds the named ENTRY of that step's record, and `step.attr` binds the step op's live
`@output`. Lowering (`to_ops`) compiles `merge_from` to the `MergeFields` context op and entry-binds
to `Apply(key=...)`; lifting (`from_ops`) round-trips both. See [graph.md](graph.md).

## Storage — the record key-group layout

All four backends (`HDF5Sink`↔`HDF5Source`, `ZarrGroupSink`↔`ZarrGroupSource`,
`ZarrBatchSink`↔`ZarrBatchSource`, `DirectorySink`↔`DirectorySource`) write a record in ONE logical
schema: per record, one group per KEY carrying the value's registered type name (`__item_type__`),
the payload as a `data` dataset, and its attrs (scalars natively — queryable; arrays as sub-datasets
under `attrs/`; structured values JSON-tagged so tuples survive). A plain (non-item) value rides the
`"plain"` type tag — an array payload as `data`, a scalar under the `value` attr. Key order is
preserved in `__field_order__`; the store is stamped `sampleflux_format = "typedrecord-v1"`.

Backends never inspect item internals — everything serializes through the item codec
(`sampleflux/io.py`: `encode_item` / `decode_item` / `encode_record` / `decode_record`), so an
externally-registered item type round-trips with zero storage edits;
`register_io(MyItem, encode=..., decode=...)` overrides the default structural codec when needed.
Decoding requires the item type to be registered (imported) in the reading process — the same
contract as Confluid's `!class:`.

```python
sink = HDF5Sink(path="out.h5", overwrite=True)
with sink:
    for record in flux:
        sink.write(record)
back = list(HDF5Source(path="out.h5"))   # exact records: keys, types, order, tuple attrs
```

`ZarrBatchSink` (the uniform single-array sink) appends the FIRST record entry's payload per row and
stores a one-time item template — per-record attr variation needs `ZarrGroupSink`.

**No backward compatibility:** a store stamped with the pre-record `typedsample-v1` tag (or carrying
no tag) raises a `ValueError` telling you to re-generate it with a current sink
(`storage/base.py::require_record_format`) — there is no legacy read path.

### Querying record stores without loading arrays

The metadata scans yield the nested `{key: {attr: value}}` shape, and a `where` expression
addresses it as `<key>.<attr>` (a plain scalar entry appears under its `value` attr):

```python
fast = MetadataFilterSource(source=HDF5Source(path="out.h5"), where="signal.samplerate > 1e6")
```

Array-valued attrs appear as shape/dtype stubs (presence/shape testable, never loaded). A key named
like a Python keyword (e.g. `class`) can't be addressed in an expression — use the programmatic
`predicate` or a non-keyword key name. Live records expose the same nested shape via
`sampleflux.storage.query.record_metadata(record)`. See [storage.md](storage.md).

## Batching — `collate_records`

`collate_records` (the collate registry's `"record"` default) turns N record dicts into ONE batched
record: per key, typed payloads stack (torch → stacked tensor, numpy → stacked array, else a list)
and each declared item attr becomes a LIST of per-record values, decoded back into one batched item
of the same type; a plain value batches as the plain list. Batches must be key-homogeneous — a
mismatch raises. See [kinds.md](kinds.md).

## What is NOT here yet (follow-ups)

A torch-`Tensor`-subclass item base (torch payloads currently ride in wrapper items or as plain
values) and confluid-native item-type discovery. See the root `TASKS.md`.
