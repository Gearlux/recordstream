# Transforms, batching & expanding ops (`sampleflux.bag` / `sampleflux.collate`)

## What a transform processes — dispatch on item type

A **sample** is a named bag of typed items (`Image`, `Mask`, `Regions`, `Label`, … — see [typed-model.md](typed-model.md)). A transform declares which item TYPES it handles and registers a per-type **kernel**; it samples its parameters ONCE per sample, then applies the matching kernel to every field whose item type it handles, passing untouched fields through:

```python
from sampleflux import Transform, Image

class Recenter(Transform):
    handles = (Image,)                       # which item types this transform touches

    def params(self):                        # sampled ONCE per sample, shared across fields
        return {"mean": 0.5}

@Recenter.kernel(Image)                       # per-type behaviour
def _(item, params):
    return item - params["mean"]
```

Because the parameters are sampled once and shared, a transform that handles several types moves those fields **consistently** — one flip decision applies to `Image`, `Mask` and `Regions` together, the thing a flat `(input, target, metadata)` triple could not express. Dispatch is MRO-aware: a kernel registered for a base item type also serves its subclasses, and a subclass transform inherits its base's kernels until it overrides them.

Two smaller shapes round it out:

- **A plain function** becomes a transform via `as_transform(fn, handles=(Image,), only=["image"])` — `only=` narrows a transform to specific field keys.
- **A type-changing transform** — read one field, write a differently-typed item (`array → Image`, `Signal → Spectrogram`, `Mask → Regions`) — subclasses `Transform` and overrides `__call__` instead of registering a same-type kernel.

Bare library transforms (torchvision `transforms.v2` dispatching by type, albumentations by keyword name) drop straight into a `Pipeline` through registered adapters — each one hits only the field(s) it handles. See [typed-model.md](typed-model.md#mixing-libraries--one-pipeline-many-worlds).

```python
from sampleflux import Sample, Image, Mask, Regions, Label, Pipeline
from torchvision.transforms import v2
import albumentations as A

out = Pipeline([
    v2.RandomHorizontalFlip(p=1.0),          # Image + Mask + Regions together (one library draw)
    v2.Normalize(mean, std),                 # Image only — wrapped by a registered adapter
    A.GaussNoise(p=1.0),                     # Image only — wrapped by a registered adapter
])(sample)
# a Label field is untouched (no kernel handles it); roles are preserved.
```

## Batching — `typed_collate` & the collate registry (`sampleflux.collate`)

Transforms are per-sample; batching is a separate stage. **`typed_collate`** (auto-dispatched for `Sample` batches) stacks each field's payload and collects each item's per-sample attributes into a list, preserving roles — the ONE batch convention:

```python
from sampleflux import typed_collate
from torch.utils.data import DataLoader

batch = typed_collate(list(flux))            # a batched Sample: payloads stacked per field
loader = DataLoader(flux, collate_fn=typed_collate)
```

Collation is a pluggable registry keyed by name, so a task can register its own convention additively:

```python
from sampleflux import register_collate, get_collate

@register_collate("yolo")                    # task aliases are additive
def yolo_collate(items): ...
loader = DataLoader(flux, collate_fn=get_collate("yolo"))
```

The string keys primarily target the MCP tool surface (JSON-serializable, enumerable collate selection) — in Python, passing the function directly stays the normal path. The full rationale is recorded in [architecture.md](architecture.md#batching-is-two-stage-collation-is-a-pluggable-registry-samplefluxcollate-2026-07-17).

## 1→N expanding ops (iterable-only pipelines)

An op may return **several** carriers — a windowing op splitting one capture into N windows is just a generator-returning op:

```python
from typing import Iterator
from sampleflux import Sample, Transform, primary, with_data

@configurable
class SlidingWindowOp(Transform):
    def __call__(self, sample: Sample) -> Iterator[Sample]:
        key, item = primary(sample, "input")
        for w in sliding_windows(item, self.size, self.stride):
            yield sample.replace_field(key, with_data(item, w))
```

Expansion is detected from the return annotation (`Iterator[...]` / `Iterable[...]` / `List[...]`; or the explicit `EXPANDS = True` marker) and flattened in every iteration route — sequential, spawn-parallel, and streamed — depth-first, so sibling order matches the nested-loop intuition. Each child continues through the remaining ops with its own (shallow-copied) Context; a child filtered to `None` just drops.

A pipeline containing an expanding op is **ITERABLE-ONLY**: `len(flux)` / `flux[i]` raise a clear `TypeError` (the expanded length is unknowable up front). Iterate it, wrap it in a torch `IterableDataset`, window at the source for random access, or materialize with `list(flux)`. `FlowGraph` steps are strictly 1→1 (a named step has one result) — expanding pipelines belong to the `Flux` engine.
