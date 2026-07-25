# Ops, batching & expanding ops (`sampleflux.transform` / `sampleflux.collate`)

## What an op processes — dispatch on value type

A **sample** is a plain record dict of typed values (`Image`, `Mask`, `Regions`, `Label`, … — see [record-model.md](record-model.md)). A native op is a `Transform`: it declares which value TYPES it handles and registers a per-type **kernel**; it samples its parameters ONCE per record (`get_params`), then applies the matching kernel to every value whose type it handles, passing untouched values through:

```python
from sampleflux import Record, Transform, Image

class Recenter(Transform):
    handles = (Image,)                       # which value types this op touches

    def get_params(self, record: Record) -> dict:
        return {"mean": 0.5}                 # sampled ONCE per record, shared across values

@Recenter.kernel(Image)                       # per-type behaviour
def _(value, params):
    return value - params["mean"]
```

Because the parameters are sampled once and shared, an op that handles several types moves those values **consistently** — one drawn decision applies to every handled value in the record. Dispatch is MRO-aware: a kernel registered for a base item type also serves its subclasses, and a subclass transform inherits its base's kernels until it overrides them.

Two smaller shapes round it out:

- **A plain function** becomes an op via `as_transform(fn, handles=(Image,), field="image")` — `field=` pins the op to one named key (still type-gated).
- **A type-changing op** — read one key, write a differently-typed item (`Threshold`: array → `Mask`, `ConvertToImage`: array → `Image`, `ConnectedComponents`: `Mask` → `Regions`) — subclasses `Transform` and overrides `__call__` instead of registering a same-type kernel.

Bare library transforms (torchvision `transforms.v2` walking the dict natively, albumentations dispatching by keyword name) drop straight into any ops list **as-is** — the engine's op-family dispatch invokes each one the way its own library expects. See [record-model.md](record-model.md#mixing-libraries--as-is-no-adapters) and [augmentation.md](augmentation.md).

```python
import albumentations as A
from sampleflux import Pipeline

out = Pipeline([
    A.HorizontalFlip(p=1.0),                 # image + mask + bboxes together (one library draw)
    A.GaussNoise(p=1.0),                     # image only — its own kwarg vocabulary
    Recenter(),                              # native op — same list
])(record)
# record["class"] (a Label) is untouched: no kernel handles it, no library key names it.
```

## Batching — `collate_records` & the collate registry (`sampleflux.collate`)

Ops are per-record; batching is a separate stage. **`collate_records`** (the registry's `"record"` default) stacks N record dicts into ONE batched record: per key, typed payloads stack (torch → stacked tensor, numpy → stacked array, else a list) and each item's declared attrs become per-record lists, decoded back into one batched item of the same type; plain values batch as plain lists. Batches must carry the same keys — a mismatch raises.

```python
from sampleflux import collate_records
from torch.utils.data import DataLoader

batch = collate_records(list(flux))          # ONE batched record: payloads stacked per key
loader = DataLoader(flux, collate_fn=collate_records)
```

Collation is a pluggable registry keyed by name, so a task can register its own convention additively:

```python
from sampleflux import register_collate, get_collate

@register_collate("yolo")                    # task aliases are additive
def yolo_collate(items): ...
loader = DataLoader(flux, collate_fn=get_collate("yolo"))
```

The string keys primarily target the MCP tool surface (JSON-serializable, enumerable collate selection) — in Python, passing the function directly stays the normal path. The full rationale is recorded in [architecture.md](architecture.md#batching-is-two-stage-collation-is-a-pluggable-registry-samplefluxcollate-2026-07-17-updated-2026-07-25).

## 1→N expanding ops (iterable-only pipelines)

An op may return **several** carriers — a windowing op splitting one capture into N windows marks itself with `EXPANDS = True` and returns an iterable of records:

```python
from typing import Iterator
from confluid import configurable
from sampleflux import Record
from sampleflux.items import item_data, with_data

@configurable(category="op")
class SlidingWindow:
    EXPANDS = True                                     # the explicit 1→N marker

    def __call__(self, record: Record) -> Iterator[Record]:
        item = record["signal"]
        for w in sliding_windows(item_data(item), self.size, self.stride):
            yield {**record, "signal": with_data(item, w)}
```

Expansion is flattened in every iteration route — sequential, spawn-parallel, and streamed — depth-first, so sibling order matches the nested-loop intuition. Each child continues through the remaining ops with its own (shallow-copied) Context; a child filtered to `None` just drops.

A pipeline containing an expanding op is **ITERABLE-ONLY**: `len(flux)` / `flux[i]` raise a clear `TypeError` (the expanded length is unknowable up front). Iterate it, wrap it in a torch `IterableDataset`, window at the source for random access, or materialize with `list(flux)`. `FlowGraph` steps are strictly 1→1 (a named step has one result) — expanding pipelines belong to the `Flux` engine.
