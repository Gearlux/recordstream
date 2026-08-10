# Ops, batching & expanding ops (`recordstream.transform` / `recordstream.collate`)

## What an op processes — dispatch on value type

A **record** is a plain dict of typed values (`Image`, `Mask`, `Boxes`, `Label`, … — see [record-model.md](record-model.md)). A native op is a `Transform`: it declares which value TYPES it handles and registers a per-type **kernel**; it samples its parameters ONCE per record (`get_params`), then applies the matching kernel to every value whose type it handles, passing untouched values through:

```python
from recordstream import Record, Transform, Image

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
- **A type-changing op** — read one key, write a differently-typed item (`Threshold`: array → `Mask`, `ConvertToImage`: array → `Image`, `ConnectedComponents`: `Mask` → `Boxes`) — subclasses `Transform` and overrides `__call__` instead of registering a same-type kernel.

Bare library transforms (torchvision `transforms.v2` walking the dict natively, albumentations dispatching by keyword name) drop straight into any ops list **as-is** — the engine's op-family dispatch invokes each one the way its own library expects. See [record-model.md](record-model.md#mixing-libraries--as-is-no-adapters) and [augmentation.md](augmentation.md).

```python
import albumentations as A
from recordstream import Pipeline

out = Pipeline([
    A.HorizontalFlip(p=1.0),                 # image + mask + bboxes together (one library draw)
    A.GaussNoise(p=1.0),                     # image only — its own kwarg vocabulary
    Recenter(),                              # native op — same list
])(record)
# record["class"] (a Label) is untouched: no kernel handles it, no library key names it.
```

## Batching — `collate_records` & the collate registry (`recordstream.collate`)

Ops are per-record; batching is a separate stage. **`collate_records`** (the registry's `"record"` default) stacks N record dicts into ONE batched record: per key, typed payloads stack (torch → stacked tensor, numpy → stacked array, else a list) and each item's declared attrs become per-record lists, decoded back into one batched item of the same type; plain values batch as plain lists. Batches must carry the same keys — a mismatch raises.

```python
from recordstream import collate_records
from torch.utils.data import DataLoader

batch = collate_records(list(stream))          # ONE batched record: payloads stacked per key
loader = DataLoader(stream, collate_fn=collate_records)
```

### The batch SHAPE is a choice — `"record"` vs `"list"`

Whether a column is STACKED is a requirement of the **model**, not a property of the data: a
torchvision detector takes `List[Tensor]` (its images differ in size), a classifier takes one
`[N, C, H, W]` tensor. Two collates ship, differing in exactly that:

```python
from recordstream import collate_list, collate_records

collate_records(records)["image"]   # Image (2, 3, 8, 8) — stacked
collate_list(records)["image"]      # [Image (3, 8, 8), Image (3, 12, 12)] — per record, items kept
```

Declare it where the batch is built — `DataLoader(stream, collate_fn=collate_list)` on torch,
`RecordSequence(source, collate="list")` on Keras. Left implicit, the shape is decided by
accident: under the default collate a column stacks if it holds array *items* and stays a list if
it holds *plain* values, so whether an op like `ToTensor` ran ends up choosing for you.

Two things make the choice free:

* **every read-back helper accepts both shapes** — `batch_values` unwraps a list-of-items
  element-wise, `batch_boxes` reads a batched `Boxes` or a list of them, `batch_metadata`
  transposes either — so a consumer never branches on which collate ran;
* **a stack failure explains itself**, naming the key, the differing shapes and the `"list"` way
  out, rather than surfacing numpy's bare *"all input arrays must have the same shape"*.

The registry is additive, so a task can register its own convention too:

```python
from recordstream import get_collate, register_collate

@register_collate("yolo")                    # task aliases are additive
def yolo_collate(items): ...
loader = DataLoader(stream, collate_fn=get_collate("yolo"))
```

The string keys primarily target the MCP tool surface (JSON-serializable, enumerable collate selection) — in Python, passing the function directly stays the normal path. The full rationale is recorded in [architecture.md](architecture.md#2-batching-is-two-stage-collation-is-a-pluggable-registry-recordstreamcollate-2026-07-17).

### Reading a batch back (`recordstream.batch`)

The inverse of `collate_records`, shipped alongside it so a model boundary never re-derives the convention:

```python
from recordstream import batch_values, batch_tensor, batch_metadata, multi_hot

batch_values(batch, "class")                             # past the wrapper item: a Label -> its .value list
batch_tensor(batch, "image", device=model.device)        # ONE torch tensor, stacked + moved
batch_tensor(batch, "class", dev, dtype=torch.int64)     # ...with the dtype your loss requires
multi_hot(batch, "class", num_classes)                   # a MultiLabel column as an [N, C] numpy matrix
batch_metadata(batch, exclude=("image", "class"))        # the remaining columns transposed into N dicts
per_record_predictions(model_output)                     # a BATCHED model output sliced into one entry per record
```

Only `batch_tensor` is torch; the rest return plain values or numpy, so a non-torch backend reuses them and converts in one line. `dtype` is a parameter, not an opinion — the same knob as `device`. What stays task-side is only WHICH call a trainer makes.

### Keras: `RecordSequence` — the batching half the framework leaves to you

A `DataLoader` needs one thing from recordstream (`collate_fn=collate_records`) and does the rest itself: row order, batch slicing, the short final batch, the per-epoch reshuffle. Keras 3 has no `DataLoader` — `keras.utils.PyDataset.__getitem__` must return a whole **batch** — so that loop is `RecordSequence`, in `recordstream.keras`:

```python
from recordstream.keras import RecordSequence

def to_xy(batch):                                            # your collate_fn equivalent
    return batch_values(batch, "image"), batch_values(batch, "class")

train = RecordSequence(stream, batch_size=32, shuffle=True, transform=to_xy)
model.fit(train, epochs=3)                                   # reshuffles between epochs itself
```

`PyDataset`'s prefetch knobs are declared parameters, at Keras's own defaults — pass `workers=4` to overlap a slow record walk (decode, resize, remote read) with the training step, `max_queue_size=` to cap how many prefetched batches may wait, and `use_multiprocessing=True` only when the per-record work is GIL-bound (each process re-pickles the sequence and its source).

`transform` is the whole task-facing surface: it maps one collated record to what the model consumes, so the shape decision stays in your code exactly as it does with a `DataLoader`. Omit it and `__getitem__` hands over the batched record itself — which is also what `seq.batches()` yields, the pairing half of prediction (a model emits `[N, ...]` while a [`PredictionsSink`](predictions.md) writes per record, so you need the batch its output came from to read that batch's [`batch_metadata`](#reading-a-batch-back-recordstreambatch)).

Needs the extra — `pip install "recordstream[keras]"`. Importing `recordstream.keras` is also what sets `KERAS_BACKEND` (Keras reads it at import time and would otherwise default to TensorFlow, which this extra does not install), so it must be the first keras-touching import in a process; never `import keras` ahead of it. Why the adapter lives here rather than in a training project is recorded in [architecture.md](architecture.md#10-the-frameworks-batching-half-lives-beside-the-collate-recordstreamkeras-2026-07-30).


## 1→N expanding ops (iterable-only pipelines)

An op may return **several** carriers — a windowing op splitting one capture into N windows marks itself with `EXPANDS = True` and returns an iterable of records:

```python
from typing import Iterator
from confluid import configurable
from recordstream import Record
from recordstream.items import item_data, with_data

@configurable(category="op")
class SlidingWindow:
    EXPANDS = True                                     # the explicit 1→N marker

    def __call__(self, record: Record) -> Iterator[Record]:
        item = record["signal"]
        for w in sliding_windows(item_data(item), self.size, self.stride):
            yield {**record, "signal": with_data(item, w)}
```

Expansion is flattened in every iteration route — sequential, spawn-parallel, and streamed — depth-first, so sibling order matches the nested-loop intuition. Each child continues through the remaining ops with its own (shallow-copied) Context; a child filtered to `None` just drops.

A pipeline containing an expanding op is **ITERABLE-ONLY**: `len(stream)` / `stream[i]` raise a clear `TypeError` (the expanded length is unknowable up front). Iterate it, wrap it in a torch `IterableDataset`, window at the source for random access, or materialize with `list(stream)`. `FlowGraph` steps are strictly 1→1 (a named step has one result) — expanding pipelines belong to the `Stream` engine.

## The training-side data helpers (`prepare_record_dataset`, `recordstream.loaders`)

Two helpers every training runnable composes, written once here:

```python
from recordstream import prepare_record_dataset
from recordstream.loaders import loader_slots            # torch-only module, not in the package root

dataset = prepare_record_dataset(source)   # ensure_record_dataset + ensure_materialized; None passes through
slots = loader_slots(batch_size=32, num_workers=0)       # deferred DataLoader triple (train shuffled)
loader = flow(slots.train, dataset=dataset)              # the dataset arrives at run time
```

`prepare_record_dataset` normalizes a wired source's TYPE and its STATE in the calling process
(the fork-safety pair — see the `ensure_materialized` docs). `loader_slots` returns the
train/val/test `LazyClass(DataLoader, ...)` markers as a named tuple; further `DataLoader`
kwargs pass through to all three, and a config can still replace any individual loader slot
wholesale. The module imports torch, so it is deliberately NOT re-exported from the package
root — `import recordstream` stays framework-free.
