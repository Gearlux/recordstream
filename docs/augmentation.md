# Augmentation — well-known libraries run AS-IS

RecordStream does not reimplement augmentations, and it does not wrap them either. A bare
[albumentations](https://albumentations.ai) transform or a bare torchvision `transforms.v2`
transform drops **as-is** into any ops list — `Stream(ops=[...])`, a `Pipeline`, a `flow:` step,
inside `RandomApply` / `Enable` — and the engine's op-family dispatch
(`recordstream.core.families._apply_op`) invokes it the way its own library expects. There are no adapter
classes and no generated per-transform op families.

```python
import albumentations as A
from torchvision.transforms import v2
from recordstream import Stream, Pipeline

stream = Stream(source=records, ops=[
    A.HorizontalFlip(p=0.5),          # bare albumentations
    A.GaussNoise(p=1.0),              # bare albumentations
    my_native_op,                     # native recordstream op — same list
])

Pipeline([v2.ToImage(), v2.RandomCrop(8)])(record)   # bare torchvision v2
```

Torchvision is optional (`pip install "recordstream[vision]"`); albumentations is a core
dependency. The family check is by MRO module name — neither library is imported until you
actually put one of its transforms in a pipeline.

## How each family is invoked

- **albumentations** dispatches by KWARG NAME: the op receives exactly its own target keys
  present in the record — `image` / `mask` / `masks` / `bboxes` / `keypoints` / `labels` — and
  nothing else, so extra record entries (scalars, domain items) never reach a library that would
  reject them. One call = **one joint draw** across those keys: image, mask and boxes move with
  the same decision. Array outputs are re-wrapped in the incoming value's item type, so an
  `Image` / `Mask` keeps its type and metadata through the library. A record with none of the
  known keys passes through untouched (logged at debug).
- **torchvision `transforms.v2`** natively walks dicts: the op is called on the record as-is,
  samples its parameters once, transforms tensor / tv_tensor / PIL leaves and passes everything
  else (labels, scalars) through.
- **everything else** is a native/wiring op `record -> Optional[Record]` (`None` drops the
  record).

## The key vocabulary — and routing into it

Key names carry meaning: albumentations sees only its own vocabulary, so a value augments only if
it rides one of those keys. If your pipeline produced the value under another name, route it with
`RenameField` (`recordstream.ops.structure`) before the library op:

```yaml
ops:
  - !class:recordstream.ops.structure.RenameField {src: spec_view, dst: image}
  - !class:albumentations.GaussNoise
    p: 1.0
```

## Boxes: use the library's own Compose

Box-carrying augmentation is albumentations' `Compose` job — drop a prebuilt `A.Compose` with its
own `bbox_params` into the ops list (the record supplies `bboxes` + `labels` under exactly those
keys):

```python
import albumentations as A

flip = A.Compose(
    [A.HorizontalFlip(p=1.0)],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)
out = Pipeline([flip])(record)      # image + mask + bboxes flipped together, one draw
```

Format handling (`pascal_voc` / `coco` / `yolo` / `albumentations`) is `BboxParams`' knob — the
engine adds nothing on top. The detection-target ops (`CocoToTorchVisionDetection` /
`MasksToDetectionBoxes`) produce a `Boxes` item for the training boundary; the plain
`bboxes`/`labels` list keys are the augmentation-time form the library consumes.

For a plain deterministic resize of the `Boxes` form there is `ResizeDetection`
(`recordstream.ops.target`) — the detection twin of the joint image+mask draw: it resizes the
image (PIL or uint8 array) to a fixed `(height, width)` AND scales the `Boxes` boxes by the
same factors in one coupled step, recording the new frame in `canvas`. Fixed-input-size
detectors need it; detectors that resize internally simply omit it. Run it BEFORE any float
conversion (e.g. before `ToTensor`):

```yaml
ops:
  - !class:recordstream.ops.target.CocoToTorchVisionDetection { bbox_format: xywh, label_offset: 1 }
  - !class:recordstream.ops.target.ResizeDetection { width: 256, height: 256 }
  - !class:recordstream.ops.torch.ToTensor { mode: RGB, normalize: true }
```

### Boxes carry the frame they are stated in

A box is only meaningful against a raster, so a `Boxes` records that raster in `canvas` —
`(H, W)` — and every op that makes or re-frames one fills it in: `CocoToTorchVisionDetection`
from the image the annotation describes, `MasksToDetectionBoxes` from the mask the boxes were
derived from, `ResizeDetection` from the size it resized to (including for an empty target, so a
negative example is not the one record whose frame is unknown). It stays `None` only when no
image is in the record to read.

That makes a desync *detectable*, which matters because it is otherwise silent. An **image-only**
resize moves pixels without moving boxes:

```yaml
ops:
  - !class:recordstream.ops.target.CocoToTorchVisionDetection { bbox_format: xywh }
  - !class:recordstream.ops.image.ConvertToImage { width: 256, height: 256 }   # ← boxes left behind
```

Every shape downstream stays valid — only the coordinates are wrong — so a model trains happily
against misplaced targets. `ConvertToImage` warns once per op when it resizes a record carrying a
`Boxes`, and a consumer that must be certain compares `canvas` against the image itself
(`recordstream.ops.image.image_frame` reads the `(H, W)` of either). Use `ResizeDetection`, which
moves both.

**A bare library transform has the same gap**, for the reason that makes the dispatch work: an
albumentations op receives exactly its own key vocabulary, and a `Boxes` is not in it. So a bare
`A.Resize` resizes the image and leaves the boxes; a bare `A.HorizontalFlip` mirrors the pixels
and leaves them — *without changing any shape at all*. The engine warns once per transform type
when a geometry-changing transform runs while a `Boxes` sat out the call, deciding "geometry-
changing" by the library's own `DualTransform` / `ImageOnlyTransform` split (so `Normalize` and
friends stay silent). Speak the library's vocabulary and it moves them for you, in the same draw:

```yaml
ops:
  - !class:recordstream.ops.structure.RenameField { src: my_boxes, dst: bboxes }
  - !class:albumentations.Compose
    transforms: [!class:albumentations.HorizontalFlip { p: 0.5 }]
    bbox_params: !class:albumentations.BboxParams { format: pascal_voc, label_fields: [labels] }
```

**torchvision v2 has it too, by the other route.** v2 walks the record natively but transforms
only its OWN `tv_tensors` types, and a `Boxes` is not one — so `v2.Resize` moves the pixels and
leaves the boxes, while the same transform over a `tv_tensors.BoundingBoxes` rescales them
correctly. The engine warns once per transform type here as well, using v2's geometric-transform
grouping so `ColorJitter` and `Normalize` stay silent. Carry boxes in v2's own type when you want
v2 to move them:

```python
from torchvision import tv_tensors

record["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w))
```

Either way, `ResizeDetection` remains the plain coupled resize over the `Boxes` form.

## YAML — bare library transforms are ordinary `!class:` nodes

No library-specific serialization format — a transform is a Confluid `!class:` node like any op,
in mapping form or call form. `Stream` flows deferred markers at route entry, and composing ops
(`Pipeline` / `Enable` / `RandomApply`) flow theirs lazily:

```yaml
ops:
  - !class:albumentations.HorizontalFlip
    p: 0.5
  - !class:albumentations.GaussNoise {p: 1.0}
  - !class:recordstream.ops.numpy.Threshold
    low_level: 0.5
```

## Layout contract (the main footgun)

The two libraries disagree about layout, and the engine keeps each library's native convention
instead of hiding it — **conversions are always explicit library transforms, never silent**:

- **albumentations** consumes and emits numpy **HWC** — run it while your values are still numpy
  arrays (an `Image`/`Mask` is an ndarray subclass, so it feeds straight in).
- **torchvision v2** wants **CHW tensors** — put the library's own `v2.ToImage()` (numpy HWC →
  CHW tv_tensor) in the list first, then any v2 transform; exactly like a plain torchvision
  pipeline.

Don't chain one library's output straight into the other without an explicit conversion step.

## Randomness & seeding

Stochasticity lives where each library puts it — the engine adds no seed plumbing:

- albumentations: `A.Compose(seed=N)` on a prebuilt Compose (individual transforms keep their own
  `p`).
- torchvision v2: the global torch RNG — `torch.manual_seed(N)`.
- per-record gating of any op (native or library): `RandomApply(op=..., probability=...,
  random_state=N)`.

## Multiprocessing: two fork hazards, both handled

A `DataLoader` worker is often a **forked** child (torch's default on Linux, and on macOS whenever
something in the process has set `fork` — `import fastai` does). A fork inherits memory but not
threads, so two things on this path would otherwise crash the child with a **SIGSEGV and no Python
traceback**, surfacing only as `DataLoader worker exited unexpectedly`:

| hazard | guard | who calls it |
|---|---|---|
| OpenCV's thread pool, inherited across the fork (albumentations runs on cv2) | `cv2.setNumThreads(0)` | the engine, automatically, the first time it invokes an albumentations op |
| a lazy source first built in the child — the download reaches `_scproxy`, which is not fork-safe | `ensure_materialized(source)` | **you**, in the parent, before building the loader |

The first needs nothing from you. The second does:

```python
from recordstream import ensure_materialized, ensure_record_dataset

dataset = ensure_materialized(ensure_record_dataset(train_set))   # reads ONE record, in the parent
loader = DataLoader(dataset, num_workers=4, collate_fn=collate_records)
```

They are **independent** — neither fixes the other. With the source warmed but the cv2 pool on, the
worker still dies; with the pool off but the source cold, it is still built in the child. Setting
`num_workers=0` avoids both by not forking at all, which is why the crash looks intermittent and
gets misread as a flaky test rather than an ordering bug.

Turning cv2's pool off costs nothing where it matters: inside a worker the *worker* is the
parallelism, so cv2's own threads oversubscribe rather than help. The engine does it at the point
of use, never at import — a process that never touches albumentations keeps its OpenCV settings.

## Other libraries — register an op family

albumentations and torchvision v2 are the built-in families, registered through the same OPEN
registry any package can use: `register_op_family(name, matcher, invoker)` teaches the engine a
new library's native calling convention (kornia, DALI, a fork extending albumentations, a
signal-processing library), and bare ops of that library then sit in ANY ops list — every engine
route and composing op, including spawn-parallel workers. Full example + rules:
[record-model.md → "A new library family"](record-model.md#a-new-library-family).

## Example

[`examples/record_pipeline.py`](../examples/record_pipeline.py) — the tour: a bare
`A.Compose` with `bbox_params` + `A.GaussNoise` + a native type-dispatched op in ONE `Pipeline`
(image/mask/bboxes moved jointly, types preserved), `field=` pinning, and torchvision v2 as-is
after an explicit `v2.ToImage()`.
