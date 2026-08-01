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
`MasksToDetectionBoxes`) produce a `Regions` item for the training boundary; the plain
`bboxes`/`labels` list keys are the augmentation-time form the library consumes.

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
