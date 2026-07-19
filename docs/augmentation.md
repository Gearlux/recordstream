# Augmentation — well-known libraries as SampleFlux ops

SampleFlux does not reimplement augmentations. Two adapter ops wrap the established
libraries — and a **generated op family** turns every individual library transform into
its own first-class op:

| Surface | What it is | Example |
|---|---|---|
| `sampleflux.ops.albumentations.AlbumentationsOp` | Adapter running one/many [albumentations](https://albumentations.ai) transforms | `AlbumentationsOp(transforms=[...], target="mask", seed=0)` |
| `sampleflux.ops.torchvision.TorchvisionTransformOp` | Adapter running one/many torchvision `transforms.v2` transforms | `TorchvisionTransformOp(transforms=[...], target="mask")` |
| `sampleflux.ops.albumentations_transforms` | **Auto-generated**: one `Alb<Name>` op per albumentations transform (~115) | `AlbHorizontalFlip(p=0.5, target="mask")` |
| `sampleflux.ops.torchvision_transforms` | **Auto-generated**: one `Tv<Name>` op per v2 transform (~55) | `TvRandomHorizontalFlip(p=0.5, target="mask")` |

All are ordinary sample-scoped ops (`__call__(sample)`): they chain in a `Flux` ops list,
inside `TransformChain` / `RandomApply` / `Enable`, in Confluid YAML, and as individual
nodes on a visual canvas (palette groups `augment`, `augment/albumentations`,
`augment/torchvision`). One library draw applies jointly to `sample.input` and — per the
`target` mode — its mask / boxes; metadata passes through untouched.

Torchvision requires the `vision` extra: `pip install "sampleflux[vision]"`
(albumentations is a core dependency; without torchvision the `Tv*` family is simply
empty and everything else works).

## Target modes

The `target` knob is a closed `Literal["none", "mask", "boxes"]` on every op above:

- `"none"` (default) — input-only augmentation (color jitter, noise, blur); the sample's
  target passes through untouched.
- `"mask"` — `sample.target` is a segmentation mask (2-D array or PIL `L` image); image
  and mask receive the SAME spatial transform.
- `"boxes"` — `sample.target` is the torchvision detection dict
  `{"boxes": [N,4] xyxy-pixel, "labels": [N]}` — exactly what `CocoToTorchVisionDetectionOp`
  and `MasksToDetectionBoxesOp` emit — and boxes move with the image. The required
  albumentations `bbox_params` are added automatically when the op builds the Compose;
  only a prebuilt `A.Compose` must carry its own.

```python
import albumentations as A
from sampleflux import Flux
from sampleflux.ops.albumentations import AlbumentationsOp
from sampleflux.ops.albumentations_transforms import AlbRandomBrightnessContrast

flux = Flux(source=samples, ops=[
    AlbumentationsOp(  # several transforms, one op
        transforms=[A.HorizontalFlip(p=0.5), A.Affine(translate_percent=0.1, p=1.0)],
        target="mask", seed=0,
    ),
    AlbRandomBrightnessContrast(p=0.5),  # or one generated op per transform
])
```

## YAML — Confluid-native, both directions

Transforms are ordinary nested `!class:` nodes (dotted paths or registered short names) —
no library-specific serialization formats. `confluid.dump` round-trips both forms.

```yaml
# Adapter with a transforms list (dotted library paths):
- !class:sampleflux.ops.albumentations.AlbumentationsOp
  target: mask
  seed: 0
  transforms:
    - !class:albumentations.HorizontalFlip
      p: 0.5
    - !class:albumentations.Affine
      translate_percent: 0.1

# Generated per-transform ops (registered short names):
- !class:AlbHorizontalFlip
  p: 0.5
  target: mask
- !class:TvRandomHorizontalFlip
  p: 0.5
  target: mask
```

## The generated op families

`sampleflux.ops._augment_bridge` walks each library's public transform classes at import
time and generates one op per transform (the waivefront-helios auto-bridge pattern): a
subclass of the adapter whose constructor mirrors the transform's own parameters (plus
`target` / `seed`), with a synthesized signature and `Args:` docstring so form-specs,
MCP schemas, and canvas widgets see the real parameters.

- The `Alb` / `Tv` name prefixes are MANDATORY: the confluid registry is flat and
  name-keyed, and the two libraries share many bare names (`ColorJitter`, `Normalize`,
  `Resize`, …).
- Zero-arg construction always works; a transform's required parameter (e.g.
  `AlbRandomCrop.height`) surfaces lazily as the library's own missing-argument error on
  first call.
- Composition/container transforms (`Compose`, `OneOf`, v2 `RandomApply`, …) are NOT
  generated — chaining ops is native SampleFlux (`ops:` lists, `TransformChain`,
  `RandomApply`).
- A generated op wired into an adapter's `transforms` list unwraps to its inner library
  transform (`raw_transform`), so canvas graphs can feed transform nodes into one
  Compose-style adapter node too.

## Layout contract (the main footgun)

The two libraries disagree about layout, and the ops keep each library's native
convention instead of hiding it:

- **albumentations** (`AlbumentationsOp`, `Alb*`) consumes numpy **HWC** (PIL converts on
  entry) and emits numpy HWC — put it BEFORE `ToTensorOp` in the chain.
- **torchvision** (`TorchvisionTransformOp`, `Tv*`) emits **CHW torch tensors** (numpy
  HWC converts on entry, PIL passes through as PIL) — no `ToTensorOp` needed after it.

Don't chain one library's output straight into the other without accounting for this.

## Randomness & seeding

All augmentation ops carry `random=True` (the confluid stochastic mark). Stochasticity
lives where each library puts it:

- albumentations: the `seed` knob (maps onto `A.Compose(seed=N)`); a prebuilt
  `A.Compose` carries its own seed instead.
- torchvision v2: the global torch RNG — `torch.manual_seed(N)`.
- per-sample gating: wrap in `RandomApply(op=..., probability=..., random_state=N)`
  (each albumentations transform also carries its own `p`).

## Examples

- [`examples/augmentation_ops.py`](../examples/augmentation_ops.py) — the tour: all
  three target modes, cross-library parity, boxes mirroring, target-side encoding, the
  generated op families, gated composition, and the Confluid-native YAML round-trip.
- [`examples/augmentation_training.py`](../examples/augmentation_training.py) — end to
  end: synthetic images+masks → joint geometric + gated photometric augmentation →
  `DataLoader` (registry collate) → a tiny CNN trained for 3 epochs with improving loss.
