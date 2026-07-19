"""Augmentation adapters: well-known libraries as SampleFlux ops, for input AND target.

Demonstrates the library-augmentation surface:
1. input-only augmentation — ``AlbumentationsOp`` flips the image, target untouched;
2. joint input+target — ``target="mask"`` flips image AND segmentation mask consistently
   (one library draw, metadata preserved);
3. the same joint flip through torchvision ``transforms.v2`` — cross-library parity;
4. detection boxes — ``MasksToDetectionBoxesOp`` derives xyxy boxes from the mask, then
   both adapters transform image AND boxes jointly (``target="boxes"``, bbox_params
   added automatically);
5. target-side augmentation/encoding — ``MetadataToTargetOp`` + ``EncodeTargetOp`` turn a
   raw metadata label into the supervised class id;
6. the GENERATED per-transform ops — every library transform is its own op
   (``AlbHorizontalFlip``, ``TvRandomHorizontalFlip``, …) chaining like any other op;
7. stochastic composition — ``TransformChain(RandomApply(AlbRandomBrightnessContrast))``
   gated per sample;
8. Confluid-NATIVE YAML — nested ``!class:albumentations.HorizontalFlip`` nodes and the
   registered short names (``!class:AlbHorizontalFlip``), with dump→load parity.

Standalone, zero-arg, exit 0 (CI runs every ``examples/*.py``).
"""

import albumentations as A
import confluid  # type: ignore[import-not-found]
import numpy as np
import torch
from torchvision.transforms import v2

from sampleflux import Flux, Sample
from sampleflux.ops.albumentations import AlbumentationsOp
from sampleflux.ops.albumentations_transforms import AlbHorizontalFlip, AlbRandomBrightnessContrast
from sampleflux.ops.random_apply import RandomApply
from sampleflux.ops.target import EncodeTargetOp, MasksToDetectionBoxesOp, MetadataToTargetOp
from sampleflux.ops.torch import ToTensorOp
from sampleflux.ops.torchvision import TorchvisionTransformOp
from sampleflux.ops.torchvision_transforms import TvRandomHorizontalFlip
from sampleflux.ops.transform_chain import TransformChain


def make_sample() -> Sample:
    """A deterministic 48x48 RGB gradient with a bright square and its binary mask."""
    height = width = 48
    image = np.linspace(0, 200, height * width * 3, dtype=np.float64).reshape(height, width, 3)
    image = image.astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    image[8:20, 4:16] = 255  # bright square, deliberately OFF-center so a flip moves it
    mask[8:20, 4:16] = 1
    return Sample(image, mask, {"label": "square", "idx": 0})


def main() -> None:
    sample = make_sample()
    image, mask = sample.input, sample.target

    # 1. Input-only augmentation: the target and metadata pass through untouched.
    out = list(Flux(source=[sample], ops=[AlbumentationsOp(A.HorizontalFlip(p=1.0))]))[0]
    assert np.array_equal(out.input, image[:, ::-1])
    assert np.array_equal(out.target, mask)
    assert out.meta == sample.meta
    print("1. albumentations input-only: image flipped, mask + metadata untouched")

    # 2. Joint input+target: ONE random draw moves image AND mask together; metadata
    #    survives verbatim.
    out_alb = list(Flux(source=[sample], ops=[AlbumentationsOp(A.HorizontalFlip(p=1.0), target="mask")]))[0]
    assert np.array_equal(out_alb.input, image[:, ::-1])
    assert np.array_equal(out_alb.target, mask[:, ::-1])
    assert out_alb.meta == sample.meta
    print("2. albumentations target='mask': image AND mask flipped consistently")

    # 3. Same augmentation via torchvision transforms.v2 — identical pixels, different
    #    layout contract (torchvision emits CHW tensors; albumentations stays HWC numpy).
    tv_op = TorchvisionTransformOp(v2.RandomHorizontalFlip(p=1.0), target="mask")
    out_tv = list(Flux(source=[sample], ops=[tv_op]))[0]
    assert np.array_equal(out_tv.input.permute(1, 2, 0).numpy(), out_alb.input)
    assert np.array_equal(out_tv.target.numpy(), out_alb.target)
    print("3. torchvision target='mask': cross-library parity (CHW tensor out)")

    # 4. Detection boxes: derive {"boxes" xyxy, "labels"} from the mask, then flip image
    #    AND boxes jointly. The adapter adds the required bbox_params automatically.
    det = MasksToDetectionBoxesOp()(sample)
    (x0, y0, x1, y1) = det.target["boxes"][0].tolist()
    mirrored = [det.input.shape[1] - x1, y0, det.input.shape[1] - x0, y1]
    out_tvb = list(Flux(source=[det], ops=[TorchvisionTransformOp(v2.RandomHorizontalFlip(p=1.0), target="boxes")]))[0]
    assert out_tvb.target["boxes"][0].tolist() == mirrored
    out_albb = list(Flux(source=[det], ops=[AlbumentationsOp(transforms=[A.HorizontalFlip(p=1.0)], target="boxes")]))[0]
    assert np.allclose(out_albb.target["boxes"][0].tolist(), mirrored, atol=1e-4)
    print(f"4. target='boxes': {[x0, y0, x1, y1]} -> {mirrored} (both libraries agree)")

    # 5. Target-side augmentation/encoding: raw label from metadata -> supervised class id.
    encode = [MetadataToTargetOp(key="label"), EncodeTargetOp(mapping={"square": 0, "disc": 1})]
    out_enc = list(Flux(source=[sample], ops=encode))[0]
    assert out_enc.target == 0
    print("5. MetadataToTargetOp + EncodeTargetOp: metadata['label'] -> class id 0")

    # 6. Generated per-transform ops: every library transform is its OWN op — no wrapper
    #    boilerplate, the transform's params are the op's params, and it chains anywhere.
    out_gen = list(Flux(source=[sample], ops=[AlbHorizontalFlip(p=1.0, target="mask")]))[0]
    assert np.array_equal(out_gen.target, mask[:, ::-1])
    out_gen_tv = list(Flux(source=[sample], ops=[TvRandomHorizontalFlip(p=1.0, target="mask")]))[0]
    assert np.array_equal(out_gen_tv.target.numpy(), mask[:, ::-1])
    print("6. generated ops: AlbHorizontalFlip / TvRandomHorizontalFlip chain like any op")

    # 7. Stochastic composition: gate a generated photometric op per sample, tensorize.
    chain = TransformChain(
        ops=[
            RandomApply(op=AlbHorizontalFlip(p=1.0, target="mask"), probability=0.5, random_state=0),
            AlbRandomBrightnessContrast(p=1.0, seed=0),
            ToTensorOp(),
        ]
    )
    source = [Sample(image.copy(), mask.copy(), {"idx": i}) for i in range(8)]
    outputs = list(Flux(source=source, ops=[chain]))
    flipped = sum(1 for s in outputs if np.array_equal(s.target, mask[:, ::-1]))
    assert all(isinstance(s.input, torch.Tensor) and s.input.shape == (3, 48, 48) for s in outputs)
    assert 0 < flipped < len(outputs)  # the gate fired for some samples, not all
    print(f"7. TransformChain(RandomApply(flip), brightness, ToTensorOp): {flipped}/{len(outputs)} flipped")

    # 8. Confluid-NATIVE YAML: transforms are nested !class: nodes (or registered short
    #    names) — dump and load round-trip with identical behavior.
    yaml_text = (
        "!class:sampleflux.ops.albumentations.AlbumentationsOp\n"
        "target: mask\n"
        "seed: 0\n"
        "transforms:\n"
        "  - !class:albumentations.HorizontalFlip\n"
        "    p: 1.0\n"
    )
    op = confluid.load(yaml_text)
    short = confluid.load("!class:AlbHorizontalFlip\np: 1.0\ntarget: mask\n")
    out_yaml = op(sample)
    out_short = short(sample)
    assert np.array_equal(out_yaml.target, out_short.target)
    reloaded = confluid.load(confluid.dump(op))
    assert np.array_equal(reloaded(sample).input, out_yaml.input)
    print("8. Confluid-native YAML (nested !class: + short names), dump->load parity:")
    print("   " + "\n   ".join(confluid.dump(op).strip().splitlines()))


if __name__ == "__main__":
    main()
