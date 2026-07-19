"""Train a tiny segmentation CNN on an augmented SampleFlux pipeline (end to end).

Demonstrates the full "augment to train" story:
1. a synthetic, network-free dataset — 64 RGB images with a bright square or disc and
   its binary segmentation mask (input AND target);
2. joint geometric augmentation — ``AlbumentationsOp(target="mask")`` flips/translates
   image AND mask with one library draw per sample;
3. gated photometric augmentation — ``RandomApply`` fires brightness/contrast on the
   input only, for half the samples;
4. tensorization — ``ToTensorOp`` for the image, a raw-callable ``.map(select="target")``
   for the mask (``WrappedOp`` under the hood);
5. ``Flux`` is a ``torch.utils.data.Dataset`` — it plugs straight into a ``DataLoader``
   with the registry collate (``get_collate("sample")``: stacked tensors + list-form
   batched metadata), and augmentations re-draw every epoch via random access;
6. a 3-epoch training loop of a tiny CNN (BCE on the mask) — losses must be finite and
   improve, proving gradients flow through the augmented pipeline.

Standalone, zero-arg, exit 0, seconds on CPU (CI runs every ``examples/*.py``).
"""

import math
from typing import List

import albumentations as A
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sampleflux import Flux, Sample
from sampleflux.collate import get_collate
from sampleflux.ops.albumentations import AlbumentationsOp
from sampleflux.ops.albumentations_transforms import AlbRandomBrightnessContrast
from sampleflux.ops.random_apply import RandomApply
from sampleflux.ops.torch import ToTensorOp

SIZE = 32  # image edge in pixels


def make_dataset(count: int = 64, seed: int = 0) -> List[Sample]:
    """Synthetic segmentation set: noisy background + one bright square or disc + its mask."""
    rng = np.random.default_rng(seed)
    samples = []
    for idx in range(count):
        image = rng.integers(0, 60, size=(SIZE, SIZE, 3), dtype=np.uint8)
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        cy, cx = rng.integers(8, SIZE - 8, size=2)
        r = int(rng.integers(3, 7))
        shape = "square" if idx % 2 == 0 else "disc"
        if shape == "square":
            region = np.zeros((SIZE, SIZE), dtype=bool)
            region[cy - r : cy + r, cx - r : cx + r] = True
        else:
            yy, xx = np.ogrid[:SIZE, :SIZE]
            region = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        image[region] = rng.integers(180, 255, size=3, dtype=np.uint8)
        mask[region] = 1
        samples.append(Sample(image, mask, {"idx": idx, "shape": shape}))
    return samples


def mask_to_float(mask: np.ndarray) -> torch.Tensor:
    """Binary HxW mask -> float32 (1, H, W) tensor, the shape BCEWithLogitsLoss expects."""
    return torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)


def build_pipeline(samples: List[Sample]) -> Flux:
    """Source -> joint geometric aug -> gated photometric aug -> tensors, all seeded."""
    geometric = AlbumentationsOp(  # image AND mask move together (one draw per sample)
        transforms=[A.HorizontalFlip(p=0.5), A.Affine(translate_percent=0.1, p=1.0)],
        target="mask",
        seed=0,
    )
    photometric = AlbRandomBrightnessContrast(p=1.0, seed=1)  # generated per-transform op
    flux = Flux(
        source=samples,
        ops=[
            geometric,
            RandomApply(op=photometric, probability=0.5, random_state=0),
            ToTensorOp(),  # image -> float CHW in [0, 1]
        ],
    )
    return flux.map(mask_to_float, select="target")  # raw-callable target map (WrappedOp)


def main() -> None:
    torch.manual_seed(0)
    samples = make_dataset()
    flux = build_pipeline(samples)

    # Flux implements the torch Dataset protocol; the registry collate stacks
    # input/target and keeps per-sample metadata as a list (Sample.is_batched).
    loader = DataLoader(flux, batch_size=8, shuffle=True, collate_fn=get_collate("sample"))

    batch = next(iter(loader))
    assert batch.input.shape == (8, 3, SIZE, SIZE) and batch.input.dtype == torch.float32
    assert batch.target.shape == (8, 1, SIZE, SIZE) and batch.target.dtype == torch.float32
    assert batch.is_batched and len(batch.batch_meta) == 8
    print(f"batch: input {tuple(batch.input.shape)}, target {tuple(batch.target.shape)}")
    print(f"       metadata (first 3): {batch.batch_meta[:3]}")

    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 1, kernel_size=3, padding=1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    epoch_means = []
    for epoch in range(3):
        losses = []
        for batch in loader:  # augmentations re-draw here: each epoch sees new variants
            optimizer.zero_grad()
            loss = criterion(model(batch.input), batch.target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        mean = sum(losses) / len(losses)
        epoch_means.append(mean)
        print(f"epoch {epoch}: mean loss {mean:.4f}")

    assert all(math.isfinite(v) for v in epoch_means)
    assert epoch_means[-1] < epoch_means[0], f"loss did not improve: {epoch_means}"
    print(f"loss improved {epoch_means[0]:.4f} -> {epoch_means[-1]:.4f} on augmented data")


if __name__ == "__main__":
    main()
