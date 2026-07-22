"""The typed-bag data model (proof of concept): a named bag of typed items, type-dispatched transforms.

Demonstrates the modality-neutral core of the redesign that steps away from
``Sample(input, target, metadata)``:

1. a ``TypedSample`` is a NAMED BAG of TYPED ITEMS, each owning its metadata — an ``Image``
   carries its layout, a ``Regions`` its canvas, a ``Label`` its classes; ``input`` /
   ``target`` are ROLE TAGS, not fixed positions;
2. the HEADLINE — ONE pipeline of BARE library transforms (each wrapped by its registered
   adapter): two torchvision ``transforms.v2`` transforms and an albumentations transform,
   each hitting only the field(s) of a type it handles. sampleflux ships NO native
   augmentation transforms — the libraries cover that through adapter coercion;
3. cross-field consistency — ONE library flip draw moves Image, Mask and Regions together,
   the Label untouched;
4. a custom transform from a plain function (``as_transform``), no library, no core edit;
5. interop — lower to a legacy ``Sample`` and lift back losslessly.

Signal-domain items (``Signal`` / ``Spectrogram``) and the ``Fourier`` transform are NOT here —
sampleflux is modality-neutral. They live in ``waivefront.bag`` and register into the SAME
registry; see ``waivefront/examples/typed_signal_pipeline.py`` for the signal + image mix.

Standalone, zero-arg, exit 0 (CI runs every ``examples/*.py``).
"""

import albumentations as A
import numpy as np
from torchvision.transforms import v2

from sampleflux import Image, Label, Mask, Pipeline, Regions, TypedSample, as_transform
from sampleflux.bag.interop import to_legacy, to_typed


def make_sample(rng: np.random.Generator) -> TypedSample:
    """A detection sample: an image, its mask, its boxes (targets), and a class label (target)."""
    return TypedSample(
        {
            "image": Image(rng.random((16, 20, 3)).astype(np.float32)),
            "mask": Mask(rng.random((16, 20)) > 0.5),
            "regions": Regions(boxes=[[2, 3, 6, 7]], labels=["drone"], canvas=(16, 20)),
            "class": Label("drone_x", classes=["noise", "drone_x"]),
        },
        roles={"mask": "target", "regions": "target", "class": "target"},
    )


def main() -> None:
    rng = np.random.default_rng(0)

    # 1. The named typed bag with role tags.
    sample = make_sample(rng)
    print("sample:      ", sample)
    print("inputs:      ", list(sample.inputs()), " targets:", list(sample.targets()))
    print("image meta:  ", f"layout={sample['image'].layout}   regions canvas={sample['regions'].canvas}")

    # 2. HEADLINE — one pipeline of BARE library transforms; a registered adapter wraps each,
    #    and every transform hits only the field(s) of its type.
    out = Pipeline(
        [
            v2.RandomHorizontalFlip(p=1.0),  # torchvision v2: Image + Mask + Regions together (one draw)
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),  # torchvision v2: Image
            A.GaussNoise(p=1.0),  # albumentations: Image
        ]
    )(sample)
    print("\n--- mixed cross-library pipeline (flip + normalize + noise) ---")
    print("image  ->", type(out["image"]).__name__, np.asarray(out["image"]).shape, "(flipped + normalized + noised)")
    print("mask   ->", type(out["mask"]).__name__, "(flipped with the image)")
    print("regions->", sample["regions"].boxes, "->", [[round(v) for v in b] for b in out["regions"].boxes], "(W=20)")
    print("class  ->", type(out["class"]).__name__, repr(out["class"].value), "(no handler — untouched)")
    assert np.array_equal(np.asarray(out["mask"]), np.asarray(sample["mask"])[:, ::-1])
    assert [round(v) for v in out["regions"].boxes[0]] == [14, 3, 18, 7]
    assert out["class"].value == "drone_x" and out.roles == sample.roles

    # 3. A custom transform from a plain function — no library, no core edit.
    brighten = as_transform(lambda d: d + 0.1, handles=(Image,), only=["image"])
    brightened = brighten(sample)
    print("\n--- custom function transform ---")
    print("image brightened:", np.allclose(np.asarray(brightened["image"]), np.asarray(sample["image"]) + 0.1))

    # 4. Interop — lossless round-trip through the legacy Sample.
    legacy = to_legacy(sample)
    back = to_typed(legacy)
    print("\n--- legacy interop ---")
    print("legacy input:", np.asarray(legacy.input).shape, " metadata keys:", list(legacy.meta))
    print("round-trip equal:", back == sample)
    assert back == sample

    print("\nOK")


if __name__ == "__main__":
    main()
