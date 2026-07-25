"""The record data model: a plain dict of typed values, type-dispatched ops, libraries as-is.

Demonstrates the modality-neutral core of the engine:

1. a sample is a PLAIN ``dict`` of TYPED values, each owning its metadata — an ``Image``
   carries its layout, a ``Label`` its classes; scalar side values are just more keys;
2. the HEADLINE — ONE pipeline mixing a BARE albumentations transform (invoked natively by
   the engine's op-family dispatch: it receives exactly its own ``image``/``mask``/``bboxes``
   keys, one call = one joint draw) with native ops. sampleflux ships NO augmentation of its
   own and NO adapter classes — the libraries run as-is;
3. cross-key consistency — ONE ``A.Compose`` draw moves image, mask and bboxes together,
   the Label untouched;
4. a native op in the torchvision-v2 authoring style: params drawn once per record in
   ``get_params``, a kernel per value type, ``field=`` pinning it to one key;
5. a torchvision ``transforms.v2`` transform as-is — after the EXPLICIT ``v2.ToImage()``
   conversion, exactly like a plain torchvision pipeline (the engine never converts silently).

Standalone, zero-arg, exit 0 (CI runs every ``examples/*.py``).
"""

import albumentations as A
import numpy as np
import torch
from torchvision.transforms import v2

from sampleflux import Image, Label, Mask, Pipeline, Record, Transform, as_transform


def make_record(rng: np.random.Generator) -> Record:
    """A detection record: image, mask, boxes (albumentations vocabulary), and a class label."""
    return {
        "image": Image(rng.random((16, 20, 3)).astype(np.float32)),
        "mask": Mask((rng.random((16, 20)) > 0.5).astype(np.uint8)),
        "bboxes": [[2, 3, 6, 7]],
        "labels": ["drone"],
        "class": Label("drone_x", classes=["noise", "drone_x"]),
        "gain_db": -3.0,  # a scalar side value is just another key
    }


class Brighten(Transform):
    """Add a per-record random offset to every Image value (the tv2 authoring pattern).

    Args:
        strength: Maximum brightness offset drawn per record.
        field: Apply only to this record key. None (default) = every Image value.
    """

    handles = (Image,)

    def __init__(self, strength: float = 0.1, field: str = None) -> None:  # type: ignore[assignment]
        super().__init__(field=field)
        self.strength = strength
        self._rng = np.random.default_rng(7)

    def get_params(self, record: Record) -> dict:
        return {"offset": self._rng.uniform(0.0, self.strength)}  # drawn ONCE per record


@Brighten.kernel(Image)
def _brighten_image(value: Image, params: dict) -> Image:
    return Image(np.asarray(value) + params["offset"], layout=value.layout)


def main() -> None:
    rng = np.random.default_rng(0)

    # 1. The record: a plain dict of typed values.
    record = make_record(rng)
    print("record keys: ", list(record))
    print("image meta:  ", f"layout={record['image'].layout}   class vocab={record['class'].classes}")

    # 2+3. HEADLINE — bare albumentations (its OWN Compose carries bbox_params) + a native
    #      op in ONE Pipeline. The engine invokes each op family natively — no wrappers.
    flip = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    out = Pipeline([flip, A.GaussNoise(p=1.0), Brighten(strength=0.2)])(record)
    assert out is not None
    print("\n--- bare albumentations + native op in one pipeline ---")
    print("image  ->", type(out["image"]).__name__, np.asarray(out["image"]).shape, "(flipped + noised + brightened)")
    print("mask   ->", type(out["mask"]).__name__, "(flipped with the image — one joint draw)")
    print("bboxes ->", record["bboxes"], "->", [[round(v) for v in b] for b in out["bboxes"]], "(W=20)")
    print("class  ->", type(out["class"]).__name__, repr(out["class"].value), "(no handler — untouched)")
    assert np.array_equal(np.asarray(out["mask"]), np.asarray(record["mask"])[:, ::-1])
    assert [round(v) for v in out["bboxes"][0]] == [14, 3, 18, 7]
    assert out["class"].value == "drone_x" and out["gain_db"] == -3.0
    assert isinstance(out["image"], Image) and isinstance(out["mask"], Mask)  # types survive the library

    # 4. field= pins a type-dispatched op to ONE key (here a no-op: "class" is not an Image).
    untouched = Brighten(strength=0.2, field="class")(record)
    assert untouched is not None and np.array_equal(np.asarray(untouched["image"]), np.asarray(record["image"]))

    # 5. torchvision v2 as-is: the EXPLICIT conversion first (v2.ToImage: numpy HWC -> CHW
    #    tv_tensor), then any v2 transform — the engine passes the dict straight through.
    tv_out = Pipeline([v2.ToImage()])({"image": np.asarray(record["image"])})
    assert tv_out is not None and isinstance(tv_out["image"], torch.Tensor) and tv_out["image"].shape == (3, 16, 20)
    cropped = Pipeline([v2.RandomCrop(8)])(tv_out)
    assert cropped is not None and tuple(cropped["image"].shape) == (3, 8, 8)
    print("\n--- torchvision v2 as-is (explicit ToImage conversion) ---")
    print("image  ->", type(cropped["image"]).__name__, tuple(cropped["image"].shape))

    # 6. A custom function op — no library, no core edit.
    doubled = as_transform(lambda d: d * 2, handles=(Mask,), field="mask")(record)
    assert doubled is not None
    print("\n--- custom function op ---")
    print("mask doubled:", np.array_equal(np.asarray(doubled["mask"]), np.asarray(record["mask"]) * 2))

    print("\nOK")


if __name__ == "__main__":
    main()
