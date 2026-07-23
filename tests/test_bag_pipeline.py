"""The headline cross-library mixed pipeline + adapter behavior + import safety."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sampleflux.bag import Image, Label, Mask, Pipeline, Regions, Sample
from tests._bag_fixtures import FixtureFlip


class TestImportSafety:
    def test_typed_imports_without_torchvision(self) -> None:
        # The top-level package must import without torchvision (adapters lazy-import their
        # library inside method bodies), so discovery stays safe on hosts missing it.
        code = "import sys; import sampleflux.bag; assert 'torchvision' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, cwd=str(Path(__file__).resolve().parents[1]))


class TestTorchvisionAdapter:
    v2 = pytest.importorskip("torchvision.transforms.v2")

    def test_normalize_touches_only_image(self) -> None:
        from sampleflux.bag.adapters import TorchvisionV2Adapter

        s = Sample(
            {"image": Image(np.ones((4, 5, 3), dtype=np.float32)), "class": Label("x")},
            roles={"class": "target"},
        )
        out = TorchvisionV2Adapter(self.v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))(s)
        assert isinstance(out["image"], Image) and out["image"].layout == "HWC"
        assert np.asarray(out["image"]).shape == (4, 5, 3)
        assert np.allclose(np.asarray(out["image"]), 1.0)  # (1-0.5)/0.5
        assert out["class"].value == "x"

    def test_flip_moves_image_mask_boxes_together(self) -> None:
        from sampleflux.bag.adapters import TorchvisionV2Adapter

        s = Sample(
            {
                "image": Image(np.arange(6 * 8 * 3).reshape(6, 8, 3).astype(np.float32)),
                "mask": Mask(np.arange(6 * 8).reshape(6, 8).astype(np.int64)),
                "regions": Regions(boxes=[[1, 1, 4, 4]], canvas=(6, 8)),
            }
        )
        out = TorchvisionV2Adapter(self.v2.RandomHorizontalFlip(p=1.0))(s)
        assert np.array_equal(np.asarray(out["image"]), np.asarray(s["image"])[:, ::-1])
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(s["mask"])[:, ::-1])
        assert out["regions"].boxes == [[4.0, 1.0, 7.0, 4.0]]  # W=8: x -> W-x

    def test_missing_transform_raises(self) -> None:
        from sampleflux.bag.adapters import TorchvisionV2Adapter

        with pytest.raises(ValueError, match="must be set"):
            TorchvisionV2Adapter()(Sample({"image": Image(np.zeros((2, 2, 3), dtype=np.float32))}))

    def test_no_handled_field_is_noop(self) -> None:
        from sampleflux.bag.adapters import TorchvisionV2Adapter

        s = Sample({"class": Label("x")})
        assert TorchvisionV2Adapter(self.v2.RandomHorizontalFlip(p=1.0))(s) == s


class TestAlbumentationsAdapter:
    def test_gaussnoise_touches_only_image(self) -> None:
        import albumentations as A

        from sampleflux.bag.adapters import AlbumentationsAdapter

        s = Sample(
            {"image": Image(np.full((6, 6, 3), 0.5, dtype=np.float32)), "class": Label("x")},
            roles={"class": "target"},
        )
        out = AlbumentationsAdapter(A.GaussNoise(p=1.0))(s)
        assert isinstance(out["image"], Image)
        assert not np.array_equal(np.asarray(out["image"]), np.asarray(s["image"]))
        assert out["class"].value == "x"

    def test_bboxes_wrapped_and_returned(self) -> None:
        import albumentations as A

        from sampleflux.bag.adapters import AlbumentationsAdapter

        s = Sample(
            {
                "image": Image(np.random.rand(10, 12, 3).astype(np.float32)),
                "regions": Regions(boxes=[[2, 3, 6, 7]], labels=[1], canvas=(10, 12)),
            }
        )
        out = AlbumentationsAdapter(A.HorizontalFlip(p=1.0))(s)
        assert out["regions"].boxes[0] == pytest.approx([6.0, 3.0, 10.0, 7.0], abs=1e-4)  # W=12
        assert out["regions"].labels == [1]

    def test_missing_transform_raises(self) -> None:
        from sampleflux.bag.adapters import AlbumentationsAdapter

        with pytest.raises(ValueError, match="must be set"):
            AlbumentationsAdapter()(Sample({"image": Image(np.zeros((2, 2, 3), dtype=np.float32))}))


class TestMixedPipeline:
    def test_cross_library_bare_transforms(self) -> None:
        # BARE library transforms drop straight into the Pipeline — the registered adapters
        # wrap them; no explicit TorchvisionV2Adapter(...) / AlbumentationsAdapter(...). Two
        # torchvision v2 transforms and an albumentations transform mix in one pipeline; each
        # hits only the field(s) of a type it handles, and the flip moves image+mask+boxes with
        # one library draw.
        v2 = pytest.importorskip("torchvision.transforms.v2")
        import albumentations as A

        rng = np.random.default_rng(0)
        sample = Sample(
            {
                "image": Image(rng.random((16, 20, 3)).astype(np.float32)),
                "mask": Mask(rng.random((16, 20)) > 0.5),
                "regions": Regions(boxes=[[2, 3, 6, 7]], labels=["a"], canvas=(16, 20)),
                "class": Label("drone_x", classes=["noise", "drone_x"]),
            },
            roles={"mask": "target", "regions": "target", "class": "target"},
        )
        out = Pipeline(
            [
                v2.RandomHorizontalFlip(p=1.0),  # torchvision v2: Image + Mask + Regions together
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),  # torchvision v2: Image
                A.GaussNoise(p=1.0),  # albumentations: Image
            ]
        )(sample)
        assert not np.array_equal(np.asarray(out["image"]), np.asarray(sample["image"]))  # flipped+normalized+noised
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(sample["mask"])[:, ::-1])  # flipped with the image
        assert out["regions"].boxes[0] == pytest.approx([14, 3, 18, 7], abs=1e-4)  # W=20: x -> W-x
        assert out["class"].value == "drone_x"  # label rode through untouched
        assert out.roles == sample.roles  # role tags preserved end-to-end


class TestCoercion:
    def test_native_transform_passes_through(self) -> None:
        from sampleflux.bag import coerce_transform

        flip = FixtureFlip()
        assert coerce_transform(flip) is flip

    def test_torchvision_bare_transform_coerced(self) -> None:
        v2 = pytest.importorskip("torchvision.transforms.v2")

        from sampleflux.bag import coerce_transform
        from sampleflux.bag.adapters.torchvision import TorchvisionV2Adapter, is_torchvision_v2_transform

        norm = v2.Normalize(mean=[0.0], std=[1.0])
        assert is_torchvision_v2_transform(norm)
        assert isinstance(coerce_transform(norm), TorchvisionV2Adapter)

    def test_albumentations_bare_transform_coerced(self) -> None:
        import albumentations as A

        from sampleflux.bag import coerce_transform
        from sampleflux.bag.adapters.albumentations import AlbumentationsAdapter, is_albumentations_transform

        noise = A.GaussNoise(p=1.0)
        assert is_albumentations_transform(noise)
        assert isinstance(coerce_transform(noise), AlbumentationsAdapter)

    def test_unknown_object_raises(self) -> None:
        from sampleflux.bag import coerce_transform

        with pytest.raises(TypeError, match="don't know how to adapt"):
            coerce_transform(object())

    def test_register_custom_adapter(self) -> None:
        # A user library object becomes droppable into a Pipeline with one register_adapter call.
        from sampleflux.bag import FunctionTransform, coerce_transform, register_adapter
        from sampleflux.bag.transform import Transform

        class MyLibDouble:  # a foreign object, not a Transform
            pass

        def factory(_obj: object) -> Transform:
            return FunctionTransform(lambda d: d * 2, handles=(Image,))

        register_adapter(lambda o: isinstance(o, MyLibDouble), factory)

        s = Sample({"image": Image(np.ones((2, 2, 3)))})
        out = Pipeline([MyLibDouble()])(s)
        assert np.allclose(np.asarray(out["image"]), 2.0)
        assert isinstance(coerce_transform(MyLibDouble()), FunctionTransform)
