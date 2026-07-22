"""Transforms — type dispatch, once-per-sample params, cross-field consistency, only filter.

Native-kernel machinery is pinned via the test fixture ``FixtureFlip`` (sampleflux ships no
native augmentation transforms — libraries cover that through adapter coercion).
"""

import numpy as np
import pytest

from sampleflux import Image, Label, Mask, Pipeline, Regions, Transform, TypedSample, as_transform
from tests._bag_fixtures import FixtureFlip


def _seg() -> TypedSample:
    return TypedSample(
        {
            "image": Image(np.arange(8 * 10 * 3).reshape(8, 10, 3).astype(np.float32)),
            "mask": Mask(np.arange(8 * 10).reshape(8, 10)),
            "regions": Regions(boxes=[[1, 1, 4, 4]], labels=["a"], canvas=(8, 10)),
            "class": Label("a"),
        }
    )


class TestKernelDispatchMachinery:
    def test_cross_field_consistency(self) -> None:
        out = FixtureFlip(p=1.0)(_seg())
        seg = _seg()
        assert np.array_equal(np.asarray(out["image"]), np.asarray(seg["image"])[:, ::-1])
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(seg["mask"])[:, ::-1])
        assert out["regions"].boxes == [[6, 1, 9, 4]]  # W=10: x -> W-x
        assert out["class"].value == "a"  # no handler — untouched

    def test_p_zero_is_identity(self) -> None:
        out = FixtureFlip(p=0.0)(_seg())
        assert np.array_equal(np.asarray(out["image"]), np.asarray(_seg()["image"]))
        assert out["regions"].boxes == [[1, 1, 4, 4]]

    def test_only_filter(self) -> None:
        out = FixtureFlip(p=1.0, only=["image"])(_seg())
        assert not np.array_equal(np.asarray(out["image"]), np.asarray(_seg()["image"]))
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(_seg()["mask"]))  # mask skipped
        assert out["regions"].boxes == [[1, 1, 4, 4]]  # regions skipped

    def test_image_layout_chw(self) -> None:
        s = TypedSample({"image": Image(np.arange(3 * 4 * 5).reshape(3, 4, 5), layout="CHW")})
        out = FixtureFlip(p=1.0)(s)
        assert np.array_equal(np.asarray(out["image"]), np.asarray(s["image"])[:, :, ::-1])

    def test_regions_uses_canvas_without_image(self) -> None:
        s = TypedSample({"regions": Regions(boxes=[[2, 0, 5, 3]], canvas=(8, 10))})
        assert FixtureFlip(p=1.0)(s)["regions"].boxes == [[5, 0, 8, 3]]

    def test_regions_without_reference_width_raises(self) -> None:
        s = TypedSample({"regions": Regions(boxes=[[2, 0, 5, 3]])})  # no image, no canvas
        with pytest.raises(ValueError, match="no reference width"):
            FixtureFlip(p=1.0)(s)

    def test_params_sampled_once(self) -> None:
        # A partial-probability flip must be all-or-nothing across fields (shared decision),
        # never per-field independent draws.
        seg = _seg()
        for _ in range(25):
            out = FixtureFlip(p=0.5)(seg)
            image_flipped = not np.array_equal(np.asarray(out["image"]), np.asarray(seg["image"]))
            regions_flipped = out["regions"].boxes != seg["regions"].boxes
            assert image_flipped == regions_flipped


class TestAdapterParity:
    def test_v2_flip_matches_native_fixture(self) -> None:
        # The bare-library path must move image/mask/boxes EXACTLY like the native fixture —
        # this is the guarantee that let sampleflux drop its native flip for the library one.
        v2 = pytest.importorskip("torchvision.transforms.v2")
        seg = _seg()
        native = FixtureFlip(p=1.0)(seg)
        adapted = Pipeline([v2.RandomHorizontalFlip(p=1.0)])(seg)
        assert np.array_equal(np.asarray(adapted["image"]), np.asarray(native["image"]))
        assert np.array_equal(np.asarray(adapted["mask"]), np.asarray(native["mask"]))
        assert adapted["regions"].boxes[0] == pytest.approx(native["regions"].boxes[0], abs=1e-4)
        assert adapted["class"].value == native["class"].value == "a"


class TestPipelineAndFunction:
    def test_pipeline_is_sequential(self) -> None:
        s = TypedSample({"x": Image(np.ones((2, 2, 3), dtype=np.float32))})
        double = as_transform(lambda d: d * 2, handles=(Image,))
        out = Pipeline([double, double])(s)
        assert np.allclose(np.asarray(out["x"]), 4.0)

    def test_function_transform_only_filter(self) -> None:
        s = TypedSample({"a": Image(np.ones((2, 2, 3))), "b": Image(np.ones((2, 2, 3)))})
        out = as_transform(lambda d: d + 1, handles=(Image,), only=["a"])(s)
        assert np.allclose(np.asarray(out["a"]), 2.0) and np.allclose(np.asarray(out["b"]), 1.0)

    def test_pipeline_repr(self) -> None:
        assert "FixtureFlip" in repr(Pipeline([FixtureFlip()]))


class TestBaseTransform:
    def test_default_get_params_and_passthrough(self) -> None:
        # A transform with no kernels leaves every field alone.
        s = TypedSample({"x": Label("v")})
        assert Transform()(s) == s

    def test_decode_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="no decode"):
            FixtureFlip().decode(TypedSample({"image": Image(np.zeros((2, 2, 3)))}))

    def test_zero_arg_construction(self) -> None:
        assert FixtureFlip().p == 0.5 and Transform().only is None
