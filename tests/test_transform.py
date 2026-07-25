"""Transforms — type dispatch, once-per-record params, cross-key consistency, ``field=`` pin.

Native-kernel machinery is pinned via the test fixture ``FixtureFlip`` (sampleflux ships no
native augmentation transforms — libraries drop into ops lists bare, invoked natively by the
engine's op-family dispatch).
"""

import numpy as np
import pytest

from sampleflux import Image, Label, Mask, Pipeline, Record, Regions, Transform, as_transform
from tests._fixtures import FixtureFlip


def _seg() -> Record:
    return {
        "image": Image(np.arange(8 * 10 * 3).reshape(8, 10, 3).astype(np.float32)),
        "mask": Mask(np.arange(8 * 10).reshape(8, 10)),
        "regions": Regions(boxes=[[1, 1, 4, 4]], labels=["a"], canvas=(8, 10)),
        "class": Label("a"),
        "gain_db": -3.0,  # a plain scalar side value is just another key
    }


class TestKernelDispatchMachinery:
    def test_cross_key_consistency(self) -> None:
        out = FixtureFlip(p=1.0)(_seg())
        seg = _seg()
        assert out is not None
        assert np.array_equal(np.asarray(out["image"]), np.asarray(seg["image"])[:, ::-1])
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(seg["mask"])[:, ::-1])
        assert out["regions"].boxes == [[6, 1, 9, 4]]  # W=10: x -> W-x
        assert out["class"].value == "a"  # no handler — untouched
        assert out["gain_db"] == -3.0  # plain value — untouched

    def test_p_zero_is_identity(self) -> None:
        out = FixtureFlip(p=0.0)(_seg())
        assert out is not None
        assert np.array_equal(np.asarray(out["image"]), np.asarray(_seg()["image"]))
        assert out["regions"].boxes == [[1, 1, 4, 4]]

    def test_field_pin(self) -> None:
        # field= pins the op to ONE key: only "image" moves, the other handled types stay.
        out = FixtureFlip(p=1.0, field="image")(_seg())
        assert out is not None
        assert not np.array_equal(np.asarray(out["image"]), np.asarray(_seg()["image"]))
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(_seg()["mask"]))  # mask skipped
        assert out["regions"].boxes == [[1, 1, 4, 4]]  # regions skipped

    def test_field_pin_on_unhandled_type_is_noop(self) -> None:
        # Still type-gated: pinning to a key whose value type has no kernel changes nothing.
        out = FixtureFlip(p=1.0, field="class")(_seg())
        assert out is not None
        assert np.array_equal(np.asarray(out["image"]), np.asarray(_seg()["image"]))
        assert out["class"].value == "a"

    def test_image_layout_chw(self) -> None:
        s = {"image": Image(np.arange(3 * 4 * 5).reshape(3, 4, 5), layout="CHW")}
        out = FixtureFlip(p=1.0)(s)
        assert out is not None
        assert np.array_equal(np.asarray(out["image"]), np.asarray(s["image"])[:, :, ::-1])

    def test_regions_uses_canvas_without_image(self) -> None:
        s = {"regions": Regions(boxes=[[2, 0, 5, 3]], canvas=(8, 10))}
        out = FixtureFlip(p=1.0)(s)
        assert out is not None and out["regions"].boxes == [[5, 0, 8, 3]]

    def test_regions_without_reference_width_raises(self) -> None:
        s = {"regions": Regions(boxes=[[2, 0, 5, 3]])}  # no image, no canvas
        with pytest.raises(ValueError, match="no reference width"):
            FixtureFlip(p=1.0)(s)

    def test_params_sampled_once(self) -> None:
        # A partial-probability flip must be all-or-nothing across keys (shared decision),
        # never per-key independent draws.
        seg = _seg()
        for _ in range(25):
            out = FixtureFlip(p=0.5)(seg)
            assert out is not None
            image_flipped = not np.array_equal(np.asarray(out["image"]), np.asarray(seg["image"]))
            regions_flipped = out["regions"].boxes != seg["regions"].boxes
            assert image_flipped == regions_flipped

    def test_input_record_not_mutated(self) -> None:
        # Transform.__call__ is copy-on-write: the incoming dict keeps its original values.
        seg = _seg()
        FixtureFlip(p=1.0)(seg)
        assert np.array_equal(np.asarray(seg["image"]), np.asarray(_seg()["image"]))


class TestPipelineAndFunction:
    def test_pipeline_is_sequential(self) -> None:
        s = {"x": Image(np.ones((2, 2, 3), dtype=np.float32))}
        double = as_transform(lambda d: d * 2, handles=(Image,))
        out = Pipeline([double, double])(s)
        assert out is not None and np.allclose(np.asarray(out["x"]), 4.0)

    def test_function_transform_field_pin(self) -> None:
        s = {"a": Image(np.ones((2, 2, 3))), "b": Image(np.ones((2, 2, 3)))}
        out = as_transform(lambda d: d + 1, handles=(Image,), field="a")(s)
        assert out is not None
        assert np.allclose(np.asarray(out["a"]), 2.0) and np.allclose(np.asarray(out["b"]), 1.0)

    def test_function_transform_preserves_item_type(self) -> None:
        s = {"m": Mask(np.ones((2, 2)))}
        out = as_transform(lambda d: d * 3, handles=(Mask,))(s)
        assert out is not None and isinstance(out["m"], Mask) and np.allclose(np.asarray(out["m"]), 3.0)

    def test_pipeline_repr(self) -> None:
        assert "FixtureFlip" in repr(Pipeline([FixtureFlip()]))


class TestBaseTransform:
    def test_default_get_params_and_passthrough(self) -> None:
        # A transform with no kernels leaves every value alone.
        s = {"x": Label("v"), "g": 1.5}
        out = Transform()(s)
        assert out == s and out is not s  # equal copy, not the same dict

    def test_zero_arg_construction(self) -> None:
        assert FixtureFlip().p == 0.5 and Transform().field is None
