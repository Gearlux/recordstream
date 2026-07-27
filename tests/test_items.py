"""Typed items — array-subclass attribute preservation, wrappers, payload accessors, registry.

Only the MODALITY-NEUTRAL core items live in recordstream (Image / Mask / Regions / Label). The
data-bearing-wrapper and multi-attribute-array paths (which the signal-domain items in a domain
package exercise for real) are covered here with small test-local item types, so the core stays
tested without importing a domain package.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from recordstream.items import (
    Image,
    Label,
    Mask,
    NDArrayItem,
    Regions,
    get_item_type,
    is_item,
    item_data,
    item_type_names,
    item_types,
    register_item,
    with_data,
)


@register_item
@dataclass
class _Blob:
    """A test-local data-bearing wrapper item (the shape a signal item takes)."""

    data: object = None
    tag: str = "x"


class _Multi(NDArrayItem):
    """A test-local array item with two extra attributes (a spectrogram-like shape)."""

    _item_attrs = ("a", "b")
    a: int = 1
    b: object = None


class TestArrayItems:
    def test_default_and_explicit_attr(self) -> None:
        assert Image(np.zeros((2, 3, 3))).layout == "HWC"
        assert Image(np.zeros((3, 2, 3)), layout="CHW").layout == "CHW"

    def test_attr_survives_numpy_ops(self) -> None:
        img = Image(np.arange(2 * 3 * 3).reshape(2, 3, 3), layout="HWC")
        flipped = np.flip(img, axis=1)
        assert isinstance(flipped, Image) and flipped.layout == "HWC"
        doubled = img * 2
        assert isinstance(doubled, Image) and doubled.layout == "HWC"
        assert isinstance(img[0], Image)  # slicing keeps the subclass + attr

    def test_multiple_attrs_survive_ufunc(self) -> None:
        item = _Multi(np.zeros((4, 8)), a=7, b={"n": 8})
        assert item.a == 7 and item.b == {"n": 8}
        shifted = item + 1  # attrs carried through the ufunc
        assert isinstance(shifted, _Multi) and shifted.a == 7 and shifted.b == {"n": 8}

    def test_unknown_attr_rejected(self) -> None:
        with pytest.raises(TypeError, match="unexpected attributes"):
            Image(np.zeros((2, 2, 3)), colorspace="rgb")

    def test_mask_has_no_extra_attrs(self) -> None:
        assert isinstance(Mask(np.zeros((4, 4))), NDArrayItem)


class TestWrapperItems:
    def test_wrapper_fields(self) -> None:
        blob = _Blob(np.ones(8), tag="sig")
        assert blob.tag == "sig" and np.asarray(blob.data).sum() == 8

    def test_zero_arg_construction(self) -> None:
        # Wrappers build with no args (fields defaulted) — the workspace lazy/zero-arg convention.
        assert _Blob().data is None and Regions().boxes == [] and Label().value is None


class TestPayloadAccessors:
    def test_item_data_array(self) -> None:
        img = Image(np.arange(4).reshape(2, 2))
        data = item_data(img)
        assert type(data) is np.ndarray and np.array_equal(data, [[0, 1], [2, 3]])

    def test_item_data_wrapper(self) -> None:
        assert np.array_equal(item_data(_Blob(np.ones(3))), np.ones(3))

    def test_item_data_no_payload_returns_self(self) -> None:
        reg = Regions(boxes=[[0, 0, 1, 1]])
        assert item_data(reg) is reg  # no `.data` slot — returns the item

    def test_item_data_plain_value_passes_through(self) -> None:
        assert item_data(3.5) == 3.5 and item_data("s") == "s"  # non-items pass through verbatim

    def test_with_data_array_preserves_attrs(self) -> None:
        img = Image(np.zeros((2, 2, 3)), layout="CHW")
        rebuilt = with_data(img, np.ones((2, 2, 3)))
        assert isinstance(rebuilt, Image) and rebuilt.layout == "CHW" and rebuilt.sum() == 12

    def test_with_data_wrapper_preserves_meta(self) -> None:
        rebuilt = with_data(_Blob(np.zeros(4), tag="t"), np.ones(4))
        assert isinstance(rebuilt, _Blob) and rebuilt.tag == "t" and np.asarray(rebuilt.data).sum() == 4

    def test_with_data_without_payload_raises(self) -> None:
        with pytest.raises(TypeError, match="no payload slot"):
            with_data(Regions(boxes=[]), [[0, 0, 1, 1]])


class TestRegistry:
    def test_builtins_registered(self) -> None:
        names = item_type_names()
        for name in ("Image", "Mask", "Regions", "Label"):
            assert name in names
        assert Image in item_types()

    def test_get_item_type_and_miss(self) -> None:
        assert get_item_type("Image") is Image
        with pytest.raises(KeyError, match="no item type registered"):
            get_item_type("Nope")

    def test_is_item(self) -> None:
        assert is_item(Image(np.zeros((1, 1, 3)))) and is_item(Label())
        assert not is_item(np.zeros((2, 2))) and not is_item(42)

    def test_register_custom_type(self) -> None:
        @register_item
        class Keypoints:  # a user type — one class + one decorator, no core edit
            def __init__(self, points: list) -> None:
                self.points = points

        assert "Keypoints" in item_type_names() and get_item_type("Keypoints") is Keypoints
