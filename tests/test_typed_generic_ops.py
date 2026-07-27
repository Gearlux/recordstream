"""The generic array→Image→Mask→Regions ops over dict records.

Pins the three native type-changing transforms that run the detection/segmentation
front-end on plain record dicts:

* :class:`recordstream.ops.image.ConvertToImage` — array-bearing key → ``Image`` item;
* :class:`recordstream.ops.numpy.Threshold` — array key → boolean ``Mask`` item;
* :class:`recordstream.ops.numpy.ConnectedComponents` — ``Mask`` → ``Regions`` item.

Each op REUSES its shared math helper, so the op output is pinned identical to the helper
(parity). recordstream-only — no domain-package import.
"""

import numpy as np
import pytest
from confluid.registry import get_registry, resolve_class

from recordstream import Image, Mask, Regions
from recordstream.ops.image import ConvertToImage, _bound_longest_side, _render_rgb
from recordstream.ops.numpy import ConnectedComponents, Threshold, connected_component_bboxes, threshold_array


def _ramp_2d() -> np.ndarray:
    return np.arange(8 * 10).reshape(8, 10).astype(np.float32)


def _blob_mask() -> np.ndarray:
    m = np.zeros((6, 6), dtype=bool)
    m[0:2, 0:2] = True  # blob A (area 4) -> (0, 1, 0, 1)
    m[4:6, 4:6] = True  # blob B (area 4) -> (4, 5, 4, 5)
    return m


# --------------------------------------------------------------------------- #
# ConvertToImage
# --------------------------------------------------------------------------- #
class TestConvertToImage:
    def test_produces_image_item_shape_dtype(self) -> None:
        out = ConvertToImage(colormap="gray")({"spec": Mask(_ramp_2d())})
        assert "image" in out
        img = out["image"]
        assert isinstance(img, Image)
        assert np.asarray(img).shape == (8, 10, 3)
        assert np.asarray(img).dtype == np.uint8
        assert img.layout == "HWC"
        # source entry untouched
        assert isinstance(out["spec"], Mask)

    def test_parity_with_render_helper_default_sizing(self) -> None:
        arr = _ramp_2d()
        out = ConvertToImage(colormap="viridis")({"spec": Mask(arr)})
        expected = _bound_longest_side(_render_rgb(arr, "viridis"), 512)
        assert np.array_equal(expected, np.asarray(out["image"]))

    def test_exact_resize_and_flip(self) -> None:
        arr = _ramp_2d()
        out = ConvertToImage(colormap="gray", width=20, height=16, flip_vertical=True)({"spec": Mask(arr)})
        assert np.asarray(out["image"]).shape == (16, 20, 3)

    def test_explicit_field_and_custom_output(self) -> None:
        rec = {"a": Mask(_ramp_2d()), "b": Mask(np.zeros((4, 4), dtype=np.float32))}
        out = ConvertToImage(field="b", output="preview")(rec)
        assert np.asarray(out["preview"]).shape == (4, 4, 3)

    def test_does_not_publish_image_dims_keys(self) -> None:
        # The Image SHAPE carries the pixel dims; no image_width_px / image_height_px entries.
        out = ConvertToImage()({"spec": Mask(_ramp_2d())})
        assert set(out.keys()) == {"spec", "image"}
        assert np.asarray(out["image"]).shape[:2] == (8, 10)

    def test_missing_explicit_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in record"):
            ConvertToImage(field="nope")({"spec": Mask(_ramp_2d())})

    def test_no_array_field_raises(self) -> None:
        with pytest.raises(ValueError, match="no array-bearing field"):
            ConvertToImage()({"lbl": Regions(boxes=[[0, 0, 1, 1]])})


# --------------------------------------------------------------------------- #
# Threshold
# --------------------------------------------------------------------------- #
class TestThreshold:
    def test_produces_mask_parity(self) -> None:
        arr = _ramp_2d()
        out = Threshold(low_level=20.0)({"spec": Mask(arr)})
        assert isinstance(out["mask"], Mask)
        assert np.asarray(out["mask"]).dtype == np.bool_
        expected = threshold_array(arr, low_level=20.0)
        assert np.array_equal(np.asarray(out["mask"]), expected)

    def test_string_literal_bound(self) -> None:
        arr = _ramp_2d()
        out = Threshold(low_level="20")({"spec": Mask(arr)})
        expected = threshold_array(arr, low_level="20")
        assert np.array_equal(np.asarray(out["mask"]), expected)

    def test_env_var_expression_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_THRESH_LEVEL", "20")
        arr = _ramp_2d()
        out = Threshold(low_level="$TEST_THRESH_LEVEL")({"spec": Mask(arr)})
        assert np.array_equal(np.asarray(out["mask"]), arr > 20.0)

    def test_meta_key_expression_has_no_source(self) -> None:
        # {key} expressions have no metadata home in the record model -> loud KeyError.
        with pytest.raises(KeyError):
            Threshold(low_level="{some_key}")({"spec": Mask(_ramp_2d())})

    def test_band_pass_both_bounds_and_ops(self) -> None:
        arr = _ramp_2d()
        out = Threshold(low_level=20.0, high_level=60.0, low_op=">=", high_op="<=")({"spec": Mask(arr)})
        expected = threshold_array(arr, low_level=20.0, high_level=60.0, low_op=">=", high_op="<=")
        assert np.array_equal(np.asarray(out["mask"]), expected)
        assert np.array_equal(np.asarray(out["mask"]), (arr >= 20.0) & (arr <= 60.0))

    def test_no_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Threshold()({"spec": Mask(_ramp_2d())})

    def test_default_field_picks_first_array(self) -> None:
        # No explicit field: first array-bearing item (insertion order).
        rec = {"raw": Mask(_ramp_2d()), "other": Regions(boxes=[])}
        out = Threshold(low_level=20.0)(rec)
        assert np.array_equal(np.asarray(out["mask"]), _ramp_2d() > 20.0)

    def test_missing_explicit_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in record"):
            Threshold(low_level=1.0, field="nope")({"spec": Mask(_ramp_2d())})

    def test_non_array_field_raises(self) -> None:
        with pytest.raises(TypeError, match="expected an array"):
            Threshold(low_level=1.0, field="reg")({"reg": Regions(boxes=[])})

    def test_no_array_field_default_raises(self) -> None:
        with pytest.raises(ValueError, match="no array-bearing field"):
            Threshold(low_level=1.0)({"reg": Regions(boxes=[])})


# --------------------------------------------------------------------------- #
# ConnectedComponents
# --------------------------------------------------------------------------- #
class TestConnectedComponents:
    def test_produces_regions_bin_box_contract(self) -> None:
        out = ConnectedComponents()({"m": Mask(_blob_mask())})
        regions = out["boxes"]
        assert isinstance(regions, Regions)
        # The pinned generic contract: (row_min, row_max, col_min, col_max) inclusive tuples.
        assert regions.boxes == [(0, 1, 0, 1), (4, 5, 4, 5)]

    def test_parity_with_helper(self) -> None:
        mask = _blob_mask()
        out = ConnectedComponents()({"m": Mask(mask)})
        expected = connected_component_bboxes(mask)
        assert out["boxes"].boxes == expected

    def test_min_area_bins_filters_small_blobs(self) -> None:
        m = np.zeros((6, 6), dtype=bool)
        m[0:2, 0:2] = True  # area 4
        m[5, 5] = True  # area 1 -> dropped when min_area_bins=2
        out = ConnectedComponents(min_area_bins=2)({"m": Mask(m)})
        assert out["boxes"].boxes == [(0, 1, 0, 1)]

    def test_connectivity_parity(self) -> None:
        # Diagonal touch: 4-connectivity keeps two blobs, 8 merges them.
        m = np.zeros((4, 4), dtype=bool)
        m[0, 0] = True
        m[1, 1] = True
        four = ConnectedComponents(connectivity=4)({"m": Mask(m)})
        eight = ConnectedComponents(connectivity=8)({"m": Mask(m)})
        assert len(four["boxes"].boxes) == 2
        assert len(eight["boxes"].boxes) == 1

    def test_default_prefers_mask_over_other_array(self) -> None:
        # An Image is inserted first, but a Mask is preferred by the default resolver.
        rec = {"img": Image(np.zeros((6, 6, 3), dtype=np.uint8)), "seg": Mask(_blob_mask())}
        out = ConnectedComponents()(rec)
        assert out["boxes"].boxes == [(0, 1, 0, 1), (4, 5, 4, 5)]

    def test_falls_back_to_first_array_when_no_mask(self) -> None:
        # No Mask item — a 2-D array item is used.
        out = ConnectedComponents()({"m": Image(_blob_mask())})
        assert out["boxes"].boxes == [(0, 1, 0, 1), (4, 5, 4, 5)]

    def test_non_2d_mask_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D mask"):
            ConnectedComponents()({"m": Mask(np.zeros((2, 2, 2), dtype=bool))})

    def test_missing_explicit_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in record"):
            ConnectedComponents(field="nope")({"m": Mask(_blob_mask())})

    def test_no_mask_or_array_raises(self) -> None:
        with pytest.raises(ValueError, match="no Mask or array-bearing field"):
            ConnectedComponents()({"reg": Regions(boxes=[])})


# --------------------------------------------------------------------------- #
# End-to-end chain: array -> Image -> Mask -> Regions, all on one record dict.
# --------------------------------------------------------------------------- #
def test_array_to_image_to_mask_to_regions_chain() -> None:
    arr = _ramp_2d()
    record = {"spec": Mask(arr)}
    out = ConnectedComponents(field="mask")(Threshold(field="spec", low_level=20.0)(ConvertToImage()(record)))
    # Every stage produced its typed entry.
    assert isinstance(out["image"], Image)
    assert isinstance(out["mask"], Mask)
    assert isinstance(out["boxes"], Regions)
    # Regions carries (row_min, row_max, col_min, col_max) bin-box tuples.
    assert out["boxes"].boxes
    for box in out["boxes"].boxes:
        assert len(box) == 4
        row_min, row_max, col_min, col_max = box
        assert row_min <= row_max and col_min <= col_max
    # The image entry carries the pixel dims via its shape (no separate metadata).
    assert np.asarray(out["image"]).shape[:2] == arr.shape


# --------------------------------------------------------------------------- #
# Discovery + zero-arg construction.
# --------------------------------------------------------------------------- #
def test_zero_arg_constructible() -> None:
    assert ConvertToImage().output == "image"
    assert Threshold().output == "mask"
    assert ConnectedComponents().output == "boxes"


@pytest.mark.parametrize(
    ("name", "cls", "group"),
    [
        ("ConvertToImage", ConvertToImage, "image"),
        ("Threshold", Threshold, "numpy"),
        ("ConnectedComponents", ConnectedComponents, "numpy"),
    ],
)
def test_discovery_tags(name: str, cls: type, group: str) -> None:
    assert cls.__confluid_category__ == "op"  # type: ignore[attr-defined]
    assert cls.__confluid_group__ == group  # type: ignore[attr-defined]
    assert resolve_class(name) is cls
    registry = get_registry()
    assert name in registry.list_classes(category="op")
    assert name in registry.list_classes(group=group)
