"""``Scale`` changes a value RANGE; ``ToType`` changes an element TYPE — and neither does the other.

Both used to hide inside ``ToTensor(normalize=True)``, which divided by 255 whenever the largest
value was above 1 — a guess that silently squashed an already-standardized image to
``[-0.008, 0.010]``. Split out, each op does one declared thing:

* ``Scale`` maps ``[source_min, source_max]`` linearly onto ``[target_min, target_max]``. A blank
  source bound is the integer type's full range; a float has no such range, so a blank bound on
  a float payload is REFUSED rather than guessed.
* ``ToType`` casts to a named element type and REFUSES the two casts numpy performs silently and
  wrongly: complex -> real (drops the imaginary part) and a value an integer type cannot hold
  (wraps round).

Both touch every ``Image`` when ``field`` is blank — a ``Mask`` holds class ids that must stay
integers — and any array entry when ``field`` names it.
"""

from dataclasses import dataclass
from typing import Any, Callable, get_args

import numpy as np
import pytest

from recordstream import Image, Label, Mask
from recordstream.ops.numpy import ELEMENT_TYPES, ElementType, Scale, ToType


def _image(values: list, dtype: str) -> Image:
    return Image(np.array([[values]], dtype=dtype))


def _values(entry: object) -> list:
    return np.asarray(entry).ravel().tolist()


@dataclass
class _Wave:
    """A wrapper item whose payload rides a ``data`` field (the shape of a domain signal item)."""

    data: np.ndarray


class TestScale:
    def test_uint8_defaults_to_its_full_range_onto_zero_one(self) -> None:
        out = Scale()({"image": _image([0, 128, 255], "uint8")})["image"]
        assert np.asarray(out).dtype == np.float32
        assert np.allclose(_values(out), [0.0, 128 / 255, 1.0])

    def test_uint8_defaults_are_bit_identical_to_the_old_divide_by_255(self) -> None:
        """A chain moving from ToTensor's removed ``normalize`` to ``Scale`` gets the same numbers."""
        pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = np.asarray(Scale()({"image": Image(pixels)})["image"])
        assert np.array_equal(out, pixels.astype(np.float32) / np.float32(255.0))

    def test_a_12_bit_sensor_in_a_uint16_container_names_its_own_range(self) -> None:
        out = Scale(source_max=4095)({"image": _image([0, 2048, 4095], "uint16")})["image"]
        assert np.allclose(_values(out), [0.0, 2048 / 4095, 1.0])

    def test_blank_bounds_are_the_containers_range_not_the_sensors(self) -> None:
        """The CON case: left blank, a 12-bit image scales by 65535 and reads dark."""
        out = Scale()({"image": _image([0, 4095], "uint16")})["image"]
        assert np.allclose(_values(out), [0.0, 4095 / 65535])

    def test_a_float_range_is_mapped_when_named(self) -> None:
        out = Scale(source_min=-140, source_max=-60)({"image": _image([-140.0, -100.0, -60.0], "float32")})["image"]
        assert np.allclose(_values(out), [0.0, 0.5, 1.0])

    def test_the_target_range_is_a_parameter(self) -> None:
        out = Scale(target_min=-1.0, target_max=1.0)({"image": _image([0, 255], "uint8")})["image"]
        assert np.allclose(_values(out), [-1.0, 1.0])

    def test_one_blank_bound_takes_the_types_value(self) -> None:
        out = Scale(source_min=55)({"image": _image([55, 255], "uint8")})["image"]
        assert np.allclose(_values(out), [0.0, 1.0])

    def test_a_blank_bound_on_a_float_is_refused_not_guessed(self) -> None:
        with pytest.raises(ValueError, match=r"Scale: 'image' is float32.*give source_min and source_max"):
            Scale()({"image": _image([0.0, 0.5, 1.0], "float32")})

    def test_an_empty_source_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"source_min and source_max are both 5\.0"):
            Scale(source_min=5, source_max=5)({"image": _image([5, 6], "uint8")})

    def test_complex_values_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"Scale: 'image' is complex"):
            Scale(source_min=0, source_max=1)({"image": Image(np.array([[[1 + 2j]]]))})

    def test_values_outside_the_source_range_are_not_clipped(self) -> None:
        out = Scale(source_max=200)({"image": _image([255], "uint8")})["image"]
        assert np.allclose(_values(out), [255 / 200])

    def test_a_float_keeps_its_own_float_type(self) -> None:
        out = Scale(source_min=0, source_max=2)({"image": _image([0.0, 2.0], "float64")})["image"]
        assert np.asarray(out).dtype == np.float64

    def test_the_item_and_its_layout_survive(self) -> None:
        chw = Image(np.zeros((3, 2, 2), dtype=np.uint8), layout="CHW")
        out = Scale()({"image": chw})["image"]
        assert isinstance(out, Image) and out.layout == "CHW"


class TestToType:
    def test_a_cast_does_not_scale(self) -> None:
        out = ToType(dtype="float64")({"image": _image([0, 128, 255], "uint8")})["image"]
        assert np.asarray(out).dtype == np.float64
        assert _values(out) == [0.0, 128.0, 255.0]

    def test_a_real_image_becomes_complex(self) -> None:
        out = ToType(dtype="complex64")({"image": _image([0.0, 0.5], "float32")})["image"]
        assert np.asarray(out).dtype == np.complex64
        assert _values(out) == [0j, 0.5 + 0j]

    def test_complex_to_real_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"ToType: 'image' is complex; float32 would drop the imaginary part"):
            ToType(dtype="float32")({"image": Image(np.array([[[1 + 2j]]]))})

    def test_a_value_an_integer_type_cannot_hold_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"ToType: 'image' holds 0\.0 \.\. 300\.0, which uint8 \(0 \.\. 255\)"):
            ToType(dtype="uint8")({"image": _image([0.0, 300.0], "float32")})

    def test_nan_cannot_become_an_integer(self) -> None:
        with pytest.raises(ValueError, match=r"ToType: 'image' holds NaN or infinity"):
            ToType(dtype="int32")({"image": _image([1.0, float("nan")], "float32")})

    def test_a_fraction_is_truncated_toward_zero(self) -> None:
        out = ToType(dtype="int16")({"image": _image([2.7, -2.7], "float32")})["image"]
        assert _values(out) == [2, -2]

    def test_an_empty_array_casts(self) -> None:
        out = ToType(dtype="uint8")({"image": Image(np.zeros((0, 2, 3), dtype=np.float32))})["image"]
        assert np.asarray(out).dtype == np.uint8

    def test_the_type_is_a_closed_choice(self) -> None:
        assert ELEMENT_TYPES == get_args(ElementType)
        assert ELEMENT_TYPES == (
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "uint8",
            "int16",
            "int32",
            "int64",
        )

    def test_an_unknown_type_name_is_refused_at_construction(self) -> None:
        with pytest.raises(Exception, match=r"dtype"):
            ToType(dtype="double")  # type: ignore[arg-type]


_MAKERS = {
    "Scale": lambda field="": Scale(field=field),
    "ToType": lambda field="": ToType(dtype="float32", field=field),
}


@pytest.mark.parametrize("make", list(_MAKERS.values()), ids=list(_MAKERS))
class TestWhichEntriesTheyTouch:
    def test_blank_field_changes_every_image_and_no_mask(self, make: Callable[..., Any]) -> None:
        record = {
            "a": _image([0, 255], "uint8"),
            "b": _image([0, 255], "uint8"),
            "target": Mask(np.array([[0, 2]], dtype=np.int64)),
        }
        out = make()(record)
        assert np.asarray(out["a"]).dtype == np.float32 and np.asarray(out["b"]).dtype == np.float32
        assert out["target"] is record["target"], "a Mask holds class ids and must be left alone"

    def test_a_named_field_reaches_a_mask(self, make: Callable[..., Any]) -> None:
        out = make("target")({"target": Mask(np.array([[0, 255]], dtype=np.uint8))})
        assert isinstance(out["target"], Mask) and np.asarray(out["target"]).dtype == np.float32

    def test_a_named_field_reaches_a_wrapper_items_payload(self, make: Callable[..., Any]) -> None:
        out = make("wave")({"wave": _Wave(data=np.array([0, 255], dtype=np.uint8))})
        assert isinstance(out["wave"], _Wave) and out["wave"].data.dtype == np.float32

    def test_a_named_field_reaches_a_plain_array(self, make: Callable[..., Any]) -> None:
        out = make("raw")({"raw": np.array([0, 255], dtype=np.uint8)})
        assert isinstance(out["raw"], np.ndarray) and out["raw"].dtype == np.float32

    def test_a_missing_field_is_named(self, make: Callable[..., Any]) -> None:
        with pytest.raises(ValueError, match=r"field 'nope' not in record \(keys: \['image'\]\)"):
            make("nope")({"image": _image([0], "uint8")})

    def test_a_field_without_an_array_is_refused(self, make: Callable[..., Any]) -> None:
        with pytest.raises(ValueError, match=r"'class' holds a Label, not an array"):
            make("class")({"class": Label(value="cat")})

    def test_other_entries_pass_through(self, make: Callable[..., Any]) -> None:
        record = {"image": _image([0, 255], "uint8"), "class": Label(value="cat"), "id": 7}
        out = make()(record)
        assert out["class"] is record["class"] and out["id"] == 7


def test_construction_stores_and_validates_nothing() -> None:
    """Zero-arg, and a bad configuration is reported at the first record, not in __init__."""
    assert Scale().source_min is None and ToType().dtype == "float32"
    Scale(source_min=5, source_max=5)  # an empty range, refused only when applied
