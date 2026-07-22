"""``TypedSample`` — role tags, views, copy-on-write mutators, array-safe equality."""

import numpy as np
import pytest

from sampleflux.bag.items import Image, Label, Regions
from sampleflux.bag.sample import ROLES, TypedSample


def _sample() -> TypedSample:
    return TypedSample(
        {"image": Image(np.ones(4)), "regions": Regions(boxes=[[0, 0, 1, 1]]), "class": Label("x")},
        roles={"regions": "target", "class": "target"},
    )


class TestRolesAndViews:
    def test_default_role_is_input(self) -> None:
        s = TypedSample({"a": Image(np.zeros((1, 1, 3))), "b": Label()})
        assert s.roles == {"a": "input", "b": "input"}

    def test_inputs_targets_aux(self) -> None:
        s = _sample()
        assert list(s.inputs()) == ["image"]
        assert list(s.targets()) == ["regions", "class"]
        assert s.aux() == {}

    def test_of_role_and_role_of(self) -> None:
        s = _sample()
        assert s.role_of("class") == "target"
        assert list(s.of_role("input")) == ["image"]

    def test_items_of_type(self) -> None:
        s = _sample()
        assert [k for k, _ in s.items_of_type(Image)] == ["image"]
        assert [k for k, _ in s.items_of_type(Image, Label)] == ["image", "class"]

    def test_roles_closed_set(self) -> None:
        assert set(ROLES) == {"input", "target", "aux", "pred"}


class TestConstruction:
    def test_role_for_unknown_field_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown field"):
            TypedSample({"a": Label()}, roles={"b": "target"})

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid role"):
            TypedSample({"a": Label()}, roles={"a": "output"})  # type: ignore[dict-item]


class TestCopyOnWrite:
    def test_set_role_returns_new(self) -> None:
        s = _sample()
        s2 = s.set_role("regions", "aux")
        assert s2.role_of("regions") == "aux"
        assert s.role_of("regions") == "target"  # original untouched

    def test_set_role_unknown_and_invalid(self) -> None:
        s = _sample()
        with pytest.raises(KeyError):
            s.set_role("nope", "aux")
        with pytest.raises(ValueError):
            s.set_role("class", "bogus")  # type: ignore[arg-type]

    def test_replace_field_preserves_role(self) -> None:
        s = _sample()
        s2 = s.replace_field("class", Label("y"))
        assert s2["class"].value == "y" and s2.role_of("class") == "target"
        assert s["class"].value == "x"

    def test_replace_new_field_defaults_input(self) -> None:
        s = _sample().replace_field("image", Image(np.zeros((2, 2, 3))))
        assert "image" in s and s.role_of("image") == "input"

    def test_drop(self) -> None:
        s = _sample().drop("class")
        assert "class" not in s and list(s.keys()) == ["image", "regions"]


class TestMappingProtocol:
    def test_len_iter_contains_getitem(self) -> None:
        s = _sample()
        assert len(s) == 3 and "image" in s and list(iter(s)) == ["image", "regions", "class"]
        assert isinstance(s["image"], Image)

    def test_fields_and_roles_are_copies(self) -> None:
        s = _sample()
        s.fields["image"] = None
        s.roles["image"] = "target"
        assert isinstance(s["image"], Image) and s.role_of("image") == "input"


class TestEquality:
    def test_equal_with_array_fields(self) -> None:
        a = TypedSample({"img": Image(np.zeros((2, 2, 3)))})
        b = TypedSample({"img": Image(np.zeros((2, 2, 3)))})
        assert a == b

    def test_unequal_arrays(self) -> None:
        a = TypedSample({"img": Image(np.zeros((2, 2, 3)))})
        b = TypedSample({"img": Image(np.ones((2, 2, 3)))})
        assert a != b

    def test_unequal_roles_or_keys(self) -> None:
        a = TypedSample({"x": Label("v")})
        assert a != TypedSample({"x": Label("v")}, roles={"x": "target"})
        assert a != TypedSample({"y": Label("v")})

    def test_not_a_sample(self) -> None:
        assert (TypedSample({"x": Label()}) == 5) is False

    def test_repr(self) -> None:
        assert "Image[input]" in repr(_sample())
