"""Typed structure ops — SetRole/RenameField/DropField/CopyField/SelectFields + primary/merge."""

import numpy as np
import pytest

from sampleflux import Image, Label, Regions, TypedSample, primary
from sampleflux.ops.structure import CopyField, DropField, RenameField, SelectFields, SetRole


def _sample() -> TypedSample:
    return TypedSample(
        {"image": Image(np.zeros((2, 2, 3))), "regions": Regions(boxes=[[0, 0, 1, 1]]), "class": Label("x")},
        roles={"regions": "target", "class": "target"},
    )


class TestSetRole:
    def test_retags(self) -> None:
        out = SetRole(key="regions", role="aux")(_sample())
        assert out.role_of("regions") == "aux" and out.role_of("class") == "target"

    def test_lazy_validation(self) -> None:
        assert SetRole().key == ""  # zero-arg constructible
        with pytest.raises(ValueError, match="'key'"):
            SetRole()(_sample())
        # An invalid Literal is rejected at CONSTRUCTION (confluid schema enforcement) …
        with pytest.raises(Exception, match="input_value='bogus'"):
            SetRole(key="class", role="bogus")  # type: ignore[arg-type]
        # … and the defensive __call__ re-check guards post-construction mutation.
        op = SetRole(key="class")
        op.role = "bogus"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="invalid role"):
            op(_sample())


class TestRenameField:
    def test_renames_role_travels(self) -> None:
        out = RenameField(src="regions", dst="boxes")(_sample())
        assert "regions" not in out and out.role_of("boxes") == "target"

    def test_rename_onto_existing_replaces(self) -> None:
        out = RenameField(src="class", dst="image")(_sample())
        assert isinstance(out["image"], Label) and out.role_of("image") == "target"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="both 'src' and 'dst'"):
            RenameField()(_sample())
        with pytest.raises(KeyError, match="unknown field"):
            RenameField(src="nope", dst="x")(_sample())


class TestDropField:
    def test_drops(self) -> None:
        out = DropField(key="class")(_sample())
        assert "class" not in out and list(out.keys()) == ["image", "regions"]

    def test_missing_raises_unless_ok(self) -> None:
        with pytest.raises(KeyError, match="unknown field"):
            DropField(key="nope")(_sample())
        assert DropField(key="nope", missing_ok=True)(_sample()) == _sample()


class TestCopyField:
    def test_copies_with_source_role(self) -> None:
        out = CopyField(src="regions", dst="regions_backup")(_sample())
        assert out["regions_backup"] is out["regions"] and out.role_of("regions_backup") == "target"

    def test_copy_with_explicit_role(self) -> None:
        out = CopyField(src="regions", dst="regions_aux", role="aux")(_sample())
        assert out.role_of("regions_aux") == "aux"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="both 'src' and 'dst'"):
            CopyField()(_sample())
        with pytest.raises(KeyError, match="unknown field"):
            CopyField(src="nope", dst="x")(_sample())


class TestSelectFields:
    def test_keeps_only_and_orders(self) -> None:
        out = SelectFields(keys=["class", "image"])(_sample())
        assert list(out.keys()) == ["class", "image"] and out.role_of("class") == "target"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="'keys'"):
            SelectFields()(_sample())
        with pytest.raises(KeyError, match="unknown fields"):
            SelectFields(keys=["image", "nope"])(_sample())


class TestPrimaryAndMerge:
    def test_primary_by_role(self) -> None:
        s = _sample()
        assert primary(s)[0] == "image"
        assert primary(s, "target")[0] == "regions"  # first target in insertion order

    def test_primary_missing_role_raises(self) -> None:
        with pytest.raises(KeyError, match="no field with role 'pred'"):
            primary(_sample(), "pred")

    def test_merge_union_last_wins(self) -> None:
        a = TypedSample({"x": Label("a"), "shared": Label("from_a")})
        b = TypedSample({"y": Label("b"), "shared": Label("from_b")}, roles={"shared": "target"})
        m = TypedSample.merge(a, b)
        assert list(m.keys()) == ["x", "shared", "y"]  # union keeps first-seen position
        assert m["shared"].value == "from_b" and m.role_of("shared") == "target"  # last wins, role travels

    def test_merge_rejects_non_sample(self) -> None:
        with pytest.raises(TypeError, match="expected TypedSample"):
            TypedSample.merge(_sample(), "nope")  # type: ignore[arg-type]

    def test_configurable_marks(self) -> None:
        for cls in (SetRole, RenameField, DropField, CopyField, SelectFields):
            assert getattr(cls, "__confluid_category__", None) == "op"
            assert getattr(cls, "__confluid_group__", None) == "structure"
