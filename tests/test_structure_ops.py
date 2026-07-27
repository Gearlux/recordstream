"""Structure ops over dict records — RenameField/DropField/CopyField/SelectFields."""

import numpy as np
import pytest

from recordstream import Image, Label, Record, Regions
from recordstream.ops.structure import CopyField, DropField, RenameField, SelectFields


def _record() -> Record:
    return {"image": Image(np.zeros((2, 2, 3))), "regions": Regions(boxes=[[0, 0, 1, 1]]), "class": Label("x")}


class TestRenameField:
    def test_renames_preserving_order(self) -> None:
        out = RenameField(src="regions", dst="boxes")(_record())
        assert "regions" not in out and isinstance(out["boxes"], Regions)
        assert list(out.keys()) == ["image", "boxes", "class"]  # renamed in place

    def test_rename_onto_existing_replaces(self) -> None:
        out = RenameField(src="class", dst="image")(_record())
        assert isinstance(out["image"], Label)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="both 'src' and 'dst'"):
            RenameField()(_record())
        with pytest.raises(KeyError, match="unknown key"):
            RenameField(src="nope", dst="x")(_record())


class TestDropField:
    def test_drops(self) -> None:
        out = DropField(key="class")(_record())
        assert "class" not in out and list(out.keys()) == ["image", "regions"]

    def test_missing_raises_unless_ok(self) -> None:
        with pytest.raises(KeyError, match="unknown key"):
            DropField(key="nope")(_record())
        rec = _record()
        assert DropField(key="nope", missing_ok=True)(rec) is rec

    def test_missing_key_config_raises(self) -> None:
        with pytest.raises(ValueError, match="'key'"):
            DropField()(_record())


class TestCopyField:
    def test_copies_same_object(self) -> None:
        out = CopyField(src="regions", dst="regions_backup")(_record())
        assert out["regions_backup"] is out["regions"]  # same value object (values are immutable)

    def test_copy_replaces_existing_dst(self) -> None:
        out = CopyField(src="class", dst="image")(_record())
        assert isinstance(out["image"], Label)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="both 'src' and 'dst'"):
            CopyField()(_record())
        with pytest.raises(KeyError, match="unknown key"):
            CopyField(src="nope", dst="x")(_record())


class TestSelectFields:
    def test_keeps_only_and_orders(self) -> None:
        out = SelectFields(keys=["class", "image"])(_record())
        assert list(out.keys()) == ["class", "image"]

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="'keys'"):
            SelectFields()(_record())
        with pytest.raises(KeyError, match="unknown keys"):
            SelectFields(keys=["image", "nope"])(_record())


class TestCopyOnWrite:
    def test_ops_never_mutate_the_incoming_record(self) -> None:
        rec = _record()
        RenameField(src="class", dst="klass")(rec)
        DropField(key="class")(rec)
        CopyField(src="class", dst="klass")(rec)
        SelectFields(keys=["image"])(rec)
        assert list(rec.keys()) == ["image", "regions", "class"]  # untouched


def test_configurable_marks() -> None:
    for cls in (RenameField, DropField, CopyField, SelectFields):
        assert getattr(cls, "__confluid_category__", None) == "op"
        assert getattr(cls, "__confluid_group__", None) == "structure"
