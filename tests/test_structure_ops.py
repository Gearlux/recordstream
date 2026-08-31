"""Structure ops over dict records — RenameField/DropField/CopyField/SelectFields."""

import numpy as np
import pytest

from recordstream import Boxes, Image, Label, Record
from recordstream.ops.structure import CopyField, DropField, PutField, RandomNumber, RenameField, SelectFields


def _record() -> Record:
    return {"image": Image(np.zeros((2, 2, 3))), "regions": Boxes(boxes=[[0, 0, 1, 1]]), "class": Label("x")}


class TestRenameField:
    def test_renames_preserving_order(self) -> None:
        out = RenameField(src="regions", dst="boxes")(_record())
        assert "regions" not in out and isinstance(out["boxes"], Boxes)
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


class TestRandomNumber:
    """A number producer — no record anywhere near it."""

    def test_each_call_is_a_fresh_draw_in_range(self) -> None:
        draw = RandomNumber(low=0.25, high=0.75)
        values = [draw() for _ in range(5)]
        assert all(0.25 <= v <= 0.75 for v in values)
        assert len(set(values)) > 1, "calls must draw fresh values"

    def test_a_seed_makes_the_sequence_reproducible(self) -> None:
        first = RandomNumber(seed=7)
        second = RandomNumber(seed=7)
        assert [first() for _ in range(4)] == [second() for _ in range(4)]

    def test_the_value_output_is_one_draw(self) -> None:
        """The declared @output a canvas can wire — NOTE it folds to a FROZEN literal at
        export; wire the NODE itself (the object output) for a fresh draw per record."""
        value = RandomNumber(low=0.0, high=1.0, seed=1).value
        assert isinstance(value, float) and 0.0 <= value <= 1.0

    def test_zero_arg_construction_is_legal(self) -> None:
        assert RandomNumber().low == 0.0


class TestPutField:
    """`record + value + name` — the one generic way to stamp a value into a record."""

    def test_a_plain_value_is_stored_under_the_key(self) -> None:
        out = PutField(key="score", value=0.7)(_record())
        assert out["score"].value == 0.7
        assert set(out) == set(_record()) | {"score"}

    def test_a_CALLABLE_value_is_drawn_PER_RECORD(self) -> None:
        """The wire that matters: a RandomNumber node feeding PutField must yield a
        different draw for each record, never one frozen number for the whole dataset."""
        put = PutField(key="score", value=RandomNumber(seed=7))
        scores = [put(_record())["score"].value for _ in range(4)]
        assert len(set(scores)) > 1

    def test_an_existing_key_is_replaced_dict_semantics(self) -> None:
        """PUT means put: a model re-stamping its own entry on a second pass must not fail.
        Protecting the gt is the prediction convention's job, not this op's."""
        out = PutField(key="class", value=1)(_record())
        assert out["class"].value == 1

    def test_an_empty_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="key"):
            PutField(value=1)(_record())

    def test_a_missing_value_is_refused_not_stored_as_none(self) -> None:
        with pytest.raises(ValueError, match="value"):
            PutField(key="score")(_record())

    def test_zero_arg_construction_is_legal(self) -> None:
        assert PutField().key == ""
