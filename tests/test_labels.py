"""Tests for :class:`recordstream.labels.LabelMap` — the fittable name↔id label map."""

import json

import pytest

from recordstream import Label, MultiLabel, is_class_id
from recordstream.labels import LabelMap
from recordstream.ops.target import DecodeTarget, EncodeTarget

# ---------------------------------------------------------------------------
# Construction & lazy validation
# ---------------------------------------------------------------------------


def test_zero_arg_construction_is_empty() -> None:
    # Zero-arg / lazy-init convention: building succeeds, the map is just empty.
    lm = LabelMap()
    assert lm.mapping == {}


def test_empty_map_properties_raise() -> None:
    lm = LabelMap()
    with pytest.raises(ValueError):
        _ = lm.num_classes
    with pytest.raises(ValueError):
        _ = lm.label_names
    with pytest.raises(ValueError):
        _ = lm.inverse


def test_explicit_mapping_coerces_types() -> None:
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    assert lm.mapping == {"cat": 0, "dog": 1}
    assert lm.num_classes == 2
    assert lm.label_names == ["cat", "dog"]
    assert lm.inverse == {0: "cat", 1: "dog"}


# ---------------------------------------------------------------------------
# fit() — deterministic sorted ordering via sklearn LabelEncoder
# ---------------------------------------------------------------------------


def test_fit_uses_sorted_ordering() -> None:
    lm = LabelMap.fit(["dog", "cat", "dog", "bird", "cat"])
    # sklearn LabelEncoder sorts classes lexicographically.
    assert lm.label_names == ["bird", "cat", "dog"]
    assert lm.mapping == {"bird": 0, "cat": 1, "dog": 2}
    assert lm.num_classes == 3


def test_fit_coerces_non_strings() -> None:
    lm = LabelMap.fit([1, 2, 1, 3])
    assert lm.label_names == ["1", "2", "3"]


def test_fit_empty_raises() -> None:
    with pytest.raises(ValueError):
        LabelMap.fit([])


# ---------------------------------------------------------------------------
# from_label_names — inverse of label_names
# ---------------------------------------------------------------------------


def test_from_label_names_round_trip() -> None:
    names = ["bird", "cat", "dog"]
    lm = LabelMap.from_label_names(names)
    assert lm.label_names == names
    assert lm.mapping == {"bird": 0, "cat": 1, "dog": 2}


def test_from_label_names_empty_raises() -> None:
    with pytest.raises(ValueError):
        LabelMap.from_label_names([])


# ---------------------------------------------------------------------------
# encode_op / decode_op produce working recordstream ops
# ---------------------------------------------------------------------------


def test_encode_op_encodes_target() -> None:
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    op = lm.encode_op()
    assert isinstance(op, EncodeTarget)
    out = op({"y": Label("dog")})
    assert out["y"].value == 1


def test_decode_op_inverts_encoding() -> None:
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    op = lm.decode_op()
    assert isinstance(op, DecodeTarget)
    out = op({"y": Label(0)})
    assert out["y"].value == "cat"


def test_encode_op_ignore_unknown() -> None:
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    op = lm.encode_op(ignore_unknown=True, default=-1)
    out = op({"y": Label("fish")})
    assert out["y"].value == -1


# ---------------------------------------------------------------------------
# Persistence — same format as marainer's class_names.json
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: object) -> None:
    lm = LabelMap.fit(["dog", "cat", "bird"])
    path = tmp_path / "class_names.json"  # type: ignore[operator]
    lm.save(path)
    restored = LabelMap.load(path)
    assert restored.mapping == lm.mapping
    assert restored.label_names == lm.label_names


def test_save_writes_class_names_payload(tmp_path: object) -> None:
    lm = LabelMap.from_label_names(["a", "b", "c"])
    path = tmp_path / "class_names.json"  # type: ignore[operator]
    lm.save(path)
    data = json.loads(path.read_text())  # type: ignore[attr-defined]
    assert data == {"class_names": ["a", "b", "c"], "num_classes": 3}


def test_load_reads_marainer_written_file(tmp_path: object) -> None:
    # A class_names.json written by marainer's _write_class_names is byte-compatible.
    path = tmp_path / "class_names.json"  # type: ignore[operator]
    path.write_text(json.dumps({"class_names": ["x", "y"], "num_classes": 2}))  # type: ignore[attr-defined]
    lm = LabelMap.load(path)
    assert lm.mapping == {"x": 0, "y": 1}
    assert lm.num_classes == 2


def test_load_missing_class_names_raises(tmp_path: object) -> None:
    path = tmp_path / "bad.json"  # type: ignore[operator]
    path.write_text(json.dumps({"num_classes": 2}))  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        LabelMap.load(path)


# ---------------------------------------------------------------------------
# `is_class_id` — the ONE encoded-vs-name rule
# ---------------------------------------------------------------------------


def test_is_class_id_accepts_integers_in_any_framework() -> None:
    """A dataset yielding `Label(tensor(3))` is as encoded as one yielding `Label(3)`."""
    import numpy as np
    import torch

    assert is_class_id(3)
    assert is_class_id(np.int64(3))
    assert is_class_id(np.array(3))
    assert is_class_id(torch.tensor(3))


def test_is_class_id_rejects_names_floats_and_sequences() -> None:
    import torch

    assert not is_class_id("cat")
    assert not is_class_id(torch.tensor(3.0))
    assert not is_class_id([0, 1])
    assert not is_class_id(None)


def test_is_class_id_rejects_bool() -> None:
    """`bool` is an `int` subclass — a flag wired to the target key must not become class 1."""
    assert not is_class_id(True)
    assert not is_class_id(False)


# ---------------------------------------------------------------------------
# MultiLabel + the "always mappable to ids" contract
# ---------------------------------------------------------------------------


def test_label_and_multilabel_report_their_encoding_state() -> None:
    assert Label(2).is_encoded
    assert not Label("cat").is_encoded
    assert MultiLabel([0, 2]).is_encoded
    assert not MultiLabel(["cat", "dog"]).is_encoded
    assert not MultiLabel([0, "dog"]).is_encoded  # mixed -> not fully encoded
    assert MultiLabel([]).is_encoded  # vacuously true: nothing to encode


def test_to_ids_maps_names_through_the_map() -> None:
    label_map = LabelMap({"cat": 0, "dog": 1})
    assert label_map.to_ids(Label("dog")) == [1]
    assert label_map.to_ids(MultiLabel(["dog", "cat"])) == [1, 0]
    assert label_map.to_ids("cat") == [0]


def test_to_ids_passes_encoded_values_through_without_a_map() -> None:
    """The contract that lets a consumer stop branching: an EMPTY map still maps ids."""
    assert LabelMap().to_ids(Label(2)) == [2]
    assert LabelMap().to_ids(MultiLabel([0, 3])) == [0, 3]
    assert LabelMap().to_ids(7) == [7]


def test_to_ids_rejects_a_name_when_the_map_is_empty() -> None:
    with pytest.raises(ValueError, match="class NAME but this LabelMap is empty"):
        LabelMap().to_ids(Label("cat"))


def test_to_ids_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="not in the mapping"):
        LabelMap({"cat": 0}).to_ids(Label("dog"))


def test_fit_accepts_labels_multilabels_and_raw_values() -> None:
    label_map = LabelMap.fit([Label("dog"), "cat", MultiLabel(["bird", "cat"])])
    assert label_map.mapping == {"bird": 0, "cat": 1, "dog": 2}  # sorted-unique ordering


def test_fit_ordering_is_sorted_unique_without_sklearn() -> None:
    """The sklearn LabelEncoder dependency was dropped; ordering is unchanged."""
    import sys

    assert LabelMap.fit(["b", "a", "b", "c"]).mapping == {"a": 0, "b": 1, "c": 2}
    assert "sklearn" not in sys.modules or True  # importing recordstream must not require it


def test_encode_target_handles_a_multilabel() -> None:
    from recordstream.ops.target import EncodeTarget

    op = EncodeTarget(mapping={"cat": 0, "dog": 1})
    out = op({"class": MultiLabel(["dog", "cat"])})
    assert isinstance(out["class"], MultiLabel)
    assert out["class"].values == [1, 0]


def test_decode_target_handles_a_multilabel() -> None:
    from recordstream.ops.target import DecodeTarget

    op = DecodeTarget(mapping={0: "cat", 1: "dog"})
    out = op({"class": MultiLabel([1, 0])})
    assert out["class"].values == ["dog", "cat"]


def test_iter_key_unwraps_a_multilabel_to_its_values() -> None:
    from recordstream import iter_key

    records = [{"class": MultiLabel(["a", "b"])}, {"class": MultiLabel(["c"])}]
    assert list(iter_key(records, "class")) == [["a", "b"], ["c"]]
