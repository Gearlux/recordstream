"""Tests for :class:`dataflux.labels.LabelMap` — the fittable name↔id label map."""

import json

import pytest

from dataflux.labels import LabelMap
from dataflux.ops.target import DecodeTargetOp, EncodeTargetOp
from dataflux.sample import Sample

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
# encode_op / decode_op produce working dataflux ops
# ---------------------------------------------------------------------------


def test_encode_op_encodes_target() -> None:
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    op = lm.encode_op()
    assert isinstance(op, EncodeTargetOp)
    out = op(Sample(input=None, target="dog", metadata={}))
    assert out.target == 1


def test_decode_op_inverts_encoding() -> None:
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    op = lm.decode_op()
    assert isinstance(op, DecodeTargetOp)
    out = op(Sample(input=None, target=0, metadata={}))
    assert out.target == "cat"


def test_encode_op_ignore_unknown() -> None:
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    op = lm.encode_op(ignore_unknown=True, default=-1)
    out = op(Sample(input=None, target="fish", metadata={}))
    assert out.target == -1


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
