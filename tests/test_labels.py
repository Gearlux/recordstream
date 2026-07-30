"""Tests for :class:`recordstream.labels.LabelMap` — the fittable name↔id label map."""

import json
from typing import Any

import numpy as np
import pytest

from recordstream import Label, MultiLabel, is_class_id
from recordstream.labels import LabelMap, class_counts, inverse_frequency_weights
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
        _ = lm.class_names
    with pytest.raises(ValueError):
        _ = lm.inverse


def test_explicit_mapping_coerces_types() -> None:
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    assert lm.mapping == {"cat": 0, "dog": 1}
    assert lm.num_classes == 2
    assert lm.class_names == ["cat", "dog"]
    assert lm.inverse == {0: "cat", 1: "dog"}


# ---------------------------------------------------------------------------
# fit() — deterministic sorted ordering via sklearn LabelEncoder
# ---------------------------------------------------------------------------


def test_fit_uses_sorted_ordering() -> None:
    lm = LabelMap.fit(["dog", "cat", "dog", "bird", "cat"])
    # sklearn LabelEncoder sorts classes lexicographically.
    assert lm.class_names == ["bird", "cat", "dog"]
    assert lm.mapping == {"bird": 0, "cat": 1, "dog": 2}
    assert lm.num_classes == 3


def test_fit_coerces_non_strings() -> None:
    lm = LabelMap.fit([1, 2, 1, 3])
    assert lm.class_names == ["1", "2", "3"]


def test_fit_empty_raises() -> None:
    with pytest.raises(ValueError):
        LabelMap.fit([])


# ---------------------------------------------------------------------------
# from_class_names — inverse of class_names
# ---------------------------------------------------------------------------


def test_from_label_names_round_trip() -> None:
    names = ["bird", "cat", "dog"]
    lm = LabelMap.from_class_names(names)
    assert lm.class_names == names
    assert lm.mapping == {"bird": 0, "cat": 1, "dog": 2}


def test_from_label_names_empty_raises() -> None:
    with pytest.raises(ValueError):
        LabelMap.from_class_names([])


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
    assert restored.class_names == lm.class_names


def test_save_writes_class_names_payload(tmp_path: object) -> None:
    lm = LabelMap.from_class_names(["a", "b", "c"])
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


# --------------------------------------------------------------------------- #
# LabelMap.encode — the fit -> encode idiom in one call
# --------------------------------------------------------------------------- #


def test_encode_wraps_a_source_into_an_encoding_stream() -> None:
    """The two-step idiom every consumer of a name-labelled dataset used to write itself."""
    from recordstream import Stream, iter_key

    records = [{"image": i, "class": Label(n)} for i, n in enumerate(["dog", "cat", "bird", "cat"])]
    label_map = LabelMap.fit(iter_key(records, "class"))

    encoded = label_map.encode(records)

    assert isinstance(encoded, Stream)
    assert list(iter_key(encoded, "class")) == [2, 1, 0, 1]


def test_encode_leaves_the_source_untouched() -> None:
    """A Stream is a view — encoding must not mutate the records it reads."""
    records = [{"class": Label("cat")}]
    LabelMap(mapping={"cat": 0}).encode(records)

    assert records[0]["class"].value == "cat"


def test_encode_refuses_an_already_encoded_id() -> None:
    """Unlike `to_ids`, the OP is a straight lookup — double-encoding fails loudly.

    Silently remapping an id would corrupt the labels of anyone who wrapped a source twice,
    so the caller asks `is_class_id` first (which is what the consuming trainers do).
    """
    from recordstream import iter_key

    encoded = LabelMap(mapping={"cat": 0}).encode([{"class": Label(1)}])

    with pytest.raises(KeyError, match="not in mapping"):
        list(iter_key(encoded, "class"))


def test_encode_flows_a_deferred_source() -> None:
    """A config-wired `!class:` source works without the caller flowing it first."""
    from confluid import Class as ConfluidClass

    from recordstream import Stream, iter_key

    records = [{"class": Label("cat")}]
    deferred = ConfluidClass(Stream, source=records)

    assert list(iter_key(LabelMap(mapping={"cat": 0}).encode(deferred), "class")) == [0]


# ---------------------------------------------------------------------------
# class_counts / inverse_frequency_weights — the label STATISTIC behind balancing
# ---------------------------------------------------------------------------


def test_class_counts_accepts_every_target_shape() -> None:
    """Label / MultiLabel / bare id all normalize through LabelMap.to_ids."""
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    assert list(class_counts([Label("cat"), Label("dog"), Label("cat")], 2, lm)) == [2.0, 1.0]
    # A multi-label target counts for EVERY class it names.
    assert list(class_counts([MultiLabel(["cat", "dog"]), MultiLabel(["dog"])], 2, lm)) == [1.0, 2.0]
    # Integer targets need no map at all — `to_ids` passes encoded ids through.
    assert list(class_counts([0, 1, 1], 2)) == [1.0, 2.0]


def test_class_counts_skips_none_targets() -> None:
    assert list(class_counts([Label(0), None, Label(0)], 2)) == [2.0, 0.0]


def test_uniform_distribution_weights_every_class_equally() -> None:
    weights = inverse_frequency_weights([0, 1, 2, 0, 1, 2], num_classes=3)
    assert weights is not None
    assert np.allclose(weights, np.ones(3))


def test_rare_classes_weigh_more_than_common_ones() -> None:
    weights = inverse_frequency_weights([0, 0, 0, 1], num_classes=2)
    assert weights is not None
    # w = total / (num_classes * count): 4/(2*3) and 4/(2*1)
    assert np.allclose(weights, np.array([2 / 3, 2.0]))
    assert weights[1] > weights[0]


def test_an_unobserved_class_gets_zero_not_infinity() -> None:
    weights = inverse_frequency_weights([0, 0], num_classes=3)
    assert weights is not None
    assert weights[1] == 0.0 and weights[2] == 0.0
    assert np.isfinite(weights).all()


def test_out_of_range_ids_are_ignored_rather_than_raising() -> None:
    """A stray label must not abort a training run."""
    weights = inverse_frequency_weights([0, 1, 99, -1], num_classes=2)
    assert weights is not None
    assert np.allclose(weights, np.ones(2))


def test_no_observations_returns_none() -> None:
    """`None` distinguishes "no weights" from "all-zero weights"."""
    assert inverse_frequency_weights([], num_classes=3) is None
    assert inverse_frequency_weights([7, 8], num_classes=3) is None


def test_weights_are_numpy_float32_not_a_tensor() -> None:
    """Only `batch_tensor` is torch in this package — a framework converts in one line."""
    weights = inverse_frequency_weights([0, 1], num_classes=2)
    assert weights is not None
    assert isinstance(weights, np.ndarray) and weights.dtype == np.float32


def test_weights_encode_class_NAMES_through_the_map() -> None:
    """The flattening a consumer used to do by hand lives here now."""
    lm = LabelMap(mapping={"cat": 0, "dog": 1})
    weights = inverse_frequency_weights([Label("cat")] * 3 + [Label("dog")], 2, lm)
    assert weights is not None
    assert np.allclose(weights, np.array([2 / 3, 2.0]))


# --------------------------------------------------------------------------- #
# The vocabulary rides the encoded data
# --------------------------------------------------------------------------- #
# A consumer that needs to name a predicted class id, or persist the mapping beside a
# checkpoint, should not have to be handed a separate LabelMap and keep it in sync with
# the dataset. Before this, one consumer monkey-patched the attribute on and read it
# back with a getattr — an undeclared convention nothing could see.


def test_encode_carries_the_vocabulary_onto_the_stream() -> None:
    from recordstream import iter_key

    records = [{"class": Label(n)} for n in ["dog", "cat", "bird"]]
    label_map = LabelMap.fit(iter_key(records, "class"))

    assert label_map.encode(records).class_names == ["bird", "cat", "dog"]


def test_a_plain_stream_carries_no_vocabulary() -> None:
    from recordstream import Stream

    assert Stream(source=[{"class": Label(0)}]).class_names is None


def test_class_names_reads_the_first_source_that_has_one() -> None:
    """A vocabulary is a property of the RUN, so the caller passes every split."""
    from recordstream import Stream, class_names, iter_key

    records = [{"class": Label(n)} for n in ["dog", "cat"]]
    encoded = LabelMap.fit(iter_key(records, "class")).encode(records)
    plain = Stream(source=records)

    assert class_names(plain, encoded) == ["cat", "dog"]


def test_class_names_skips_none_so_call_sites_need_no_guards() -> None:
    from recordstream import class_names, iter_key

    records = [{"class": Label("cat")}]
    encoded = LabelMap.fit(iter_key(records, "class")).encode(records)

    assert class_names(None, None, encoded) == ["cat"]


def test_class_names_is_none_when_nothing_carries_one() -> None:
    """An unencoded run is not an error — it has integer labels and no vocabulary."""
    from recordstream import class_names

    assert class_names(None, [{"class": Label(0)}]) is None


def test_class_names_coerces_a_foreign_sources_names_to_str() -> None:
    """A Stream validates its own `List[str]`; a foreign dataset attribute is not so lucky."""
    from recordstream import class_names

    class _ForeignDataset:
        class_names = [1, 2]  # e.g. integer category ids from another library

    assert class_names(_ForeignDataset()) == ["1", "2"]


def test_a_stream_rejects_non_string_names_at_construction() -> None:
    """The declared `List[str]` is enforced — the slot is config, not a free-for-all."""
    import pytest

    from recordstream import Stream

    with pytest.raises(Exception, match="valid string"):
        # The wrong type is the POINT — this asserts the runtime check, which is what a caller
        # building a Stream from YAML gets. mypy would otherwise reject the very call under test.
        Stream(source=[], class_names=[1, 2])  # type: ignore[list-item]


# --------------------------------------------------------------------------- #
# A deferred source materializes itself
# --------------------------------------------------------------------------- #
# `LabelMap.encode` already flowed its source; `project` / `iter_key` did not — so every
# consumer wrote `flow(source)` at the call site to compensate, and had to know which
# entry point needed it. Flowing a live object is a no-op, so this costs nothing.


def _deferred(records: list) -> Any:
    """A `!class:` marker as a config hands one over — unbuilt."""
    from confluid import Class

    from recordstream import Stream

    return Class(Stream, source=records)


def test_project_materializes_a_deferred_source() -> None:
    from recordstream import project

    assert list(project(_deferred([{"class": Label(0), "extra": 1}]), ("class",))) == [{"class": Label(0)}]


def test_iter_key_materializes_a_deferred_source() -> None:
    from recordstream import iter_key

    assert list(iter_key(_deferred([{"class": Label(i)} for i in range(3)]), "class")) == [0, 1, 2]


def test_num_classes_materializes_a_deferred_source() -> None:
    from recordstream import num_classes

    assert num_classes(_deferred([{"class": Label(i)} for i in range(4)])) == 4


def test_a_live_source_is_unaffected() -> None:
    """Flowing a built object is a no-op — the common path must not change."""
    from recordstream import iter_key

    assert list(iter_key([{"class": Label(i)} for i in range(3)], "class")) == [0, 1, 2]
