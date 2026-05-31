"""Tests for the target movers / encoders (``dataflux.ops.target``)."""

import pytest

from dataflux.ops.target import DecodeTargetOp, EncodeTargetOp, MetadataToTargetOp
from dataflux.sample import Sample


# --------------------------------------------------------------------------- #
# MetadataToTargetOp
# --------------------------------------------------------------------------- #
def test_metadata_to_target_moves_value() -> None:
    out = MetadataToTargetOp(key="drone")(Sample(input=0, metadata={"drone": "DJI MINI3"}))
    assert out.target == "DJI MINI3"


def test_metadata_to_target_leaves_metadata_untouched_without_target_key() -> None:
    sample = Sample(input=0, metadata={"drone": "DJI MINI3"})
    out = MetadataToTargetOp(key="drone")(sample)
    assert set(out.metadata) == {"drone"}


def test_metadata_to_target_copies_to_target_key() -> None:
    sample = Sample(input=0, metadata={"drone": "DJI MINI3"})
    out = MetadataToTargetOp(key="drone", target_key="raw_label")(sample)
    assert out.target == "DJI MINI3"
    assert out.metadata["raw_label"] == "DJI MINI3"


def test_metadata_to_target_missing_key_raises() -> None:
    with pytest.raises(KeyError, match="no key 'drone'"):
        MetadataToTargetOp(key="drone")(Sample(input=0, metadata={"other": 1}))


# --------------------------------------------------------------------------- #
# EncodeTargetOp
# --------------------------------------------------------------------------- #
def test_encode_target_maps_known_value() -> None:
    op = EncodeTargetOp(mapping={"DJI AVATA2": 2, "DJI MINI3": 5})
    assert op(Sample(input=0, target="DJI MINI3")).target == 5


def test_encode_target_class_zero_allowed() -> None:
    op = EncodeTargetOp(mapping={"first": 0, "second": 1})
    assert op(Sample(input=0, target="first")).target == 0


def test_encode_target_unknown_raises() -> None:
    op = EncodeTargetOp(mapping={"a": 1})
    with pytest.raises(KeyError, match="not in mapping"):
        op(Sample(input=0, target="missing"))


def test_encode_target_unknown_substitutes_default_when_ignored() -> None:
    op = EncodeTargetOp(mapping={"a": 1}, ignore_unknown=True, default=7)
    assert op(Sample(input=0, target="missing")).target == 7


def test_encode_target_empty_mapping_rejected() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        EncodeTargetOp(mapping={})


# --------------------------------------------------------------------------- #
# DecodeTargetOp
# --------------------------------------------------------------------------- #
def test_decode_target_inverts_encode() -> None:
    mapping = {"DJI AVATA2": 2, "DJI MINI3": 5}
    inverse = {v: k for k, v in mapping.items()}
    sample = Sample(input=0, target="DJI MINI3")
    encoded = EncodeTargetOp(mapping=mapping)(sample)
    decoded = DecodeTargetOp(mapping=inverse)(encoded)
    assert decoded.target == "DJI MINI3"


def test_decode_target_unknown_default_is_none() -> None:
    op = DecodeTargetOp(mapping={1: "a"}, ignore_unknown=True)
    assert op(Sample(input=0, target=999)).target is None


def test_decode_target_empty_mapping_rejected() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        DecodeTargetOp(mapping={})


# --------------------------------------------------------------------------- #
# Composed chain (the decomposed classification label path)
# --------------------------------------------------------------------------- #
def test_metadata_to_target_then_encode() -> None:
    label_to_index = {"DJI AVATA2": 2, "DJI MINI3": 5}
    sample = Sample(input=0, metadata={"drone": "DJI AVATA2"})
    sample = MetadataToTargetOp(key="drone", target_key="raw_label")(sample)
    sample = EncodeTargetOp(mapping=label_to_index)(sample)
    assert sample.target == 2
    # raw label preserved for decode/reporting
    assert sample.metadata["raw_label"] == "DJI AVATA2"
