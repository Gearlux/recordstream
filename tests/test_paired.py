"""Tests for dataflux.paired.PairedSource."""

from typing import Any, Dict, Iterator, Optional

import confluid  # type: ignore[import-not-found]
import pytest

from dataflux.discovery import get_callable_path
from dataflux.paired import PairedSource
from dataflux.sample import Sample

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@confluid.configurable
class PairedIndexedSource:
    """Indexable primary source producing Samples keyed by integer id."""

    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Sample:
        return Sample(input=index * 10, target=None, metadata={"id": f"s{index}"})

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.size):
            yield self[i]


@confluid.configurable
class DictStore:
    """Minimal mapping-shaped secondary for tests."""

    def __init__(self, records: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.records: Dict[str, Dict[str, Any]] = records or {}

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.records[key]

    def keys(self) -> Any:
        return self.records.keys()


# Module-level callables so resolve_callable can find them
def sample_id_key(sample: Sample) -> str:
    return str(sample.metadata["id"])


def identity_extract(record: Dict[str, Any], sample: Sample) -> Optional[Dict[str, Any]]:
    return record


def none_extract(record: Dict[str, Any], sample: Sample) -> Optional[Dict[str, Any]]:
    return None


def odd_only_extract(record: Dict[str, Any], sample: Sample) -> Optional[Dict[str, Any]]:
    """Return the record only for odd-id samples, None otherwise."""
    idx = int(str(sample.metadata["id"])[1:])
    if idx % 2 == 1:
        return record
    return None


def resolve_by_id(key: str, primary: PairedIndexedSource) -> Sample:
    """Reverse-lookup: key 's<N>' -> primary[N]."""
    idx = int(key[1:])
    return primary[idx]


# ---------------------------------------------------------------------------
# left_outer
# ---------------------------------------------------------------------------


def test_left_outer_emits_all_primary() -> None:
    primary = PairedIndexedSource(size=4)
    store = DictStore({"s0": {"label": "a"}, "s2": {"label": "c"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key)

    samples = list(paired)

    assert len(samples) == 4
    assert [s.input for s in samples] == [0, 10, 20, 30]
    assert [s.metadata["annotated"] for s in samples] == [True, False, True, False]


def test_left_outer_flattens_record_into_metadata() -> None:
    primary = PairedIndexedSource(size=2)
    store = DictStore({"s0": {"label": "dog", "confidence": 0.9}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key)

    samples = list(paired)

    assert samples[0].metadata["label"] == "dog"
    assert samples[0].metadata["confidence"] == 0.9
    assert samples[0].metadata["annotation_key"] == "s0"
    assert "label" not in samples[1].metadata
    assert samples[1].metadata["annotation_key"] == "s1"


def test_left_outer_preserves_original_metadata() -> None:
    primary = PairedIndexedSource(size=1)
    store = DictStore({"s0": {"label": "x"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key)

    sample = list(paired)[0]
    assert sample.metadata["id"] == "s0"  # primary metadata survived
    assert sample.metadata["label"] == "x"


def test_left_outer_prefix() -> None:
    primary = PairedIndexedSource(size=1)
    store = DictStore({"s0": {"label": "x"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key, prefix="ann_")

    sample = list(paired)[0]
    assert sample.metadata["ann_label"] == "x"
    assert "label" not in sample.metadata


def test_left_outer_store_full_under() -> None:
    primary = PairedIndexedSource(size=1)
    store = DictStore({"s0": {"label": "x", "score": 0.5}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key, store_full_under="raw_annotation")

    sample = list(paired)[0]
    assert sample.metadata["raw_annotation"] == {"label": "x", "score": 0.5}
    # Still flattened too
    assert sample.metadata["label"] == "x"


def test_left_outer_len_delegates_to_primary() -> None:
    primary = PairedIndexedSource(size=7)
    store = DictStore({"s0": {"label": "a"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key)

    assert len(paired) == 7


def test_left_outer_getitem_matched_and_unmatched() -> None:
    primary = PairedIndexedSource(size=3)
    store = DictStore({"s1": {"label": "y"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key)

    matched = paired[1]
    assert matched.metadata["annotated"] is True
    assert matched.metadata["label"] == "y"

    unmatched = paired[0]
    assert unmatched.metadata["annotated"] is False
    assert "label" not in unmatched.metadata


# ---------------------------------------------------------------------------
# inner
# ---------------------------------------------------------------------------


def test_inner_emits_only_matched() -> None:
    primary = PairedIndexedSource(size=4)
    store = DictStore({"s0": {"label": "a"}, "s3": {"label": "d"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key, policy="inner")

    samples = list(paired)

    assert len(samples) == 2
    assert {s.metadata["annotation_key"] for s in samples} == {"s0", "s3"}
    assert all(s.metadata["annotated"] for s in samples)


def test_inner_len_is_cached_scan() -> None:
    primary = PairedIndexedSource(size=10)
    store = DictStore({f"s{i}": {"label": "x"} for i in (0, 2, 4, 6)})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key, policy="inner")

    assert len(paired) == 4
    # Second call hits cache; should still be correct.
    assert len(paired) == 4


def test_inner_rejects_getitem() -> None:
    primary = PairedIndexedSource(size=2)
    store = DictStore({"s0": {"label": "a"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=sample_id_key, policy="inner")

    with pytest.raises(TypeError, match="left_outer"):
        _ = paired[0]


# ---------------------------------------------------------------------------
# extract_fn
# ---------------------------------------------------------------------------


def test_extract_fn_transforms_record() -> None:
    primary = PairedIndexedSource(size=2)
    store = DictStore({"s0": {"label": "a"}, "s1": {"label": "b"}})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        extract_fn=identity_extract,
    )

    samples = list(paired)

    assert samples[0].metadata["label"] == "a"
    assert samples[1].metadata["label"] == "b"


def test_extract_fn_returning_none_marks_unannotated() -> None:
    primary = PairedIndexedSource(size=3)
    store = DictStore({"s0": {"label": "x"}, "s1": {"label": "y"}, "s2": {"label": "z"}})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        extract_fn=odd_only_extract,
    )

    samples = list(paired)

    # Every sample emitted under left_outer; only odd ones are annotated
    assert [s.metadata["annotated"] for s in samples] == [False, True, False]


def test_extract_fn_with_inner_policy_filters() -> None:
    primary = PairedIndexedSource(size=4)
    store = DictStore({f"s{i}": {"label": "x"} for i in range(4)})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        extract_fn=odd_only_extract,
        policy="inner",
    )

    samples = list(paired)

    assert {s.metadata["annotation_key"] for s in samples} == {"s1", "s3"}


def test_extract_fn_none_suppresses_flattening() -> None:
    primary = PairedIndexedSource(size=1)
    store = DictStore({"s0": {"label": "x"}})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        extract_fn=none_extract,
    )

    sample = list(paired)[0]
    assert sample.metadata["annotated"] is False
    assert "label" not in sample.metadata


# ---------------------------------------------------------------------------
# Coarser-key / broadcast
# ---------------------------------------------------------------------------


def pack_key(sample: Sample) -> str:
    """Every sample shares the key 'pack' - simulates pack-level annotation."""
    return "pack"


def test_coarser_key_broadcasts_to_all_matching_samples() -> None:
    primary = PairedIndexedSource(size=3)
    store = DictStore({"pack": {"drone": "dji_mavic"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=pack_key)

    samples = list(paired)

    assert all(s.metadata["annotated"] for s in samples)
    assert all(s.metadata["drone"] == "dji_mavic" for s in samples)


# ---------------------------------------------------------------------------
# right_driven
# ---------------------------------------------------------------------------


def test_right_driven_iterates_annotation_keys() -> None:
    primary = PairedIndexedSource(size=10)
    store = DictStore({"s0": {"label": "a"}, "s3": {"label": "d"}})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        policy="right_driven",
        primary_resolver=resolve_by_id,
    )

    samples = list(paired)

    assert len(samples) == 2
    assert [s.metadata["annotation_key"] for s in samples] == ["s0", "s3"]
    assert [s.input for s in samples] == [0, 30]


def test_right_driven_len_is_secondary_len() -> None:
    primary = PairedIndexedSource(size=100)
    store = DictStore({"s1": {"label": "a"}, "s5": {"label": "b"}, "s9": {"label": "c"}})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        policy="right_driven",
        primary_resolver=resolve_by_id,
    )

    assert len(paired) == 3


def test_right_driven_skips_when_extract_fn_returns_none() -> None:
    primary = PairedIndexedSource(size=4)
    store = DictStore({f"s{i}": {"label": "x"} for i in range(4)})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        policy="right_driven",
        primary_resolver=resolve_by_id,
        extract_fn=odd_only_extract,
    )

    samples = list(paired)

    assert {s.metadata["annotation_key"] for s in samples} == {"s1", "s3"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_policy_raises() -> None:
    with pytest.raises(ValueError, match="Invalid policy"):
        PairedSource(
            primary=PairedIndexedSource(),
            secondary=DictStore(),
            key_fn=sample_id_key,
            policy="outer_join",
        )


def test_right_driven_requires_primary_resolver() -> None:
    with pytest.raises(ValueError, match="primary_resolver"):
        PairedSource(
            primary=PairedIndexedSource(),
            secondary=DictStore(),
            key_fn=sample_id_key,
            policy="right_driven",
        )


def test_right_driven_requires_secondary_keys_method() -> None:
    class NoKeys:
        def __contains__(self, k: str) -> bool:  # pragma: no cover - defensive
            return False

        def __getitem__(self, k: str) -> Any:  # pragma: no cover - defensive
            raise KeyError(k)

    with pytest.raises(TypeError, match="keys"):
        PairedSource(
            primary=PairedIndexedSource(),
            secondary=NoKeys(),
            key_fn=sample_id_key,
            policy="right_driven",
            primary_resolver=resolve_by_id,
        )


def test_left_outer_requires_mapping_interface() -> None:
    class NoContains:
        pass

    with pytest.raises(TypeError, match="__contains__"):
        PairedSource(
            primary=PairedIndexedSource(),
            secondary=NoContains(),
            key_fn=sample_id_key,
        )


# ---------------------------------------------------------------------------
# String-path resolution
# ---------------------------------------------------------------------------


def test_key_fn_accepts_string_path() -> None:
    primary = PairedIndexedSource(size=1)
    store = DictStore({"s0": {"label": "x"}})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=get_callable_path(sample_id_key),
    )

    sample = list(paired)[0]
    assert sample.metadata["label"] == "x"


def test_extract_fn_accepts_string_path() -> None:
    primary = PairedIndexedSource(size=1)
    store = DictStore({"s0": {"label": "x"}})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        extract_fn=get_callable_path(identity_extract),
    )

    sample = list(paired)[0]
    assert sample.metadata["label"] == "x"


def test_callable_is_stored_as_string() -> None:
    paired = PairedSource(
        primary=PairedIndexedSource(),
        secondary=DictStore(),
        key_fn=sample_id_key,
    )

    assert isinstance(paired.key_fn, str)
    assert ":sample_id_key" in paired.key_fn


# ---------------------------------------------------------------------------
# Chained / composed
# ---------------------------------------------------------------------------


def test_chained_paired_sources_compose() -> None:
    """Pack-level + window-level annotations merged via two PairedSources."""
    primary = PairedIndexedSource(size=3)
    pack_store = DictStore({"pack": {"drone": "mavic"}})
    window_store = DictStore({"s1": {"event": "takeoff"}})

    pack_paired = PairedSource(primary=primary, secondary=pack_store, key_fn=pack_key)
    full_paired = PairedSource(primary=pack_paired, secondary=window_store, key_fn=sample_id_key)

    samples = list(full_paired)

    # Every sample has drone (from pack), s1 also has event
    assert all(s.metadata["drone"] == "mavic" for s in samples)
    assert samples[1].metadata["event"] == "takeoff"
    assert "event" not in samples[0].metadata


# ---------------------------------------------------------------------------
# Confluid serialization round-trip
# ---------------------------------------------------------------------------


def test_confluid_roundtrip_preserves_behavior() -> None:
    primary = PairedIndexedSource(size=3)
    store = DictStore({"s0": {"label": "a"}, "s2": {"label": "c"}})
    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=sample_id_key,
        policy="left_outer",
        prefix="ann_",
    )

    yaml_state = confluid.dump(paired)
    restored = confluid.load(yaml_state)

    original_result = [(s.input, s.metadata.get("ann_label"), s.metadata["annotated"]) for s in paired]
    restored_result = [(s.input, s.metadata.get("ann_label"), s.metadata["annotated"]) for s in restored]

    assert original_result == restored_result
