"""Tests for DataFlux sources, in particular DatasetSplit."""

from typing import Any, Iterator, List

import confluid  # type: ignore[import-not-found]
import pytest

from dataflux.core import Flux
from dataflux.sample import Sample
from dataflux.sources import DatasetSplit


@confluid.configurable
class IndexedSource:
    """A configurable indexable source for DatasetSplit tests.

    Stores a list of integers; each `__getitem__` returns ``Sample(input=i)``.
    """

    def __init__(self, size: int = 100) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Sample:
        return Sample(input=index, target=None, metadata={"idx": index})

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.size):
            yield self[i]


# ---------------------------------------------------------------------------
# Fraction mode
# ---------------------------------------------------------------------------


def test_fraction_mode_partitions_cleanly() -> None:
    """train + val cover the full source with no overlap."""
    source = IndexedSource(size=100)
    train = DatasetSplit(source=source, split="train", val_fraction=0.1, seed=42)
    val = DatasetSplit(source=source, split="val", val_fraction=0.1, seed=42)

    train_idx = {s.input for s in train}
    val_idx = {s.input for s in val}

    assert len(train) == 90
    assert len(val) == 10
    assert train_idx.isdisjoint(val_idx)
    assert train_idx | val_idx == set(range(100))


def test_fraction_mode_is_deterministic_across_instances() -> None:
    """Same seed + source -> identical shuffle -> stable split."""
    source = IndexedSource(size=50)
    first = list(DatasetSplit(source=source, split="val", val_fraction=0.2, seed=7))
    second = list(DatasetSplit(source=source, split="val", val_fraction=0.2, seed=7))
    assert [s.input for s in first] == [s.input for s in second]


def test_fraction_mode_different_seeds_differ() -> None:
    source = IndexedSource(size=50)
    a = {s.input for s in DatasetSplit(source=source, split="val", val_fraction=0.2, seed=1)}
    b = {s.input for s in DatasetSplit(source=source, split="val", val_fraction=0.2, seed=2)}
    # Overwhelmingly likely to differ on 50 elements
    assert a != b


def test_fraction_mode_requires_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        DatasetSplit(source=IndexedSource(size=10), split="train", val_fraction=0.1)


def test_fraction_mode_requires_val_fraction() -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        DatasetSplit(source=IndexedSource(size=10), split="train", seed=0)


def test_fraction_mode_rejects_invalid_split() -> None:
    with pytest.raises(ValueError, match="split"):
        DatasetSplit(source=IndexedSource(size=10), split="test", val_fraction=0.1, seed=0)


def test_fraction_mode_rejects_out_of_range_fraction() -> None:
    src = IndexedSource(size=10)
    with pytest.raises(ValueError, match="val_fraction"):
        DatasetSplit(source=src, split="train", val_fraction=1.5, seed=0)


# ---------------------------------------------------------------------------
# Range mode
# ---------------------------------------------------------------------------


def test_range_mode_slices_source() -> None:
    source = IndexedSource(size=20)
    view = DatasetSplit(source=source, start=5, end=15)
    values = [s.input for s in view]
    assert values == list(range(5, 15))
    assert len(view) == 10


def test_range_mode_open_ended() -> None:
    source = IndexedSource(size=20)
    head = DatasetSplit(source=source, end=10)
    tail = DatasetSplit(source=source, start=10)
    assert [s.input for s in head] == list(range(10))
    assert [s.input for s in tail] == list(range(10, 20))


def test_range_mode_clamps_out_of_bounds() -> None:
    source = IndexedSource(size=5)
    view = DatasetSplit(source=source, start=-10, end=100)
    assert len(view) == 5


def test_passthrough_returns_full_source() -> None:
    source = IndexedSource(size=8)
    view = DatasetSplit(source=source)
    assert [s.input for s in view] == list(range(8))


def test_mixed_modes_rejected() -> None:
    source = IndexedSource(size=10)
    with pytest.raises(ValueError, match="not both"):
        DatasetSplit(source=source, split="train", val_fraction=0.1, seed=0, start=0, end=5)


def test_invalid_source_type() -> None:
    with pytest.raises(TypeError, match="__len__"):

        class _Plain:
            pass

        DatasetSplit(source=_Plain())


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_getitem_resolves_through_underlying_source() -> None:
    source = IndexedSource(size=30)
    view = DatasetSplit(source=source, start=10, end=20)
    # First element of the view is index 10 of the source
    assert view[0].input == 10
    assert view[-1].input == 19  # Python list indexing supports negatives


# ---------------------------------------------------------------------------
# Works inside a Flux pipeline
# ---------------------------------------------------------------------------


def _scale(value: int, factor: int = 1) -> int:
    return value * factor


def test_dataset_split_inside_flux() -> None:
    source = IndexedSource(size=20)
    view = DatasetSplit(source=source, start=0, end=5)
    flux = Flux(source=view).map(_scale, factor=10)
    results: List[Sample] = flux.collect()
    assert [s.input for s in results] == [0, 10, 20, 30, 40]


# ---------------------------------------------------------------------------
# Confluid serialization round-trip
# ---------------------------------------------------------------------------


def test_serialization_roundtrip_preserves_split() -> None:
    source = IndexedSource(size=40)
    split = DatasetSplit(source=source, split="val", val_fraction=0.25, seed=123)
    yaml_state = confluid.dump(split)
    assert "!class:DatasetSplit" in yaml_state
    assert "val_fraction: 0.25" in yaml_state
    assert "seed: 123" in yaml_state

    restored: Any = confluid.load(yaml_state)
    # Reloaded source is a fresh IndexedSource with size=40 — feed it identically.
    assert len(restored) == len(split)
    assert [s.input for s in restored] == [s.input for s in split]


def test_ref_based_splits_share_source_and_partition_cleanly() -> None:
    """Two DatasetSplits referencing the same YAML source via ``!ref:`` share
    the underlying source by identity (single load) and produce disjoint,
    complementary views."""
    yaml_state = """
hf_train: !class:IndexedSource()
  size: 50

train_set: !class:DatasetSplit()
  source: !ref:hf_train
  split: train
  val_fraction: 0.2
  seed: 9

val_set: !class:DatasetSplit()
  source: !ref:hf_train
  split: val
  val_fraction: 0.2
  seed: 9
"""
    state: Any = confluid.load(yaml_state)
    train = state["train_set"]
    val = state["val_set"]

    # Confluid !ref: resolves to the same live object — critical for
    # expensive sources like HuggingFaceSource so the dataset loads once.
    assert train.source is val.source
    assert train.source is state["hf_train"]

    train_idx = {s.input for s in train}
    val_idx = {s.input for s in val}
    assert train_idx.isdisjoint(val_idx)
    assert train_idx | val_idx == set(range(50))
