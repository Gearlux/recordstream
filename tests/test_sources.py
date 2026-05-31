"""Tests for DataFlux sources: DatasetSplit (+ cached split views), RangeSource, ConcatSource."""

import inspect
from typing import Any, Iterator, List

import confluid  # type: ignore[import-not-found]
import pytest

from dataflux.core import Flux
from dataflux.sample import Sample
from dataflux.sources import ConcatSource, DatasetSplit, RangeSource


@confluid.configurable
class IndexedSource:
    """A configurable indexable source for the tests.

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
# DatasetSplit — select-one API (split=)
# ---------------------------------------------------------------------------


def test_fraction_mode_partitions_cleanly() -> None:
    """split='train' + split='val' cover the full source with no overlap."""
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
    split = DatasetSplit(source=IndexedSource(size=10), split="train", val_fraction=0.1)  # lazy ctor
    with pytest.raises(ValueError, match="seed"):
        _ = split.train


def test_fraction_mode_rejects_invalid_split() -> None:
    with pytest.raises(ValueError, match="split"):
        DatasetSplit(source=IndexedSource(size=10), split="holdout", val_fraction=0.1, seed=0)  # type: ignore[arg-type]


def test_fraction_mode_rejects_out_of_range_fraction() -> None:
    src = IndexedSource(size=10)
    split = DatasetSplit(source=src, split="train", val_fraction=1.5, seed=0)  # lazy ctor
    with pytest.raises(ValueError, match="val_fraction"):
        _ = split.train


# ---------------------------------------------------------------------------
# DatasetSplit — three-way (select-one)
# ---------------------------------------------------------------------------


def test_three_way_split_partitions_cleanly() -> None:
    """split='train'/'val'/'test' cover the full source with no overlap."""
    source = IndexedSource(size=100)
    kw = dict(source=source, val_fraction=0.2, test_fraction=0.1, seed=42)
    train = DatasetSplit(split="train", **kw)  # type: ignore[arg-type]
    val = DatasetSplit(split="val", **kw)  # type: ignore[arg-type]
    test = DatasetSplit(split="test", **kw)  # type: ignore[arg-type]

    train_idx = {s.input for s in train}
    val_idx = {s.input for s in val}
    test_idx = {s.input for s in test}

    assert len(val) == 20
    assert len(test) == 10
    assert len(train) == 70
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx | val_idx | test_idx == set(range(100))


def test_test_fraction_out_of_range_rejected() -> None:
    split = DatasetSplit(source=IndexedSource(size=10), split="test", test_fraction=1.5, seed=0)  # lazy ctor
    with pytest.raises(ValueError, match="test_fraction"):
        _ = split.test


def test_val_plus_test_fraction_must_be_under_one() -> None:
    split = DatasetSplit(source=IndexedSource(size=10), split="train", val_fraction=0.6, test_fraction=0.5, seed=0)
    with pytest.raises(ValueError, match="must be < 1"):
        _ = split.train


# ---------------------------------------------------------------------------
# DatasetSplit — cached property API (.train / .val / .test)
# ---------------------------------------------------------------------------


def test_property_api_partitions_cleanly() -> None:
    """A single DatasetSplit exposes the three disjoint, complementary views."""
    source = IndexedSource(size=100)
    split = DatasetSplit(source=source, val_fraction=0.2, test_fraction=0.1, seed=42)

    train_idx = {s.input for s in split.train}
    val_idx = {s.input for s in split.val}
    test_idx = {s.input for s in split.test}

    assert len(split.train) == 70 and len(split.val) == 20 and len(split.test) == 10
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx | val_idx | test_idx == set(range(100))


def test_property_views_are_cached() -> None:
    split = DatasetSplit(source=IndexedSource(size=20), val_fraction=0.25, seed=1)
    assert split.train is split.train  # memoized — same object each access
    assert split.val is split.val
    assert split.test is split.test


def test_property_and_select_one_agree() -> None:
    """``split='val'`` (select-one) yields the same indices as the ``.val`` property."""
    source = IndexedSource(size=40)
    selected = DatasetSplit(source=source, split="val", val_fraction=0.25, seed=3)
    split = DatasetSplit(source=source, val_fraction=0.25, seed=3)
    assert [s.input for s in selected] == [s.input for s in split.val]


def test_no_fractions_train_is_full_val_test_empty() -> None:
    """No fractions (and no seed needed) → train is the whole source (unshuffled), val/test empty."""
    source = IndexedSource(size=8)
    split = DatasetSplit(source=source)
    assert [s.input for s in split.train] == list(range(8))
    assert len(split.val) == 0
    assert len(split.test) == 0


def test_default_iteration_is_train() -> None:
    """Iterating a DatasetSplit with no ``split`` delegates to the ``train`` view."""
    split = DatasetSplit(source=IndexedSource(size=10), val_fraction=0.2, seed=1)
    assert [s.input for s in split] == [s.input for s in split.train]


def test_datasetsplit_dropped_range_params() -> None:
    """Range mode moved to RangeSource — DatasetSplit no longer accepts start/end."""
    params = set(inspect.signature(DatasetSplit).parameters)
    assert {"source", "split", "val_fraction", "test_fraction", "seed"} == params


def test_invalid_source_type() -> None:
    class _Plain:
        pass

    split = DatasetSplit(source=_Plain())  # lazy: construction succeeds
    with pytest.raises(TypeError, match="__len__"):
        _ = split.train


# ---------------------------------------------------------------------------
# RangeSource (the extracted contiguous-slice mode)
# ---------------------------------------------------------------------------


def test_range_source_slices() -> None:
    source = IndexedSource(size=20)
    view = RangeSource(source=source, start=5, end=15)
    assert [s.input for s in view] == list(range(5, 15))
    assert len(view) == 10


def test_range_source_open_ended() -> None:
    source = IndexedSource(size=20)
    head = RangeSource(source=source, end=10)
    tail = RangeSource(source=source, start=10)
    assert [s.input for s in head] == list(range(10))
    assert [s.input for s in tail] == list(range(10, 20))


def test_range_source_clamps_out_of_bounds() -> None:
    view = RangeSource(source=IndexedSource(size=5), start=-10, end=100)
    assert len(view) == 5


def test_range_source_getitem_resolves_through_underlying_source() -> None:
    view = RangeSource(source=IndexedSource(size=30), start=10, end=20)
    assert view[0].input == 10
    assert view[-1].input == 19  # Python list indexing supports negatives


def test_range_source_invalid_source_type() -> None:
    class _Plain:
        pass

    rng = RangeSource(source=_Plain())  # lazy: construction succeeds
    with pytest.raises(TypeError, match="__len__"):
        len(rng)


def _scale(value: int, factor: int = 1) -> int:
    return value * factor


def test_range_source_inside_flux() -> None:
    view = RangeSource(source=IndexedSource(size=20), start=0, end=5)
    flux = Flux(source=view).map(_scale, factor=10)
    results: List[Sample] = flux.collect()
    assert [s.input for s in results] == [0, 10, 20, 30, 40]


# ---------------------------------------------------------------------------
# ConcatSource (indexable join of multiple sources)
# ---------------------------------------------------------------------------


def test_concat_source_len_and_iter() -> None:
    cat = ConcatSource(sources=[IndexedSource(size=3), IndexedSource(size=2)])
    assert len(cat) == 5
    # Each sub-source yields its own 0..n-1 inputs, walked in order.
    assert [s.input for s in cat] == [0, 1, 2, 0, 1]


def test_concat_source_getitem_maps_to_subsource() -> None:
    cat = ConcatSource(sources=[IndexedSource(size=3), IndexedSource(size=2)])
    assert [cat[i].input for i in range(5)] == [0, 1, 2, 0, 1]  # 3 from src0, 2 from src1
    assert cat[-1].input == 1  # last item of src1


def test_concat_source_out_of_bounds() -> None:
    cat = ConcatSource(sources=[IndexedSource(size=3)])
    with pytest.raises(IndexError):
        cat[3]


def test_concat_source_empty() -> None:
    cat = ConcatSource(sources=[])
    assert len(cat) == 0
    with pytest.raises(IndexError):
        cat[0]


def test_concat_source_rejects_non_indexable() -> None:
    cat = ConcatSource(sources=[object()])  # lazy: construction succeeds
    with pytest.raises(TypeError, match="__len__"):
        len(cat)


def test_concat_source_is_splittable() -> None:
    """A ConcatSource is indexable, so DatasetSplit can partition the joined sources."""
    cat = ConcatSource(sources=[IndexedSource(size=30), IndexedSource(size=20)])
    split = DatasetSplit(source=cat, val_fraction=0.2, seed=1)
    assert len(split.train) == 40 and len(split.val) == 10
    assert len(split.train) + len(split.val) == 50


# ---------------------------------------------------------------------------
# Confluid serialization round-trips
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


def test_serialization_roundtrip_preserves_three_way_split() -> None:
    source = IndexedSource(size=60)
    split = DatasetSplit(source=source, split="test", val_fraction=0.2, test_fraction=0.1, seed=5)
    yaml_state = confluid.dump(split)
    assert "!class:DatasetSplit" in yaml_state
    assert "test_fraction: 0.1" in yaml_state

    restored: Any = confluid.load(yaml_state)
    assert len(restored) == len(split)
    assert [s.input for s in restored] == [s.input for s in split]


def test_property_split_roundtrip_via_views() -> None:
    """A property-style DatasetSplit (no ``split``) round-trips; views recompute identically."""
    source = IndexedSource(size=40)
    split = DatasetSplit(source=source, val_fraction=0.25, seed=123)
    yaml_state = confluid.dump(split)
    assert "!class:DatasetSplit" in yaml_state
    restored: Any = confluid.load(yaml_state)
    assert [s.input for s in restored.val] == [s.input for s in split.val]
    assert [s.input for s in restored.train] == [s.input for s in split.train]


def test_property_refs_share_one_instance_and_partition_cleanly() -> None:
    """ONE DatasetSplit referenced via ``!ref:my_split.train`` / ``.val`` — the new pattern.

    Both attribute-refs resolve from the SAME flowed DatasetSplit (Confluid's dotted-ref now reuses
    the materialized instance — see confluid ``test_dotted_attribute_ref_reuses_single_instance``),
    so the cached views share the single underlying source — it is the document's ``hf`` instance,
    loaded exactly once — and form a disjoint, complementary partition.
    """
    yaml_state = """
hf: !class:IndexedSource()
  size: 50

my_split: !class:DatasetSplit()
  source: !ref:hf
  val_fraction: 0.2
  seed: 9

train_set: !class:dataflux.core.Flux()
  source: !ref:my_split.train

val_set: !class:dataflux.core.Flux()
  source: !ref:my_split.val
"""
    state: Any = confluid.load(yaml_state)
    train_flux = state["train_set"]
    val_flux = state["val_set"]

    # Both Flux sources are views off the SAME DatasetSplit, wrapping the SINGLE ``hf`` instance
    # (one load), and the shared split's cached property IS the view the Flux received.
    assert train_flux.source.source is val_flux.source.source
    assert train_flux.source.source is state["hf"]
    assert state["my_split"].train is train_flux.source

    train_idx = {s.input for s in train_flux}
    val_idx = {s.input for s in val_flux}
    assert len(val_idx) == 10
    assert train_idx.isdisjoint(val_idx)
    assert train_idx | val_idx == set(range(50))


def test_select_one_refs_share_source_and_partition_cleanly() -> None:
    """Select-one pattern: two DatasetSplits over a shared ``!ref:source`` — a single load.

    Both ``!ref:hf_train`` resolve (by Confluid's instance memo) to the SAME source instance, so
    the source is materialized exactly once and the two views partition it cleanly. This is the
    load-once-guaranteed pattern when a single shared instance matters (e.g. ``HuggingFaceSource``).
    """
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

    assert train.source is val.source
    assert train.source is state["hf_train"]

    train_idx = {s.input for s in train}
    val_idx = {s.input for s in val}
    assert train_idx.isdisjoint(val_idx)
    assert train_idx | val_idx == set(range(50))


def test_concat_source_roundtrip() -> None:
    cat = ConcatSource(sources=[IndexedSource(size=3), IndexedSource(size=4)])
    yaml_state = confluid.dump(cat)
    assert "!class:ConcatSource" in yaml_state
    restored: Any = confluid.load(yaml_state)
    assert len(restored) == 7
    assert [s.input for s in restored] == [s.input for s in cat]


# ---------------------------------------------------------------------------
# HuggingFaceSource lazy / zero-arg construction (no network in __init__)
# ---------------------------------------------------------------------------


def test_hf_source_zero_arg_construction_does_no_work() -> None:
    # Per the lazy / zero-arg convention: building the source must not touch the network and
    # must succeed with no constructor arguments. Nothing is materialized until first use.
    from dataflux.sources import HuggingFaceSource

    src = HuggingFaceSource()
    assert src._dataset is None  # nothing loaded at construction time
    # Even a fully-configured source stays unmaterialized until the dataset is accessed.
    configured = HuggingFaceSource(path="some/dataset", split="test", count=7)
    assert configured._dataset is None
    assert configured.path == "some/dataset" and configured.split == "test" and configured.count == 7


def test_hf_source_dataset_without_path_raises() -> None:
    # The zero-arg constructor allows an unconfigured source, but materializing one without a
    # dataset id cannot succeed — the error surfaces lazily, at the `dataset` property, not in __init__.
    from dataflux.sources import HuggingFaceSource

    src = HuggingFaceSource()
    with pytest.raises(ValueError, match="path is empty"):
        _ = src.dataset


# ---------------------------------------------------------------------------
# HuggingFaceSource.__len__ / count semantics (lazy `_dataset` pre-seeded, no network)
# ---------------------------------------------------------------------------


def _hf_source_with_count(count: Any, dataset_len: int = 13) -> Any:
    """Build a HuggingFaceSource and pre-seed its lazy cache so `dataset` never hits the network."""
    from dataflux.sources import HuggingFaceSource

    src: Any = HuggingFaceSource(count=count)
    src._dataset = list(range(dataset_len))  # short-circuits the lazy load in the `dataset` property
    return src


def test_hf_source_len_count_zero_means_all() -> None:
    # Regression: count=0 must report the full length (matching __iter__'s ``count or len``),
    # not 0 — otherwise a len()-based stepper (e.g. FluxStudio's WalkDataset) sees an empty
    # source even though iteration yields every sample.
    assert len(_hf_source_with_count(0)) == 13


def test_hf_source_len_count_none_means_all() -> None:
    assert len(_hf_source_with_count(None)) == 13


def test_hf_source_len_positive_count_caps() -> None:
    assert len(_hf_source_with_count(5)) == 5


# ---------------------------------------------------------------------------
# HuggingFaceSource.metadata_features resolution ("*" sentinel = the rest)
# ---------------------------------------------------------------------------


def test_resolve_metadata_features_none_and_empty_mean_none() -> None:
    from dataflux.sources import _resolve_metadata_features

    cols = ["image", "label", "id", "source_file"]
    assert _resolve_metadata_features(None, cols, "image", "label") == []
    assert _resolve_metadata_features([], cols, "image", "label") == []


def test_resolve_metadata_features_explicit_list_verbatim() -> None:
    from dataflux.sources import _resolve_metadata_features

    cols = ["image", "label", "id", "source_file"]
    assert _resolve_metadata_features(["id"], cols, "image", "label") == ["id"]
    # used verbatim — names need not exist in column_names (caller's choice)
    assert _resolve_metadata_features(["id", "extra"], cols, "image", "label") == ["id", "extra"]


def test_resolve_metadata_features_star_is_the_rest() -> None:
    from dataflux.sources import _resolve_metadata_features

    cols = ["image", "label", "id", "source_file"]
    # the rest = every column except input/target, order preserved
    assert _resolve_metadata_features(["*"], cols, "image", "label") == ["id", "source_file"]
    # bare string form accepted (YAML users may write `metadata_features: "*"`)
    assert _resolve_metadata_features("*", cols, "image", "label") == ["id", "source_file"]


def test_resolve_metadata_features_star_plus_extras_union() -> None:
    from dataflux.sources import _resolve_metadata_features

    cols = ["image", "label", "id"]
    # "*" plus a name already in the rest -> no duplicate; an out-of-columns extra is appended
    assert _resolve_metadata_features(["*", "id", "note"], cols, "image", "label") == ["id", "note"]


def test_resolve_metadata_features_star_without_columns_degrades() -> None:
    from dataflux.sources import _resolve_metadata_features

    # No column_names available (e.g. a non-Dataset backing) -> "*" yields just the extras.
    assert _resolve_metadata_features(["*"], None, "image", "label") == []
    assert _resolve_metadata_features(["*", "note"], None, "image", "label") == ["note"]


class _StubHFDataset(list):
    """A list of row-dicts that also exposes ``column_names`` like a real ``datasets.Dataset``.

    Lets the lazy ``resolved_metadata_features`` property expand the ``"*"`` sentinel against the
    backing columns without touching the network — iterable + indexable + ``len``-able for free.
    """

    def __init__(self, rows: List[Any], column_names: List[str]) -> None:
        super().__init__(rows)
        self.column_names = column_names


def test_hf_source_iter_metadata_features_star_expands_on_real_dataset() -> None:
    # End-to-end through __iter__: a dataset with extra columns + metadata_features="*" carries
    # every non-input/target column onto Sample.metadata (plus the synthetic hf_path/hf_split).
    # The "*" expansion is now lazy (resolved_metadata_features reads dataset.column_names).
    from dataflux.sources import HuggingFaceSource

    rows = [{"image": i, "label": i % 2, "id": f"r{i}", "src": "a"} for i in range(3)]
    src = HuggingFaceSource(path="fake/ds", split="train", metadata_features=["*"])
    src._dataset = _StubHFDataset(rows, ["image", "label", "id", "src"])  # pre-seed: no network

    samples = list(src)
    assert [s.input for s in samples] == [0, 1, 2]
    md = samples[0].metadata
    assert md["id"] == "r0" and md["src"] == "a"
    assert "image" not in md and "label" not in md  # input/target excluded from metadata
    assert md["hf_path"] == "fake/ds" and md["hf_split"] == "train"
