"""Dataset identity — ``recordstream.uri`` and ``HuggingFaceSource``'s implementation of it."""

from pathlib import Path
from typing import Any, List, Optional

import pytest

from recordstream import (
    ConcatSource,
    DatasetSplit,
    HuggingFaceSource,
    RangeSource,
    Stream,
    SupportsDatasetIdentity,
    dataset_uri,
    dataset_uris,
    dataset_url,
)
from recordstream.uri import MAX_WRAPPER_DEPTH


class _FakeSource:
    """A minimal indexable source, so a view can wrap something with a known length."""

    def __init__(self, size: int = 6, uri: Optional[str] = None) -> None:
        self._size = size
        self.dataset_uri = uri  # type: ignore[assignment]
        self.dataset_url = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Any:
        return {"class": index}

    def __iter__(self) -> Any:
        return iter(self[i] for i in range(self._size))


# --- HuggingFaceSource's identity ------------------------------------------------------------


def test_a_hub_repo_id_becomes_an_hf_uri_and_a_viewer_url() -> None:
    source = HuggingFaceSource(path="ylecun/mnist", split="train")
    assert source.dataset_uri == "hf://datasets/ylecun/mnist?split=train"
    assert source.dataset_url == "https://huggingface.co/datasets/ylecun/mnist/viewer/default/train"


def test_config_name_and_revision_ride_the_uri_and_the_url() -> None:
    source = HuggingFaceSource(path="ylecun/mnist", name="fashion", split="test", revision="abc123")
    assert source.dataset_uri == "hf://datasets/ylecun/mnist?name=fashion&revision=abc123&split=test"
    assert source.dataset_url == "https://huggingface.co/datasets/ylecun/mnist/viewer/fashion/test"


def test_query_parameters_are_sorted_so_one_config_has_exactly_one_uri() -> None:
    """Two identically-configured sources must produce the SAME string, whatever the kwarg order."""
    one = HuggingFaceSource(path="a/b", split="train", name="cfg", revision="r1")
    other = HuggingFaceSource(path="a/b", revision="r1", name="cfg", split="train")
    assert one.dataset_uri == other.dataset_uri


def test_a_local_directory_becomes_a_file_uri_with_no_browsable_url(tmp_path: Path) -> None:
    source = HuggingFaceSource(path=str(tmp_path), split="train")
    assert source.dataset_uri == f"{tmp_path.resolve().as_uri()}?split=train"
    # Data on disk has no web page, and inventing a Hub URL for it would be a lie.
    assert source.dataset_url is None


def test_an_unconfigured_source_has_no_identity() -> None:
    source = HuggingFaceSource()
    assert source.dataset_uri is None
    assert source.dataset_url is None


def test_a_source_without_a_split_links_to_the_dataset_page() -> None:
    source = HuggingFaceSource(path="ylecun/mnist", split="")
    assert source.dataset_uri == "hf://datasets/ylecun/mnist"
    assert source.dataset_url == "https://huggingface.co/datasets/ylecun/mnist"


def test_asking_for_identity_never_loads_the_dataset() -> None:
    """The whole point of reading stored config: a source that is never iterated still answers."""
    source = HuggingFaceSource(path="ylecun/mnist", split="train")
    assert source.dataset_uri is not None
    assert source._dataset is None  # nothing was materialized


def test_huggingface_source_satisfies_the_protocol() -> None:
    assert isinstance(HuggingFaceSource(path="a/b"), SupportsDatasetIdentity)


# --- the free functions ----------------------------------------------------------------------


def test_the_free_functions_read_a_sources_own_identity() -> None:
    source = HuggingFaceSource(path="ylecun/mnist", split="train")
    assert dataset_uri(source) == source.dataset_uri
    assert dataset_url(source) == source.dataset_url


def test_none_and_an_identity_less_source_answer_none() -> None:
    assert dataset_uri(None) is None
    assert dataset_url(None) is None
    assert dataset_uri(_FakeSource()) is None


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda src: Stream(source=src), id="stream"),
        pytest.param(lambda src: RangeSource(source=src, start=0, stop=2), id="range"),
        pytest.param(lambda src: DatasetSplit(source=src, val_fraction=0.5, seed=0), id="split"),
        pytest.param(lambda src: Stream(source=RangeSource(source=src, start=0, stop=2)), id="nested"),
    ],
)
def test_a_wrapper_reports_the_wrapped_datasets_identity_verbatim(wrap: Any) -> None:
    """A view identifies the same DATASET; how much of it this run reads is a separate fact."""
    source = HuggingFaceSource(path="ylecun/mnist", split="train")
    wrapped = wrap(source)
    assert dataset_uri(wrapped) == "hf://datasets/ylecun/mnist?split=train"
    assert dataset_url(wrapped) == "https://huggingface.co/datasets/ylecun/mnist/viewer/default/train"


def test_a_split_view_reports_the_split_sources_identity() -> None:
    """The three views are private objects only ever read off a live split — they follow too.

    Wrapping a local fake rather than a Hub source on purpose: reading ``.train`` PARTITIONS,
    which calls ``len(source)``, which would make a Hub source download.
    """
    inner = _FakeSource(size=8, uri="hf://datasets/a/b?split=train")
    split = DatasetSplit(source=inner, val_fraction=0.25, seed=0)
    assert dataset_uri(split.train) == "hf://datasets/a/b?split=train"
    assert dataset_uri(split.val) == dataset_uri(split.train)


def test_a_wrapper_with_its_own_identity_wins_over_the_one_it_wraps() -> None:
    inner = HuggingFaceSource(path="ylecun/mnist", split="train")
    outer = _FakeSource(uri="store://curated/v2")
    outer.source = inner  # type: ignore[attr-defined]
    assert dataset_uri(outer) == "store://curated/v2"


def test_a_cyclic_wrapper_chain_terminates() -> None:
    """The depth cap exists so a tracking call can never hang on a self-referential source."""
    node = _FakeSource()
    node.source = node  # type: ignore[attr-defined]
    assert dataset_uri(node) is None


def test_a_chain_deeper_than_the_cap_gives_up_rather_than_walking_forever() -> None:
    deepest = HuggingFaceSource(path="a/b", split="train")
    chain: Any = deepest
    for _ in range(MAX_WRAPPER_DEPTH + 2):
        wrapper = _FakeSource()
        wrapper.source = chain  # type: ignore[attr-defined]
        chain = wrapper
    assert dataset_uri(chain) is None


def test_a_deferred_config_marker_is_materialized_before_being_asked() -> None:
    """A `!class:` marker straight out of a config must not need a `flow()` at the call site."""
    from confluid import load

    node = load(
        "!class:recordstream.sources.huggingface.HuggingFaceSource()\n  path: ylecun/mnist\n  split: train\n",
        flow=False,
    )
    assert dataset_uri(node) == "hf://datasets/ylecun/mnist?split=train"


# --- concatenation ---------------------------------------------------------------------------


def test_a_concatenation_declines_to_name_one_dataset() -> None:
    concat = ConcatSource(sources=[HuggingFaceSource(path="a/b"), HuggingFaceSource(path="c/d")])
    assert dataset_uri(concat) is None
    assert dataset_url(concat) is None


def test_dataset_uris_fans_out_over_the_members() -> None:
    concat = ConcatSource(sources=[HuggingFaceSource(path="a/b"), HuggingFaceSource(path="c/d")])
    assert dataset_uris(concat) == ["hf://datasets/a/b?split=train", "hf://datasets/c/d?split=train"]


def test_dataset_uris_deduplicates_and_descends_into_nested_concatenations() -> None:
    same = HuggingFaceSource(path="a/b")
    inner = ConcatSource(sources=[same, HuggingFaceSource(path="c/d")])
    outer = ConcatSource(sources=[inner, HuggingFaceSource(path="a/b")])
    assert dataset_uris(outer) == ["hf://datasets/a/b?split=train", "hf://datasets/c/d?split=train"]


def test_dataset_uris_of_a_single_source_is_its_one_uri() -> None:
    source = HuggingFaceSource(path="ylecun/mnist", split="train")
    assert dataset_uris(source) == ["hf://datasets/ylecun/mnist?split=train"]


def test_dataset_uris_skips_members_that_have_no_identity() -> None:
    concat = ConcatSource(sources=[_FakeSource(), HuggingFaceSource(path="a/b")])
    found: List[str] = dataset_uris(concat)
    assert found == ["hf://datasets/a/b?split=train"]
