"""``ensure_record_dataset`` / ``RecordSource`` — normalizing a wired dataset slot.

A consumer's ``train_set`` / ``val_set`` / ``test_set`` may be wired to a ``Stream``, another
torch ``Dataset``, a bare source, or a plain list. Normalizing once up front lets the rest of a
pipeline assume record items unconditionally.
"""

from typing import Any, Dict, Iterator, List

import numpy as np
import torch

from recordstream import Label, Record, Stream, ensure_materialized, ensure_record_dataset


class _RowDataset(torch.utils.data.Dataset):
    """A map-style dataset of raw (non-record) rows."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


class _IterableSource:
    """A bare iterable source — no ``__getitem__``, no ``Dataset`` base."""

    def __iter__(self) -> Iterator[Record]:
        yield {"image": np.zeros(2), "class": Label(0)}
        yield {"image": np.ones(2), "class": Label(1)}


def _records(n: int = 2) -> List[Record]:
    return [{"image": np.zeros(2), "class": Label(i)} for i in range(n)]


def test_a_stream_is_returned_as_is() -> None:
    """Identity matters: a subclass's own wrap (e.g. its ``class_names``) must survive."""
    stream = Stream(source=_records())
    assert ensure_record_dataset(stream) is stream


def test_a_stream_subclass_keeps_its_attributes() -> None:
    stream = Stream(source=_records())
    stream.class_names = ["a", "b"]  # type: ignore[attr-defined]
    assert ensure_record_dataset(stream).class_names == ["a", "b"]  # type: ignore[attr-defined]


def test_a_plain_list_becomes_a_map_style_record_dataset() -> None:
    dataset = ensure_record_dataset(_records(3))
    assert isinstance(dataset, Stream)
    assert len(dataset) == 3  # type: ignore[arg-type]
    assert isinstance(dataset[0], dict) and "image" in dataset[0]  # type: ignore[index]


def test_a_torch_dataset_of_rows_is_wrapped() -> None:
    dataset = ensure_record_dataset(_RowDataset(_records(2)))
    assert isinstance(dataset, Stream)
    assert len(dataset) == 2  # type: ignore[arg-type]


def test_a_bare_iterable_source_is_wrapped_and_iterates() -> None:
    dataset = ensure_record_dataset(_IterableSource())
    assert isinstance(dataset, Stream)
    assert [int(record["class"].value) for record in dataset] == [0, 1]  # type: ignore[union-attr]


def test_it_is_idempotent() -> None:
    once = ensure_record_dataset(_records())
    assert ensure_record_dataset(once) is once


def test_exported_from_the_package_root() -> None:
    import recordstream

    assert "ensure_record_dataset" in recordstream.__all__
    assert "RecordSource" in recordstream.__all__


# ---------------------------------------------------------------------------
# ensure_materialized — normalize a source's STATE, not its type
# ---------------------------------------------------------------------------
class _LazySource:
    """A source that does its real work on first read, like every source in this package."""

    def __init__(self, rows: int = 3) -> None:
        self.rows = rows
        self.reads = 0
        self.built = False

    def __len__(self) -> int:
        return self.rows  # deliberately does NOT build: len() was measured not to be enough

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index >= self.rows:
            raise IndexError(index)
        self.built = True
        self.reads += 1
        return {"image": index}


def test_ensure_materialized_builds_what_a_read_needs() -> None:
    """The point: after this, a forked child inherits a source that needs nothing."""
    source = _LazySource()

    returned = ensure_materialized(source)

    assert source.built, "the source was never actually read"
    assert returned is source, "it returns the source so it composes"


def test_len_is_not_enough_which_is_why_this_reads_a_record() -> None:
    """Pins the measurement the docstring cites, so the implementation cannot be 'simplified'.

    Loading a dataset object is not the same as building everything a read needs — a real
    HuggingFaceSource still went to the Hub from inside a worker after `len()` in the parent.
    """
    source = _LazySource()

    len(source)

    assert not source.built, "if len() built it, this test's premise is wrong, not the code"


def test_it_reads_exactly_one_record() -> None:
    """One is enough, and more would make warming a large split expensive."""
    source = _LazySource(rows=100)

    ensure_materialized(source)

    assert source.reads == 1


def test_an_empty_source_is_not_an_error() -> None:
    """A split with no rows has nothing to build, and a caller should need no guard."""
    assert ensure_materialized(_LazySource(rows=0)) is not None


def test_an_iterable_only_source_is_warmed_too() -> None:
    """Not every source is map-style; the iterable path must build the same way."""

    class _IterableOnly:
        def __init__(self) -> None:
            self.built = False

        def __iter__(self) -> Iterator[Dict[str, Any]]:
            self.built = True
            yield {"image": 0}

    source = _IterableOnly()
    ensure_materialized(source)
    assert source.built


# --------------------------------------------------------------------------- #
# prepare_record_dataset — the TYPE + STATE composition
# --------------------------------------------------------------------------- #


def test_prepare_none_passes_through() -> None:
    """An unwired optional split needs no guard at the call site."""
    from recordstream import prepare_record_dataset

    assert prepare_record_dataset(None) is None


def test_prepare_wraps_and_warms_in_one_call() -> None:
    """The composition IS ensure_record_dataset + ensure_materialized: a lazy source comes
    back as a record Stream whose build already happened in THIS process."""
    from recordstream import Stream, prepare_record_dataset

    source = _LazySource(rows=2)
    prepared = prepare_record_dataset(source)
    assert isinstance(prepared, Stream)
    assert source.built, "the lazy build must happen here, not in a forked worker"


def test_prepare_returns_a_stream_as_the_same_stream() -> None:
    """Identity matters: a label-encoding Stream keeps its class_names."""
    from recordstream import Stream, prepare_record_dataset

    stream = Stream(source=[{"class": 1}])
    assert prepare_record_dataset(stream) is stream


def test_prepare_is_exported_from_the_package_root() -> None:
    import recordstream

    assert "prepare_record_dataset" in recordstream.__all__
