"""``ensure_record_dataset`` / ``RecordSource`` — normalizing a wired dataset slot.

A consumer's ``train_set`` / ``val_set`` / ``test_set`` may be wired to a ``Stream``, another
torch ``Dataset``, a bare source, or a plain list. Normalizing once up front lets the rest of a
pipeline assume record items unconditionally.
"""

from typing import Any, Dict, Iterator, List

import numpy as np
import torch

from recordstream import Label, Record, Stream, ensure_record_dataset


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
    """Identity matters: a subclass's own wrap (e.g. its ``label_names``) must survive."""
    stream = Stream(source=_records())
    assert ensure_record_dataset(stream) is stream


def test_a_stream_subclass_keeps_its_attributes() -> None:
    stream = Stream(source=_records())
    stream.label_names = ["a", "b"]  # type: ignore[attr-defined]
    assert ensure_record_dataset(stream).label_names == ["a", "b"]  # type: ignore[attr-defined]


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
