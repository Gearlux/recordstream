"""Reading a batched record back (``recordstream.batch``) — the inverse of ``collate_records``.

These pin the CONVENTION, not a task: the helpers answer "what did the collate put under this
key?" and hand back values / one tensor / per-record dicts. Shaping for a particular loss is
the caller's job, so there is deliberately no dtype promotion or multi-hot test here.
"""

import numpy as np
import pytest
import torch

from recordstream import (
    Image,
    Label,
    Mask,
    MultiLabel,
    Record,
    batch_metadata,
    batch_tensor,
    batch_values,
    collate_records,
    multi_hot,
)

# --------------------------------------------------------------------------- #
# batch_values — getting past the wrapper item
# --------------------------------------------------------------------------- #


def _chw(size: int = 4) -> np.ndarray:
    return np.random.rand(3, size, size).astype("float32")


def test_label_values_come_back_as_the_per_record_list() -> None:
    """A wrapper item's payload is NOT stacked by the collate — it stays a list."""
    batch = collate_records([{"class": Label(i % 2)} for i in range(4)])
    assert batch_values(batch, "class") == [0, 1, 0, 1]


def test_multilabel_values_come_back_as_a_list_of_lists() -> None:
    batch = collate_records([{"class": MultiLabel([0, 2])}, {"class": MultiLabel([1])}])
    assert batch_values(batch, "class") == [[0, 2], [1]]


def test_array_item_values_come_back_as_the_stacked_payload() -> None:
    batch = collate_records([{"image": Image(_chw(), layout="CHW")} for _ in range(3)])
    values = batch_values(batch, "image")
    assert getattr(values, "shape", None) == (3, 3, 4, 4)


def test_plain_values_come_back_as_the_gathered_list() -> None:
    batch = collate_records([{"idx": i} for i in range(3)])
    assert list(batch_values(batch, "idx")) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# batch_tensor — one tensor, whichever shape the collate left
# --------------------------------------------------------------------------- #


def test_stacked_array_payload_becomes_a_tensor() -> None:
    batch = collate_records([{"image": Image(_chw(8), layout="CHW")} for _ in range(4)])
    x = batch_tensor(batch, "image")
    assert isinstance(x, torch.Tensor) and x.shape == (4, 3, 8, 8)


def test_per_record_list_is_stacked() -> None:
    """The Label path: a list of scalars becomes an [N] tensor."""
    batch = collate_records([{"class": Label(i % 3)} for i in range(6)])
    y = batch_tensor(batch, "class")
    assert y.shape == (6,) and y.tolist() == [0, 1, 2, 0, 1, 2]


def test_a_prestacked_tensor_is_used_verbatim() -> None:
    """A hand-built batch (tests, a custom collate) must not be re-stacked."""
    pinned = torch.arange(4)
    assert batch_tensor({"class": Label(value=pinned)}, "class") is pinned


def test_mask_item_stacks_without_a_dtype_opinion() -> None:
    """No int64 promotion here — that is the segmenter's model boundary, not the convention."""
    batch = collate_records([{"target": Mask(np.zeros((4, 4), dtype="uint8"))} for _ in range(2)])
    assert batch_tensor(batch, "target").dtype is torch.uint8


def test_device_moves_the_result() -> None:
    batch = collate_records([{"class": Label(i)} for i in range(2)])
    assert batch_tensor(batch, "class", device="cpu").device.type == "cpu"


def test_dtype_is_honoured() -> None:
    """`dtype` is a PARAMETER, not an opinion — the caller names its loss's contract."""
    batch = collate_records([{"class": Label(torch.tensor(i, dtype=torch.int32))} for i in range(3)])

    assert batch_tensor(batch, "class").dtype is torch.int32  # unasked: as-is
    assert batch_tensor(batch, "class", dtype=torch.int64).dtype is torch.int64


def test_dtype_matters_because_the_loss_rejects_the_wrong_one() -> None:
    """Why the knob exists: int32 class ids are legal and CrossEntropyLoss refuses them."""
    import torch.nn as nn

    batch = collate_records([{"class": Label(torch.tensor(i, dtype=torch.int32))} for i in range(3)])
    logits = torch.randn(3, 4)

    # Matched loosely on purpose: torch reworded this in 2.13 ("expected scalar type Long but
    # found Int" -> "expected target dtype to be Long or Byte, but got Int"). The REJECTION is
    # the contract this test pins; the exact phrasing is torch's to change.
    with pytest.raises(RuntimeError, match="(scalar type|target dtype).*Long"):
        nn.CrossEntropyLoss()(logits, batch_tensor(batch, "class"))

    nn.CrossEntropyLoss()(logits, batch_tensor(batch, "class", dtype=torch.int64))  # no raise


# --------------------------------------------------------------------------- #
# multi_hot — MultiLabel rendered as a matrix, framework-free
# --------------------------------------------------------------------------- #


def test_multi_hot_marks_every_label_of_every_record() -> None:
    batch = collate_records([{"class": MultiLabel([0, 2])}, {"class": MultiLabel([1])}])

    assert multi_hot(batch, "class", 3).tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]


def test_multi_hot_returns_numpy_so_any_framework_can_use_it() -> None:
    """A torch return type would have forced a second implementation for the next backend."""
    batch = collate_records([{"class": MultiLabel([0])}])
    result = multi_hot(batch, "class", 2)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32  # what the multi-label losses consume
    assert torch.as_tensor(result).dtype is torch.float32  # one line into torch


def test_multi_hot_dtype_is_selectable() -> None:
    assert multi_hot(collate_records([{"class": MultiLabel([0])}]), "class", 2, dtype="int64").dtype == np.int64


def test_an_empty_label_set_is_an_all_zero_row_not_an_error() -> None:
    """ "This record has no classes" is a meaningful multi-label target."""
    batch = collate_records([{"class": MultiLabel([1])}, {"class": MultiLabel([])}])

    assert multi_hot(batch, "class", 2).tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_out_of_range_ids_are_ignored_rather_than_raising() -> None:
    """A stray label must not abort a training run."""
    batch = collate_records([{"class": MultiLabel([0, 99, -1])}])

    assert multi_hot(batch, "class", 2).tolist() == [[1.0, 0.0]]


def test_multi_hot_width_is_num_classes_not_the_observed_max() -> None:
    """The head's width decides the matrix, never the batch's contents."""
    batch = collate_records([{"class": MultiLabel([0])}])

    assert multi_hot(batch, "class", 5).shape == (1, 5)


# --------------------------------------------------------------------------- #
# batch_metadata — the collate's transpose
# --------------------------------------------------------------------------- #


def test_columns_transpose_into_per_record_dicts() -> None:
    records: list = [
        {"image": Image(_chw(), layout="CHW"), "class": Label(0), "idx": i, "src": f"f{i}"} for i in range(3)
    ]
    metas = batch_metadata(collate_records(records), exclude=("image", "class"))
    assert metas == [{"idx": 0, "src": "f0"}, {"idx": 1, "src": "f1"}, {"idx": 2, "src": "f2"}]


def test_excluded_keys_are_omitted() -> None:
    batch = collate_records([{"image": Image(_chw(), layout="CHW"), "idx": i} for i in range(2)])
    assert batch_metadata(batch, exclude=("image",)) == [{"idx": 0}, {"idx": 1}]
    assert "image" not in (batch_metadata(batch, exclude=("image",)) or [{}])[0]


def test_no_remaining_keys_returns_none() -> None:
    """`None` is the "nothing to correlate" answer a sink reads as skip."""
    batch = collate_records([{"image": Image(_chw(), layout="CHW")} for _ in range(2)])
    assert batch_metadata(batch, exclude=("image",)) is None


def test_a_hand_built_metadata_key_is_returned_verbatim() -> None:
    batch: Record = {"image": torch.zeros(2, 3, 4, 4), "metadata": [{"idx": 0}, {"idx": 1}]}
    assert batch_metadata(batch, exclude=("image",)) == [{"idx": 0}, {"idx": 1}]


def test_a_ragged_batch_truncates_rather_than_raising() -> None:
    """A metadata detail must never take down an inference run."""
    batch: Record = {"a": [1, 2, 3], "b": [1, 2]}
    assert batch_metadata(batch) == [{"a": 1, "b": 1}, {"a": 2, "b": 2}]


def test_a_label_column_contributes_its_values() -> None:
    """Metadata carried as a Label unwraps like any other key."""
    batch = collate_records([{"image": Image(_chw(), layout="CHW"), "split": Label("train")} for _ in range(2)])
    assert batch_metadata(batch, exclude=("image",)) == [{"split": "train"}, {"split": "train"}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
