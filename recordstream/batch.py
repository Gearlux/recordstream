"""Reading a batched record back — the inverse of :func:`~recordstream.collate.collate_records`.

:mod:`recordstream.collate` writes the batch convention; this module reads it. They are two
halves of ONE piece of knowledge (an item's payload stacks, its declared attrs become
per-record lists, a plain value becomes a plain list), so they live side by side — a consumer
that had to re-derive the read-back would be re-deriving the collate.

The three primitives are deliberately TASK-AGNOSTIC. They answer "what did the collate put
under this key?", never "what shape does my loss want?" — a classification trainer wanting
``[N]`` int64 class ids, a segmenter wanting an ``[N, H, W]`` int64 mask, and a multi-label
trainer wanting an ``[N, C]`` float multi-hot all start from the same unwrapped values and
shape them at their own model boundary. Putting that shaping here would mean one function
with a task switch.

Typical use at a model boundary::

    x = batch_tensor(batch, "image", device=self.device)   # [N, 3, H, W]
    meta = batch_metadata(batch, exclude=("image", "class"))  # per-record dicts for a sink
"""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import numpy as np

from recordstream.items import Label, MultiLabel, Record, item_data

if TYPE_CHECKING:  # torch is imported lazily at call time — this is annotation-only
    from torch import Tensor

__all__ = ["batch_metadata", "batch_tensor", "batch_values"]


def batch_values(batch: Record, key: str) -> Any:
    """The raw batched values under ``key``, unwrapped from their item type.

    The one place that knows how to get *past* a wrapper item: a :class:`~recordstream.Label`
    yields its ``.value`` (the collate leaves it a per-record LIST — a wrapper item's payload
    is not stacked), a :class:`~recordstream.MultiLabel` its ``.values`` (a list OF lists),
    and anything else goes through :func:`~recordstream.item_data` (an array item yields its
    stacked payload, a plain value the per-record list the collate gathered).

    No torch, no stacking, no dtype opinion — just the values. Use :func:`batch_tensor` when
    a tensor is what you need.

    Example::

        batch_values(collate_records([{"class": Label(0)}, {"class": Label(1)}]), "class")
        # [0, 1]
    """
    item = batch[key]
    if isinstance(item, MultiLabel):
        return item.values
    if isinstance(item, Label):
        return item.value
    return item_data(item)


def batch_tensor(batch: Record, key: str, device: Any = None) -> "Tensor":
    """The batched values under ``key`` as ONE torch tensor.

    Normalizes the two shapes the collate can leave behind — a stacked array payload, or a
    per-record LIST (a wrapper item's payload, or a plain value) — into a single tensor. An
    already-stacked tensor is used verbatim; anything else goes through the cheap,
    memory-sharing ``torch.as_tensor``.

    Args:
        batch: A batched record (the output of :func:`~recordstream.collate_records`).
        key: The record key to read.
        device: Optional target device. A tensor built HERE from a per-record list is created
            on the CPU regardless of a framework's own batch move, so pass the module's device
            when the result feeds a model.

    Returns:
        A ``torch.Tensor``. The dtype is whatever the values carry — shaping (an int64 class-id
        promotion, a float multi-hot) belongs to the caller's model boundary.

    Example::

        x = batch_tensor(batch, "image", device=self.device)   # [N, 3, H, W]
    """
    import torch  # local: recordstream stays importable without touching torch

    value = batch_values(batch, key)
    if isinstance(value, list):
        tensor = torch.stack([torch.as_tensor(v) for v in value])
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(np.asarray(value))
    return tensor if device is None else tensor.to(device)


def batch_metadata(batch: Record, exclude: Iterable[str] = ()) -> Optional[List[Dict[str, Any]]]:
    """Per-record metadata dicts recovered from a batched record — the collate's transpose.

    The collate turns N records into ONE record of per-key columns; this turns those columns
    back into N dicts, so a predictions sink can correlate a model's per-record output with
    the record it came from. Keys named in ``exclude`` (typically the input and target keys,
    which the model already consumed) are left out.

    A hand-built batch that instead carries a single list-valued ``"metadata"`` key is
    returned verbatim — that shape is already the answer.

    Args:
        batch: A batched record.
        exclude: Keys to omit from the per-record dicts.

    Returns:
        One dict per record, or ``None`` when nothing remains after ``exclude`` (there is no
        metadata to correlate, which a caller reads as "skip"). The list is as long as the
        SHORTEST column — a ragged batch truncates rather than raising, so a metadata detail
        cannot take down an inference run.

    Example::

        batch_metadata(batch, exclude=("image", "class"))   # [{"idx": 0}, {"idx": 1}]
    """
    if "metadata" in batch and isinstance(batch["metadata"], list):
        return list(batch["metadata"])
    skip = set(exclude)
    columns = {key: batch_values(batch, key) for key in batch if key not in skip}
    if not columns:
        return None
    n = min(len(v) for v in columns.values())
    return [{k: columns[k][i] for k in columns} for i in range(n)]
