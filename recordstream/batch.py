"""Reading a batched record back — the inverse of :func:`~recordstream.collate.collate_records`.

:mod:`recordstream.collate` writes the batch convention; this module reads it. They are two
halves of ONE piece of knowledge (an item's payload stacks, its declared attrs become
per-record lists, a plain value becomes a plain list), so they live side by side — a consumer
that had to re-derive the read-back would be re-deriving the collate.

The primitives are deliberately TASK-AGNOSTIC. They answer "what did the collate put under
this key?", never "which of these does my loss want?" — a trainer picks the call and names
the dtype its contract requires:

* :func:`batch_values` — the raw values, past the wrapper item. Framework-free.
* :func:`multi_hot` — a :class:`~recordstream.MultiLabel` column as an ``[N, C]`` matrix.
  Framework-free (numpy).
* :func:`batch_tensor` — the torch adapter: stack, optional dtype, optional device.
* :func:`batch_metadata` — the collate's transpose, for prediction sinks. Framework-free.

**Only `batch_tensor` is torch.** Everything else returns plain values or numpy, so a
non-torch backend uses the same code and converts in one line
(``tf.convert_to_tensor(m)`` / ``torch.as_tensor(m)``, the latter sharing memory). ``dtype``
is a PARAMETER, not an opinion — the same knob as ``device``: recordstream never decides the
contract, it honours the one the caller names.

Typical use at a model boundary::

    x = batch_tensor(batch, "image", device=self.device)                     # [N, 3, H, W]
    y = batch_tensor(batch, "class", device=self.device, dtype=torch.int64)  # [N] class ids
    y = torch.as_tensor(multi_hot(batch, "class", num_classes)).to(self.device)   # [N, C]
    meta = batch_metadata(batch, exclude=("image", "class"))  # per-record dicts for a sink
"""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import numpy as np

from recordstream.items import Label, MultiLabel, Record, item_data

if TYPE_CHECKING:  # torch is imported lazily at call time — this is annotation-only
    from torch import Tensor

__all__ = ["batch_metadata", "batch_tensor", "batch_values", "multi_hot"]


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


def multi_hot(batch: Record, key: str, num_classes: int, dtype: Any = "float32") -> np.ndarray:
    """A :class:`~recordstream.MultiLabel` column as an ``[N, num_classes]`` multi-hot matrix.

    The encoding a multi-label target needs: row ``i`` has a 1 in every column that record's
    label set contains. It is the natural rendering of :class:`~recordstream.MultiLabel`, so it
    belongs beside the item rather than in whichever consumer needed it first.

    **Returns NUMPY, deliberately.** Nothing about counting labels into a matrix is
    framework-specific, and numpy is what every framework converts from in one line —
    ``torch.as_tensor(m)`` (which shares memory, no copy) or the TensorFlow/JAX equivalent. A
    torch-typed return would have forced a second implementation for the next backend.

    Args:
        batch: A batched record (the output of :func:`~recordstream.collate_records`).
        key: The record key holding the multi-label target.
        num_classes: Matrix width. Ids outside ``[0, num_classes)`` are IGNORED rather than
            raising — a stray label must not abort a training run (the same rule
            ``marainer.torch.inverse_frequency_weights`` applies to class counting).
        dtype: Result dtype, default ``"float32"`` — the multi-label losses
            (``BCEWithLogitsLoss`` and friends) want float targets shaped like the logits, not
            integer class ids.

    Returns:
        An ``[N, num_classes]`` numpy array. A record whose label set is empty yields an
        all-zero row, which is a meaningful multi-label target (this record has no classes) and
        not an error.

    Example::

        y = torch.as_tensor(multi_hot(batch, "class", num_classes=3)).to(self.device)
        # MultiLabel([0, 2]), MultiLabel([1])  ->  [[1, 0, 1], [0, 1, 0]]
    """
    value = batch_values(batch, key)
    rows = value if isinstance(value, list) else [value]
    out = np.zeros((len(rows), int(num_classes)), dtype=dtype)
    for row, ids in enumerate(rows):
        for class_id in ids if isinstance(ids, (list, tuple, set)) else [ids]:
            index = int(class_id)
            if 0 <= index < num_classes:
                out[row, index] = 1
    return out


def batch_tensor(batch: Record, key: str, device: Any = None, dtype: Any = None) -> "Tensor":
    """The batched values under ``key`` as ONE torch tensor.

    Normalizes the two shapes the collate can leave behind — a stacked array payload, or a
    per-record LIST (a wrapper item's payload, or a plain value) — into a single tensor. An
    already-stacked tensor is used verbatim; anything else goes through the cheap,
    memory-sharing ``torch.as_tensor``.

    This is the TORCH adapter over :func:`batch_values`. A non-torch backend calls
    ``batch_values`` (or :func:`multi_hot`) and converts with its own one-liner; nothing here
    is duplicated for it.

    Args:
        batch: A batched record (the output of :func:`~recordstream.collate_records`).
        key: The record key to read.
        device: Optional target device. A tensor built HERE from a per-record list is created
            on the CPU regardless of a framework's own batch move, so pass the module's device
            when the result feeds a model.
        dtype: Optional target dtype — a PARAMETER, not an opinion: the caller names the
            contract its loss requires and this honours it. Pass ``torch.int64`` for class ids
            (``CrossEntropyLoss`` raises *"expected scalar type Long but found Int"* on an
            int32 target, and a dataset yielding int32 label tensors is perfectly legal) or for
            a pixel-class mask. ``None`` keeps whatever the values carry.

    Returns:
        A ``torch.Tensor``.

    Example::

        x = batch_tensor(batch, "image", device=self.device)                     # [N, 3, H, W]
        y = batch_tensor(batch, "class", device=self.device, dtype=torch.int64)  # [N] class ids
    """
    import torch  # local: recordstream stays importable without touching torch

    value = batch_values(batch, key)
    if isinstance(value, list):
        tensor = torch.stack([torch.as_tensor(v) for v in value])
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
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
