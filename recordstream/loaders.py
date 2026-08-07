"""Deferred torch ``DataLoader`` slots for a training runnable — the torch half of batching.

Batching has two halves (see ``docs/architecture.md`` §10): WHAT a batch contains (a collate)
and WHICH ROWS go in which batch (order, slicing, the short final batch, the per-epoch
reshuffle). torch's second half is the ``DataLoader``, and every training runnable in the
workspace baked the same three deferred loader slots into its constructor — train shuffled,
val/test not, one shared kwarg set. :func:`loader_slots` is that construction written once.

The slots are :class:`confluid.LazyClass` markers, not live loaders, on purpose: the lazy-init
mandate forbids functional work in a constructor, and the dataset does not exist yet — the run
method flows each slot with ``dataset=`` at run time (``flow(self.train_loader, dataset=ds)``).

**This module is torch-only and deliberately NOT re-exported from the package root** (the
``recordstream.ops.torch`` pattern): ``import recordstream`` keeps pulling no ML framework, and
a consumer — which is by definition a torch trainer — imports it directly::

    from recordstream.loaders import loader_slots
"""

from typing import Any, Callable, List, NamedTuple

from confluid import Lazy, LazyClass

from recordstream.collate import collate_records
from recordstream.items import Record

try:
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - exercised only on a torch-free install
    raise ImportError(
        "recordstream.loaders needs torch (it declares DataLoader slots) — install `recordstream[torch]`."
    ) from exc

__all__ = ["LoaderSlots", "loader_slots"]


class LoaderSlots(NamedTuple):
    """The three deferred loader markers, addressed by split (``slots.train`` / ``.val`` / ``.test``)."""

    train: Lazy[DataLoader[Any]]
    val: Lazy[DataLoader[Any]]
    test: Lazy[DataLoader[Any]]


def loader_slots(
    batch_size: int,
    num_workers: int,
    *,
    collate_fn: Callable[[List[Record]], Record] = collate_records,
    **loader_kw: Any,
) -> LoaderSlots:
    """The train/val/test deferred ``DataLoader`` triple every torch training runnable declares.

    Configurability is TWO-CHANNELLED, and this helper narrows neither:

    * **From code** — any further ``DataLoader`` kwarg (``pin_memory``, ``drop_last``,
      ``prefetch_factor``, a ``worker_init_fn``) passes through ``**loader_kw`` and is baked
      into all three markers.
    * **From config** — the runnable's ``train_loader`` / ``val_loader`` / ``test_loader``
      slots stay whole-value replaceable in YAML (``train_loader: !class:torch.utils.data.DataLoader
      {shuffle: false, pin_memory: true, ...}``), exactly as with the inline construction this
      replaces; the run method only injects ``dataset=`` at flow time, so every knob a replaced
      slot sets survives.

    Args:
        batch_size: Rows per batch, baked into all three loaders.
        num_workers: Worker processes per loader. ``persistent_workers`` derives from it
            (``num_workers != 0``) — there is deliberately no separate knob, because persistent
            workers with zero workers is a torch error and the pairing never varies.
        collate_fn: The batch-shape choice (see the collate registry) — ``collate_records``
            stacks, ``collate_list`` does not (what a detection consumer passes), and a task
            collate is any callable.
        **loader_kw: Further ``DataLoader`` kwargs, baked into ALL THREE markers. ``shuffle``
            is refused here because it is the ONE per-split kwarg this helper owns (train
            ``True``, eval ``False``) — a different split policy is a whole-slot replacement,
            not a shared kwarg.

    Returns:
        A :class:`LoaderSlots` named tuple — three ``LazyClass(DataLoader, ...)`` markers
        (``slots.train`` with ``shuffle=True``, ``slots.val`` / ``slots.test`` with
        ``shuffle=False``). Assign them to the runnable's ``train_loader`` / ``val_loader`` /
        ``test_loader`` slots and flow each with ``dataset=`` at run time.
    """
    if "shuffle" in loader_kw:
        raise ValueError(
            "loader_slots: 'shuffle' is per-split (train shuffles, val/test do not) and cannot be "
            "a shared kwarg — replace the individual loader slot instead "
            "(e.g. train_loader: !class:torch.utils.data.DataLoader {shuffle: false, ...})."
        )
    shared = dict(
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers != 0,
        **loader_kw,
    )
    return LoaderSlots(
        train=LazyClass(DataLoader, shuffle=True, **shared),
        val=LazyClass(DataLoader, shuffle=False, **shared),
        test=LazyClass(DataLoader, shuffle=False, **shared),
    )
