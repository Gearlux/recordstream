"""Deferred torch ``DataLoader`` slots for a training runnable — the torch half of batching.

Batching has two halves (see ``docs/architecture.md`` §10): WHAT a batch contains (a collate)
and WHICH ROWS go in which batch (order, slicing, the short final batch, the per-epoch
reshuffle). torch's second half is the ``DataLoader``, and every training runnable in the
workspace baked the same three deferred loader slots into its constructor — train shuffled,
val/test not, one shared kwarg set. :func:`loader_slots` is that construction written once.

The slots are :class:`confluid.PartialClass` markers, not live loaders, on purpose: the lazy-init
mandate forbids functional work in a constructor, and the dataset does not exist yet — the run
method flows each slot with ``dataset=`` at run time (``flow(self.train_loader, dataset=ds)``).

**This module is torch-only and deliberately NOT re-exported from the package root** (the
``recordstream.ops.torch`` pattern): ``import recordstream`` keeps pulling no ML framework, and
a consumer — which is by definition a torch trainer — imports it directly::

    from recordstream.loaders import loader_slots
"""

from typing import Any, Callable, List, NamedTuple, Optional

from confluid import Partial, PartialClass

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

    train: Partial[DataLoader[Any]]
    val: Partial[DataLoader[Any]]
    test: Partial[DataLoader[Any]]


def loader_slots(
    batch_size: int,
    num_workers: int,
    *,
    collate_fn: Callable[[List[Record]], Record] = collate_records,
    persistent_workers: Optional[bool] = None,
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
        num_workers: Worker processes per loader. ``persistent_workers`` DERIVES from it
            (``num_workers != 0``) unless the caller says otherwise.
        persistent_workers: Override the derived pairing. ``None`` (the default) derives as
            above and is what nearly every run wants; an explicit value is for the case the
            derivation cannot see — a HOST fact. macOS terminates persistent workers slowly
            enough that a short run spends longer stopping than training, so a config there
            says ``persistent_workers: false`` while keeping its workers. ``True`` with
            ``num_workers=0`` is refused HERE (torch raises for that pairing when the loader
            is first iterated, minutes into a run); it is the one invariant the previously
            derived-only value protected by construction.
        collate_fn: The batch-shape choice (see the collate registry) — ``collate_records``
            stacks, ``collate_list`` does not (what a detection consumer passes), and a task
            collate is any callable.
        **loader_kw: Further ``DataLoader`` kwargs, baked into ALL THREE markers. ``shuffle``
            is refused here because it is the ONE per-split kwarg this helper owns (train
            ``True``, eval ``False``) — a different split policy is a whole-slot replacement,
            not a shared kwarg.

    Returns:
        A :class:`LoaderSlots` named tuple — three ``PartialClass(DataLoader, ...)`` markers
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
    if persistent_workers and num_workers == 0:
        raise ValueError(
            "loader_slots: persistent_workers=True needs num_workers > 0 — torch keeps worker "
            "processes alive between epochs and there are none. Raise num_workers, or leave "
            "persistent_workers unset to derive it."
        )
    shared = dict(
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers != 0) if persistent_workers is None else persistent_workers,
        **loader_kw,
    )
    return LoaderSlots(
        train=PartialClass(DataLoader, shuffle=True, **shared),
        val=PartialClass(DataLoader, shuffle=False, **shared),
        test=PartialClass(DataLoader, shuffle=False, **shared),
    )
