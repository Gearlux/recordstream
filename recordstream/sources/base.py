"""Shared internals for the view sources (``split`` / ``range`` / ``concat``)."""

from typing import Any

from confluid.fluid import Fluid as _ConfluidFluid

from recordstream.core import _fluid_source_guidance


def _guard_live_source(source: Any, slot: str) -> None:
    """Raise Stream's actionable deferred-marker error when ``source`` is still a Fluid.

    A still-deferred marker in a ``source:`` slot is a CONFIG error — the slot needs a live
    object — and the view sources answer it exactly as ``Stream._guard_live_source`` does,
    naming the slot, the deferred target and the fix (drop ``_partial_: true``), instead of
    the cryptic ``got Partial`` a bare ``hasattr`` check produces. Deliberately
    message-only: the slot is never flowed here (the raise-with-guidance convention for
    ``source:`` slots, distinct from the free functions ``project`` / ``dataset_uri``,
    which do materialize a marker first).
    """
    if isinstance(source, _ConfluidFluid):
        raise TypeError(_fluid_source_guidance(source, slot=slot))


def _pass_through(item: Any) -> Any:
    """Pass a wrapped source's item through verbatim.

    Every carrier is a plain dict; the view sources
    (:class:`~recordstream.sources.DatasetSplit` / :class:`~recordstream.sources.RangeSource` /
    :class:`~recordstream.sources.ConcatSource`) only slice/index, they never inspect payloads,
    so a source's records flow through them unchanged.
    """
    return item
