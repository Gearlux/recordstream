"""Shared internals for the view sources (``split`` / ``range`` / ``concat``)."""

from typing import Any


def _pass_through(item: Any) -> Any:
    """Pass a wrapped source's item through verbatim.

    Every carrier is a plain dict; the view sources
    (:class:`~recordstream.sources.DatasetSplit` / :class:`~recordstream.sources.RangeSource` /
    :class:`~recordstream.sources.ConcatSource`) only slice/index, they never inspect payloads,
    so a source's records flow through them unchanged.
    """
    return item
