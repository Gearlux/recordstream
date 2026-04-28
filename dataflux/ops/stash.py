"""Stash / unstash ``sample.input`` to / from ``metadata``.

Use ``StashInputOp(key)`` to snapshot the current ``sample.input`` under a
metadata key without changing ``sample.input``. Use ``UnstashInputOp(key)``
later (e.g. inside another ``Tee`` branch) to restore that value into
``sample.input``.

``UnstashInputOp`` defaults to ``copy=True`` (deepcopy) so two branches
that both unstash the same key are independent — each gets its own array
to mutate. Without the copy, an in-place op like ``ClipPercentilesOp``
in the first branch would silently corrupt the stashed value seen by the
second branch.
"""

import copy as _copy

from confluid import configurable

from dataflux.sample import Sample


@configurable
class StashInputOp:
    """Copy ``sample.input`` into ``metadata[key]``; ``sample.input`` unchanged.

    Args:
        key: Metadata key to write.
        copy: When ``True``, deepcopy ``sample.input`` before stashing.
            Defaults to ``False`` (cheap pointer alias) — the typical case
            is that downstream ops use ``sample._replace(input=...)`` and
            don't mutate the shared array in place.
    """

    def __init__(self, key: str, copy: bool = False) -> None:
        self.key = key
        self.copy = copy

    def __call__(self, sample: Sample) -> Sample:
        sample.metadata[self.key] = _copy.deepcopy(sample.input) if self.copy else sample.input
        return sample


@configurable
class UnstashInputOp:
    """Set ``sample.input := metadata[key]``.

    Args:
        key: Metadata key to read.
        copy: When ``True`` (default), deepcopy the stashed value before
            assigning. This prevents two branches that unstash the same
            key from corrupting each other through downstream in-place
            mutations. Set ``False`` only when the caller has audited
            that no downstream op mutates the array in place.
    """

    def __init__(self, key: str, copy: bool = True) -> None:
        self.key = key
        self.copy = copy

    def __call__(self, sample: Sample) -> Sample:
        value = sample.metadata[self.key]
        if self.copy:
            value = _copy.deepcopy(value)
        return sample._replace(input=value)
