"""Stash / unstash ``sample.input`` / ``sample.target`` to / from ``metadata``.

Use ``StashInputOp(key)`` to snapshot the current ``sample.input`` under a
metadata key without changing ``sample.input``; ``UnstashInputOp(key)``
restores it later. ``StashTargetOp`` / ``UnstashTargetOp`` are the exact
``sample.target`` counterparts.

Graph WIRING is the job of the context ops (``sampleflux.ops.context`` —
``Save``/``Use``/``Mix`` over per-sample Context cells, see ``docs/graph.md``).
The stash family remains for the two jobs cells cannot do, because the
snapshot rides ``sample.metadata`` WITH the sample:

* crossing a ``Parallel`` boundary — metadata travels through the stream
  split/join; Context cells deliberately raise there;
* deliberately PERSISTING a snapshot into a sink (the metadata key is
  serialised alongside the sample unless an ``Unstash*Op`` removes it).

The ``Unstash*Op``\\ s default to ``copy=True`` (deepcopy) so two readers
of the same key are independent — each gets its own array to mutate.
Without the copy, an in-place op like ``ClipPercentilesOp`` after the first
restore would silently corrupt the stashed value seen by the second.
"""

import copy as _copy

from confluid import configurable

from sampleflux.sample import Sample


@configurable(category="op", group="structure")
class StashInputOp:
    """Copy ``sample.input`` into ``metadata[key]``; ``sample.input`` unchanged.

    Args:
        key: Metadata key to write.
        copy: When ``True``, deepcopy ``sample.input`` before stashing.
            Defaults to ``False`` (cheap pointer alias) — the typical case
            is that downstream ops use ``sample._replace(input=...)`` and
            don't mutate the shared array in place.
    """

    def __init__(self, key: str = "", copy: bool = False) -> None:
        # Lazy / zero-arg: store config only.
        self.key = key
        self.copy = copy

    def __call__(self, sample: Sample) -> Sample:
        sample.meta[self.key] = _copy.deepcopy(sample.input) if self.copy else sample.input
        return sample


@configurable(category="op", group="structure")
class UnstashInputOp:
    """Set ``sample.input := metadata[key]``.

    Args:
        key: Metadata key to read.
        copy: When ``True`` (default), deepcopy the stashed value before
            assigning. This prevents two branches that unstash the same
            key from corrupting each other through downstream in-place
            mutations. Set ``False`` only when the caller has audited
            that no downstream op mutates the array in place.
        remove: When ``True`` (default), DELETE the key from metadata after
            restoring it — so the snapshot doesn't linger on the bus and
            leak into a downstream sink. Set ``False`` to keep it (required
            when the SAME key is unstashed again later, e.g. a fan-out that
            restores the fork before several branches — only the LAST
            unstash of a key may remove it).
    """

    def __init__(self, key: str = "", copy: bool = True, remove: bool = True) -> None:
        # Lazy / zero-arg: store config only; a missing key surfaces lazily as a KeyError in __call__.
        self.key = key
        self.copy = copy
        self.remove = remove

    def __call__(self, sample: Sample) -> Sample:
        value = sample.meta[self.key]
        if self.copy:
            value = _copy.deepcopy(value)
        if self.remove:
            del sample.meta[self.key]  # key exists (just read above)
        return sample._replace(input=value)


@configurable(category="op", group="structure")
class StashTargetOp:
    """Copy ``sample.target`` into ``metadata[key]``; ``sample.target`` unchanged.

    Args:
        key: Metadata key to write.
        copy: When ``True``, deepcopy ``sample.target`` before stashing.
            Defaults to ``False`` (cheap pointer alias) — the typical case
            is that downstream ops use ``sample._replace(target=...)`` and
            don't mutate the shared value in place.
    """

    def __init__(self, key: str = "", copy: bool = False) -> None:
        # Lazy / zero-arg: store config only.
        self.key = key
        self.copy = copy

    def __call__(self, sample: Sample) -> Sample:
        sample.meta[self.key] = _copy.deepcopy(sample.target) if self.copy else sample.target
        return sample


@configurable(category="op", group="structure")
class UnstashTargetOp:
    """Set ``sample.target := metadata[key]``.

    Args:
        key: Metadata key to read.
        copy: When ``True`` (default), deepcopy the stashed value before
            assigning. This prevents two branches that unstash the same
            key from corrupting each other through downstream in-place
            mutations. Set ``False`` only when the caller has audited
            that no downstream op mutates the value in place.
        remove: When ``True`` (default), DELETE the key from metadata after
            restoring it — so the snapshot doesn't linger on the bus and
            leak into a downstream sink. Set ``False`` to keep it (required
            when the SAME key is unstashed again later, e.g. a fan-out that
            restores the fork before several branches — only the LAST
            unstash of a key may remove it).
    """

    def __init__(self, key: str = "", copy: bool = True, remove: bool = True) -> None:
        # Lazy / zero-arg: store config only; a missing key surfaces lazily as a KeyError in __call__.
        self.key = key
        self.copy = copy
        self.remove = remove

    def __call__(self, sample: Sample) -> Sample:
        value = sample.meta[self.key]
        if self.copy:
            value = _copy.deepcopy(value)
        if self.remove:
            del sample.meta[self.key]  # key exists (just read above)
        return sample._replace(target=value)
