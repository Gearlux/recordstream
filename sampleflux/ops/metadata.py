"""Metadata-manipulation ops."""

import fnmatch
from typing import List, Optional, Tuple

from confluid import configurable

from sampleflux.sample import Sample


def _matches_any(key: str, patterns: Tuple[str, ...]) -> bool:
    """True if ``key`` matches ANY ``fnmatch`` glob in ``patterns`` (case-sensitive)."""
    return any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)


@configurable(category="op", group="structure")
class DropMetadataOp:
    """Remove metadata keys matching glob patterns (a pass-through ``Sample -> Sample`` op).

    A key is DROPPED when it matches an ``exclude`` pattern AND does NOT match any ``include``
    pattern — so ``include`` PROTECTS keys and takes priority over ``exclude`` (the rsync /
    gitignore include-wins model). Strips bookkeeping you don't want a downstream sink to
    serialise — e.g. a bulky ``Stash*Op`` snapshot (a stashed complex signal) kept on the
    metadata bus for a ``Parallel`` crossing. The replacement metadata is a
    fresh dict (copy-on-write); ``input`` / ``target`` are untouched. Single-sample only (reads
    ``sample.meta``), like ``CopyMetadataOp`` — drop keys before collation.

    Args:
        exclude: Glob patterns (``fnmatch``: ``*`` = any run, ``?`` = one char, ``[seq]`` = a set)
            for keys to REMOVE. A pattern with NO wildcards matches that key exactly. Case-sensitive.
            E.g. ``spec_*`` removes every spec-prefixed snapshot; with no ``exclude`` nothing
            is dropped.
        include: Glob patterns for keys to KEEP even when they match ``exclude`` — higher priority,
            so it carves exceptions out of ``exclude``. E.g. ``exclude=["spec_*"]`` +
            ``include=["spec_keep"]`` drops every ``spec_*`` key EXCEPT the protected one. ``include``
            only ever protects against ``exclude`` (with no ``exclude`` it has no effect).
    """

    def __init__(self, exclude: Optional[List[str]] = None, include: Optional[List[str]] = None) -> None:
        self.exclude = exclude
        self.include = include

    def __call__(self, sample: Sample) -> Sample:
        exclude = tuple(self.exclude or ())
        include = tuple(self.include or ())
        kept = {
            key: value
            for key, value in sample.meta.items()
            if not (_matches_any(key, exclude) and not _matches_any(key, include))
        }
        return sample._replace(metadata=kept)
