"""Defensive deepcopy ops.

Use these when a downstream op mutates ``sample.input`` / ``sample.target``
in place and you want subsequent branches (or external references) to see
the pre-mutation value. ``Tee`` shares a single ``Sample`` across all
branches by design, so isolation is opt-in via these ops.
"""

import copy
from typing import Any

from confluid import configurable

from dataflux.sample import Sample


@configurable
class CopySampleOp:
    """Deepcopy of input, target, and metadata."""

    def __call__(self, sample: Sample) -> Sample:
        return Sample(
            input=copy.deepcopy(sample.input),
            target=copy.deepcopy(sample.target),
            metadata=copy.deepcopy(sample.metadata),
        )


@configurable
class CopyInputOp:
    """Deepcopy ``sample.input``."""

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=copy.deepcopy(sample.input))


@configurable
class CopyTargetOp:
    """Deepcopy ``sample.target``."""

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(target=copy.deepcopy(sample.target))


@configurable
class CopyMetadataOp:
    """Deepcopy ``sample.metadata``.

    The replacement dict is a fresh object, so subsequent in-place writes
    on the new metadata won't be seen by other holders of the old dict.
    """

    def __call__(self, sample: Sample) -> Sample:
        new_meta: Any = copy.deepcopy(sample.metadata)
        return sample._replace(metadata=new_meta)
