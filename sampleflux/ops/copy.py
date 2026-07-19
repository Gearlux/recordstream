"""Defensive deepcopy ops.

Use these when a downstream op mutates ``sample.input`` / ``sample.target``
in place and you want later readers (or external references) to see the
pre-mutation value — e.g. before handing a sample to an in-place library
call, or to decouple a snapshot from the live stream.
"""

import copy
from typing import Any

from confluid import configurable

from sampleflux.sample import Sample


@configurable(category="op", group="structure")
class CopySampleOp:
    """Deepcopy of input, target, and metadata."""

    def __call__(self, sample: Sample) -> Sample:
        return Sample(
            input=copy.deepcopy(sample.input),
            target=copy.deepcopy(sample.target),
            metadata=copy.deepcopy(sample.meta),
        )


@configurable(category="op", group="structure")
class CopyInputOp:
    """Deepcopy ``sample.input``."""

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=copy.deepcopy(sample.input))


@configurable(category="op", group="structure")
class CopyTargetOp:
    """Deepcopy ``sample.target``."""

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(target=copy.deepcopy(sample.target))


@configurable(category="op", group="structure")
class CopyMetadataOp:
    """Deepcopy ``sample.meta``.

    The replacement dict is a fresh object, so subsequent in-place writes
    on the new metadata won't be seen by other holders of the old dict.
    """

    def __call__(self, sample: Sample) -> Sample:
        new_meta: Any = copy.deepcopy(sample.meta)
        return sample._replace(metadata=new_meta)
