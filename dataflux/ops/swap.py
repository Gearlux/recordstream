"""``SwapInputTargetOp`` — exchange ``sample.input`` and ``sample.target``.

Useful when an op operates on ``sample.input`` but you want it applied to
the target instead: swap, run the op, swap back.
"""

from confluid import configurable

from dataflux.sample import Sample


@configurable
class SwapInputTargetOp:
    """Exchange ``sample.input`` ↔ ``sample.target``. Metadata unchanged."""

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.target, target=sample.input)
