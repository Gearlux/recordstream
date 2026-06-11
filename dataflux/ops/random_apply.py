"""``RandomApply`` — apply an op with a given probability.

A compose-group op (alongside ``Enable`` / ``Tee`` / ``Parallel``):
wrap any single ``Sample → Sample`` op so it fires only *p* fraction of
the time. Samples that are skipped pass through unchanged.

Modality-neutral — it threads any ``Sample`` through any op — so it lives
in core dataflux, not a domain package.
"""

import random
from typing import Optional, cast

from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)


@configurable(category="op", group="compose", random=True)
class RandomApply:
    """Gate any op behind a Bernoulli coin flip.

    On each call, a uniform ``U ~ [0, 1)`` is drawn; if ``U < probability``
    the inner ``op`` is applied, otherwise the sample passes through unchanged.

    ``op`` is flowed lazily on first use (Confluid ``!class:`` / ``!lazy:``
    markers are resolved at call-time, not at construction), so building a
    ``RandomApply()`` with no arguments costs nothing.

    YAML:

    .. code-block:: yaml

        - !class:dataflux.ops.random_apply.RandomApply
          probability: 0.5
          op: !class:dataflux.ops.numpy.RescaleOp
            in_min: -1.0
            in_max: 1.0

    Args:
        op: Inner ``Sample → Sample`` callable to gate.  Defaults to ``None``
            (identity); validated lazily on first call.
        probability: Gate probability in ``[0, 1]``.  ``0.0`` = never apply;
            ``1.0`` = always apply.  Defaults to ``0.5``.
        random_state: Seed for the Bernoulli gate RNG. ``None`` = non-deterministic (default).
    """

    def __init__(
        self,
        op: Optional[object] = None,
        probability: float = 0.5,
        random_state: Optional[int] = None,
    ) -> None:
        self.op = op
        self.probability = probability
        self.random_state = random_state
        self._gate_rng: Optional[random.Random] = None

    def __call__(self, sample: Sample) -> Sample:
        if self.op is None:
            raise ValueError("RandomApply requires 'op' to be set before calling.")
        if self._gate_rng is None:
            self._gate_rng = random.Random(self.random_state)
        if self._gate_rng.random() >= self.probability:
            return sample
        from confluid import flow
        from confluid.fluid import Fluid

        op = flow(self.op) if isinstance(self.op, Fluid) else self.op
        self.op = op  # cache the flowed op so we only flow once
        return cast(Sample, op(sample))  # type: ignore[operator]


__all__ = ["RandomApply"]
