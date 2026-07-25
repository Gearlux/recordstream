"""``RandomApply`` — apply an op with a given probability.

A compose-group op (alongside ``Enable`` / ``Pipeline`` / ``Parallel``):
wrap any single op (native or a bare library transform) so it fires only *p*
fraction of the time. Records that are skipped pass through unchanged.

Modality-neutral — it threads any record through any op — so it lives
in core sampleflux, not a domain package.
"""

import random
from typing import Optional

from confluid import configurable
from loggair import get_logger

from sampleflux.items import Record

logger = get_logger(__name__)


@configurable(category="op", group="compose", random=True)
class RandomApply:
    """Gate any op behind a Bernoulli coin flip.

    On each call, a uniform ``U ~ [0, 1)`` is drawn; if ``U < probability``
    the inner ``op`` is applied, otherwise the record passes through unchanged.

    ``op`` is flowed lazily on first use (Confluid ``!class:`` / ``!lazy:``
    markers are resolved at call-time, not at construction), so building a
    ``RandomApply()`` with no arguments costs nothing.

    YAML:

    .. code-block:: yaml

        - !class:sampleflux.ops.random_apply.RandomApply
          probability: 0.5
          op: !class:albumentations.HorizontalFlip {p: 1.0}

    Args:
        op: Inner op to gate (native op or bare library transform).  Defaults to ``None``
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

    def __call__(self, record: Record) -> Optional[Record]:
        if self.op is None:
            raise ValueError("RandomApply requires 'op' to be set before calling.")
        if self._gate_rng is None:
            self._gate_rng = random.Random(self.random_state)
        if self._gate_rng.random() >= self.probability:
            return record
        from confluid import flow
        from confluid.fluid import Fluid

        # _apply_op is the engine's op-family dispatch — routing through it (instead of
        # op(record)) lets a bare albumentations / torchvision-v2 transform nest inside
        # the gate exactly as it would sit in a bare ops list.
        from sampleflux.core import _apply_op

        op = flow(self.op) if isinstance(self.op, Fluid) else self.op
        self.op = op  # cache the flowed op so we only flow once
        return _apply_op(record, op)


__all__ = ["RandomApply"]
