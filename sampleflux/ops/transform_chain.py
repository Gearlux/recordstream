"""``TransformChain`` — group a sequence of ops into a single named unit.

A compose-group op (alongside ``Enable`` / ``Parallel``):
wrap an ordered list of ``Sample → Sample`` callables so they appear as
one node on a visual canvas (dynamic ``op_0``, ``op_1``, … ``SAMPLEFLUX_OP``
inputs instead of N wired ``SAMPLEFLUX_SAMPLE`` connections) and one named
block in a Confluid YAML.

Unlike ``Enable`` there is no boolean gate — the chain always fires.
Unlike ``Parallel`` there is no worker pool — ops run sequentially in the
calling thread.  If any op returns ``None`` the chain stops early and
propagates ``None`` (consistent with ``FilterOp`` semantics).
"""

from typing import List, Optional

from confluid import configurable
from loggair import get_logger

from sampleflux.bag.sample import Sample

logger = get_logger(__name__)


@configurable(category="op", group="compose")
class TransformChain:
    """Apply a fixed sequence of ops to every sample, always.

    Wrap a list of ops into one named unit so they appear as a single node
    on a visual canvas (dynamic ``op_0``, ``op_1``, … ``SAMPLEFLUX_OP`` inputs)
    and one block in Confluid YAML instead of N separate connections.

    If any op in the chain returns ``None`` the remaining ops are skipped
    and ``None`` is propagated (consistent with ``FilterOp`` semantics —
    the sample is dropped).

    Inner ops keep full autonomy over their own randomness; ``TransformChain``
    itself is deterministic.  Nest a
    :class:`~sampleflux.ops.random_apply.RandomApply` inside the chain to
    gate individual ops stochastically.

    YAML example::

        - !class:sampleflux.ops.transform_chain.TransformChain
          ops:
            - !class:sampleflux.ops.random_apply.RandomApply
              op: !class:waivefront.torchsig.processing.AWGNOp {}
              probability: 0.8
            - !class:sampleflux.ops.torch.ToTensorOp {}

    Args:
        ops: Ordered list of callables ``Sample -> Optional[Sample]`` applied
            in sequence.  Defaults to ``[]`` (identity — the chain passes
            every sample through unchanged).
    """

    def __init__(self, ops: Optional[List] = None) -> None:
        self.ops: List = list(ops) if ops else []

    def __call__(self, sample: Sample) -> Optional[Sample]:
        from confluid import flow
        from confluid.fluid import Fluid

        # _apply_op = the engine's contract-aware chokepoint, so field-scoped ops
        # (e.g. a pair-scoped op from the kinds grid) chain exactly as in a bare ops list.
        from sampleflux.core import _apply_op

        current: Optional[Sample] = sample
        for i, op in enumerate(self.ops):
            if current is None:
                return None
            if isinstance(op, Fluid):
                op = flow(op)
                self.ops[i] = op
            if op is None:
                continue
            current = _apply_op(current, op)
        return current

    def close(self) -> None:
        """Propagate close() to inner ops that own resources (e.g. SampleSinkOp)."""
        for op in self.ops:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["TransformChain"]
