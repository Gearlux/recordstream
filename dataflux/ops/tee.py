"""``Tee`` — fan-out side effects across N branches on a single sample.

Branches run sequentially on the **same** ``Sample`` object and **same**
``metadata`` dict. There are no automatic copies; if isolation is needed,
place an explicit ``CopyInputOp`` / ``CopySampleOp`` at the start of a
branch. Later branches see what earlier branches wrote to ``metadata``.

If any op in any branch returns ``None``, ``Tee`` propagates ``None``
(consistent with ``FilterOp`` semantics — the whole sample is dropped).
"""

from typing import Any, List, Optional

from confluid import configurable, flow
from confluid.fluid import Fluid

from dataflux.sample import Sample


@configurable
class Tee:
    """Run N op-list branches sequentially on the same sample / metadata."""

    def __init__(self, branches: List[List[Any]]) -> None:
        self.branches = [list(b) for b in branches]

    def __call__(self, sample: Sample) -> Optional[Sample]:
        current: Optional[Sample] = sample
        for branch in self.branches:
            for i, op in enumerate(branch):
                if current is None:
                    return None
                if isinstance(op, Fluid):
                    op = flow(op)
                    branch[i] = op
                if op is None:
                    continue
                current = op(current)
        return current

    def close(self) -> None:
        """Propagate close() to inner ops that own resources."""
        for branch in self.branches:
            for op in branch:
                close_fn = getattr(op, "close", None)
                if callable(close_fn):
                    close_fn()
