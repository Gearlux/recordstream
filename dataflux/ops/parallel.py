"""``Parallel`` — explicit parallel sub-pipeline op.

Place inside a :class:`~dataflux.core.Flux`'s ops list to dispatch each
upstream sample through an inner sub-pipeline (``self.ops``) in a
spawn-context worker pool. Bounded prefetch caps outstanding work so the
executor queue can't grow unboundedly with source length.

Falls back to inline sequential application when invoked as a regular
per-sample op (e.g. via :meth:`Flux.__getitem__`) so random access remains
correct.

Note:
    Do not nest a ``Parallel`` op inside another ``Parallel.ops`` — workers
    must not themselves spawn workers. ``Tee``, ``Enable``, and any
    pickle-safe per-sample op are fine inside.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from collections import deque
from typing import Any, Iterable, Iterator, List, Optional

from confluid import configurable, flow
from confluid.fluid import Fluid
from dataflux.core import _worker_task
from dataflux.sample import Sample


@configurable
class Parallel:
    """Run an inner op sub-pipeline in a worker pool with bounded prefetch.

    Args:
        ops: Sequential sub-pipeline applied to each sample inside a worker.
        workers: Number of worker processes (spawn context). Must be >= 1.
    """

    def __init__(self, ops: List[Any], workers: int = 4) -> None:
        if workers < 1:
            raise ValueError(f"Parallel(workers={workers!r}): must be >= 1")
        self.ops = list(ops)
        self.workers = int(workers)

    def _materialize_ops(self) -> None:
        # Confluid post-construction paradigm leaves nested ops as Fluid
        # markers; resolve them in-place on first use, mirroring Tee.
        for i, op in enumerate(self.ops):
            if isinstance(op, Fluid):
                self.ops[i] = flow(op)

    def __call__(self, sample: Sample) -> Optional[Sample]:
        # Inline fallback for non-streaming callers (e.g. Flux.__getitem__).
        self._materialize_ops()
        current: Optional[Sample] = sample
        for op in self.ops:
            if current is None:
                return None
            current = op(current)
        return current

    def stream(self, samples: Iterable[Optional[Sample]]) -> Iterator[Optional[Sample]]:
        """Stream-level dispatch with bounded prefetch (in-order yield)."""
        self._materialize_ops()
        ctx = multiprocessing.get_context("spawn")
        limit = max(2 * self.workers, self.workers + 1)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.workers, mp_context=ctx
        ) as executor:
            pending: "deque[concurrent.futures.Future[Optional[Sample]]]" = deque()
            for s in samples:
                if s is None:
                    continue
                pending.append(executor.submit(_worker_task, s, self.ops))
                if len(pending) >= limit:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    def close(self) -> None:
        """Propagate close() to inner ops that own resources."""
        for op in self.ops:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["Parallel"]
