"""``Parallel`` — explicit parallel sub-pipeline op.

Place inside a :class:`~sampleflux.core.Flux`'s ops list to dispatch each
upstream sample through an inner sub-pipeline (``self.ops``) in a
spawn-context worker pool. Bounded prefetch caps outstanding work so the
executor queue can't grow unboundedly with source length.

Falls back to inline sequential application when invoked as a regular
per-sample op (e.g. via :meth:`Flux.__getitem__`) so random access remains
correct.

Note:
    Do not nest a ``Parallel`` op inside another ``Parallel.ops`` — workers
    must not themselves spawn workers. ``Pipeline``, ``Enable``, and any
    pickle-safe per-record op are fine inside.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from collections import deque
from typing import Any, Iterable, Iterator, List, Optional

from confluid import configurable, flow
from confluid.fluid import Fluid

from sampleflux.core import _worker_task
from sampleflux.items import Record


@configurable(category="op", group="compose")
class Parallel:
    """Run an inner op sub-pipeline in a worker pool with bounded prefetch.

    Args:
        ops: Sequential sub-pipeline applied to each record inside a worker.
        workers: Number of worker processes (spawn context). Must be >= 1.
    """

    def __init__(self, ops: Optional[List[Any]] = None, workers: int = 4) -> None:
        # Lazy / zero-arg: store config only; ``workers >= 1`` is validated lazily in ``stream``.
        self.ops = list(ops) if ops else []
        self.workers = int(workers)

    def _materialize_ops(self) -> None:
        # Confluid post-construction paradigm leaves nested ops as Fluid
        # markers; resolve them in-place on first use, mirroring Pipeline.
        for i, op in enumerate(self.ops):
            if isinstance(op, Fluid):
                self.ops[i] = flow(op)

    def __call__(self, record: Record) -> Optional[Record]:
        # Inline fallback for non-streaming callers (e.g. Flux.__getitem__). Routed through
        # _apply_op — the same op-family dispatch the streamed route's _worker_task uses —
        # so bare library transforms behave identically.
        from sampleflux.core import _apply_op

        self._materialize_ops()
        current: Optional[Record] = record
        for op in self.ops:
            if current is None:
                return None
            current = _apply_op(current, op)
        return current

    def stream(self, samples: Iterable[Optional[Record]]) -> Iterator[Optional[Record]]:
        """Stream-level dispatch with bounded prefetch (in-order yield)."""
        if self.workers < 1:
            raise ValueError(f"Parallel(workers={self.workers!r}): must be >= 1")
        self._materialize_ops()
        ctx = multiprocessing.get_context("spawn")
        limit = max(2 * self.workers, self.workers + 1)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx) as executor:
            pending: "deque[concurrent.futures.Future[Optional[Record]]]" = deque()
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
