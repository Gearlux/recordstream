"""``Parallel`` — explicit parallel sub-pipeline op.

Place inside a :class:`~recordstream.core.stream.Stream`'s ops list to dispatch each
upstream record through an inner sub-pipeline (``self.ops``) in a
spawn-context worker pool. At most ``window × workers`` records are in flight
and each is dropped once yielded, so memory stays flat however long the source.

Falls back to inline sequential application when invoked as a regular
per-record op (e.g. via :meth:`Stream.__getitem__`) so random access remains
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
from typing import Annotated, Any, Iterable, Iterator, List, Optional

from annotated_types import Interval
from confluid import configurable, flow
from confluid.fluid import Fluid

from recordstream.core import _worker_task
from recordstream.items import Record

# A count of worker processes, or of records in flight per worker: a node form offers it from 1 up.
AtLeastOne = Annotated[int, Interval(ge=1)]


@configurable(category="op", group="compose")
class Parallel:
    """Run an inner op sub-pipeline in a worker pool with bounded prefetch.

    Args:
        ops: Sequential sub-pipeline applied to each record inside a worker.
        workers: Number of worker processes (spawn context). Must be >= 1.
        window: Records in flight per worker (at most ``window × workers``); widen it for a few very slow records.
    """

    def __init__(self, ops: Optional[List[Any]] = None, workers: AtLeastOne = 4, window: AtLeastOne = 2) -> None:
        # Partial / zero-arg: store config only. The range marks refuse a value below 1 when the op is built;
        # ``stream`` checks again, for a value set on the attribute afterwards.
        self.ops = list(ops) if ops else []
        self.workers = int(workers)
        self.window = int(window)

    def _materialize_ops(self) -> None:
        # Confluid post-construction paradigm leaves nested ops as Fluid
        # markers; resolve them in-place on first use, mirroring Pipeline.
        for i, op in enumerate(self.ops):
            if isinstance(op, Fluid):
                self.ops[i] = flow(op)

    def __call__(self, record: Record) -> Optional[Record]:
        # Inline fallback for non-streaming callers (e.g. Stream.__getitem__). Routed through
        # _apply_op — the same op-family dispatch the streamed route's _worker_task uses —
        # so bare library transforms behave identically.
        from recordstream.core import _apply_op

        self._materialize_ops()
        current: Optional[Record] = record
        for op in self.ops:
            if current is None:
                return None
            current = _apply_op(current, op)
        return current

    def stream(self, records: Iterable[Optional[Record]]) -> Iterator[Optional[Record]]:
        """Stream-level dispatch with bounded prefetch (in-order yield)."""
        if self.workers < 1:
            raise ValueError(f"Parallel(workers={self.workers!r}): must be >= 1")
        if self.window < 1:
            raise ValueError(
                f"Parallel(window={self.window!r}): must be >= 1 — the records each worker may have in flight"
            )
        self._materialize_ops()
        ctx = multiprocessing.get_context("spawn")
        limit = self.window * self.workers

        from recordstream.core import _extra_op_families

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx) as executor:
            pending: "deque[concurrent.futures.Future[Optional[Record]]]" = deque()
            extra_families = _extra_op_families()  # ship third-party op families to the workers
            for s in records:
                if s is None:
                    continue
                pending.append(executor.submit(_worker_task, s, self.ops, extra_families))
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
