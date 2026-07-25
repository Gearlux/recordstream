"""Per-sample named-cell store — the graph data plane for graph-shaped pipelines.

A :class:`Context` holds named **cells** for exactly one sample's trip through the op
list: branch snapshots (a cell holding a record dict), captured
``@output`` values, and per-sample parameters. The context ops in
:mod:`sampleflux.ops.context` (``Save`` / ``Use`` / ``Drop`` / ``Apply`` / ``Capture`` /
``Mix``) move data between the linear sample stream and these cells, which is what lets
a plain sequential op list execute a fan-out/fan-in graph.

The context ops route graph data through these per-sample Context CELLS and never touch
the sample's own fields — each typed item still owns its own metadata inside the sample.
The Context is the *wiring* plane — engine-created, per sample, empty again by the end of
a well-formed graph (every cell freed after its last read). Nothing here is
``@configurable``; a Context never appears in YAML.

The engine (``Flux`` — and ``FlowGraph``, which manages its env directly) creates one
Context per source item and activates it around the op loop via a
:class:`contextvars.ContextVar`, so ops reach it inside ``__call__`` with no signature
change (:func:`current` / :func:`require`). A hand-rolled loop outside an engine opts in
explicitly::

    with activate(Context()):
        for op in ops:
            sample = op(sample)
"""

import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

__all__ = ["Context", "activate", "current", "require"]


class Context:
    """Named-cell store for one sample's trip through a graph-shaped pipeline.

    Cells are stored and returned **by reference** — copy semantics are the reading
    op's decision (``Use`` deep-copies unless it drops the cell).
    """

    __slots__ = ("_cells",)

    def __init__(self) -> None:
        self._cells: Dict[str, Any] = {}

    def put(self, name: str, value: Any) -> None:
        """Store ``value`` under ``name`` (overwrites an existing cell)."""
        self._cells[name] = value

    def get(self, name: str) -> Any:
        """Return the cell's value by reference; a missing cell is an actionable error."""
        if name not in self._cells:
            live = ", ".join(sorted(self._cells)) or "<none>"
            raise KeyError(
                f"Context has no cell {name!r} (live cells: {live}). "
                f"A cell must be written (Save / Capture) before it is read, and is gone after "
                f"a drop — check the op order and drop flags."
            )
        return self._cells[name]

    def delete(self, name: str) -> None:
        """Free the cell; deleting a missing cell is an error (it flags a liveness bug)."""
        if name not in self._cells:
            live = ", ".join(sorted(self._cells)) or "<none>"
            raise KeyError(f"Context cannot drop missing cell {name!r} (live cells: {live}).")
        del self._cells[name]

    def live(self) -> Tuple[str, ...]:
        """Names of all currently-held cells (sorted, for stable error messages/tests)."""
        return tuple(sorted(self._cells))

    def copy(self) -> "Context":
        """Shallow copy — same cell values, independent cell *set* (for 1→N expansion children)."""
        clone = Context()
        clone._cells = dict(self._cells)
        return clone

    def clear(self) -> None:
        """Drop every cell."""
        self._cells.clear()

    def __contains__(self, name: object) -> bool:
        return name in self._cells

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Context(cells={sorted(self._cells)})"


_CURRENT: contextvars.ContextVar[Optional[Context]] = contextvars.ContextVar("sampleflux_context", default=None)


def current() -> Optional[Context]:
    """The active per-sample :class:`Context`, or ``None`` outside an engine/`activate` block."""
    return _CURRENT.get()


def require(op_name: str = "context op") -> Context:
    """The active Context, or an actionable error naming the op that needed it."""
    ctx = _CURRENT.get()
    if ctx is None:
        raise RuntimeError(
            f"{op_name}: no active Context. Context ops need the per-sample Context the engine "
            f"creates — run the pipeline through Flux/FlowGraph, or wrap a manual loop in "
            f"`with sampleflux.context.activate(Context()):`."
        )
    return ctx


@contextmanager
def activate(ctx: Context) -> Iterator[Context]:
    """Activate ``ctx`` as the current per-sample Context for the enclosed block."""
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)
