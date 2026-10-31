"""The root of every error recordstream raises.

Each of the package's exceptions derives from :class:`RecordstreamError` AND from the builtin it stands for
(``class DrawSpecError(RecordstreamError, ValueError)``), so ``except RecordstreamError`` catches every one recordstream
raises, and an ``except ValueError`` written before the root existed keeps catching what it caught.

A leaf module: it imports nothing, so every module of the package can import it without a cycle. The two private
control-flow signals (``draws._NoFingerprint``, ``flow.trace._Breakpoint``) are not errors — they never leave the
module that raises them — and stay outside it.
"""


class RecordstreamError(Exception):
    """An error recordstream raises; each subclass is also the builtin it replaces (``ValueError``, ``RuntimeError``,
    ``MemoryError``)."""


__all__ = ["RecordstreamError"]
