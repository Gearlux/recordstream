"""``RangeSource`` — a contiguous ``[start:stop)`` index slice over an indexable source."""

from typing import Any, Iterator, List, Optional

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record
from recordstream.sources.base import _pass_through

logger = get_logger(__name__)


@configurable(category="source")
class RangeSource:
    """A contiguous index slice ``[start:stop)`` over an indexable source.

    The plain-slice counterpart to :class:`~recordstream.sources.DatasetSplit` (which shuffles
    + partitions) — extracted from DatasetSplit's old "range mode". Negative ``start`` /
    ``stop`` count from the end; both are clamped to ``[0, len(source)]``. Lazy: only index
    arithmetic happens up front; records are produced on demand.

    The wrapped source must implement ``__len__`` and ``__getitem__``.

    Args:
        source: The underlying indexable source (defaults to ``None``; validated lazily on first use).
        start: Inclusive start index (``None`` ⇒ 0; a negative value counts from the end).
        stop: Exclusive stop index (``None`` ⇒ len(source); a negative value counts from the end).
    """

    def __init__(self, source: Any = None, start: Optional[int] = None, stop: Optional[int] = None) -> None:
        # Lazy / zero-arg: store config only; the index arithmetic (and source validation) is deferred
        # to the ``indices`` property so the source can be configured post-construction.
        self.source = source
        self.start = start
        self.stop = stop
        self._indices: Optional[List[int]] = None

    @property
    def indices(self) -> List[int]:
        """The contiguous ``[start:stop)`` source indices, computed lazily on first access and cached."""
        if self._indices is None:
            source = self.source
            if source is None or not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
                raise TypeError(
                    "RangeSource requires a source supporting __len__ and __getitem__; " f"got {type(source).__name__}"
                )
            n = len(source)
            s = 0 if self.start is None else self.start
            e = n if self.stop is None else self.stop
            if s < 0:
                s = max(0, n + s)
            if e < 0:
                e = max(0, n + e)
            s = max(0, min(s, n))
            e = max(s, min(e, n))
            self._indices = list(range(s, e))
            logger.debug("RangeSource: size=%d source_size=%d", len(self._indices), n)
        return self._indices

    def __iter__(self) -> Iterator[Record]:
        for idx in self.indices:
            yield _pass_through(self.source[idx])

    def __getitem__(self, index: int) -> Any:
        return _pass_through(self.source[self.indices[index]])

    def __len__(self) -> int:
        return len(self.indices)
