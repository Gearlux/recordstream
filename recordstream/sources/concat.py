"""``ConcatSource`` — several indexable sources presented end to end as one."""

import bisect
from typing import Any, Iterator, List, Optional

from confluid import configurable

from recordstream.items import Record
from recordstream.sources.base import _guard_live_source, _pass_through


@configurable(category="source")
class ConcatSource:
    """Concatenates multiple indexable sources into one longer indexable source.

    The indexable counterpart to :class:`recordstream.core.JointStream` (which is iteration-only):
    ``len`` is the sum of the parts and ``source[i]`` maps a global index onto the owning
    sub-source, so a ``ConcatSource`` can itself be wrapped by
    :class:`~recordstream.sources.DatasetSplit` / :class:`~recordstream.sources.RangeSource`.
    (Distinct from an annotation-join source, which *column-joins* annotations onto records —
    this one *concatenates* sequences end to end.)

    Each sub-source must implement ``__len__`` and ``__getitem__``.

    Args:
        sources: The indexable sources to concatenate, walked in order (defaults to ``None`` ⇒ empty).
    """

    def __init__(self, sources: Optional[List[Any]] = None) -> None:
        # Partial / zero-arg: store config only; sub-source validation + the cumulative-offset precompute
        # are deferred to the ``offsets`` property so sources can be configured post-construction.
        self.sources = list(sources) if sources else []
        self._offsets: Optional[List[int]] = None

    @property
    def offsets(self) -> List[int]:
        """Cumulative END offsets per sub-source, computed lazily on first access and cached.

        Computing them validates each sub-source (``__len__`` / ``__getitem__``); enables an
        O(log k) global-index → (sub-source, local index) map.
        """
        if self._offsets is None:
            offsets: List[int] = []
            total = 0
            for i, src in enumerate(self.sources):
                _guard_live_source(src, f"ConcatSource.sources[{i}]")
                if not hasattr(src, "__len__") or not hasattr(src, "__getitem__"):
                    raise TypeError(
                        "ConcatSource requires sources supporting __len__ and __getitem__; "
                        f"source[{i}] is {type(src).__name__}"
                    )
                total += len(src)
                offsets.append(total)
            self._offsets = offsets
        return self._offsets

    def __len__(self) -> int:
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(self, index: int) -> Any:
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(index)
        j = bisect.bisect_right(self.offsets, index)
        start = self.offsets[j - 1] if j > 0 else 0
        return _pass_through(self.sources[j][index - start])

    def __iter__(self) -> Iterator[Record]:
        for src in self.sources:
            for item in src:
                yield _pass_through(item)
