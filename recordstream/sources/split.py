"""``DatasetSplit`` — reproducible train/val/test views over an indexable source."""

import random
from typing import Any, Dict, Iterator, List, Literal, Optional, get_args

from confluid import configurable

from recordstream.items import Record
from recordstream.sources.base import _pass_through

# Closed set of split names for DatasetSplit's fraction mode (workspace mandate: prefer
# closed Literals over bare strings — self-documenting + machine-introspectable by UIs /
# navigaitor form-spec / MCP schemas via ``typing.get_args``). The runtime-validation tuple
# is derived from the Literal so there is ONE source of truth — never restate the values.
SplitName = Literal["train", "val", "test"]
_SPLIT_NAMES = get_args(SplitName)


@configurable(category="source")
class DatasetSplit:
    """
    Splits an indexable source into reproducible ``train`` / ``val`` / ``test`` views.

    A ``source`` (it yields records and is wired into a trainer's ``source:`` slot),
    not an engine — it applies no ops, it just exposes a reproducible partition of another
    source. (For a contiguous index slice use :class:`~recordstream.sources.RangeSource`; to
    concatenate several sources use :class:`~recordstream.sources.ConcatSource`.)

    **Property API (preferred).** Configure ONE ``DatasetSplit`` with ``seed`` and the
    held-out fraction(s) (``val_fraction`` and/or ``test_fraction``) and read the three
    cached view sources off it::

        split = DatasetSplit(source=src, val_fraction=0.1, test_fraction=0.1, seed=42)
        split.train   # ≈80% — the remainder
        split.val     # ≈10%
        split.test    # ≈10%

    The views are disjoint and complementary, computed once (cached) over a single
    deterministic shuffle, so the underlying source is consumed once. In Confluid YAML the
    views are reachable by **attribute reference** — ``!ref:my_split.train`` / ``.val`` /
    ``.test`` — and because two ``!ref:`` to the same key flow the *same* instance, the
    partition and the source load are shared across all three references::

        my_split: !class:recordstream.sources.split.DatasetSplit()
          source: !ref:hf_train
          val_fraction: 0.1
          test_fraction: 0.1
          seed: 42

        train_set: !class:recordstream.core.Stream()
          source: !ref:my_split.train
        val_set: !class:recordstream.core.Stream()
          source: !ref:my_split.val

    **Select-one API.** Passing ``split`` makes the ``DatasetSplit`` itself iterate that one
    view (``split=None`` ⇒ ``train``), so it is directly usable as a single ``source:``.

    Omit ``test_fraction`` for a plain two-way train/val split; omit both fractions for a
    degenerate split where ``train`` is the whole source and ``val`` / ``test`` are empty.

    The wrapped source must implement ``__len__`` and ``__getitem__``. Lazy: only index
    arithmetic happens up front; records are produced on demand.

    Args:
        source: The underlying indexable source (defaults to ``None``; validated lazily on first use).
        split: View this iterates as a source — ``train`` / ``val`` / ``test`` (``None`` ⇒ ``train``).
        val_fraction: Fraction of records assigned to the ``val`` view. Must be in ``(0, 1)``.
        test_fraction: Fraction of records assigned to the ``test`` view. Must be in ``(0, 1)``.
        seed: Seed for the deterministic shuffle. Required when any fraction is set.
    """

    def __init__(
        self,
        source: Any = None,
        split: Optional[SplitName] = None,
        val_fraction: Optional[float] = None,
        test_fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        # Lazy / zero-arg: store config only. All validation is deferred to first materialization
        # (``_validate``, invoked from ``_view``) so the source can be configured post-construction.
        self.source = source
        self.split = split
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        # Cache of materialized split views. Underscore-prefixed so confluid's
        # vars(obj)-based discovery / dump ignores it (the `train`/`val`/`test`
        # @property descriptors live on the class, not in vars(obj), so they never
        # surface as configurable attributes either).
        self._views: Dict[str, "_SplitView"] = {}

    def _validate(self) -> None:
        """Validate the (post-construction) configuration. Called lazily before the first partition."""
        source = self.source
        if source is None or not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
            raise TypeError(
                "DatasetSplit requires a source supporting __len__ and __getitem__; " f"got {type(source).__name__}"
            )
        if self.split is not None and self.split not in _SPLIT_NAMES:
            raise ValueError(f"split must be one of {_SPLIT_NAMES}; got {self.split!r}")
        if (self.val_fraction is not None or self.test_fraction is not None) and self.seed is None:
            raise ValueError("DatasetSplit requires `seed` when a fraction is set, so the partition is reproducible.")
        if self.val_fraction is not None and not (0.0 < self.val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0, 1); got {self.val_fraction}")
        if self.test_fraction is not None and not (0.0 < self.test_fraction < 1.0):
            raise ValueError(f"test_fraction must be in (0, 1); got {self.test_fraction}")
        if (self.val_fraction or 0.0) + (self.test_fraction or 0.0) >= 1.0:
            raise ValueError(
                "val_fraction + test_fraction must be < 1 (to leave a non-empty train split); "
                f"got val_fraction={self.val_fraction}, test_fraction={self.test_fraction}"
            )

    def _partition(self) -> Dict[str, List[int]]:
        """Deterministically partition the source indices into ``train`` / ``val`` / ``test``.

        One shuffle seeded by ``seed`` (skipped when no fraction is set, so the degenerate
        "all train" case keeps source order); layout is ``[train | val | test]``. ``max(1, …)``
        guarantees a held-out split gets at least one record on tiny sources.
        """
        n = len(self.source)
        val_fraction = self.val_fraction or 0.0
        test_fraction = self.test_fraction or 0.0
        shuffled = list(range(n))
        if val_fraction or test_fraction:
            random.Random(self.seed).shuffle(shuffled)
        val_count = max(1, int(round(n * val_fraction))) if val_fraction else 0
        test_count = max(1, int(round(n * test_fraction))) if test_fraction else 0
        train_count = max(0, n - val_count - test_count)
        return {
            "train": shuffled[:train_count],
            "val": shuffled[train_count : train_count + val_count],
            "test": shuffled[train_count + val_count :],
        }

    def _view(self, split: SplitName) -> "_SplitView":
        if split not in self._views:
            self._validate()
            self._views[split] = _SplitView(self.source, self._partition()[split])
        return self._views[split]

    @property
    def train(self) -> "_SplitView":
        """Cached training-split view (the remainder after ``val`` / ``test`` are held out)."""
        return self._view("train")

    @property
    def val(self) -> "_SplitView":
        """Cached validation-split view (≈ ``val_fraction`` of the source)."""
        return self._view("val")

    @property
    def test(self) -> "_SplitView":
        """Cached test-split view (≈ ``test_fraction`` of the source)."""
        return self._view("test")

    def __iter__(self) -> Iterator[Record]:
        return iter(self._view(self.split or "train"))

    def __getitem__(self, index: int) -> Any:
        return self._view(self.split or "train")[index]

    def __len__(self) -> int:
        return len(self._view(self.split or "train"))


class _SplitView:
    """An indexable view of ``source`` restricted (and reordered) to ``indices``.

    Internal to :class:`DatasetSplit` — produced by its ``train`` / ``val`` / ``test``
    properties (and reachable in Confluid YAML via ``!ref:my_split.train``). Deliberately
    NOT a ``@configurable``: it is never constructed directly in a config, only read off a
    live ``DatasetSplit`` instance, so it carries no discovery surface of its own. It stays
    in this module for the same reason — it is DatasetSplit's own return type, not a
    separately-exported source.
    """

    def __init__(self, source: Any, indices: List[int]) -> None:
        self.source = source
        self.indices = indices

    def __iter__(self) -> Iterator[Record]:
        for idx in self.indices:
            yield _pass_through(self.source[idx])

    def __getitem__(self, index: int) -> Any:
        return _pass_through(self.source[self.indices[index]])

    def __len__(self) -> int:
        return len(self.indices)
