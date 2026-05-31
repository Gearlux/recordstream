import bisect
import random
from typing import Any, Dict, Iterator, List, Literal, Optional, get_args

from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)

# Closed set of split names for DatasetSplit's fraction mode (workspace mandate: prefer
# closed Literals over bare strings — self-documenting + machine-introspectable by UIs /
# navigaitor form-spec / MCP schemas via ``typing.get_args``). The runtime-validation tuple
# is derived from the Literal so there is ONE source of truth — never restate the values.
SplitName = Literal["train", "val", "test"]
_SPLIT_NAMES = get_args(SplitName)

# Sentinel for ``HuggingFaceSource.metadata_features`` meaning "every dataset column except the
# input/target features" — the full-traceability option, kept OPT-IN (``None`` / ``[]`` still = no
# extra metadata) so existing configs are unaffected. Resolved against the loaded dataset's
# ``column_names`` at construction. Accepted bare (``"*"``) or as the one-element list (``["*"]``);
# FluxStudio's metadata picker offers it as a selectable "*" entry.
METADATA_ALL_FEATURES = "*"


def _resolve_metadata_features(
    requested: Optional[Any],
    column_names: Optional[List[str]],
    input_feature: str,
    target_feature: str,
) -> List[str]:
    """Resolve a ``metadata_features`` spec into a concrete, order-preserving column list.

    ``None`` / ``[]`` -> ``[]`` (no extra metadata — the backward-compatible default). The sentinel
    ``"*"`` (bare or inside a list) -> every column in ``column_names`` except ``input_feature`` /
    ``target_feature`` (full traceability). An explicit list of names is used verbatim. ``"*"`` may
    be combined with extra names (union, order-preserving: the "rest" first, then the extras).
    """
    if not requested:
        return []
    if isinstance(requested, str):
        requested = [requested]
    if METADATA_ALL_FEATURES not in requested:
        return list(requested)
    excluded = {input_feature, target_feature}
    rest = [c for c in (column_names or []) if c not in excluded]
    extras = [r for r in requested if r != METADATA_ALL_FEATURES and r not in excluded and r not in rest]
    return rest + extras


@configurable(category="source")
class HuggingFaceSource:
    """
    DataFlux Source for Hugging Face Datasets.
    Configurable mapping of dataset features to DataFlux Sample triplets.

    Lazy & zero-arg per the workspace class-design convention (see confluid AGENTS.md
    "Lazy Initialization & Zero-Arg Construction"): the constructor only stores values and
    does NO functional work — ``HuggingFaceSource()`` is valid, and the dataset is downloaded
    only on first access to :attr:`dataset` (cached thereafter; reset ``_dataset`` to reload).
    ``path`` is therefore optional at construction and validated lazily when the data is needed.

    Args:
        path: HF dataset identifier — a Hub repo id (e.g. ``kitofrank/RFUAV``) or a local imagefolder path.
        split: HF split name (``train`` / ``validation`` / ``test`` / etc.).
        input_feature: Dataset feature column to map onto ``Sample.input``.
        target_feature: Dataset feature column to map onto ``Sample.target``.
        metadata_features: Columns onto ``Sample.metadata``; ``None``=none, ``"*"``=all but input/target, else a list.
        count: Optional cap on the number of samples yielded (useful for fast smoke runs).
        name: Optional HF subset/config name (e.g. for multi-config datasets).
    """

    def __init__(
        self,
        path: str = "",
        split: str = "train",
        input_feature: str = "image",
        target_feature: str = "label",
        metadata_features: Optional[List[str]] = None,
        count: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Lazy constructor: store config only — never load here. Real work (the network/disk
        # download) is deferred to the ``dataset`` property so the object is cheap to build and
        # configurable post-construction.
        self.path = path
        self.split = split
        self.input_feature = input_feature
        self.target_feature = target_feature
        # Stored as the RAW spec (``None`` / ``"*"`` / list) — resolved against the loaded dataset's
        # columns lazily by the ``resolved_metadata_features`` property, not eagerly here.
        self.metadata_features = metadata_features
        self.count = count
        self.name = name
        # Extra kwargs forwarded verbatim to ``datasets.load_dataset`` at load time (e.g. ``token``,
        # ``trust_remote_code``). Captured now, applied lazily in the ``dataset`` property.
        self._load_kwargs = dict(kwargs)
        # Lazy cache for the materialized dataset (see the ``dataset`` property).
        self._dataset: Any = None

    @property
    def dataset(self) -> Any:
        """The HF dataset, loaded on first access and cached. Resetting ``_dataset`` to None reloads.

        Raises ``ValueError`` if ``path`` was never set — the zero-arg constructor allows building an
        unconfigured source, but materializing one without a dataset id cannot succeed.
        """
        if self._dataset is None:
            if not self.path:
                raise ValueError(
                    "HuggingFaceSource.path is empty — set it (constructor arg, YAML, or configure()) "
                    "before iterating or indexing the source."
                )
            from datasets import load_dataset

            logger.info(f"HuggingFaceSource: Loading {self.path} ({self.split})...")
            self._dataset = load_dataset(self.path, name=self.name, split=self.split, **self._load_kwargs)
        return self._dataset

    @property
    def resolved_metadata_features(self) -> List[str]:
        """``metadata_features`` resolved against the live dataset's columns (expands the ``"*"`` sentinel).

        Lazy because the ``"*"`` expansion needs the loaded dataset's ``column_names``; ``None`` / ``[]``
        stays "no extra metadata" (backward-compatible).
        """
        return _resolve_metadata_features(
            self.metadata_features, getattr(self.dataset, "column_names", None), self.input_feature, self.target_feature
        )

    def __iter__(self) -> Iterator[Sample]:
        counter = 0
        dataset = self.dataset
        metadata_features = self.resolved_metadata_features
        limit = self.count or len(dataset)

        for item in dataset:
            if counter >= limit:
                break

            # 1. Extract Input
            input_val = item.get(self.input_feature)

            # 2. Extract Target
            target_val = item.get(self.target_feature)

            # 3. Build Metadata
            metadata = {f: item.get(f) for f in metadata_features}
            metadata["hf_path"] = self.path
            metadata["hf_split"] = self.split

            yield Sample(input=input_val, target=target_val, metadata=metadata)
            counter += 1

    def __getitem__(self, index: int) -> Sample:
        item = self.dataset[index]
        metadata = {f: item.get(f) for f in self.resolved_metadata_features}
        metadata["hf_path"] = self.path
        metadata["hf_split"] = self.split
        return Sample(
            input=item.get(self.input_feature),
            target=item.get(self.target_feature),
            metadata=metadata,
        )

    def __len__(self) -> int:
        # A ``count`` of 0 (or None) means "all samples", matching __iter__'s
        # ``limit = self.count or len(...)``. Returning a bare ``self.count`` here
        # would report 0 for the common "0 == unlimited" case, making the source
        # look empty (e.g. a downstream len()-based stepper raising ``len == 0``)
        # even though iteration yields every sample.
        return self.count or len(self.dataset)


@configurable(category="source")
class DatasetSplit:
    """
    Splits an indexable source into reproducible ``train`` / ``val`` / ``test`` views.

    A ``source`` (it yields ``Sample``s and is wired into a trainer's ``source:`` slot),
    not an engine — it applies no ops, it just exposes a reproducible partition of another
    source. (For a contiguous index slice use :class:`RangeSource`; to concatenate several
    sources use :class:`ConcatSource`.)

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

        my_split: !class:dataflux.sources.DatasetSplit()
          source: !ref:hf_train
          val_fraction: 0.1
          test_fraction: 0.1
          seed: 42

        train_set: !class:dataflux.core.Flux()
          source: !ref:my_split.train
        val_set: !class:dataflux.core.Flux()
          source: !ref:my_split.val

    **Select-one API.** Passing ``split`` makes the ``DatasetSplit`` itself iterate that one
    view (``split=None`` ⇒ ``train``), so it is directly usable as a single ``source:``.

    Omit ``test_fraction`` for a plain two-way train/val split; omit both fractions for a
    degenerate split where ``train`` is the whole source and ``val`` / ``test`` are empty.

    The wrapped source must implement ``__len__`` and ``__getitem__``. Lazy: only index
    arithmetic happens up front; samples are produced on demand.

    Args:
        source: The underlying indexable source.
        split: View this iterates as a source — ``train`` / ``val`` / ``test`` (``None`` ⇒ ``train``).
        val_fraction: Fraction of samples assigned to the ``val`` view. Must be in ``(0, 1)``.
        test_fraction: Fraction of samples assigned to the ``test`` view. Must be in ``(0, 1)``.
        seed: Seed for the deterministic shuffle. Required when any fraction is set.
    """

    def __init__(
        self,
        source: Any,
        split: Optional[SplitName] = None,
        val_fraction: Optional[float] = None,
        test_fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
            raise TypeError(
                "DatasetSplit requires a source supporting __len__ and __getitem__; " f"got {type(source).__name__}"
            )
        if split is not None and split not in _SPLIT_NAMES:
            raise ValueError(f"split must be one of {_SPLIT_NAMES}; got {split!r}")
        if (val_fraction is not None or test_fraction is not None) and seed is None:
            raise ValueError("DatasetSplit requires `seed` when a fraction is set, so the partition is reproducible.")
        if val_fraction is not None and not (0.0 < val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
        if test_fraction is not None and not (0.0 < test_fraction < 1.0):
            raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")
        if (val_fraction or 0.0) + (test_fraction or 0.0) >= 1.0:
            raise ValueError(
                "val_fraction + test_fraction must be < 1 (to leave a non-empty train split); "
                f"got val_fraction={val_fraction}, test_fraction={test_fraction}"
            )

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

    def _partition(self) -> Dict[str, List[int]]:
        """Deterministically partition the source indices into ``train`` / ``val`` / ``test``.

        One shuffle seeded by ``seed`` (skipped when no fraction is set, so the degenerate
        "all train" case keeps source order); layout is ``[train | val | test]``. ``max(1, …)``
        guarantees a held-out split gets at least one sample on tiny sources.
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

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._view(self.split or "train"))

    def __getitem__(self, index: int) -> Sample:
        return self._view(self.split or "train")[index]

    def __len__(self) -> int:
        return len(self._view(self.split or "train"))


class _SplitView:
    """An indexable view of ``source`` restricted (and reordered) to ``indices``.

    Internal to :class:`DatasetSplit` — produced by its ``train`` / ``val`` / ``test``
    properties (and reachable in Confluid YAML via ``!ref:my_split.train``). Deliberately
    NOT a ``@configurable``: it is never constructed directly in a config, only read off a
    live ``DatasetSplit`` instance, so it carries no discovery surface of its own.
    """

    def __init__(self, source: Any, indices: List[int]) -> None:
        self.source = source
        self.indices = indices

    def __iter__(self) -> Iterator[Sample]:
        for idx in self.indices:
            yield Sample.from_any(self.source[idx])

    def __getitem__(self, index: int) -> Sample:
        return Sample.from_any(self.source[self.indices[index]])

    def __len__(self) -> int:
        return len(self.indices)


@configurable(category="source")
class RangeSource:
    """A contiguous index slice ``[start:end)`` over an indexable source.

    The plain-slice counterpart to :class:`DatasetSplit` (which shuffles + partitions) —
    extracted from DatasetSplit's old "range mode". Negative ``start`` / ``end`` count from
    the end; both are clamped to ``[0, len(source)]``. Lazy: only index arithmetic happens
    up front; samples are produced on demand.

    The wrapped source must implement ``__len__`` and ``__getitem__``.

    Args:
        source: The underlying indexable source.
        start: Inclusive start index (``None`` ⇒ 0; a negative value counts from the end).
        end: Exclusive end index (``None`` ⇒ len(source); a negative value counts from the end).
    """

    def __init__(self, source: Any, start: Optional[int] = None, end: Optional[int] = None) -> None:
        if not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
            raise TypeError(
                "RangeSource requires a source supporting __len__ and __getitem__; " f"got {type(source).__name__}"
            )
        self.source = source
        self.start = start
        self.end = end
        n = len(source)
        s = 0 if start is None else start
        e = n if end is None else end
        if s < 0:
            s = max(0, n + s)
        if e < 0:
            e = max(0, n + e)
        s = max(0, min(s, n))
        e = max(s, min(e, n))
        self._indices: List[int] = list(range(s, e))
        logger.debug("RangeSource: size=%d source_size=%d", len(self._indices), n)

    def __iter__(self) -> Iterator[Sample]:
        for idx in self._indices:
            yield Sample.from_any(self.source[idx])

    def __getitem__(self, index: int) -> Sample:
        return Sample.from_any(self.source[self._indices[index]])

    def __len__(self) -> int:
        return len(self._indices)


@configurable(category="source")
class ConcatSource:
    """Concatenates multiple indexable sources into one longer indexable source.

    The indexable counterpart to :class:`dataflux.core.JointFlux` (which is iteration-only):
    ``len`` is the sum of the parts and ``source[i]`` maps a global index onto the owning
    sub-source, so a ``ConcatSource`` can itself be wrapped by :class:`DatasetSplit` /
    :class:`RangeSource`. (Distinct from :class:`dataflux.paired.AnnotationJoinSource`, which
    *column-joins* annotations onto samples — this one *concatenates* sequences end to end.)

    Each sub-source must implement ``__len__`` and ``__getitem__``.

    Args:
        sources: The indexable sources to concatenate, walked in order.
    """

    def __init__(self, sources: List[Any]) -> None:
        for i, src in enumerate(sources):
            if not hasattr(src, "__len__") or not hasattr(src, "__getitem__"):
                raise TypeError(
                    "ConcatSource requires sources supporting __len__ and __getitem__; "
                    f"source[{i}] is {type(src).__name__}"
                )
        self.sources = list(sources)
        # Cumulative END offsets, for an O(log k) global-index → (sub-source, local index) map.
        self._offsets: List[int] = []
        total = 0
        for src in self.sources:
            total += len(src)
            self._offsets.append(total)

    def __len__(self) -> int:
        return self._offsets[-1] if self._offsets else 0

    def __getitem__(self, index: int) -> Sample:
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(index)
        j = bisect.bisect_right(self._offsets, index)
        start = self._offsets[j - 1] if j > 0 else 0
        return Sample.from_any(self.sources[j][index - start])

    def __iter__(self) -> Iterator[Sample]:
        for src in self.sources:
            for item in src:
                yield Sample.from_any(item)
