import bisect
import random
from typing import Any, Collection, Dict, Iterator, List, Literal, Optional, get_args

from confluid import configurable
from loggair import get_logger

from sampleflux.items import Image, Label, Record

logger = get_logger(__name__)


def _pass_through(item: Any) -> Any:
    """Pass a wrapped source's item through verbatim.

    Every carrier is a plain record dict; the view sources
    (:class:`DatasetSplit` / :class:`RangeSource` / :class:`ConcatSource`) only slice/index,
    they never inspect payloads, so a source's records flow through them unchanged.
    """
    return item


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
# Visual editors offer it as a selectable "*" entry in a metadata picker.
METADATA_ALL_FEATURES = "*"


def _resolve_metadata_features(
    requested: Optional[List[str] | str],
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
    SampleFlux Source for Hugging Face Datasets, yielding plain record dicts.

    Key mapping (the record layout):

    * the ``input_feature`` value (image / array) -> an :class:`~sampleflux.Image` under the
      record key ``"image"``;
    * the ``target_feature`` value (label) -> a :class:`~sampleflux.Label` under the record key
      ``"class"``;
    * each ``metadata_features`` column -> its own :class:`~sampleflux.Label` keyed by the column
      name, plus the source-provenance ``hf_path`` / ``hf_split`` Labels.

    Lazy & zero-arg per the workspace class-design convention (see confluid AGENTS.md
    "Lazy Initialization & Zero-Arg Construction"): the constructor only stores values and
    does NO functional work — ``HuggingFaceSource()`` is valid, and the dataset is downloaded
    only on first access to :attr:`dataset` (cached thereafter; reset ``_dataset`` to reload).
    ``path`` is therefore optional at construction and validated lazily when the data is needed.

    Args:
        path: HF dataset identifier — a Hub repo id (e.g. ``kitofrank/RFUAV``) or a local imagefolder path.
        split: HF split name (``train`` / ``validation`` / ``test`` / etc.).
        input_feature: Dataset feature column mapped onto the ``"image"`` record key (an ``Image`` item).
        target_feature: Dataset feature column mapped onto the ``"class"`` record key (a ``Label`` item).
        metadata_features: Columns -> per-column ``Label`` entries; ``None``=none, ``"*"``=all-but-i/o, else a list.
        count: Optional cap on the number of samples yielded (useful for fast smoke runs).
        name: Optional HF subset/config name (e.g. for multi-config datasets).
    """

    def __init__(
        self,
        path: str = "",
        split: str = "train",
        input_feature: str = "image",
        target_feature: str = "label",
        metadata_features: Optional[List[str] | str] = "*",
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

    def _to_record(
        self,
        item: Any,
        metadata_features: List[str],
        keys: Optional[Collection[str]] = None,
    ) -> Record:
        """Assemble one record dict from a raw HF row dict (see the class docstring for the key mapping).

        ``keys`` gates which record entries are built (``None`` = all) — the projection path
        (:meth:`project`) passes only the requested ones, so an unwanted image is never decoded.
        ``metadata_features`` arrives pre-filtered on the projection path.
        """
        record: Record = {}
        if keys is None or "image" in keys:
            # The input value (image/array) becomes an ``Image`` item; a PIL image / list is coerced
            # to an ndarray by ``Image.__new__`` (np.asarray), preserving the default HWC layout.
            record["image"] = Image(item.get(self.input_feature))
        if keys is None or "class" in keys:
            record["class"] = Label(item.get(self.target_feature))
        # Each requested metadata column rides its OWN Label entry keyed by the column name (the
        # metadata a value needs travels WITH it). Source provenance follows the same shape.
        for feature in metadata_features:
            record[feature] = Label(item.get(feature))
        if keys is None or "hf_path" in keys:
            record["hf_path"] = Label(self.path)
        if keys is None or "hf_split" in keys:
            record["hf_split"] = Label(self.split)
        return record

    def __iter__(self) -> Iterator[Record]:
        dataset = self.dataset
        metadata_features = self.resolved_metadata_features
        limit = self.count or len(dataset)

        for counter, item in enumerate(dataset):
            if counter >= limit:
                break
            yield self._to_record(item, metadata_features)

    def __getitem__(self, index: int) -> Record:
        return self._to_record(self.dataset[index], self.resolved_metadata_features)

    def project(self, keys: Collection[str]) -> Iterator[Record]:
        """Yield key-restricted records — the ``SupportsProjection`` efficient path.

        Only the requested keys are built, so a label-only walk (e.g. :func:`~sampleflux.num_classes`)
        skips decoding the image entirely: ``"image"`` -> the input feature, ``"class"`` -> the target
        Label, plus any requested metadata-column / provenance keys.
        """
        want = frozenset(keys)
        dataset = self.dataset
        # Resolve (and pre-filter) the metadata columns only when a key beyond the fixed image/class
        # pair is requested — the "*" expansion needs the loaded dataset's columns.
        meta_requested = bool(want - {"image", "class"})
        metadata_features = [f for f in self.resolved_metadata_features if f in want] if meta_requested else []
        limit = self.count or len(dataset)
        for counter, item in enumerate(dataset):
            if counter >= limit:
                break
            yield self._to_record(item, metadata_features, keys=want)

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

    A ``source`` (it yields records and is wired into a trainer's ``source:`` slot),
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

        my_split: !class:sampleflux.sources.DatasetSplit()
          source: !ref:hf_train
          val_fraction: 0.1
          test_fraction: 0.1
          seed: 42

        train_set: !class:sampleflux.core.Flux()
          source: !ref:my_split.train
        val_set: !class:sampleflux.core.Flux()
          source: !ref:my_split.val

    **Select-one API.** Passing ``split`` makes the ``DatasetSplit`` itself iterate that one
    view (``split=None`` ⇒ ``train``), so it is directly usable as a single ``source:``.

    Omit ``test_fraction`` for a plain two-way train/val split; omit both fractions for a
    degenerate split where ``train`` is the whole source and ``val`` / ``test`` are empty.

    The wrapped source must implement ``__len__`` and ``__getitem__``. Lazy: only index
    arithmetic happens up front; samples are produced on demand.

    Args:
        source: The underlying indexable source (defaults to ``None``; validated lazily on first use).
        split: View this iterates as a source — ``train`` / ``val`` / ``test`` (``None`` ⇒ ``train``).
        val_fraction: Fraction of samples assigned to the ``val`` view. Must be in ``(0, 1)``.
        test_fraction: Fraction of samples assigned to the ``test`` view. Must be in ``(0, 1)``.
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
    live ``DatasetSplit`` instance, so it carries no discovery surface of its own.
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


@configurable(category="source")
class RangeSource:
    """A contiguous index slice ``[start:stop)`` over an indexable source.

    The plain-slice counterpart to :class:`DatasetSplit` (which shuffles + partitions) —
    extracted from DatasetSplit's old "range mode". Negative ``start`` / ``stop`` count from
    the end; both are clamped to ``[0, len(source)]``. Lazy: only index arithmetic happens
    up front; samples are produced on demand.

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


@configurable(category="source")
class ConcatSource:
    """Concatenates multiple indexable sources into one longer indexable source.

    The indexable counterpart to :class:`sampleflux.core.JointFlux` (which is iteration-only):
    ``len`` is the sum of the parts and ``source[i]`` maps a global index onto the owning
    sub-source, so a ``ConcatSource`` can itself be wrapped by :class:`DatasetSplit` /
    :class:`RangeSource`. (Distinct from :class:`waivefront.paired.AnnotationJoinSource`, which
    *column-joins* annotations onto samples — this one *concatenates* sequences end to end.)

    Each sub-source must implement ``__len__`` and ``__getitem__``.

    Args:
        sources: The indexable sources to concatenate, walked in order (defaults to ``None`` ⇒ empty).
    """

    def __init__(self, sources: Optional[List[Any]] = None) -> None:
        # Lazy / zero-arg: store config only; sub-source validation + the cumulative-offset precompute
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
