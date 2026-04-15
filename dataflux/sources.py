import random
from typing import Any, Iterator, List, Optional

from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)


@configurable
class HuggingFaceSource:
    """
    DataFlux Source for Hugging Face Datasets.
    Configurable mapping of dataset features to DataFlux Sample triplets.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        input_feature: str = "image",
        target_feature: str = "label",
        metadata_features: Optional[List[str]] = None,
        count: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        from datasets import load_dataset

        self.path = path
        self.split = split
        self.input_feature = input_feature
        self.target_feature = target_feature
        self.metadata_features = metadata_features or []
        self.count = count

        logger.info(f"HuggingFaceSource: Loading {path} ({split})...")
        self._dataset = load_dataset(path, name=name, split=split, **kwargs)

    def __iter__(self) -> Iterator[Sample]:
        counter = 0
        limit = self.count or len(self._dataset)

        for item in self._dataset:
            if counter >= limit:
                break

            # 1. Extract Input
            input_val = item.get(self.input_feature)

            # 2. Extract Target
            target_val = item.get(self.target_feature)

            # 3. Build Metadata
            metadata = {f: item.get(f) for f in self.metadata_features}
            metadata["hf_path"] = self.path
            metadata["hf_split"] = self.split

            yield Sample(input=input_val, target=target_val, metadata=metadata)
            counter += 1

    def __getitem__(self, index: int) -> Sample:
        item = self._dataset[index]
        metadata = {f: item.get(f) for f in self.metadata_features}
        metadata["hf_path"] = self.path
        metadata["hf_split"] = self.split
        return Sample(input=item.get(self.input_feature), target=item.get(self.target_feature), metadata=metadata)

    def __len__(self) -> int:
        if self.count is not None:
            return self.count
        return len(self._dataset)


@configurable
class DatasetSplit:
    """
    Selects a subset view of an indexable source (e.g. ``HuggingFaceSource``).

    Supports three mutually exclusive modes:

    1. **Fraction mode** — pick a reproducible train/val split from a single source.
       Pass ``split`` (``"train"`` or ``"val"``), ``val_fraction``, and ``seed``.
       Two ``DatasetSplit`` instances sharing a source (by Python identity or
       by ``!ref:`` in YAML) and the *same* ``seed`` + ``val_fraction`` yield
       disjoint, complementary views. With Confluid ``!ref:``, the underlying
       source is loaded exactly once and shared by identity between the splits.

    2. **Range mode** — plain index slice ``[start:end)`` over the source.
       Pass ``start`` and/or ``end``.

    3. **Passthrough** — no split args yields a full view (rarely useful,
       mostly for symmetry in YAML templates).

    The wrapped source must implement ``__len__`` and ``__getitem__``.
    Lazy evaluation is preserved: only index arithmetic happens at
    construction time; samples are produced on demand.

    Args:
        source: The underlying indexable source.
        split: ``"train"`` or ``"val"`` (fraction mode only).
        val_fraction: Fraction of samples assigned to the ``"val"`` view
            (fraction mode). Must be in ``(0, 1)``.
        seed: Seed for the deterministic shuffle (fraction mode).
            Required when ``val_fraction`` is set so that sibling splits
            stay consistent.
        start: Inclusive start index (range mode).
        end: Exclusive end index (range mode).

    Example (fraction mode, YAML)::

        hf_train: !class:dataflux.sources.HuggingFaceSource()
          path: mnist
          split: train

        train_set: !class:dataflux.sources.DatasetSplit()
          source: !ref:hf_train
          split: train
          val_fraction: 0.1
          seed: 42

        val_set: !class:dataflux.sources.DatasetSplit()
          source: !ref:hf_train
          split: val
          val_fraction: 0.1
          seed: 42

    Alternative (no shared load, HuggingFace native slicing)::

        # Loads the HF dataset twice — kept for clarity if sharing is not desired.
        train_src: !class:dataflux.sources.HuggingFaceSource()
          path: mnist
          split: "train[:90%]"
        val_src: !class:dataflux.sources.HuggingFaceSource()
          path: mnist
          split: "train[90%:]"
    """

    def __init__(
        self,
        source: Any,
        split: Optional[str] = None,
        val_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        if not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
            raise TypeError(
                "DatasetSplit requires a source supporting __len__ and __getitem__; " f"got {type(source).__name__}"
            )

        fraction_args = val_fraction is not None or split is not None
        range_args = start is not None or end is not None
        if fraction_args and range_args:
            raise ValueError(
                "DatasetSplit accepts either fraction-mode args (split, val_fraction, seed) "
                "or range-mode args (start, end), not both."
            )

        self.source = source
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed
        self.start = start
        self.end = end

        self._indices: List[int] = self._compute_indices()
        logger.debug(
            "DatasetSplit: mode=%s size=%d source_size=%d",
            "fraction" if fraction_args else ("range" if range_args else "passthrough"),
            len(self._indices),
            len(source),
        )

    def _compute_indices(self) -> List[int]:
        n = len(self.source)

        # Fraction mode
        if self.val_fraction is not None or self.split is not None:
            if self.val_fraction is None:
                raise ValueError("DatasetSplit fraction mode requires `val_fraction`.")
            if self.seed is None:
                raise ValueError("DatasetSplit fraction mode requires `seed` so that sibling splits stay consistent.")
            if not (0.0 < self.val_fraction < 1.0):
                raise ValueError(f"val_fraction must be in (0, 1); got {self.val_fraction}")
            if self.split not in ("train", "val"):
                raise ValueError(f"split must be 'train' or 'val' in fraction mode; got {self.split!r}")

            rng = random.Random(self.seed)
            shuffled = list(range(n))
            rng.shuffle(shuffled)
            val_count = max(1, int(round(n * self.val_fraction)))
            train_count = n - val_count
            if self.split == "train":
                return shuffled[:train_count]
            return shuffled[train_count:]

        # Range mode
        if self.start is not None or self.end is not None:
            start = 0 if self.start is None else self.start
            end = n if self.end is None else self.end
            if start < 0:
                start = max(0, n + start)
            if end < 0:
                end = max(0, n + end)
            start = max(0, min(start, n))
            end = max(start, min(end, n))
            return list(range(start, end))

        # Passthrough
        return list(range(n))

    def __iter__(self) -> Iterator[Sample]:
        for idx in self._indices:
            yield Sample.from_any(self.source[idx])

    def __getitem__(self, index: int) -> Sample:
        return Sample.from_any(self.source[self._indices[index]])

    def __len__(self) -> int:
        return len(self._indices)
