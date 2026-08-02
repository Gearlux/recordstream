"""``HuggingFaceSource`` — a Hugging Face dataset as a stream of record dicts."""

from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Optional
from urllib.parse import quote, urlencode

from confluid import configurable
from loggair import get_logger

from recordstream.items import Image, Label, Record

logger = get_logger(__name__)

#: Scheme of the canonical identifier for a Hub dataset. Matches the convention hosted
#: tracking services already use for a dataset source, so a URI recorded here is the one
#: their UI expects rather than a spelling invented for this package.
HF_URI_PREFIX = "hf://datasets/"

#: Where a Hub dataset is browsable. The ``/viewer/<config>/<split>`` suffix opens the
#: dataset viewer on exactly the rows this source reads.
HF_BROWSE_PREFIX = "https://huggingface.co/datasets/"

#: The config name the Hub viewer uses when a dataset declares no named configs.
HF_DEFAULT_CONFIG = "default"

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
    RecordStream Source for Hugging Face Datasets, yielding plain record dicts.

    Key mapping (the record layout):

    * the ``input_feature`` value (image / array) -> an :class:`~recordstream.Image` under the
      record key ``"image"``;
    * the ``target_feature`` value (label) -> a :class:`~recordstream.Label` under the record key
      ``"class"``;
    * each ``metadata_features`` column -> its own :class:`~recordstream.Label` keyed by the column
      name, plus the source-provenance ``hf_path`` / ``hf_split`` Labels.

    Lazy & zero-arg per the workspace class-design convention (see confluid AGENTS.md
    "Lazy Initialization & Zero-Arg Construction"): the constructor only stores values and
    does NO functional work — ``HuggingFaceSource()`` is valid, and the dataset is downloaded
    only on first access to :attr:`dataset` (cached thereafter; reset ``_dataset`` to reload).
    ``path`` is therefore optional at construction and validated lazily when the data is needed.

    It also carries its own IDENTITY (:mod:`recordstream.uri`): :attr:`dataset_uri` names the
    dataset canonically (``hf://datasets/ylecun/mnist?split=train``, or a ``file://`` URI for a
    local imagefolder) and :attr:`dataset_url` links to the Hub viewer for the same rows. Both
    read stored configuration only — asking either never loads anything.

    Args:
        path: HF dataset identifier — a Hub repo id (e.g. ``kitofrank/RFUAV``) or a local imagefolder path.
        split: HF split name (``train`` / ``validation`` / ``test`` / etc.).
        input_feature: Dataset feature column mapped onto the ``"image"`` record key (an ``Image`` item).
        target_feature: Dataset feature column mapped onto the ``"class"`` record key (a ``Label`` item).
        metadata_features: Columns -> per-column ``Label`` entries; ``None``=none, ``"*"``=all-but-i/o, else a list.
        count: Optional cap on the number of records yielded (useful for fast smoke runs).
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

            # The identifier goes in the LOAD line: it is the one moment a reader of the log
            # can tie the run to a specific dataset, and the browsable URL is what makes that
            # tie followable rather than merely recorded.
            logger.info(f"HuggingFaceSource: Loading {self.dataset_url or self.dataset_uri}...")
            self._dataset = load_dataset(self.path, name=self.name, split=self.split, **self._load_kwargs)
        return self._dataset

    # -- identity (see recordstream.uri) ------------------------------------------------------

    @property
    def _identity_query(self) -> str:
        """The ``name`` / ``revision`` / ``split`` selection as a sorted query string.

        Sorted so two identically-configured sources produce the SAME string — a URI whose
        parameter order depended on insertion would not compare equal to itself.
        """
        parts: Dict[str, str] = {}
        if self.name:
            parts["name"] = str(self.name)
        revision = self._load_kwargs.get("revision")
        if revision:
            parts["revision"] = str(revision)
        if self.split:
            parts["split"] = str(self.split)
        return urlencode(sorted(parts.items()))

    @property
    def dataset_uri(self) -> Optional[str]:
        """Canonical identifier for the dataset this source reads — ``None`` without a ``path``.

        A Hub repo id becomes ``hf://datasets/<path>?…``; a local directory becomes its
        ``file://`` URI. Which one applies is decided by whether ``path`` exists on disk —
        the same question ``datasets.load_dataset`` itself answers. Pure string work over the
        stored configuration: nothing is loaded, so an unconsumed source still answers.
        """
        if not self.path:
            return None
        local = Path(self.path)
        base = local.resolve().as_uri() if local.exists() else HF_URI_PREFIX + quote(str(self.path).strip("/"))
        query = self._identity_query
        return f"{base}?{query}" if query else base

    @property
    def dataset_url(self) -> Optional[str]:
        """Browsable Hub link, or ``None`` for a local dataset (which has no web page).

        Points at the dataset VIEWER on this source's config + split when a split is
        configured, so the link opens on the rows this source reads rather than on the
        repository's front page.
        """
        if not self.path or Path(self.path).exists():
            return None
        page = HF_BROWSE_PREFIX + quote(str(self.path).strip("/"))
        if not self.split:
            return page
        config = quote(str(self.name)) if self.name else HF_DEFAULT_CONFIG
        return f"{page}/viewer/{config}/{quote(str(self.split))}"

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

        Only the requested keys are built, so a label-only walk (e.g. :func:`~recordstream.num_classes`)
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
        # A ``count`` of 0 (or None) means "all records", matching __iter__'s
        # ``limit = self.count or len(...)``. Returning a bare ``self.count`` here
        # would report 0 for the common "0 == unlimited" case, making the source
        # look empty (e.g. a downstream len()-based stepper raising ``len == 0``)
        # even though iteration yields every record.
        return self.count or len(self.dataset)
