"""``HuggingFaceSource`` — a Hugging Face dataset as a stream of record dicts."""

from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Optional
from urllib.parse import quote, urlencode

from confluid import configurable, output
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


def _source_file(value: Any) -> str:
    """The file a raw HF cell came from — ``""`` when it did not come from one.

    ``datasets`` reports it two ways depending on whether the column is decoded: a decoded PIL
    image carries ``.filename``, an undecoded one is ``{"bytes": …, "path": …}``. Both are read
    here so the caller does not have to know which form it has.
    """
    path = getattr(value, "filename", None)
    if not path and isinstance(value, dict):
        path = value.get("path")
    return str(path) if path else ""


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

    Partial & zero-arg per the workspace class-design convention (see confluid AGENTS.md
    "Partial Initialization & Zero-Arg Construction"): the constructor only stores values and
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
        revision: Optional dataset git revision (branch / tag / commit). Declared rather than
            left to ``load_kwargs`` because it is part of the source's IDENTITY — ``dataset_uri``
            reads it.
        load_kwargs: The remaining ``datasets.load_dataset`` options (``token``, ``cache_dir``,
            ``trust_remote_code``, …) as an explicit mapping. A dict rather than ``**kwargs``:
            see the note in ``__init__`` — a ``**kwargs`` constructor accepts every broadcast key
            there is, and ``load_dataset`` turns an unknown keyword into a builder config name.
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
        revision: Optional[str] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Partial constructor: store config only — never load here. Real work (the network/disk
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
        self.revision = revision
        # The long tail of `datasets.load_dataset` options (``token``, ``cache_dir``,
        # ``trust_remote_code``), as an EXPLICIT dict rather than the `**kwargs` this took until
        # 2026-08-02. That `**kwargs` was a live hazard, not merely untidy: confluid/liquifai
        # broadcast a key into any node whose constructor ACCEPTS it, and a `**kwargs` constructor
        # accepts every key there is — so unrelated run identity landed in `load_dataset`, which
        # turns unknown keyword arguments into a builder CONFIG NAME. A single
        # `sonair train … --run_name my_run` therefore looked for `mnist` under a config named
        # `default-<hash of the kwargs>`, missed the cache, and went to the Hub. Measured: the same
        # command passes with no name override and fails with one.
        self._load_kwargs = dict(load_kwargs or {})
        # Partial cache for the materialized dataset (see the ``dataset`` property).
        self._dataset: Any = None

    @property
    def load_options(self) -> Dict[str, Any]:
        """``load_kwargs`` with the declared ``revision`` folded in — what reaches ``load_dataset``.

        Recomputed on every read rather than merged in ``__init__``, per the workspace
        derived-state rule: ``revision`` is a DECLARED parameter, so the config layer may set it
        post-construction (that is how a broadcast key arrives), and a dict assembled once in the
        constructor would silently keep the value the object was born with.
        """
        options = dict(self._load_kwargs)
        if self.revision is not None:
            options["revision"] = self.revision
        return options

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
            self._dataset = load_dataset(self.path, name=self.name, split=self.split, **self.load_options)
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
        if self.revision:
            parts["revision"] = str(self.revision)
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
    @output
    def class_names(self) -> List[str]:
        """The class vocabulary this dataset declares for :attr:`target_feature`; ``[]`` if none.

        A HuggingFace ``ClassLabel`` carries its own ``names``, so a classification dataset can
        answer this from METADATA — no records read. That is what makes it worth exposing: a
        consumer connects the source and gets the vocabulary for free, instead of walking the
        label column (seconds per thousand records) or being told to type it in.

        Read from the CONFIGURED ``target_feature``, never "the first ClassLabel": a dataset may
        carry several (``oxford_iiit_pet`` has ``label`` and ``species``) and only the caller
        knows which one this source targets.

        EMPTY when there is nothing to report — the feature is absent, or is not a ``ClassLabel``
        (a detection set nests its classes inside an object field). That is not an error: most
        sources have no vocabulary, and :func:`recordstream.class_names` skips a source whose
        answer is empty.

        A list rather than ``Optional``: it is a declared ``@output``, and a visual editor types
        an output socket from this annotation — a container renders as a readable STRING you can
        preview, while an ``Optional`` falls through to an opaque object socket that invites a
        wire this value cannot satisfy.
        """
        features = getattr(self._dataset, "features", None) if self._dataset is not None else None
        if features is None:
            # The dataset has not been loaded yet, so ask the BUILDER: its info carries the
            # feature schema without downloading a single row. That is what keeps this cheap
            # enough for a visual editor to read while exporting a graph.
            try:
                from datasets import load_dataset_builder

                features = load_dataset_builder(self.path, name=self.name or None).info.features
            except Exception as exc:
                logger.debug(f"class_names: builder lookup failed ({type(exc).__name__}: {exc})")
            if not features:
                # A LOCAL folder has no schema until a split is prepared, and the builder
                # reports None rather than failing — so an empty answer here is "not known
                # yet", not "no classes". Loading settles it (and is cached from then on).
                features = getattr(self.dataset, "features", None)
        names = getattr((features or {}).get(self.target_feature), "names", None)
        return [str(entry) for entry in names] if names else []

    @property
    def resolved_metadata_features(self) -> List[str]:
        """``metadata_features`` resolved against the live dataset's columns (expands the ``"*"`` sentinel).

        Partial because the ``"*"`` expansion needs the loaded dataset's ``column_names``; ``None`` / ``[]``
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
        if keys is None or "hf_file" in keys:
            # Where this record CAME FROM, when the dataset is file-backed (a local imagefolder,
            # an audio/image folder). Present only when there is a file: a Hub dataset is
            # parquet-backed and has none, and an entry saying "" would claim otherwise. A
            # consumer that rewrites or moves the original needs this and can get it nowhere
            # else -- the decoded item is an array with no memory of its origin.
            source_file = _source_file(item.get(self.input_feature))
            if source_file:
                record["hf_file"] = Label(source_file)
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
        for counter, item in enumerate(self._narrowed_rows(dataset, want, metadata_features, limit)):
            if counter >= limit:
                break
            yield self._to_record(item, metadata_features, keys=want)

    def _narrowed_rows(self, dataset: Any, want: "frozenset[str]", metadata_features: List[str], limit: int) -> Any:
        """The rows a projection iterates — narrowed to the COLUMNS the keys actually need.

        This is where the projection promise is kept. Iterating the full dataset decodes every
        column whether or not ``_to_record`` uses it, so the old path saved nothing — measured
        on a real image set (2000 rows, warm): 5.6111 s projected against 5.6556 s for the full
        walk, while selecting the label column first costs 0.0092 s (~570x).

        Three shapes:

        * keys that need NO row data (``hf_path``/``hf_split`` come from config) — empty dicts,
          one per row, touching no column. NOT ``select_columns([])``: that yields ZERO rows
          (measured), which would silently answer "no records" to "what split is each from".
        * a dataset that can narrow — ``select_columns`` over exactly the needed columns, with
          the input column cast to ``decode=False`` when only the FILE of the image is wanted
          (``hf_file`` without ``image``): the path arrives, the pixels never do.
        * anything else — the dataset unchanged (a streaming/older dataset has no
          ``select_columns``; a failed cast must degrade to correct-but-slow, never to wrong).
        """
        columns: List[str] = []
        if "image" in want:
            columns.append(self.input_feature)
        if "class" in want:
            columns.append(self.target_feature)
        columns.extend(metadata_features)
        undecoded_file = "hf_file" in want and "image" not in want
        if "hf_file" in want and self.input_feature not in columns:
            columns.append(self.input_feature)
        if not columns:
            return ({} for _ in range(min(limit, len(dataset))))
        if not callable(getattr(dataset, "select_columns", None)):
            return dataset
        known = getattr(dataset, "column_names", None)
        keep = [c for c in dict.fromkeys(columns) if known is None or c in known]
        try:
            if undecoded_file:
                from datasets import Image as HFImage

                dataset = dataset.cast_column(self.input_feature, HFImage(decode=False))
            return dataset.select_columns(keep)
        except Exception as exc:  # noqa: BLE001 - slow is acceptable, a changed answer is not
            logger.debug(f"projection narrowing failed ({type(exc).__name__}: {exc}); walking the full rows")
            return dataset

    def __len__(self) -> int:
        # A ``count`` of 0 (or None) means "all records", matching __iter__'s
        # ``limit = self.count or len(...)``. Returning a bare ``self.count`` here
        # would report 0 for the common "0 == unlimited" case, making the source
        # look empty (e.g. a downstream len()-based stepper raising ``len == 0``)
        # even though iteration yields every record.
        #
        # A ``count`` LARGER than the split is clamped, never reported verbatim: a map-style
        # consumer trusts ``len()`` for its index space, so a lying length surfaces as an
        # ``IndexError`` deep inside a DataLoader worker, one epoch in. (Found the hard way:
        # ``count: 32`` over cppe-5's 29-row test split killed the first validation pass.)
        n = len(self.dataset)
        return min(self.count, n) if self.count else n
