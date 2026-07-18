"""Queryable metadata — filter stored samples by metadata predicates WITHOUT loading arrays.

Two pieces (mirroring the ``sampleflux.projection`` protocol-plus-fallback design):

- :class:`SupportsMetadataScan` — a source opts in by implementing
  ``iter_metadata() -> Iterator[(key, metadata_dict)]`` that reads ONLY the metadata
  (HDF5 attrs, Zarr ``.zattrs``, a SigMF ``.sigmf-meta`` JSON) — never the data arrays.
  Free-function scanners for the shipped sources live here (``scan_hdf5_metadata`` /
  ``scan_zarr_metadata``); ``SigMFSource.iter_metadata`` implements the protocol
  directly.
- :class:`MetadataFilterSource` — a view source (``category="source"``) yielding only
  the samples whose metadata passes a predicate: the YAML-friendly ``where`` expression
  (the same restricted-eval namespace as ``FormulaOp`` — metadata keys become variables)
  and/or a programmatic ``predicate`` callable. The matching index set is computed
  lazily from the metadata scan (cached), so arrays load only for matches; a source
  without the protocol falls back to a full iteration filter.

Existing HDF5/Zarr files are queryable with NO rewrite — their metadata already lives in
attrs/``.zattrs``. (A ``.metaindex`` sidecar accelerator is a TASKS.md follow-up if
scans ever become hot.)
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, cast, runtime_checkable

import h5py
from confluid import configurable
from loggair import get_logger

from sampleflux.ops.formula import _FORMULA_NAMESPACE
from sampleflux.sample import Sample

logger = get_logger("sampleflux.storage.query")

__all__ = ["MetadataFilterSource", "SupportsMetadataScan", "scan_hdf5_metadata", "scan_zarr_metadata"]


@runtime_checkable
class SupportsMetadataScan(Protocol):
    """A source that can enumerate per-sample metadata WITHOUT loading data arrays."""

    def iter_metadata(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield ``(sample key, metadata dict)`` pairs, array payloads untouched."""
        ...  # pragma: no cover - protocol


def scan_hdf5_metadata(path: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Scan an ``HDF5Sink`` file's metadata: dataset attrs + array-metadata SHAPE/DTYPE stubs.

    Array-valued metadata (datasets under ``{prefix}_meta/``) is represented by a stub
    string ``"<array shape=... dtype=...>"`` — queries can test presence/shape without a
    single array read.
    """
    with h5py.File(str(path), "r") as handle:
        prefixes = sorted(k.split("_data")[0] for k in handle.keys() if k.endswith("_data"))
        for prefix in prefixes:
            metadata: Dict[str, Any] = dict(handle[f"{prefix}_data"].attrs)
            meta_grp = handle.get(f"{prefix}_meta")
            if isinstance(meta_grp, h5py.Group):
                for key, dset in meta_grp.items():
                    metadata[key] = f"<array shape={tuple(dset.shape)} dtype={dset.dtype}>"
            yield prefix, metadata


def scan_zarr_metadata(path: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Scan a ``ZarrGroupSink`` store's metadata: each sample group's ``.zattrs``."""
    import zarr

    root = zarr.open_group(str(path), mode="r")
    for name in sorted(root.group_keys()):
        yield name, dict(root[name].attrs)


def _where_predicate(where: str) -> Callable[[Dict[str, Any]], bool]:
    """Compile a ``where`` expression into a metadata predicate.

    The expression evaluates in the FormulaOp restricted namespace (``math.*`` +
    ``abs``/``min``/``max``/``round``/``pow``, no builtins) with the metadata KEYS bound
    as variables — e.g. ``"snr_db > 10 and drone == 'DJI'"``. A missing key (NameError)
    means the sample does not match (logged at trace-equivalent debug); any other
    evaluation error raises (a malformed expression must fail loudly).
    """

    def _predicate(metadata: Dict[str, Any]) -> bool:
        namespace = {**_FORMULA_NAMESPACE, **metadata}
        try:
            return bool(eval(where, {"__builtins__": {}}, namespace))  # noqa: S307 - restricted namespace
        except NameError as exc:
            logger.debug(f"MetadataFilterSource: where={where!r} — {exc}; sample treated as non-matching")
            return False
        except Exception as exc:
            raise ValueError(f"MetadataFilterSource: where expression {where!r} failed: {exc}") from exc

    return _predicate


@configurable(category="source")
class MetadataFilterSource:
    """A view source yielding only the samples whose metadata matches.

    Filtering uses the wrapped source's :class:`SupportsMetadataScan` protocol when
    available (metadata-only scan — data arrays load ONLY for matching samples, via the
    source's ``__getitem__``), else falls back to full-iteration filtering (the
    projection-module pattern). Match criteria compose with AND: the ``where`` expression
    and the programmatic ``predicate`` must both pass when both are set.

    Args:
        source: The wrapped source; required at use time, validated lazily.
        where: Restricted boolean expression over metadata keys (e.g. ``"snr_db > 10"``). Blank = no expression.
        predicate: Programmatic ``metadata -> bool`` callable (not serialized; the ``FilterOp.p`` convention).
    """

    def __init__(
        self,
        source: Optional[Any] = None,
        where: str = "",
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        # Lazy / zero-arg: store config only; matching indices compute lazily on first access.
        self.source = source
        self.where = str(where)
        self.predicate = predicate
        self._matches: Optional[List[int]] = None

    def _match(self, metadata: Dict[str, Any]) -> bool:
        if self.where and not _where_predicate(self.where)(metadata):
            return False
        if self.predicate is not None and not self.predicate(metadata):
            return False
        return True

    @property
    def matches(self) -> List[int]:
        """Indices of matching samples (computed once per instance; ``_matches = None`` resets)."""
        if self._matches is None:
            if self.source is None:
                raise ValueError("MetadataFilterSource: a 'source' is required")
            if not self.where and self.predicate is None:
                raise ValueError("MetadataFilterSource: set 'where' and/or 'predicate' — an empty filter is a bug")
            if isinstance(self.source, SupportsMetadataScan):
                self._matches = [i for i, (_key, meta) in enumerate(self.source.iter_metadata()) if self._match(meta)]
            else:
                logger.debug(
                    f"MetadataFilterSource: {type(self.source).__name__} has no iter_metadata — "
                    "falling back to full-iteration filtering (arrays load for every sample)."
                )
                self._matches = [i for i, sample in enumerate(self.source) if self._match(dict(sample.meta))]
        return self._matches

    def __iter__(self) -> Iterator[Sample]:
        matches = self.matches  # validates the source before iteration
        source: Any = self.source
        if hasattr(source, "__getitem__"):
            for index in matches:
                yield cast(Sample, source[index])
        else:
            match_set = set(matches)
            for i, sample in enumerate(source):
                if i in match_set:
                    yield cast(Sample, sample)

    def __len__(self) -> int:
        return len(self.matches)

    def __getitem__(self, index: int) -> Sample:
        source_index = self.matches[index]
        source: Any = self.source
        if hasattr(source, "__getitem__"):
            return cast(Sample, source[source_index])
        for i, sample in enumerate(source):
            if i == source_index:
                return cast(Sample, sample)
        raise IndexError(index)
