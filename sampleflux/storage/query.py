"""Queryable metadata — filter stored records by metadata predicates WITHOUT loading arrays.

Two pieces (mirroring the ``sampleflux.projection`` protocol-plus-fallback design):

- :class:`SupportsMetadataScan` — a source opts in by implementing
  ``iter_metadata() -> Iterator[(key, metadata_dict)]`` that reads ONLY the metadata
  (HDF5 attrs, Zarr ``.zattrs``, a sidecar JSON) — never the data arrays.
  Free-function scanners for the shipped sources live here (``scan_hdf5_metadata`` /
  ``scan_zarr_metadata``); any external storage source can implement the protocol
  directly (it is structural — no import of this module required).
- :class:`MetadataFilterSource` — a view source (``category="source"``) yielding only
  the records whose metadata passes a predicate: the YAML-friendly ``where`` expression
  (the same restricted-eval namespace as ``FormulaOp`` — metadata keys become variables)
  and/or a programmatic ``predicate`` callable. The matching index set is computed
  lazily from the metadata scan (cached), so arrays load only for matches; a source
  without the protocol falls back to a full iteration filter.

Record-layout HDF5/Zarr stores are queryable with NO extra index — their metadata already
lives in attrs/``.zattrs``. (A ``.metaindex`` sidecar accelerator is a TASKS.md follow-up
if scans ever become hot.)
"""

import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, cast, runtime_checkable

import h5py
import numpy as np
from confluid import configurable
from loggair import get_logger

from sampleflux.io import PLAIN_TYPE, encode_item
from sampleflux.items import Record
from sampleflux.ops.formula import _FORMULA_NAMESPACE
from sampleflux.storage.base import PLAIN_VALUE, require_record_format, restore_attrs

logger = get_logger("sampleflux.storage.query")

__all__ = [
    "MetadataFilterSource",
    "SupportsMetadataScan",
    "record_metadata",
    "scan_hdf5_metadata",
    "scan_zarr_metadata",
]


class _AttrView(dict):
    """A metadata sub-dict that ALSO answers attribute access — so a record scan's per-key
    attrs evaluate naturally in a ``where`` expression (``signal.samplerate > 1e6``) while
    staying a plain dict for programmatic predicates."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors normal attribute-miss semantics
            raise AttributeError(name) from exc


def _viewed(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap dict-valued entries in :class:`_AttrView` (one level — the record key/attr shape)."""
    return {k: _AttrView(v) if isinstance(v, dict) else v for k, v in metadata.items()}


def record_metadata(record: Record) -> Dict[str, Dict[str, Any]]:
    """A live record's queryable metadata: ``{key: {attr: value}}`` (attrs via the io codec,
    payloads untouched) — the same nested shape the record storage scans yield. A ``"plain"``
    value contributes ``{"value": <payload>}`` when its payload is a scalar, else ``{}``."""
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in record.items():
        encoded = encode_item(value)
        if encoded.type_name == PLAIN_TYPE:
            payload = encoded.payload
            if isinstance(payload, np.generic):
                payload = payload.item()
            out[key] = {PLAIN_VALUE: payload} if isinstance(payload, (bool, int, float, str)) else {}
        else:
            out[key] = dict(encoded.attrs)
    return out


@runtime_checkable
class SupportsMetadataScan(Protocol):
    """A source that can enumerate per-record metadata WITHOUT loading data arrays."""

    def iter_metadata(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield ``(record key, metadata dict)`` pairs, array payloads untouched."""
        ...  # pragma: no cover - protocol


def scan_hdf5_metadata(path: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Scan an ``HDF5Sink`` file's metadata WITHOUT loading payload arrays.

    Per record the NESTED shape ``{key: {attr: value}}`` (plain attrs decoded; array-valued
    attrs as SHAPE/DTYPE stubs ``"<array shape=... dtype=...>"`` — queries can test
    presence/shape without an array read; a ``"plain"`` value's scalar payload appears under
    its ``value`` attr) — a ``where`` expression addresses it as ``"<key>.<attr>"``
    (e.g. ``"signal.samplerate > 1e6"``).
    """
    with h5py.File(str(path), "r") as handle:
        require_record_format(handle.attrs.get("sampleflux_format"), "scan_hdf5_metadata")
        for name in sorted(k for k in handle.keys() if k.startswith("s")):
            group = handle[name]
            nested: Dict[str, Any] = {}
            for field in json.loads(group.attrs["__field_order__"]):
                fgrp = group[field]
                plain = {k: v for k, v in fgrp.attrs.items() if k != "__item_type__"}
                attrs = restore_attrs(dict(plain), {})
                agrp = fgrp.get("attrs")
                if isinstance(agrp, h5py.Group):
                    for key, dset in agrp.items():
                        attrs[key] = f"<array shape={tuple(dset.shape)} dtype={dset.dtype}>"
                nested[field] = attrs
            yield name, nested


def scan_zarr_metadata(path: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Scan a ``ZarrGroupSink`` store's metadata: ``.zattrs`` only, no payload arrays.

    Yields the same NESTED ``{key: {attr: value}}`` shape as the HDF5 scan (array-valued
    attrs as name stubs).
    """
    import zarr

    root = zarr.open_group(str(path), mode="r")
    require_record_format(root.attrs.get("sampleflux_format"), "scan_zarr_metadata")
    for name in sorted(root.group_keys()):
        group = cast(Any, root[name])
        nested: Dict[str, Any] = {}
        for field in json.loads(group.attrs["__field_order__"]):
            fgrp = group[field]
            plain = {k: v for k, v in dict(fgrp.attrs).items() if k != "__item_type__"}
            attrs = restore_attrs(plain, {})
            if "attrs" in fgrp:
                for key in fgrp["attrs"].array_keys():
                    attrs[key] = f"<array {key}>"
            nested[field] = attrs
        yield name, nested


def _where_predicate(where: str) -> Callable[[Dict[str, Any]], bool]:
    """Compile a ``where`` expression into a metadata predicate.

    The expression evaluates in the FormulaOp restricted namespace (``math.*`` +
    ``abs``/``min``/``max``/``round``/``pow``, no builtins) with the metadata KEYS bound
    as variables — e.g. ``"snr_db > 10 and drone == 'DJI'"`` (a flat external scan) or
    ``"signal.samplerate > 1e6"`` (a record scan's per-key attrs). A missing key/attr
    (NameError/AttributeError) means the record does not match (logged at debug); any
    other evaluation error raises (a malformed expression must fail loudly).

    NOTE: a record key named like a Python keyword (e.g. ``class``) cannot be addressed in
    an expression — query such keys via the programmatic ``predicate`` (metadata is plain
    nested dicts there), or give queryable keys non-keyword names.
    """

    def _predicate(metadata: Dict[str, Any]) -> bool:
        # Dict-valued entries (the record scans' per-key attr dicts) evaluate through
        # _AttrView so "<key>.<attr>" reads naturally; flat external metadata is untouched.
        namespace = {**_FORMULA_NAMESPACE, **_viewed(metadata)}
        try:
            return bool(eval(where, {"__builtins__": {}}, namespace))  # noqa: S307 - restricted namespace
        except (NameError, AttributeError, KeyError) as exc:
            logger.debug(f"MetadataFilterSource: where={where!r} — {exc}; record treated as non-matching")
            return False
        except Exception as exc:
            raise ValueError(f"MetadataFilterSource: where expression {where!r} failed: {exc}") from exc

    return _predicate


@configurable(category="source")
class MetadataFilterSource:
    """A view source yielding only the records whose metadata matches.

    Filtering uses the wrapped source's :class:`SupportsMetadataScan` protocol when
    available (metadata-only scan — data arrays load ONLY for matching records, via the
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
        """Indices of matching records (computed once per instance; ``_matches = None`` resets)."""
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
                    "falling back to full-iteration filtering (arrays load for every record)."
                )
                self._matches = [i for i, record in enumerate(self.source) if self._match(record_metadata(record))]
        return self._matches

    def __iter__(self) -> Iterator[Record]:
        matches = self.matches  # validates the source before iteration
        source: Any = self.source
        if hasattr(source, "__getitem__"):
            for index in matches:
                yield cast(Record, source[index])
        else:
            match_set = set(matches)
            for i, record in enumerate(source):
                if i in match_set:
                    yield cast(Record, record)

    def __len__(self) -> int:
        return len(self.matches)

    def __getitem__(self, index: int) -> Record:
        source_index = self.matches[index]
        source: Any = self.source
        if hasattr(source, "__getitem__"):
            return cast(Record, source[source_index])
        for i, record in enumerate(source):
            if i == source_index:
                return cast(Record, record)
        raise IndexError(index)
