"""Keyed-join source pairing a primary DataSource with a secondary annotation mapping."""

from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

from confluid import configurable
from logflow import get_logger

from dataflux.discovery import get_callable_path, resolve_callable
from dataflux.sample import Sample

logger = get_logger(__name__)

VALID_POLICIES = ("left_outer", "inner", "right_driven")


@configurable
class PairedSource:
    """Pair a primary DataSource with a secondary annotation mapping via a key function.

    Produces ``Sample`` values where the matched annotation record is flattened into
    ``Sample.metadata``. Supports three join policies:

    - ``left_outer``: iterate primary; attach annotation when the key matches,
      otherwise emit the sample unannotated. Preserves primary's ``__len__`` and
      ``__getitem__``. (Scenario A.)
    - ``inner``: same as left_outer, filtered to annotated samples only.
      (Scenario B.)
    - ``right_driven``: iterate ``secondary.keys()``; resolve each primary sample
      via ``primary_resolver(key, primary)``. Use when annotations are sparse.

    Coarser-granularity joins are expressed by returning a coarser key from
    ``key_fn`` so multiple primary samples map to the same annotation record.
    Use ``extract_fn`` to project the record down to each sample's scope (e.g.
    trim a pack-level time-ranged annotation to a single window). Returning
    ``None`` from ``extract_fn`` marks the sample as unannotated.

    Args:
        primary: Any iterable (or ``DataSource``) yielding raw items that
            ``Sample.from_any`` can coerce into samples.
        secondary: For ``left_outer``/``inner``, a mapping-like object supporting
            ``__contains__`` and ``__getitem__``. For ``right_driven``, an object
            also supporting ``keys()``.
        key_fn: ``"module:function"`` path (or a callable) producing the join key
            from a sample. Signature: ``(sample: Sample) -> str``.
        policy: One of ``"left_outer"``, ``"inner"``, ``"right_driven"``.
        extract_fn: Optional ``"module:function"`` path (or callable) called as
            ``extract_fn(record, sample) -> dict | None`` to project the record
            per sample. Returning ``None`` marks the sample unannotated.
        prefix: Optional string prefix applied to every annotation field when
            flattening into ``Sample.metadata``.
        store_full_under: If set, also stash the (extracted) record under
            ``Sample.metadata[store_full_under]``.
        primary_resolver: Required for ``right_driven``. ``"module:function"``
            path (or callable) invoked as ``primary_resolver(key, primary)``
            to fetch the primary sample for a given annotation key.
    """

    def __init__(
        self,
        primary: Any,
        secondary: Any,
        key_fn: Union[str, Callable[[Sample], str]],
        policy: str = "left_outer",
        extract_fn: Optional[Union[str, Callable[[Dict[str, Any], Sample], Optional[Dict[str, Any]]]]] = None,
        prefix: str = "",
        store_full_under: Optional[str] = None,
        primary_resolver: Optional[Union[str, Callable[[str, Any], Any]]] = None,
    ) -> None:
        if policy not in VALID_POLICIES:
            raise ValueError(f"Invalid policy {policy!r}; must be one of {VALID_POLICIES}")

        if policy in ("left_outer", "inner"):
            if not hasattr(secondary, "__contains__") or not hasattr(secondary, "__getitem__"):
                raise TypeError(
                    f"policy={policy!r} requires secondary to support __contains__ and __getitem__; "
                    f"got {type(secondary).__name__}"
                )

        if policy == "right_driven":
            if primary_resolver is None:
                raise ValueError("policy='right_driven' requires primary_resolver")
            if not hasattr(secondary, "keys"):
                raise TypeError(
                    f"policy='right_driven' requires secondary to support keys(); " f"got {type(secondary).__name__}"
                )

        self.primary = primary
        self.secondary = secondary
        self.key_fn = get_callable_path(key_fn) if callable(key_fn) else key_fn
        self.policy = policy
        self.extract_fn = get_callable_path(extract_fn) if callable(extract_fn) else extract_fn
        self.prefix = prefix
        self.store_full_under = store_full_under
        self.primary_resolver = get_callable_path(primary_resolver) if callable(primary_resolver) else primary_resolver

        self._key_fn_cache: Optional[Callable[[Sample], str]] = None
        self._extract_fn_cache: Optional[Callable[[Dict[str, Any], Sample], Optional[Dict[str, Any]]]] = None
        self._primary_resolver_cache: Optional[Callable[[str, Any], Any]] = None
        self._inner_length: Optional[int] = None

    @property
    def _resolved_key_fn(self) -> Callable[[Sample], str]:
        if self._key_fn_cache is None:
            self._key_fn_cache = resolve_callable(self.key_fn)
        return self._key_fn_cache

    @property
    def _resolved_extract_fn(self) -> Optional[Callable[[Dict[str, Any], Sample], Optional[Dict[str, Any]]]]:
        if self.extract_fn is None:
            return None
        if self._extract_fn_cache is None:
            self._extract_fn_cache = resolve_callable(self.extract_fn)
        return self._extract_fn_cache

    @property
    def _resolved_primary_resolver(self) -> Callable[[str, Any], Any]:
        if self.primary_resolver is None:
            raise ValueError("primary_resolver is not set")
        if self._primary_resolver_cache is None:
            self._primary_resolver_cache = resolve_callable(self.primary_resolver)
        return self._primary_resolver_cache

    def _attach(self, sample: Sample, record: Optional[Dict[str, Any]], key: str) -> Sample:
        metadata = dict(sample.metadata)
        metadata["annotation_key"] = key
        metadata["annotated"] = record is not None

        if record is not None:
            for k, v in record.items():
                metadata[f"{self.prefix}{k}"] = v
            if self.store_full_under is not None:
                metadata[self.store_full_under] = record

        return sample._replace(metadata=metadata)

    def _lookup(self, sample: Sample) -> Tuple[str, Optional[Dict[str, Any]]]:
        key = self._resolved_key_fn(sample)
        if key not in self.secondary:
            return key, None
        record: Optional[Dict[str, Any]] = self.secondary[key]
        extract_fn = self._resolved_extract_fn
        if extract_fn is not None and record is not None:
            record = extract_fn(record, sample)
        return key, record

    def __iter__(self) -> Iterator[Sample]:
        if self.policy == "right_driven":
            yield from self._iter_right_driven()
            return

        for item in self.primary:
            sample = Sample.from_any(item)
            key, record = self._lookup(sample)
            if self.policy == "inner" and record is None:
                continue
            yield self._attach(sample, record, key)

    def _iter_right_driven(self) -> Iterator[Sample]:
        resolver = self._resolved_primary_resolver
        for key in self.secondary.keys():
            raw = resolver(key, self.primary)
            sample = Sample.from_any(raw)
            record: Optional[Dict[str, Any]] = self.secondary[key]
            extract_fn = self._resolved_extract_fn
            if extract_fn is not None and record is not None:
                record = extract_fn(record, sample)
            if record is None:
                # extract_fn signalled "not applicable"; skip this entry
                continue
            yield self._attach(sample, record, key)

    def __len__(self) -> int:
        if self.policy == "left_outer":
            return len(self.primary)
        if self.policy == "right_driven":
            return len(list(self.secondary.keys()))

        if self._inner_length is None:
            count = 0
            for item in self.primary:
                sample = Sample.from_any(item)
                _, record = self._lookup(sample)
                if record is not None:
                    count += 1
            self._inner_length = count
        return self._inner_length

    def __getitem__(self, index: int) -> Sample:
        if self.policy != "left_outer":
            raise TypeError(f"__getitem__ is only supported for policy='left_outer'; got {self.policy!r}")
        if not hasattr(self.primary, "__getitem__"):
            raise TypeError("primary must support __getitem__ for PairedSource.__getitem__")
        sample = Sample.from_any(self.primary[index])
        key, record = self._lookup(sample)
        return self._attach(sample, record, key)
