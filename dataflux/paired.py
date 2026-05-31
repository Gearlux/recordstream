"""Re-join raw data samples with a sidecar annotation store (the annotation loop).

The recurring pattern this solves: you have raw data samples (RFUAV I/Q windows,
images, …) coming out of a ``DataSource``, and *separately* a sidecar store of
annotations covering some of them — typically a LabelStudio export that annotaide
writes as a ``sample_id -> record`` JSON mapping. :class:`AnnotationJoinSource`
re-joins the two by a key function so each matched annotation record is attached
to ``Sample.metadata``, ready for training.

    raw data ──annotate (LabelStudio)──▶ annotation store ──AnnotationJoinSource──▶ annotated samples
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

from confluid import configurable
from logflow import get_logger

from dataflux.discovery import get_callable_path, resolve_callable
from dataflux.sample import Sample

logger = get_logger(__name__)

# Join policy is a closed set. As a Literal it is enforced two ways with no extra
# code: static checkers reject bad values, and Confluid's @configurable validates
# it through pydantic at construction (both the Python and YAML/load paths), so a
# bad policy fails before __init__ runs. It also renders as an enum dropdown in
# the navigaitor form-spec.
Policy = Literal["left_outer", "inner", "right_driven"]


@runtime_checkable
class AnnotationStore(Protocol):
    """The read contract :class:`AnnotationJoinSource` needs from its annotation store:
    membership + lookup + key enumeration (``key -> record``).

    Structural (a ``Protocol``), so it does NOT couple dataflux to annotaide:
    annotaide's ``JSONFileAnnotationStore`` satisfies it — and so does a plain
    ``dict`` — without any import or inheritance. The write side (``save`` /
    ``delete``) lives in annotaide, not here.

    It is ``@runtime_checkable`` on purpose: Confluid's ``@configurable`` layer
    isinstance-validates it at construction, so a non-conforming ``annotations``
    is rejected before ``__init__`` runs (the type does the enforcement — no
    manual shape guard needed, mirroring how the ``policy`` ``Literal`` is
    validated). Every real store (``dict``, ``JSONFileAnnotationStore``) provides
    all three methods.
    """

    def __contains__(self, key: str) -> bool: ...

    def __getitem__(self, key: str) -> Dict[str, Any]: ...

    def keys(self) -> Iterable[str]: ...


@configurable
class AnnotationJoinSource:
    """Join a data source with a sidecar annotation store via a key function.

    Produces ``Sample`` values where the matched annotation record is flattened into
    ``Sample.metadata``. Three join policies cover the scenarios we actually see:

    - ``left_outer``: iterate ``data``; attach the annotation when the key matches,
      otherwise emit the sample unannotated. Preserves the data source's ``__len__``
      and ``__getitem__``. (Every sample, annotated where available.)
    - ``inner``: same as left_outer, filtered to annotated samples only.
      (The labeled subset.)
    - ``right_driven``: iterate ``annotations.keys()``; resolve each data sample
      via ``data_resolver(key, data)``. Use when annotations are sparse relative
      to the data.

    Coarser-granularity joins are expressed by returning a coarser key from
    ``key_fn`` so multiple data samples map to the same annotation record.
    Use ``extract_fn`` to project the record down to each sample's scope (e.g.
    trim a pack-level time-ranged annotation to a single window). Returning
    ``None`` from ``extract_fn`` marks the sample as unannotated.

    Args:
        data: The data source — any iterable (or ``DataSource``) yielding raw items that
            ``Sample.from_any`` can coerce into samples.
        annotations: The annotation store — a read-mapping (``key -> record``)
            satisfying :class:`AnnotationStore` (``__contains__`` + ``__getitem__`` +
            ``keys()``); validated at construction. A plain ``dict`` or annotaide's
            ``JSONFileAnnotationStore`` qualifies.
        key_fn: ``"module:function"`` path (or a callable) producing the join key
            from a sample. Signature: ``(sample: Sample) -> str``. Stored as a path so
            the source round-trips through Confluid YAML.
        policy: Join policy — one of ``"left_outer"``, ``"inner"``, ``"right_driven"``.
        extract_fn: Optional ``"module:function"`` path (or callable) called as
            ``extract_fn(record, sample) -> dict | None`` to project the record
            per sample. Returning ``None`` marks the sample unannotated.
        prefix: Optional string prefix applied to every annotation field when
            flattening into ``Sample.metadata``.
        store_full_under: If set, also stash the (extracted) record under
            ``Sample.metadata[store_full_under]``.
        data_resolver: Required for ``right_driven``. ``"module:function"`` path
            (or callable) invoked as ``data_resolver(key, data)`` to fetch the data
            sample for a given annotation key.
    """

    def __init__(
        self,
        data: Iterable[Any],
        annotations: AnnotationStore,
        key_fn: Union[str, Callable[[Sample], str]],
        policy: Policy = "left_outer",
        extract_fn: Optional[Union[str, Callable[[Dict[str, Any], Sample], Optional[Dict[str, Any]]]]] = None,
        prefix: str = "",
        store_full_under: Optional[str] = None,
        # data arg is Any (not Iterable[Any]): resolvers are written against a
        # concrete source type (e.g. RFUAVSource) and contravariance would reject
        # those signatures against a broader annotation.
        data_resolver: Optional[Union[str, Callable[[str, Any], Any]]] = None,
    ) -> None:
        # `annotations` shape is enforced by the AnnotationStore Protocol via
        # pydantic at construction. Only the policy-conditional requirement that
        # right_driven needs a resolver is checked here (a Protocol can't express it).
        if policy == "right_driven" and data_resolver is None:
            raise ValueError("policy='right_driven' requires data_resolver")

        self.data = data
        self.annotations = annotations
        self.key_fn = get_callable_path(key_fn) if callable(key_fn) else key_fn
        self.policy = policy
        self.extract_fn = get_callable_path(extract_fn) if callable(extract_fn) else extract_fn
        self.prefix = prefix
        self.store_full_under = store_full_under
        self.data_resolver = get_callable_path(data_resolver) if callable(data_resolver) else data_resolver

        self._key_fn_cache: Optional[Callable[[Sample], str]] = None
        self._extract_fn_cache: Optional[Callable[[Dict[str, Any], Sample], Optional[Dict[str, Any]]]] = None
        self._data_resolver_cache: Optional[Callable[[str, Any], Any]] = None
        self._inner_length: Optional[int] = None

    @property
    def _resolved_key_fn(self) -> Callable[[Sample], str]:
        if self._key_fn_cache is None:
            self._key_fn_cache = resolve_callable(self.key_fn)
        return self._key_fn_cache

    @property
    def _resolved_extract_fn(
        self,
    ) -> Optional[Callable[[Dict[str, Any], Sample], Optional[Dict[str, Any]]]]:
        if self.extract_fn is None:
            return None
        if self._extract_fn_cache is None:
            self._extract_fn_cache = resolve_callable(self.extract_fn)
        return self._extract_fn_cache

    @property
    def _resolved_data_resolver(self) -> Callable[[str, Any], Any]:
        if self.data_resolver is None:
            raise ValueError("data_resolver is not set")
        if self._data_resolver_cache is None:
            self._data_resolver_cache = resolve_callable(self.data_resolver)
        return self._data_resolver_cache

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
        if key not in self.annotations:
            return key, None
        record: Optional[Dict[str, Any]] = self.annotations[key]
        extract_fn = self._resolved_extract_fn
        if extract_fn is not None and record is not None:
            record = extract_fn(record, sample)
        return key, record

    def __iter__(self) -> Iterator[Sample]:
        if self.policy == "right_driven":
            yield from self._iter_right_driven()
            return

        for item in self.data:
            sample = Sample.from_any(item)
            key, record = self._lookup(sample)
            if self.policy == "inner" and record is None:
                continue
            yield self._attach(sample, record, key)

    def _iter_right_driven(self) -> Iterator[Sample]:
        resolver = self._resolved_data_resolver
        for key in self.annotations.keys():
            raw = resolver(key, self.data)
            sample = Sample.from_any(raw)
            record: Optional[Dict[str, Any]] = self.annotations[key]
            extract_fn = self._resolved_extract_fn
            if extract_fn is not None and record is not None:
                record = extract_fn(record, sample)
            if record is None:
                # extract_fn signalled "not applicable"; skip this entry
                continue
            yield self._attach(sample, record, key)

    def __len__(self) -> int:
        from collections.abc import Sized

        if self.policy == "left_outer":
            # left_outer preserves the data source's length; it must be sized.
            if not isinstance(self.data, Sized):
                raise TypeError(
                    f"policy='left_outer' requires a sized data source for len(); " f"got {type(self.data).__name__}"
                )
            return len(self.data)
        if self.policy == "right_driven":
            return len(list(self.annotations.keys()))

        if self._inner_length is None:
            count = 0
            for item in self.data:
                sample = Sample.from_any(item)
                _, record = self._lookup(sample)
                if record is not None:
                    count += 1
            self._inner_length = count
        return self._inner_length

    def __getitem__(self, index: int) -> Sample:
        if self.policy != "left_outer":
            raise TypeError(f"__getitem__ is only supported for policy='left_outer'; got {self.policy!r}")
        # Duck-typed on __getitem__ (not isinstance Sequence): workspace sources
        # like RFUAVSource / HuggingFaceSource expose __getitem__ without
        # subclassing collections.abc.Sequence.
        if not hasattr(self.data, "__getitem__"):
            raise TypeError("data must support __getitem__ for AnnotationJoinSource.__getitem__")
        sample = Sample.from_any(cast(Sequence[Any], self.data)[index])
        key, record = self._lookup(sample)
        return self._attach(sample, record, key)
