"""The engine's own callable wrappers — ``FilterOp`` and ``WrappedOp``.

Op-shaped, but NOT in :mod:`recordstream.ops`: both are construction targets of ``Stream``'s
fluent API (``.filter`` appends a ``FilterOp``, ``.map`` a ``WrappedOp``), and moving them
into ``ops/`` would make core import from a package that imports core. They carry no
discovery ``category`` on purpose — they wrap a RAW Python callable, which no GUI can wire.
Rationale: ``docs/architecture.md`` §5.
"""

from typing import Any, Callable, Dict, Optional, Union, cast

from confluid import configurable

from recordstream.items import Record, item_data, with_data


@configurable
class FilterOp:
    """Configurable filter operation.

    The op form of :meth:`~recordstream.core.Stream.filter` — a predicate gate over the
    stream: the record passes when the predicate returns ``True`` and is dropped otherwise
    (``__call__`` returns ``None``, which every engine route treats as "skip this record").

    Args:
        p: Predicate ``record -> bool``; the record passes through when it returns ``True``, else is dropped.
            Defaults to ``None`` (zero-arg construction); a predicate must be set before the op runs.
    """

    def __init__(self, p: Optional[Callable[[Record], bool]] = None):
        # Lazy / zero-arg: store config only; a missing predicate is validated lazily in __call__.
        self.p = p

    def __call__(self, record: Record) -> Optional[Record]:
        if self.p is None:
            raise ValueError("FilterOp.p (predicate) is not set — provide a record->bool callable before use.")
        return record if self.p(record) else None


@configurable
class WrappedOp:
    """Configurable transformation wrapper with smart mapping.

    The op form of :meth:`~recordstream.core.Stream.map` — lifts a plain function over one
    record value. The callable is ALWAYS stored as its importable ``module:function`` path
    (via :mod:`recordstream.discovery`), so the op pickles across ``spawn`` workers and
    serializes into Confluid YAML verbatim; the live function resolves lazily on first
    call.

    Args:
        f: The wrapped callable, or its importable ``module:function`` path (stored as a string for serialization).
            Defaults to ``""`` (zero-arg construction); resolving an empty path fails lazily on first call.
        key: The record key whose value payload the function transforms (item metadata preserved).
            ``None`` (default) = the function receives the WHOLE record dict and returns the new record.
        kw: Extra keyword arguments forwarded to the wrapped callable on every call (defaults to none).
    """

    def __init__(self, f: Union[str, Callable] = "", key: Optional[str] = None, kw: Optional[Dict[str, Any]] = None):
        from recordstream.discovery import get_callable_path

        # Lazy / zero-arg: store config only (the empty-path default resolves lazily via the `func`
        # property). EXPLICIT: always store the string path for serialization.
        self.f = get_callable_path(f) if callable(f) else f
        self.key = key
        self.kw = dict(kw) if kw else {}
        # Internal cache for the live callable
        self._func_cache: Optional[Callable] = None

    @property
    def func(self) -> Callable:
        if self._func_cache is None:
            from recordstream.discovery import resolve_callable

            self._func_cache = resolve_callable(self.f)
        return self._func_cache

    def __call__(self, record: Record) -> Optional[Record]:
        if self.key is None:
            return cast(Optional[Record], self.func(record, **self.kw))
        if self.key not in record:
            raise KeyError(f"WrappedOp: record has no key {self.key!r} (keys: {list(record)})")
        value = record[self.key]
        new_data = self.func(item_data(value), **self.kw)
        try:
            new_value = with_data(value, new_data)
        except TypeError:
            new_value = new_data  # a plain (non-item) value is replaced verbatim
        return {**record, self.key: new_value}
