import concurrent.futures
import multiprocessing
from contextlib import nullcontext
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

from confluid import configurable
from confluid import load as _confluid_load
from confluid import materialize as _confluid_materialize
from confluid.fluid import Fluid as _ConfluidFluid
from loggair import get_logger

from recordstream.items import NDArrayItem, Record, item_data, with_data

logger = get_logger(__name__)


def _op_expands(op: Any) -> bool:
    """True when an op is a 1→N expanding op (explicit ``EXPANDS = True`` class attribute)."""
    return bool(getattr(op, "EXPANDS", False))


#: An op-family MATCHER recognises a library's op objects. Keep it IMPORT-FREE — inspect
#: ``type(op).__mro__`` module names rather than importing the library.
OpMatcher = Callable[[Any], bool]
#: An op-family INVOKER applies one foreign op with its library's native calling
#: convention: ``(record, op) -> Optional[Record]`` (``None`` = drop the record).
OpInvoker = Callable[[Record, Any], Optional[Record]]

#: The registered op families, in registration order. Dispatch checks LAST-registered
#: first, so a later (more specific) family can shadow an earlier one.
_OP_FAMILIES: List[Tuple[str, OpMatcher, OpInvoker]] = []


def register_op_family(name: str, matcher: OpMatcher, invoker: OpInvoker) -> None:
    """Teach the engine to invoke a NEW library's ops natively — the open extension point.

    ``matcher(op) -> bool`` recognises the family's op objects (keep it import-free —
    inspect ``type(op).__mro__`` module names); ``invoker(record, op)`` applies one op with
    the library's own calling convention and returns the new record (``None`` drops it).
    Re-registering a ``name`` REPLACES that family in place; otherwise the family is
    appended, and dispatch checks last-registered first (a more specific family shadows an
    earlier one — register yours after the built-ins to win an overlap).

    Both callables MUST be module-level functions (picklable by reference): the engine's
    spawn-parallel routes ship non-builtin families to worker processes by pickling them.

    Example — kornia augmentations (``nn.Module``s over batched BCHW tensors)::

        def is_kornia(op) -> bool:
            return any(c.__module__.startswith("kornia.augmentation") for c in type(op).__mro__)

        def invoke_kornia(record, op):
            img = record["image"]                    # a CHW torch.Tensor (e.g. after ToTensor)
            out = op(img.unsqueeze(0)).squeeze(0)    # kornia draws once per batch call
            return {**record, "image": out}

        register_op_family("kornia", is_kornia, invoke_kornia)
    """
    entry = (str(name), matcher, invoker)
    for i, (existing, _, _) in enumerate(_OP_FAMILIES):
        if existing == name:
            _OP_FAMILIES[i] = entry
            return
    _OP_FAMILIES.append(entry)


def registered_op_families() -> Tuple[str, ...]:
    """The registered op-family names, in registration/dispatch-precedence order."""
    return tuple(name for name, _, _ in _OP_FAMILIES)


def _sync_op_families(families: Optional[List[Tuple[str, OpMatcher, OpInvoker]]]) -> None:
    """Merge families shipped from the parent process into this process's registry.

    Spawn workers import this module (built-ins present) but never re-run the user's
    registration side effects — the parallel routes therefore pass the parent's
    non-builtin entries along and merge them here (idempotent by name).
    """
    for name, matcher, invoker in families or []:
        register_op_family(name, matcher, invoker)


def _extra_op_families() -> List[Tuple[str, OpMatcher, OpInvoker]]:
    """The non-builtin registry entries — what a spawn worker cannot rebuild by import alone."""
    return [entry for entry in _OP_FAMILIES if entry[0] not in _BUILTIN_FAMILIES]


#: The record keys albumentations understands — its OWN target vocabulary. An albumentations
#: op receives exactly these keys (the ones present) and nothing else, so extra record
#: entries (scalars, domain items) never reach a library that would reject them.
_ALB_KEYS: Tuple[str, ...] = ("image", "mask", "masks", "bboxes", "keypoints", "labels")


def _is_albumentations(op: Any) -> bool:
    """True for an albumentations transform / ``Compose`` — by MRO module name (no import here)."""
    return any(getattr(cls, "__module__", "").startswith("albumentations") for cls in type(op).__mro__)


def _invoke_albumentations(record: Record, op: Any) -> Optional[Record]:
    """albumentations dispatches by KWARG NAME: hand the op exactly its own target keys
    present in the record (one call = one joint draw across them); array outputs are
    re-wrapped in the incoming value's item type (``with_data``) so ``Image``/``Mask``
    keep their type and metadata. Box-carrying augmentation belongs in albumentations' own
    ``A.Compose(..., bbox_params=...)`` — format handling is Compose's job in that library.
    """
    kwargs = {k: record[k] for k in _ALB_KEYS if k in record}
    if not kwargs:
        logger.debug(
            f"albumentations op {type(op).__name__} received no known keys "
            f"({', '.join(_ALB_KEYS)}) — record keys: {list(record)}; passing through."
        )
        return record
    out = op(**kwargs)
    merged = dict(record)
    for key, value in out.items():
        original = record.get(key)
        if isinstance(original, NDArrayItem) and not isinstance(value, NDArrayItem):
            value = with_data(original, value)
        merged[key] = value
    return merged


def _is_torchvision_v2(op: Any) -> bool:
    """True for a torchvision ``transforms.v2`` transform — by MRO module name (no import here)."""
    return any(getattr(cls, "__module__", "").startswith("torchvision.transforms.v2") for cls in type(op).__mro__)


def _invoke_torchvision_v2(record: Record, op: Any) -> Optional[Record]:
    """torchvision v2 natively walks a dict: params sampled once, tensor/tv_tensor/PIL
    leaves transformed, everything else passed through — called as-is."""
    return cast(Record, op(record))


# The built-in families register through the SAME open registry third parties use —
# one mechanism, no privileged code path. Registered at import, so spawn workers
# rebuild them by importing this module.
register_op_family("albumentations", _is_albumentations, _invoke_albumentations)
register_op_family("torchvision_v2", _is_torchvision_v2, _invoke_torchvision_v2)
_BUILTIN_FAMILIES: Tuple[str, ...] = ("albumentations", "torchvision_v2")


def _apply_op(record: Record, op: Any) -> Optional[Record]:
    """Apply one op to the record dict — the engine's op-FAMILY dispatch.

    The single op-application chokepoint shared by the sequential, parallel (via
    :func:`_worker_task`), streamed, and random-access (``__getitem__``) paths; composing
    ops (``Pipeline`` / ``Parallel`` / ``Enable`` / ``RandomApply`` / ``ConfigureOp``)
    route their inner ops through here so every op is applied identically. Each op family
    is invoked the way its library expects — no wrapper/adapter classes: the registered
    families (:func:`register_op_family`; built-ins ``albumentations`` /
    ``torchvision_v2``) are checked LAST-registered first, and an op matching none of
    them is a native/wiring op called ``op(record) -> Optional[Record]`` (``None`` drops
    the record — filter semantics).
    """
    for _name, matcher, invoker in reversed(_OP_FAMILIES):
        try:
            matched = matcher(op)
        except Exception:  # pragma: no cover - a defensive matcher never breaks dispatch
            matched = False
        if matched:
            return invoker(record, op)
    return cast(Optional[Record], op(record))


@runtime_checkable
class MapStyle(Protocol):
    """A map-style dataset: ``len(ds)`` and ``ds[i]``.

    What recordstream MEANS by "a dataset", said structurally so the engine never imports a
    framework to express it. ``Stream`` used to inherit ``torch.utils.data.Dataset``, which made
    torch a hard dependency of a package whose own work is numpy — for nothing: that base is not
    load-bearing. ``DataLoader`` duck-types its argument (a plain object with these two methods
    works), nothing in the workspace does ``isinstance(x, Dataset)``, and the annotation is the
    only thing the inheritance ever bought.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


def _describe_deferred_source(source: Any) -> str:
    """Return a human-friendly description of a still-deferred Confluid source.

    Surfaces the tag/target so the error explains WHAT was deferred instead
    of just noting it isn't a live object.
    """
    target = getattr(source, "target", "<unknown>")
    target_name = target if isinstance(target, str) else getattr(target, "__qualname__", str(target))
    return f"{type(source).__name__}(target={target_name!r})"


def _fluid_source_guidance(source: Any) -> str:
    """Build an actionable message when Stream.source is still a Confluid Fluid."""
    return (
        f"Stream.source is still a deferred Confluid marker: {_describe_deferred_source(source)}. "
        "Confluid has not materialized it yet. Fixes: (a) in YAML, write the source as "
        "`!class:X()` (with parens) instead of `!class:X` so it becomes an Instance and is "
        "materialized at load time; (b) or call `flow(source)` on the source before handing "
        "it to Stream."
    )


def _fluid_op_guidance(op: Any, index: int) -> str:
    """Build an actionable message when a Stream op marker cannot be materialized."""
    return (
        f"Stream.ops[{index}] is a deferred Confluid marker that could not be materialized: "
        f"{_describe_deferred_source(op)}. Fixes: (a) in YAML, write the op as `!class:X()` "
        "(with parens) so it becomes an Instance and is materialized at load time; (b) or "
        "call `flow(op)` on the op before handing it to Stream."
    )


def _check_ops_materialized(ops: List[Any]) -> None:
    """Flow any still-deferred Confluid op markers IN PLACE at engine-route entry.

    The same lazy-flow convention the composing ops (``Pipeline`` / ``Enable`` /
    ``RandomApply``) use — so a YAML ops doc may list bare ``!class:`` mapping-form
    entries (e.g. a bare albumentations transform) directly under ``ops:``. The in-place
    write is the cache: later routes (and the spawn pickler) see live ops. A marker that
    cannot build raises ONE actionable error naming the offending index.
    """
    from confluid import flow

    for i, op in enumerate(ops):
        if isinstance(op, _ConfluidFluid):
            try:
                ops[i] = flow(op)
            except Exception as exc:
                raise TypeError(_fluid_op_guidance(op, i)) from exc


@configurable
class FilterOp:
    """Configurable filter operation.

    The op form of :meth:`Stream.filter` — a predicate gate over the stream: the record
    passes when the predicate returns ``True`` and is dropped otherwise (``__call__``
    returns ``None``, which every engine route treats as "skip this record").

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

    The op form of :meth:`Stream.map` — lifts a plain function over one record value. The
    callable is ALWAYS stored as its importable ``module:function`` path (via
    :mod:`recordstream.discovery`), so the op pickles across ``spawn`` workers and
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


def _expand(op: Any, record: Any) -> List[Any]:
    """Run a 1→N EXPANDING op and return its flattened children."""
    raw = op(record)
    if raw is None:
        return []
    return [child for child in raw if child is not None]


def linear_steps(ops: Sequence[Any]) -> Tuple[List[Any], str]:
    """Compile a flat op list into the linear step graph the engine executes.

    A sequence IS a graph — every step reads the previous one — so an ``ops:`` list needs no
    lifting to run on the graph kernel, just names. The names are positional (``s0``, ``s1``,
    …) and never surface: nothing in an ``ops:`` document can reference a step, so they exist
    only to key the step environment. Positional (not op-class) naming is deliberate — the
    same op twice in a row is two distinct steps, which a name-keyed mapping would collapse.

    Returns ``(steps, output_step)``; an empty list yields ``([], "")``, the identity graph.
    """
    from recordstream.flow import FlowStep

    steps = [FlowStep(name=f"s{i}", op=op, from_=None, bind={}, merge_from=()) for i, op in enumerate(ops)]
    return cast(List[Any], steps), (steps[-1].name if steps else "")


def _worker_task(
    record: Any, ops: List[Any], families: Optional[List[Tuple[str, OpMatcher, OpInvoker]]] = None
) -> Optional[Any]:
    """Single-result worker for STRICTLY 1→1 op lists (the ``Parallel`` op's contract).

    Kept for callers that need exactly one carrier back; expanding ops raise here —
    route expanding pipelines through the iterating engine.
    """
    from recordstream.flow import run_steps

    _sync_op_families(families)
    steps, outputs = linear_steps(ops)
    return run_steps(record, steps, outputs)


@configurable(category="engine")
class JointStream:
    """
    Aggregates multiple Stream streams into a single joint stream.
    Each sub-stream maintains its own unique transformation chain.

    The iteration-only fan-in engine behind :meth:`Stream.joint`: each sub-stream applies
    its OWN op chain, so differently-processed streams concatenate lazily without
    materialization. For an indexable (random-access) concatenation of raw sources,
    use ``ConcatSource`` instead.

    Args:
        streams: The Stream streams to concatenate; iteration walks them in order and length is their sum.
            Defaults to ``None`` ⇒ an empty joint stream (zero-arg construction).
    """

    def __init__(self, streams: Optional[List["Stream"]] = None) -> None:
        # Lazy / zero-arg: store config only; no sub-streams ⇒ an empty stream.
        self.streams = streams if streams is not None else []

    def __iter__(self) -> Iterator[Record]:
        """Iterate through all sub-streams sequentially."""
        for stream in self.streams:
            yield from stream

    def __len__(self) -> int:
        """Total length is the sum of all sub-streams."""
        return sum(len(f) for f in self.streams)


@configurable(category="engine")
class Stream:
    """
    The primary stream engine for RecordStream.
    Wraps any iterable or indexed dataset and provides a functional API.

    Every carrier is a plain record ``dict`` of typed values, and every op is applied
    through the op-FAMILY dispatch (:func:`_apply_op`) — so native recordstream ops,
    bare albumentations transforms, and bare torchvision ``transforms.v2`` transforms
    all sit in ONE ``ops`` list as-is. ``source`` is duck-typed (any iterable; the
    Indexable protocol if ``__getitem__``/``__len__`` are present).

    Args:
        source: Any iterable or indexable dataset (duck-typed) yielding record dicts; ``None`` = empty stream.
        ops: Ordered ops applied lazily on access — native ops and bare library transforms alike (``None`` = no ops).
        chunk_size: Parallel-processing chunk size; ``0`` (the default) processes sequentially.
        class_names: Optional ordered class vocabulary this stream's labels index into. Set by
            :meth:`~recordstream.LabelMap.encode` so the vocabulary travels WITH the encoded
            data — a consumer that needs to name a predicted class id, or persist the mapping
            beside a checkpoint, reads it via :func:`~recordstream.class_names` instead of
            being handed a separate LabelMap it has to keep in sync.
    """

    def __init__(
        self,
        source: Optional[Iterable[Any]] = None,
        ops: Optional[List[Any]] = None,
        chunk_size: Optional[int] = 0,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.source = source
        self.ops: List[Any] = ops or []
        self.class_names: Optional[List[str]] = class_names
        self._workers = 1
        self._chunk_size = chunk_size or 0
        # Populated on first random access when the source is iterable-only
        # (has ``__len__`` but not ``__getitem__``).
        self._indexable_cache: Optional[List[Any]] = None

    def _guard_live_source(self) -> Any:
        """Return the source, surfacing a clear error when it's still a Fluid marker."""
        if isinstance(self.source, _ConfluidFluid):
            raise TypeError(_fluid_source_guidance(self.source))
        return self.source

    @classmethod
    def from_source(cls, source: Any) -> "Stream":
        """Create a Stream from a DataSource."""
        return cls(source=source)

    @classmethod
    def joint(cls, streams: List["Stream"]) -> "Stream":
        """Create a new Stream that aggregates multiple other Stream streams."""
        return cls(source=JointStream(streams))

    @classmethod
    def from_ops_yaml(cls, path: str, source: Optional[Iterable[Any]] = None) -> "Stream":
        """Attach an ops-only Confluid YAML (e.g. one exported by a pipeline-authoring tool) to ``source``.

        ``path`` is the ``{ops: [!class:...()]}`` document produced by an external graph
        exporter's ops-export. Op markers are materialized to live callables before being
        attached (``confluid.load`` leaves ``!class:`` markers nested under a mapping key
        deferred, so ``confluid.materialize`` flows them into live ops).
        """
        loaded = _confluid_load(path)
        raw_ops = loaded.get("ops", []) if isinstance(loaded, dict) else []
        ops = list(_confluid_materialize(raw_ops))
        return cls(source=source, ops=ops)

    @property
    def _expands(self) -> bool:
        """True when any (materialized) op is a 1→N expanding op — the pipeline is then iterable-only."""
        return any(not isinstance(op, _ConfluidFluid) and _op_expands(op) for op in self.ops)

    def _guard_not_expanding(self, operation: str) -> None:
        if self._expands:
            culprit = next(
                type(op).__name__ for op in self.ops if not isinstance(op, _ConfluidFluid) and _op_expands(op)
            )
            raise TypeError(
                f"Stream.{operation}: the pipeline contains the 1→N expanding op {culprit!r}, so the "
                "expanded length/index mapping is unknowable up front — the pipeline is ITERABLE-ONLY. "
                "Iterate it (or wrap in a torch IterableDataset); for random access, window/expand at "
                "the source instead, or materialize with list(stream) first."
            )

    def __len__(self) -> int:
        """Return the length of the underlying source if available."""
        from collections.abc import Sized

        source = self._guard_live_source()
        self._guard_not_expanding("__len__")
        if isinstance(source, Sized):
            return len(source)
        return 0

    def __getitem__(self, index: int) -> Any:
        """Random access: get the i-th record with ops applied."""
        source = self._guard_live_source()
        if source is None:
            raise TypeError("Stream source is None — cannot index. Pass a DataSource / iterable to Stream(source=...).")
        self._guard_not_expanding("__getitem__")

        if hasattr(source, "__getitem__"):
            raw = source[index]
        elif hasattr(source, "__len__"):
            if self._indexable_cache is None:
                logger.debug(f"Stream: materializing iterable-only source {type(source).__name__} for random access.")
                self._indexable_cache = list(source)
            raw = self._indexable_cache[index]
        else:
            raise TypeError(
                f"Stream source {type(source).__name__} does not support indexing and has no __len__ "
                "(bare iterator). Map-style DataLoader random access is unsafe on a one-shot "
                "iterator; give the source a __len__ (then Stream caches on first access) or wrap "
                "it in ``list(...)`` before handing it to Stream."
            )
        from recordstream.flow import run_steps

        _check_ops_materialized(self.ops)
        steps, outputs = linear_steps(self.ops)
        record = run_steps(raw, steps, outputs)
        if record is None:
            raise IndexError(f"Record {index} filtered out by the pipeline")
        return cast(Record, record)

    def to_sink(self, sink: Any) -> None:
        """Write the entire stream to a DataSink."""
        from recordstream.storage.base import Storage

        target_sink: Any = sink if isinstance(sink, Storage) else nullcontext()

        with target_sink:
            for record in self:
                sink.write(record)
            sink.flush()

    def parallel(self, workers: int = 4) -> "Stream":
        """Enable multiprocess execution for the pipeline."""
        self._workers = workers
        return self

    def batch(self, chunk_size: int) -> "Stream":
        """Group records into chunks (lists of N records)."""
        self._chunk_size = chunk_size
        return self

    def map(self, func: Callable, key: Optional[str] = None, **kwargs: Any) -> "Stream":
        """Append a transformation to the stream.

        ``key`` names the record entry whose payload ``func`` transforms; ``None`` hands
        ``func`` the whole record dict.
        """
        op = WrappedOp(func, key, kwargs)
        self.ops.append(op)
        return self

    def filter(self, predicate: Callable[[Record], bool]) -> "Stream":
        """Filter the stream based on a predicate."""
        self.ops.append(FilterOp(predicate))
        return self

    def __iter__(self) -> Iterator[Any]:
        """Execute the pipeline lazily."""
        if not self._guard_live_source():
            return

        if any(hasattr(op, "stream") and callable(op.stream) for op in self.ops):
            it = self._iter_streamed()
        elif self._workers > 1:
            it = self._iter_parallel()
        else:
            it = self._iter_sequential()

        if self._chunk_size > 0:
            batch = []
            for record in it:
                batch.append(record)
                if len(batch) == self._chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            yield from it

    def _iter_streamed(self) -> Iterator[Record]:
        """Mixed per-record / stream-level op chain (a stream-level op exposes ``.stream``).

        A stream-level op (``Parallel``) sees the WHOLE stream rather than one record, so it
        cannot be a step in the per-record graph — the chain is split at each such op and the
        per-record runs between them go through the ordinary kernel. Records travel as plain
        records: the per-record Context they used to be paired with is gone, and with it the
        "cells cannot cross a stream-op boundary" restriction that pairing imposed.
        """
        from recordstream.flow import run_steps_multi

        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)

        def per_record(stream: Iterator[Optional[Record]], op: Any) -> Iterator[Optional[Record]]:
            steps, outputs = linear_steps([op])
            for record in stream:
                if record is None:
                    continue
                yield from run_steps_multi(record, steps, outputs)

        carried: Iterator[Optional[Record]] = iter(source)
        for op in self.ops:
            if hasattr(op, "stream") and callable(op.stream):
                carried = op.stream(carried)
            else:
                carried = per_record(carried, op)

        for record in carried:
            if record is not None:
                yield record

    def _iter_sequential(self) -> Iterator[Record]:
        """Standard single-threaded execution — the flat op list run as a linear step graph."""
        from recordstream.flow import _result_readers, run_steps_multi

        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)
        # Compile + analyse ONCE per iteration, never per record (see run_steps_multi).
        steps, outputs = linear_steps(self.ops)
        readers = _result_readers(steps, outputs)
        for item in source:
            yield from run_steps_multi(item, steps, outputs, readers)

    def _iter_parallel(self) -> Iterator[Record]:
        """Multiprocess execution engine."""
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)

        # We use 'spawn' to be consistent with Loggair and prevent CI deadlocks
        ctx = multiprocessing.get_context("spawn")

        from recordstream.flow import _graph_worker_task

        steps, outputs = linear_steps(self.ops)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._workers, mp_context=ctx) as executor:
            futures = []
            extra_families = _extra_op_families()  # ship third-party op families to the workers
            for item in source:
                futures.append(executor.submit(_graph_worker_task, item, steps, outputs, extra_families))

            for future in futures:
                yield from future.result()

    def collect(self) -> List[Record]:
        """Materialize the full stream into a list."""
        return list(self)

    def project(self, keys: Collection[str]) -> Iterator[Record]:
        """Yield pipeline-output records carrying only ``keys`` (the projection primitive).

        Implements :class:`recordstream.projection.SupportsProjection`. Stream must run its op
        chain to produce each record (an op may consume the input), so this is the generic
        "iterate, then keep only the requested keys" form. Lazy: a generator.
        """
        want = set(keys)
        for record in self:
            yield {k: v for k, v in record.items() if k in want}


#: What a wired dataset slot may hold — the contract :func:`ensure_record_dataset` enforces,
#: What a wired dataset slot may hold, named ONCE here rather than restated by every consumer:
#: anything MAP-STYLE (``__len__`` + ``__getitem__`` — which a :class:`Stream` is), or any
#: iterable of records (a recordstream source, a plain list of record dicts). Consumers annotate
#: their slots ``Optional[Lazy[RecordSource]]`` — ``Lazy`` because they flow the slot at run time.
#:
#: Expressed with the structural :class:`MapStyle` rather than ``torch.utils.data.Dataset`` so the
#: engine can say "a dataset" without importing a framework; torch's ``DataLoader`` is itself
#: duck-typed and consumes either.
RecordSource = Union[MapStyle, Iterable[Record]]


def ensure_record_dataset(source: Optional[Union[_ConfluidFluid, RecordSource]]) -> "Stream":
    """Normalize any wired source into a map-style ``Dataset`` that yields record dicts.

    A wired ``train_set`` / ``val_set`` / ``test_set`` may be a :class:`Stream`, another torch
    ``Dataset``, a recordstream source (``HuggingFaceSource``), or a plain list — and its items
    may be record dicts or raw rows. A ``Stream`` already coerces every item to a record, so:

    * a ``Stream`` is returned as-is (already a ``Dataset`` of records; this preserves a
      subclass's own wrap, e.g. a label-encoding Stream with its ``class_names``), and
    * anything else is wrapped in a ``Stream``, which makes it both a map-style ``Dataset``
      AND a record-yielding one.

    Calling this once up front lets the rest of a training pipeline (target detection, label
    fitting/encoding, collate, metrics) assume record items — no per-call "is this a record?"
    checks. It lives beside :class:`Stream` because that is the only type it knows: the whole
    body is "already a Stream? else wrap in one".

    The parameter admits a ``Fluid`` and ``None`` because both genuinely occur at the call
    sites: a config hands over a deferred marker, and an optional split may be unwired (which
    yields an empty stream, so a caller needs no guard).

    A DEFERRED source (a ``!class:`` marker straight out of a config) is materialized first,
    matching :func:`~recordstream.project` and :meth:`~recordstream.LabelMap.encode`. Without
    it, wrapping a marker produced a ``Stream`` whose source was still a Fluid — which fails
    later, at first iteration, with an error about the Stream rather than about the config that
    caused it. Flowing a live object is a no-op.
    """
    from confluid import flow

    source = flow(source)
    if isinstance(source, Stream):
        return source
    # `cast`: a map-style `Dataset` iterates through Python's legacy `__getitem__` protocol,
    # which mypy does not model — so it is not `Iterable` statically even though `Stream`
    # consumes it correctly at runtime (`Stream.source` accepts "any iterable or indexable
    # dataset"). Widening that annotation with a Protocol breaks `to_pydantic` for every
    # Stream, so the exception is documented here instead. See TASKS.md.
    return Stream(source=cast(Iterable[Any], source))
