"""The ``Stream`` engine, its fan-in sibling ``JointStream``, and the ops-list plumbing.

``Stream`` is the dataset-surface facade over the one step-graph kernel: an ``ops:`` list is
compiled to POSITIONAL steps by :func:`linear_steps` and run by
:mod:`recordstream.flow`, exactly as a ``flow:`` document's author-named steps are. The two
spellings are the same thing (``docs/architecture.md`` §3).

What else lives here and why:

* ``JointStream`` — 20 lines that exist to be ``Stream.joint``'s return value (§5).
* :func:`linear_steps` / :func:`_worker_task` — both compile or run an ops LIST, which is
  ``Stream``'s spelling of a pipeline.
* :func:`ensure_record_dataset` — its whole body is "already a Stream? else wrap in one".
* the deferred-source guidance helpers — they phrase ``Stream``'s own Fluid errors.

The imports of :mod:`recordstream.flow` are body-local ON PURPOSE: flow imports the op
dispatch from :mod:`recordstream.core.families` at module level, so a top-level import back
would close the cycle.
"""

import concurrent.futures
import multiprocessing
from contextlib import nullcontext
from typing import Any, Callable, Collection, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, cast

from confluid import configurable
from confluid import load as _confluid_load
from confluid import materialize as _confluid_materialize
from confluid.fluid import Fluid as _ConfluidFluid
from loggair import get_logger

from recordstream.core.families import OpInvoker, OpMatcher, _extra_op_families, _op_expands, _sync_op_families
from recordstream.core.mapstyle import RecordSource
from recordstream.core.wrappers import FilterOp, WrappedOp
from recordstream.items import Record

logger = get_logger(__name__)


def _describe_deferred_source(source: Any) -> str:
    """Return a human-friendly description of a still-deferred Confluid source.

    Surfaces the tag/target so the error explains WHAT was deferred instead
    of just noting it isn't a live object.
    """
    target = getattr(source, "target", "<unknown>")
    target_name = target if isinstance(target, str) else getattr(target, "__qualname__", str(target))
    return f"{type(source).__name__}(target={target_name!r})"


def _fluid_source_guidance(source: Any, slot: str = "Stream.source") -> str:
    """Build an actionable message when a source slot is still a Confluid Fluid.

    ``slot`` names the owning slot (``"Stream.source"``, ``"RangeSource.source"``, …) so the
    view sources raise the SAME guidance Stream does — a still-deferred ``source:`` is a
    CONFIG error (the parens-less ``!class:X`` spelling), never something the engine flows.
    """
    return (
        f"{slot} is still a deferred Confluid marker: {_describe_deferred_source(source)}. "
        "Confluid has not materialized it yet. Fixes: (a) in YAML, write the source as "
        "`!class:X()` (with parens) instead of `!class:X` so it becomes an Instance and is "
        "materialized at load time; (b) or call `flow(source)` on the source before wiring it."
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
    through the op-FAMILY dispatch (:func:`~recordstream.core.families._apply_op`) — so
    native recordstream ops, bare albumentations transforms, and bare torchvision
    ``transforms.v2`` transforms all sit in ONE ``ops`` list as-is. ``source`` is
    duck-typed (any iterable; the Indexable protocol if ``__getitem__``/``__len__`` are
    present).

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


def ensure_materialized(source: RecordSource) -> RecordSource:
    """Read ONE whole record, so everything this source builds lazily is built in THIS process.

    The companion to :func:`ensure_record_dataset`: that one normalizes a source's TYPE, this one
    normalizes its STATE. Returns the source, so it composes.

    A source in this package is lazy on purpose — a constructor does no work, and the download /
    file open / client construction happens on first read. That is right until the first read
    happens somewhere it must not, and there is one such place: **a forked child process.**

    Measured, on macOS, 2026-08-02. A ``DataLoader`` worker was the first to touch a
    ``HuggingFaceSource``, so ``load_dataset`` ran in the child, called ``hf_hub_download`` ->
    ``httpx.Client()`` -> ``urllib.request.getproxies`` -> ``_scproxy`` -> CoreFoundation, which
    is not fork-safe: **SIGSEGV**, with no Python traceback, surfacing only as
    ``DataLoader worker exited unexpectedly``. Calling this in the parent first makes the child
    inherit a source that needs nothing, and the crash does not happen. The consumer cannot
    always choose spawn instead — a framework may set the start method globally (fastai sets
    ``fork`` at import), and spawning has its own cost (a model on Apple's MPS cannot be shared
    to a spawned worker at all).

    It reads a whole RECORD rather than asking a cheaper question, and that is the point:
    ``len(source)`` was measured NOT to be enough, because loading the dataset object is not the
    same as building everything a read needs. :func:`~recordstream.first_value` is no substitute
    either — it is projection-aware, so it deliberately avoids building the values it was not
    asked for.

    An empty source is not an error: there is nothing to build, and a caller should not need a
    guard for a split that happens to have no rows.
    """
    try:
        if hasattr(source, "__getitem__"):
            source[0]  # type: ignore[index]
        else:
            next(iter(source), None)  # type: ignore[call-overload]
    except (IndexError, StopIteration):
        pass  # empty source — nothing to warm, and not an error
    return source


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


def prepare_record_dataset(source: Optional[Union[_ConfluidFluid, RecordSource]]) -> Optional["Stream"]:
    """Normalize a wired dataset slot's TYPE and its STATE, in THIS process.

    The composition every forking consumer writes:
    :func:`ensure_record_dataset` normalizes the TYPE (any wired source into a record-yielding
    map-style ``Stream``) and :func:`ensure_materialized` normalizes the STATE (one whole record
    read here, so a lazy source is built in the CALLING process rather than in a forked
    ``DataLoader`` worker — where a first read SIGSEGVs through ``_scproxy`` / CoreFoundation on
    macOS with no Python traceback; the full account lives on :func:`ensure_materialized`).

    ``None`` passes through, so an unwired optional split (``val_set``) needs no guard at the
    call site. This function exists because the pair was re-composed identically in every
    training runnable across the workspace ("``_prepare``"); a consumer needing only one half
    still calls that half directly.
    """
    if source is None:
        return None
    return cast("Stream", ensure_materialized(ensure_record_dataset(source)))
