import concurrent.futures
import json
import multiprocessing
from contextlib import nullcontext
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch.utils.data
from confluid import configurable
from confluid import load as _confluid_load
from confluid import materialize as _confluid_materialize
from confluid.fluid import Fluid as _ConfluidFluid
from loggair import get_logger

from sampleflux.context import Context, activate
from sampleflux.projection import ProjectionField
from sampleflux.sample import FEATURES_KEY, SPEC_KEY, TYPE_KEYS, InputMeta, Pair, Sample, TargetMeta

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sampleflux.typespec import SampleType

logger = get_logger(__name__)


@lru_cache(maxsize=None)
def _serialized_type_keys(produces: "SampleType") -> Tuple[str, str]:
    """Serialize an op's ``PRODUCES`` to the two stored-type JSON strings, memoized per spec so the
    ``datasets.Features`` build happens once per distinct spec rather than once per sample."""
    features, extras = produces.to_hf_features()
    return json.dumps(features.to_dict()), json.dumps(extras)


def _refresh_type(sample: Sample, op: Any) -> Sample:
    """Keep a sample's stored type honest after an op — only when the sample already carries one.

    Default (untracked) pipelines never stamp a type, so this is a no-op and metadata is byte-identical
    to before. When a stored type IS present (set via :meth:`Sample.with_type` or loaded from a typed
    dataset), an op that declares ``PRODUCES`` refreshes it; an op that declares none drops it so
    :meth:`Sample.describe` falls back to inference rather than reporting a stale type.
    """
    if not any(key in sample.meta for key in TYPE_KEYS):
        return sample
    produces = getattr(op, "PRODUCES", None)
    if produces is not None:
        features_key, spec_key = _serialized_type_keys(produces)
        return sample._replace(metadata={**sample.meta, FEATURES_KEY: features_key, SPEC_KEY: spec_key})
    return sample._replace(metadata={k: v for k, v in sample.meta.items() if k not in TYPE_KEYS})


def _apply_op(sample: Sample, op: Any) -> Optional[Sample]:
    """Apply one op and refresh the stored type. The single op-application chokepoint shared by the
    sequential, parallel (via :func:`_worker_task`), streamed, and random-access (``__getitem__``) paths.

    The op's introspected contract (:func:`sampleflux.kinds.op_contract`) picks the BINDING:
    a classic sample/untyped op receives the Sample verbatim (today's fast path); a
    field-scoped op (``input`` / ``target`` / ``pair`` / ``input_meta`` / ``target_meta`` —
    packed views or unpacked separate arguments) receives exactly its declared view and the
    result merges back with the untouched fields preserved (:func:`_apply_view`).
    """
    from sampleflux.kinds import op_contract

    contract = op_contract(op)
    if contract.accepts in ("sample", "any", "value") and contract.style == "packed":
        result = op(sample)
        if result is None:
            return None
        return _refresh_type(result, op)
    return _apply_view(sample, op, contract)


def _view_error(op: Any, scope: str, result: Any) -> TypeError:
    return TypeError(
        f"{type(op).__name__}: a {scope!r}-scope op must return the matching view/tuple, a full "
        f"Sample, or None — got {type(result).__name__}"
    )


# One argument of an unpacked op, bound from the sample per its declared field scope.
_BIND_GET: Dict[str, Callable[[Sample], Any]] = {
    "input": lambda s: s.input,
    "target": lambda s: s.target,
    "metadata": lambda s: s.meta,
    "input_meta": lambda s: s.input_meta(),
    "target_meta": lambda s: s.target_meta(),
}


def _apply_bindings(sample: Sample, op: Any, bindings: Tuple[str, ...]) -> Optional[Sample]:
    """Apply an UNPACKED op — each argument bound per its declared field scope — and merge back.

    Handles EVERY combination the binding resolver produces: the classic
    ``f(input, target)`` / ``f(input, target, metadata)``, the meta forms
    ``f(input, metadata)`` / ``f(target, metadata)``, and mixed VIEW arguments like
    ``f(im: InputMeta, tm: TargetMeta)`` or ``f(x: Input, tm: TargetMeta)``. The result
    must be ``None`` (drop), a full ``Sample`` (takes over), or a tuple of the SAME arity
    — each element merged per its binding (a view/2-tuple element for a ``*_meta`` binding
    replaces value + metadata; a bare element replaces only the value). Metadata-bearing
    elements merge left-to-right (the LAST metadata write wins — they usually share the
    one live dict anyway, which the op may also mutate in place).
    """
    args = [_BIND_GET[b](sample) for b in bindings]
    result = op(*args)
    if result is None:
        return None
    if isinstance(result, Sample):
        return _refresh_type(result, op)
    # A NAMED view is itself a tuple — returning ONE view from a multi-binding op would be
    # silently misread as two elements, so it only counts as the whole result at arity 1.
    is_single_view = isinstance(result, (InputMeta, TargetMeta, Pair))
    if (is_single_view and len(bindings) != 1) or not (isinstance(result, tuple) and len(result) == len(bindings)):
        raise TypeError(
            f"{type(op).__name__}: an unpacked op bound as {bindings!r} must return a tuple of the "
            f"same arity, a full Sample, or None — got {type(result).__name__}"
        )
    updates: Dict[str, Any] = {}
    for binding, element in zip(bindings, result):
        if binding in ("input", "target", "metadata"):
            updates[binding] = element
        else:  # input_meta / target_meta
            field = "input" if binding == "input_meta" else "target"
            if isinstance(element, tuple) and len(element) == 2:
                updates[field] = element[0]
                updates["metadata"] = element[1]
            else:  # bare value: only the field changes (in-place meta mutation is already live)
                updates[field] = element
    return _refresh_type(sample._replace(**updates), op)


def _apply_view(sample: Sample, op: Any, contract: Any) -> Optional[Sample]:
    """Bind a field-scoped op's declared view from ``sample``, apply, and merge the result back.

    Unpacked ops route through :func:`_apply_bindings` (per-argument scopes). Packed
    single-view scopes (``None`` always drops; a returned ``Sample`` always takes over;
    metadata dicts are handed live, so in-place mutation propagates):

    - ``input`` / ``target`` — the bare value in, the new value out (other fields kept);
    - ``metadata`` — the dict in, the (new) dict out;
    - ``pair`` — a `Pair` in (a plain-tuple-annotated op indexes it identically), a
      2-tuple out replaces input+target (metadata kept);
    - ``input_meta`` / ``target_meta`` — the named view in; a view/2-tuple out replaces
      value + metadata; a bare value out replaces only the value.
    """
    if contract.style == "unpacked" and contract.bindings:
        return _apply_bindings(sample, op, contract.bindings)
    scope = contract.accepts

    if scope == "input" or scope == "target":
        field = scope
        result = op(getattr(sample, field))
        if result is None:
            return None
        return _refresh_type(sample._replace(**{field: result}), op)

    if scope == "metadata":
        result = op(sample.meta)
        if result is None:
            return None
        if isinstance(result, Sample):
            return _refresh_type(result, op)
        if isinstance(result, dict):
            return _refresh_type(sample._replace(metadata=result), op)
        raise _view_error(op, scope, result)

    if scope == "pair":
        result = op(Pair(sample.input, sample.target))
        if result is None:
            return None
        if isinstance(result, Sample):
            return _refresh_type(result, op)
        if isinstance(result, tuple) and len(result) == 2:
            return _refresh_type(sample._replace(input=result[0], target=result[1]), op)
        raise _view_error(op, scope, result)

    if scope in ("input_meta", "target_meta"):
        field = "input" if scope == "input_meta" else "target"
        result = op(sample.input_meta() if scope == "input_meta" else sample.target_meta())
        if result is None:
            return None
        if isinstance(result, Sample):
            return _refresh_type(result, op)
        if isinstance(result, tuple) and len(result) == 2:
            return _refresh_type(sample._replace(**{field: result[0], "metadata": result[1]}), op)
        return _refresh_type(sample._replace(**{field: result}), op)

    # A packed "sample"-scope op took the _apply_op fast path; anything else is defensive.
    result = op(sample)  # pragma: no cover
    return None if result is None else _refresh_type(result, op)  # pragma: no cover


def _describe_deferred_source(source: Any) -> str:
    """Return a human-friendly description of a still-deferred Confluid source.

    Surfaces the tag/target so the error explains WHAT was deferred instead
    of just noting it isn't a live object.
    """
    target = getattr(source, "target", "<unknown>")
    target_name = target if isinstance(target, str) else getattr(target, "__qualname__", str(target))
    return f"{type(source).__name__}(target={target_name!r})"


def _fluid_source_guidance(source: Any) -> str:
    """Build an actionable message when Flux.source is still a Confluid Fluid."""
    return (
        f"Flux.source is still a deferred Confluid marker: {_describe_deferred_source(source)}. "
        "Confluid has not materialized it yet. Fixes: (a) in YAML, write the source as "
        "`!class:X()` (with parens) instead of `!class:X` so it becomes an Instance and is "
        "materialized at load time; (b) or call `flow(source)` on the source before handing "
        "it to Flux."
    )


def _fluid_op_guidance(op: Any, index: int) -> str:
    """Build an actionable message when a Flux op is still a Confluid Fluid."""
    return (
        f"Flux.ops[{index}] is still a deferred Confluid marker: {_describe_deferred_source(op)}. "
        "Ops must be live callables at iteration time. Fixes: (a) in YAML, write each op as "
        "`!class:X()` (with parens) so it becomes an Instance and is materialized at load "
        "time; (b) or call `flow(op)` on the op before handing it to Flux."
    )


def _check_ops_materialized(ops: List[Any]) -> None:
    """Raise a single actionable error if any op is still a Confluid Fluid marker."""
    for i, op in enumerate(ops):
        if isinstance(op, _ConfluidFluid):
            raise TypeError(_fluid_op_guidance(op, i))


@configurable
class FilterOp:
    """Configurable filter operation.

    Args:
        p: Predicate ``Sample -> bool``; the sample passes through when it returns ``True``, else is dropped.
            Defaults to ``None`` (zero-arg construction); a predicate must be set before the op runs.
    """

    def __init__(self, p: Optional[Callable[[Sample], bool]] = None):
        # Lazy / zero-arg: store config only; a missing predicate is validated lazily in __call__.
        self.p = p

    def __call__(self, s: Sample) -> Optional[Sample]:
        if self.p is None:
            raise ValueError("FilterOp.p (predicate) is not set — provide a Sample->bool callable before use.")
        return s if self.p(s) else None


@configurable
class WrappedOp:
    """Configurable transformation wrapper with smart mapping.

    Args:
        f: The wrapped callable, or its importable ``module:function`` path (stored as a string for serialization).
            Defaults to ``""`` (zero-arg construction); resolving an empty path fails lazily on first call.
        s: Which Sample slot to transform — ``"input"`` (default), ``"target"``, or ``"all"`` (the whole Sample).
        kw: Extra keyword arguments forwarded to the wrapped callable on every call (defaults to none).
    """

    def __init__(self, f: Union[str, Callable] = "", s: str = "input", kw: Optional[Dict[str, Any]] = None):
        from sampleflux.discovery import get_callable_path

        # Lazy / zero-arg: store config only (the empty-path default resolves lazily via the `func`
        # property). EXPLICIT: always store the string path for serialization.
        self.f = get_callable_path(f) if callable(f) else f
        self.s = s
        self.kw = dict(kw) if kw else {}
        # Internal cache for the live callable
        self._func_cache: Optional[Callable] = None

    @property
    def func(self) -> Callable:
        if self._func_cache is None:
            from sampleflux.discovery import resolve_callable

            self._func_cache = resolve_callable(self.f)
        return self._func_cache

    def __call__(self, sample: Sample) -> Optional[Sample]:
        try:
            if self.s == "input":
                new_input = self.func(sample.input, **self.kw)
                return sample._replace(input=new_input)
            elif self.s == "target":
                new_target = self.func(sample.target, **self.kw)
                return sample._replace(target=new_target)
            elif self.s == "all":
                return cast(Sample, self.func(sample, **self.kw))
            return sample
        except Exception as e:
            raise e


class _Carried(NamedTuple):
    """A carrier travelling the streamed route together with its per-sample Context.

    ``sample`` is a :class:`Sample` on the default route; under ``Flux(native=True)`` it
    may be any native carrier (a metadata-free pair, a bare value).
    """

    sample: Any
    ctx: Context


def _apply_op_native(carrier: Any, op: Any) -> Any:
    """Apply one op to a NATIVE carrier (Sample / pair / bare value / a field view).

    Adaptation rules (the op's contract via :func:`sampleflux.kinds.op_contract`):

    - a **Sample** carrier routes through :func:`_apply_op` (which binds every scope);
    - an **any-op** receives the carrier verbatim (untyped ops behave exactly as today);
    - two NATIVE fast lanes keep metadata-free data metadata-free: a pair-scope op on a
      pair carrier (result stays a pair) and an input-scope op on a bare value (result
      stays a bare value);
    - everything else PROMOTES the carrier to a Sample view (``Sample.from_any`` — view
      types like ``InputMeta`` coerce field-correctly) — promotion is one-way and sticky,
      so op-written metadata is never dropped.
    """
    from sampleflux.kinds import classify_carrier, op_contract

    contract = op_contract(op)
    if isinstance(carrier, Sample):
        return _apply_op(carrier, op)
    if contract.accepts == "any":
        return op(carrier)
    kind = classify_carrier(carrier)
    if contract.accepts == "pair" and kind == "pair":
        result = op(carrier[0], carrier[1]) if contract.style == "unpacked" else op(tuple(carrier))
        if result is None:
            return None
        if isinstance(result, (Sample, tuple)):
            return result
        raise _view_error(op, "pair", result)
    if contract.accepts == "input" and kind == "value":
        return op(carrier)
    return _apply_op(Sample.from_any(carrier), op)  # promotion is sticky


def _expand(op: Any, carrier: Any) -> List[Any]:
    """Run a 1→N EXPANDING op and return its flattened, type-refreshed children."""
    raw = op(carrier)
    if raw is None:
        return []
    children: List[Any] = []
    for child in raw:
        if child is None:
            continue
        children.append(_refresh_type(child, op) if isinstance(child, Sample) else child)
    return children


def _worker_task(sample: Any, ops: List[Any], native: bool = False) -> Optional[Any]:
    """Single-result worker for STRICTLY 1→1 op lists (the ``Parallel`` op's contract).

    Kept for callers that need exactly one carrier back; expanding ops raise here —
    route expanding pipelines through :func:`_worker_task_multi`.
    """
    results = _worker_task_multi(sample, ops, native=native, allow_expansion=False)
    return results[0] if results else None


def _worker_task_multi(sample: Any, ops: List[Any], native: bool = False, allow_expansion: bool = True) -> List[Any]:
    """Top-level helper for multiprocess workers. Must be at top level for pickling.

    Runs one source carrier through the op list and returns EVERY resulting carrier —
    usually one, zero when filtered, several when a 1→N EXPANDING op fired (detected via
    :func:`sampleflux.kinds.op_contract`; each expansion child continues through the
    REMAINING ops with a shallow copy of the per-sample Context, depth-first so sibling
    order matches the nested-loop intuition).

    Activates ONE fresh per-carrier :class:`~sampleflux.context.Context` around the op
    loop so context ops (``Save``/``Use``/``Apply``/``Capture``/``Mix``) can move data
    between the linear stream and named cells — the executor itself stays a plain
    ``for op in ops`` loop. Contexts are created inside the worker (spawn-safe: ops
    pickle, a Context never crosses a process boundary).

    ``native=True`` keeps the carrier's own kind (Sample / pair / value) and adapts it
    per op via :func:`_apply_op_native` instead of coercing everything to Sample.
    """
    from collections import deque

    from sampleflux.kinds import op_contract

    pending: "deque[Tuple[Any, Context, int]]" = deque([(sample, Context(), 0)])
    out: List[Any] = []
    while pending:
        current, ctx, start = pending.popleft()
        alive = True
        with activate(ctx):
            i = start
            while i < len(ops):
                op = ops[i]
                i += 1
                if op_contract(op).expands:
                    if not allow_expansion:
                        raise TypeError(
                            f"op {type(op).__name__!r} is a 1→N expanding op, which this strictly "
                            "1→1 route cannot carry — run it through the Flux iteration paths."
                        )
                    children = _expand(op, current)
                    if not children:
                        alive = False
                        break
                    # Depth-first: the first child continues inline; its siblings go to the
                    # FRONT of the queue (reversed, so sibling order is preserved) — output
                    # order matches the nested-loop intuition even for chained expansions.
                    for child in reversed(children[1:]):
                        pending.appendleft((child, ctx.copy(), i))
                    current = children[0]
                    continue
                result = _apply_op_native(current, op) if native else _apply_op(current, op)
                if result is None:
                    alive = False
                    break
                current = result
        if alive and current is not None:
            out.append(current)
    return out


@configurable(category="engine")
class JointFlux:
    """
    Aggregates multiple Flux streams into a single joint stream.
    Each sub-flux maintains its own unique transformation chain.

    Args:
        fluxes: The Flux streams to concatenate; iteration walks them in order and length is their sum.
            Defaults to ``None`` ⇒ an empty joint stream (zero-arg construction).
    """

    def __init__(self, fluxes: Optional[List["Flux"]] = None) -> None:
        # Lazy / zero-arg: store config only; no sub-fluxes ⇒ an empty stream.
        self.fluxes = fluxes if fluxes is not None else []

    def __iter__(self) -> Iterator[Sample]:
        """Iterate through all sub-fluxes sequentially."""
        for flux in self.fluxes:
            yield from flux

    def __len__(self) -> int:
        """Total length is the sum of all sub-fluxes."""
        return sum(len(f) for f in self.fluxes)


@configurable(category="engine")
class Flux(torch.utils.data.Dataset[Sample]):
    """
    The primary stream engine for SampleFlux.
    Wraps any iterable or indexed dataset and provides a functional API.

    Annotation design (kept intentionally ``Any``):
        Per the SampleFlux mandate "Functional Purity: Transforms are plain
        Python callables. Never introduce base classes or complex inheritance
        for data operations.", ``source`` is duck-typed (any iterable; the
        Indexable protocol if ``__getitem__``/``__len__`` are present) and
        ``ops`` is a list of bare callables ``Sample -> Optional[Sample]``.
        No ``Source`` or ``Op`` ABC is introduced.

        Downstream auto-gen pydantic mirrors (``confluid.to_pydantic``)
        coerce abstract iterable types to ``Any`` so identity-tracked
        serialization (e.g. shared-source dataset-split patterns in
        navigaitor) works correctly — see
        ``confluid/pydantic_export.py:_ITER_TYPES_AS_ANY``.

    Args:
        source: Any iterable or indexable dataset (duck-typed) to wrap; ``None`` yields an empty stream.
        ops: Ordered callables ``Sample -> Optional[Sample]`` applied lazily on access (``None`` = no ops).
        chunk_size: Parallel-processing chunk size; ``0`` (the default) processes sequentially.
        native: Opt-in multi-type mode — carriers keep their own kind (Sample / metadata-free pair /
            bare value) and each op is adapted per its introspected contract (``sampleflux.kinds``).
            ``False`` (the default) coerces every item to ``Sample`` exactly as before.
    """

    def __init__(
        self,
        source: Optional[Iterable[Any]] = None,
        ops: Optional[List[Any]] = None,
        chunk_size: Optional[int] = 0,
        native: bool = False,
    ) -> None:
        self.source = source
        self.ops: List[Any] = ops or []
        self.native = bool(native)
        self._workers = 1
        self._chunk_size = chunk_size or 0
        # Populated on first random access when the source is iterable-only
        # (has ``__len__`` but not ``__getitem__``).
        self._indexable_cache: Optional[List[Any]] = None

    def _guard_live_source(self) -> Any:
        """Return the source, surfacing a clear error when it's still a Fluid marker.

        Flux does not materialize deferred Confluid markers itself — that's
        Confluid's job — but if a user hands Flux a deferred Class/Instance
        marker we raise with an actionable message instead of letting the
        failure surface as ``num_samples=0`` or a generic ``TypeError`` deep
        inside torch's DataLoader.
        """
        if isinstance(self.source, _ConfluidFluid):
            raise TypeError(_fluid_source_guidance(self.source))
        return self.source

    @classmethod
    def from_source(cls, source: Any) -> "Flux":
        """Create a Flux from a DataSource."""
        return cls(source=source)

    @classmethod
    def joint(cls, fluxes: List["Flux"]) -> "Flux":
        """Create a new Flux that aggregates multiple other Flux streams."""
        return cls(source=JointFlux(fluxes))

    @classmethod
    def from_ops_yaml(cls, path: str, source: Optional[Iterable[Any]] = None) -> "Flux":
        """Attach an ops-only Confluid YAML (e.g. one exported by a pipeline-authoring tool) to ``source``.

        ``path`` is the ``{ops: [!class:...()]}`` document produced by
        an external graph exporter's ops-export (the CLI or the
        canvas Export button). It also accepts an inline YAML string (``confluid.load``
        handles both).

        The op markers are **materialized to live callables** before being attached:
        ``confluid.load`` leaves ``!class:`` markers nested under a mapping key deferred
        (its final flow pass doesn't descend dict→list), so a plain ``load(path)["ops"]``
        would hand :class:`Flux` deferred ``Instance`` markers — which iteration rejects by
        design (see :meth:`_guard_live_source` / ``_check_ops_materialized``). Routing through
        :func:`confluid.materialize` flows the top-level list of markers into live ops.
        """
        loaded = _confluid_load(path)
        raw_ops = loaded.get("ops", []) if isinstance(loaded, dict) else []
        ops = list(_confluid_materialize(raw_ops))
        return cls(source=source, ops=ops)

    @classmethod
    def from_flow_yaml(cls, path: str, source: Optional[Iterable[Any]] = None) -> "Flux":
        """Attach a ``{flow: {...}}`` graph document to ``source``, LOWERED to the serial form.

        The named-step flow document (see :mod:`sampleflux.flow`) is compiled into a flat
        context-ops list via :func:`sampleflux.flow.to_ops`, so the graph executes on this
        plain serial engine. ``FlowGraph.from_yaml`` is the native-engine twin.
        """
        from sampleflux.flow import flow_yaml_to_flux

        return cast("Flux", flow_yaml_to_flux(path, source=source))

    @property
    def _expands(self) -> bool:
        """True when any (materialized) op is a 1→N expanding op — the pipeline is then iterable-only."""
        from sampleflux.kinds import op_contract

        return any(not isinstance(op, _ConfluidFluid) and op_contract(op).expands for op in self.ops)

    def _guard_not_expanding(self, operation: str) -> None:
        if self._expands:
            from sampleflux.kinds import op_contract

            culprit = next(
                type(op).__name__ for op in self.ops if not isinstance(op, _ConfluidFluid) and op_contract(op).expands
            )
            raise TypeError(
                f"Flux.{operation}: the pipeline contains the 1→N expanding op {culprit!r}, so the "
                "expanded length/index mapping is unknowable up front — the pipeline is ITERABLE-ONLY. "
                "Iterate it (or wrap in a torch IterableDataset); for random access, window/expand at "
                "the source instead, or materialize with list(flux) first."
            )

    def __len__(self) -> int:
        """Return the length of the underlying source if available.

        Surfaces a clear error when the source is still a deferred Confluid
        marker so downstream callers (e.g. torch's DataLoader) don't end up
        reporting the opaque ``num_samples=0``, and when the pipeline contains
        a 1→N expanding op (iterable-only — the true length is unknowable).
        """
        from collections.abc import Sized

        source = self._guard_live_source()
        self._guard_not_expanding("__len__")
        if isinstance(source, Sized):
            return len(source)
        return 0

    def __getitem__(self, index: int) -> Sample:
        """Random access: get the i-th sample with ops applied.

        Supports three source shapes:

        - **Indexable** (``__getitem__`` present) — delegates directly.
        - **Iterable with ``__len__``** (map-style-but-stream, like
          :class:`waivefront.regions_source.RegionsJsonSource`) — materializes
          the full source into a list on first access, caches it on the Flux
          instance, and indexes into the cache on every subsequent call.
          The list is built once per Flux lifetime, not once per epoch.
        - **Bare iterator** (no ``__len__``) — raises ``TypeError``. Caching
          a one-shot iterator silently would consume the user's source; if
          random access is genuinely needed, either give the source a
          ``__len__`` or wrap with ``list(source)`` explicitly at the call
          site.
        """
        source = self._guard_live_source()
        if source is None:
            raise TypeError("Flux source is None — cannot index. Pass a DataSource / iterable to Flux(source=...).")
        self._guard_not_expanding("__getitem__")

        if hasattr(source, "__getitem__"):
            raw = source[index]
        elif hasattr(source, "__len__"):
            if self._indexable_cache is None:
                logger.debug(
                    f"Flux: materializing iterable-only source " f"{type(source).__name__} for map-style random access."
                )
                self._indexable_cache = list(source)
            raw = self._indexable_cache[index]
        else:
            raise TypeError(
                f"Flux source {type(source).__name__} does not support indexing and has no __len__ "
                "(bare iterator). Map-style DataLoader random access is unsafe on a one-shot "
                "iterator; give the source a __len__ (then Flux caches on first access) or wrap "
                "it in ``list(...)`` before handing it to Flux."
            )
        _check_ops_materialized(self.ops)
        sample: Any = raw if self.native else Sample.from_any(raw)
        with activate(Context()):
            for op in self.ops:
                result = _apply_op_native(sample, op) if self.native else _apply_op(sample, op)
                if result is None:
                    raise IndexError(f"Sample {index} filtered out by {op}")
                sample = result
        return cast(Sample, sample)

    def to_sink(self, sink: Any) -> None:
        """Write the entire flux to a DataSink."""
        from sampleflux.storage.base import Storage

        # Open sink if it's a context-aware storage; otherwise no-op context.
        target_sink: Any = sink if isinstance(sink, Storage) else nullcontext()

        with target_sink:
            for sample in self:
                sink.write(sample)
            sink.flush()

    def parallel(self, workers: int = 4) -> "Flux":
        """
        Enable multiprocess execution for the pipeline.

        Args:
            workers: Number of worker processes to spawn.
        """
        self._workers = workers
        return self

    def batch(self, chunk_size: int) -> "Flux":
        """
        Group samples into chunks (lists of N samples).

        Args:
            chunk_size: Number of samples per chunk.
        """
        self._chunk_size = chunk_size
        return self

    def map(self, func: Callable, select: str = "input", **kwargs: Any) -> "Flux":
        """
        Append a transformation to the flux.
        """
        op = WrappedOp(func, select, kwargs)
        self.ops.append(op)
        return self

    def filter(self, predicate: Callable[[Sample], bool]) -> "Flux":
        """Filter the flux based on a predicate."""
        self.ops.append(FilterOp(predicate))
        return self

    def __iter__(self) -> Iterator[Any]:
        """Execute the pipeline lazily.

        Routing:
          * Any op exposes a callable ``stream`` attribute (e.g.
            :class:`sampleflux.ops.parallel.Parallel`) → :meth:`_iter_streamed`,
            which composes the upstream iterator through stream-level ops.
          * Else ``self._workers > 1`` → legacy :meth:`_iter_parallel`.
          * Else :meth:`_iter_sequential`.
        """
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
            for sample in it:
                batch.append(sample)
                if len(batch) == self._chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            yield from it

    def _iter_streamed(self) -> Iterator[Sample]:
        """Mixed per-sample / stream-level op chain.

        Per-sample ops are applied via ``op(sample)``. Ops that implement
        ``.stream(sample_iter)`` (e.g.
        :class:`sampleflux.ops.parallel.Parallel`) are handed the upstream
        generator and yield transformed samples themselves. ``None`` results
        are filtered, matching :meth:`_iter_sequential`.

        Each sample travels with its own per-sample :class:`Context` (a private
        ``(sample, ctx)`` carrier between per-sample stages), activated around
        every ``_apply_op`` call. A stream-level op is a Context boundary: the
        carrier is stripped to a bare sample before ``op.stream(...)`` (raising
        if cells are still live — cross-``Parallel`` graphs are a documented v1
        limit; ``Parallel``'s INNER chain gets its own contexts via
        :func:`_worker_task`), and samples emerging downstream get fresh
        contexts.
        """
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)

        def to_carried() -> Iterator[Optional[_Carried]]:
            for item in source:
                yield _Carried(item if self.native else Sample.from_any(item), Context())

        def per_sample(stream: Iterator[Optional[_Carried]], op: Any) -> Iterator[Optional[_Carried]]:
            from sampleflux.kinds import op_contract

            expands = op_contract(op).expands
            for c in stream:
                if c is None:
                    continue
                with activate(c.ctx):
                    if expands:
                        children = _expand(op, c.sample)
                    else:
                        s = _apply_op_native(c.sample, op) if self.native else _apply_op(c.sample, op)
                if expands:
                    for j, child in enumerate(children):
                        yield _Carried(child, c.ctx if j == 0 else c.ctx.copy())
                else:
                    yield None if s is None else _Carried(s, c.ctx)

        def strip(stream: Iterator[Optional[_Carried]], op: Any) -> Iterator[Optional[Sample]]:
            for c in stream:
                if c is None:
                    yield None
                    continue
                if c.ctx.live():
                    raise RuntimeError(
                        f"Flux: context cells {c.ctx.live()!r} are still live at the stream-level op "
                        f"{type(op).__name__!r}. Context cells cannot cross a stream-op boundary "
                        f"(e.g. Parallel) — drop them before it, or move the whole graph inside it."
                    )
                yield c.sample

        def wrap(stream: Iterator[Optional[Sample]]) -> Iterator[Optional[_Carried]]:
            for s in stream:
                yield None if s is None else _Carried(s, Context())

        carried: Iterator[Optional[_Carried]] = to_carried()
        for op in self.ops:
            if hasattr(op, "stream") and callable(op.stream):
                carried = wrap(op.stream(strip(carried, op)))
            else:
                carried = per_sample(carried, op)

        for c in carried:
            if c is not None:
                yield c.sample

    def _iter_sequential(self) -> Iterator[Sample]:
        """Standard single-threaded execution."""
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)
        for item in source:
            sample = item if self.native else Sample.from_any(item)
            yield from _worker_task_multi(sample, self.ops, native=self.native)

    def _iter_parallel(self) -> Iterator[Sample]:
        """Multiprocess execution engine."""
        source = self._guard_live_source()
        if source is None:
            return
        _check_ops_materialized(self.ops)

        # We use 'spawn' to be consistent with Loggair and prevent CI deadlocks
        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._workers, mp_context=ctx) as executor:
            futures = []
            for item in source:
                sample = item if self.native else Sample.from_any(item)
                futures.append(executor.submit(_worker_task_multi, sample, self.ops, self.native))

            for future in futures:
                yield from future.result()

    def collect(self) -> List[Sample]:
        """Materialize the full flux into a list."""
        return list(self)

    def project(self, fields: Collection[ProjectionField]) -> Iterator[Sample]:
        """Yield pipeline-output Samples carrying only ``fields`` (the projection primitive).

        Implements :class:`sampleflux.projection.SupportsProjection`. Flux must run
        its op chain to produce each Sample (an op may consume the input), so this
        is the generic "iterate, then drop unrequested fields" form — it cannot
        skip input construction the way a leaf source (e.g. an image dataset that
        reads only the label column) can. Lazy: a generator. ``fields`` is a
        subset of ``{"input", "target", "metadata"}``.
        """
        want = frozenset(fields)
        for sample in self:
            yield Sample(
                input=sample.input if "input" in want else None,
                target=sample.target if "target" in want else None,
                metadata=sample.meta if "metadata" in want else {},
            )
