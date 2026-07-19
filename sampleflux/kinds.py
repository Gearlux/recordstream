"""Op-kind introspection — WHAT a transform processes and HOW it wants to be called.

The taxonomy is a grid over two axes, detected from ``__call__``'s signature so ops stay
plain callables (no base classes) and a visual editor can surface the names later:

**Field scope** (:data:`SampleKind`) — which part of the ``Sample(input, target,
metadata)`` triple the transform processes:

========================  ==========================  ============================
scope                     without metadata            with metadata
========================  ==========================  ============================
input only                ``input`` (bare value)      ``input_meta`` (`InputMeta`)
target only               ``target`` (bare value)     ``target_meta`` (`TargetMeta`)
both                      ``pair`` (`(input,target)`)  ``sample`` (the full triple)
========================  ==========================  ============================

plus ``value`` (a bare carrier of unknown role, runtime classification only) and ``any``
(untyped — receives whatever flows, exactly today's behavior).

**Call style** (:data:`CallStyle`) — packed (ONE argument: the ``Sample`` / a tuple / a
view) or unpacked (the fields as SEPARATE arguments):

- ``__call__(self, sample: Sample)``                  → sample, packed
- ``__call__(self, input, target, metadata)``         → sample, unpacked (3 required args)
- ``__call__(self, pair: tuple)`` / ``(p: Pair)``     → pair, packed
- ``__call__(self, input, target)``                   → pair, unpacked (2 required args)
- ``__call__(self, v: InputMeta)``                    → input_meta, packed
- ``__call__(self, input, metadata)``                 → input_meta, unpacked (2nd arg named ``metadata``/``meta``)
- ``__call__(self, target, metadata)``                → target_meta, unpacked (1st arg named ``target``)
- ``__call__(self, x: Input)``                        → input, bare value (`Input`/`Target` Annotated aliases,
  or mark your own type: ``Annotated[np.ndarray, INPUT]``)
- untyped single argument                             → any (unchanged)

The ENGINE binds the declared view from whatever carrier flows and merges the result
back, preserving untouched fields (see ``core._apply_op``). Arity counts REQUIRED
parameters only, so an existing op with optional extras keeps today's behavior. Explicit
class attributes (``SAMPLE_KIND_IN`` / ``SAMPLE_KIND_OUT`` / ``EXPANDS`` /
``CALL_STYLE``) override detection for callables introspection can't read.

The same introspection powers 1→N detection: a ``-> Iterator[Sample]`` /
``-> Iterable[Sample]`` return annotation (or ``EXPANDS = True``) marks an EXPANDING op,
which makes the pipeline iterable-only (see ``Flux.__len__``/``__getitem__``).
"""

import collections.abc
import inspect
from dataclasses import dataclass
from typing import Annotated, Any, Dict, Literal, Tuple, Union, get_args, get_origin, get_type_hints

from sampleflux.sample import InputMeta, Pair, Sample, TargetMeta

SampleKind = Literal["sample", "pair", "input", "target", "metadata", "input_meta", "target_meta", "value", "any"]
"""The field-scope taxonomy — see the module docstring grid."""

CallStyle = Literal["packed", "unpacked"]
"""How the op wants its view: one packed argument, or the fields as separate arguments."""

SAMPLE_KINDS: Tuple[str, ...] = get_args(SampleKind)
CALL_STYLES: Tuple[str, ...] = get_args(CallStyle)

_META_PARAM_NAMES = frozenset({"metadata", "meta"})
_TARGET_PARAM_NAMES = frozenset({"target"})

__all__ = [
    "CALL_STYLES",
    "CallStyle",
    "INPUT",
    "Input",
    "METADATA",
    "MetaDict",
    "OpContract",
    "SAMPLE_KINDS",
    "SampleKind",
    "TARGET",
    "Target",
    "classify_carrier",
    "op_contract",
]

_EXPANDING_ORIGINS = (
    list,
    set,
    frozenset,
    collections.abc.Iterable,
    collections.abc.Iterator,
    collections.abc.Generator,
    collections.abc.Sequence,
)


class _KindMark:
    """PEP-593 marker naming the field a bare-value annotation binds (``Annotated[T, INPUT]``)."""

    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"KindMark({self.kind})"


INPUT = _KindMark("input")
TARGET = _KindMark("target")
METADATA = _KindMark("metadata")

Input = Annotated[Any, INPUT]
"""Annotation alias: the op processes the BARE input value (``Annotated[T, INPUT]`` keeps a real T)."""

Target = Annotated[Any, TARGET]
"""Annotation alias: the op processes the BARE target value (``Annotated[T, TARGET]`` keeps a real T)."""

MetaDict = Annotated[Any, METADATA]
"""Annotation alias: the op processes the metadata DICT (a plain ``dict`` annotation works too)."""


@dataclass(frozen=True)
class OpContract:
    """What an op consumes/produces, how it is called, and whether it expands 1→N.

    For an UNPACKED op, ``bindings`` lists each required parameter's field scope in
    order (e.g. ``("input_meta", "target_meta")`` for ``f(im: InputMeta, tm: TargetMeta)``)
    — the engine binds each argument independently and merges each returned element back.
    ``accepts`` stays the grid SUMMARY of the covered fields (what a visual editor surfaces).
    """

    accepts: SampleKind = "any"
    produces: SampleKind = "any"
    expands: bool = False
    style: CallStyle = "packed"
    bindings: Tuple[str, ...] = ()


_ANY_CONTRACT = OpContract()
_contract_cache: Dict[type, OpContract] = {}


def classify_carrier(obj: Any) -> SampleKind:
    """The carrier kind of a runtime object.

    The named views are checked BEFORE the generic tuple rule — an ``InputMeta`` IS a
    2-tuple and would otherwise misclassify as a pair.
    """
    if isinstance(obj, Sample):
        return "sample"
    if isinstance(obj, InputMeta):
        return "input_meta"
    if isinstance(obj, TargetMeta):
        return "target_meta"
    if isinstance(obj, Pair):
        return "pair"
    if isinstance(obj, tuple) and len(obj) == 2:
        return "pair"
    return "value"


def _unwrap_optional(anno: Any) -> Any:
    """``Optional[X]`` / ``Union[X, None]`` -> ``X`` (multi-arm unions are left as-is)."""
    if get_origin(anno) is Union:
        args = [a for a in get_args(anno) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return anno


def _kind_mark_of(anno: Any) -> Any:
    """The ``_KindMark`` on an ``Annotated[...]`` layer, or None."""
    if get_origin(anno) is Annotated:
        for meta in get_args(anno)[1:]:
            if isinstance(meta, _KindMark):
                return meta
    return None


def _kind_of(anno: Any) -> SampleKind:
    """The field scope an annotation names; unknown/absent/Any -> ``any``."""
    anno = _unwrap_optional(anno)
    mark = _kind_mark_of(anno)
    if mark is not None:
        return mark.kind  # type: ignore[no-any-return]
    if get_origin(anno) is Annotated:
        return _kind_of(get_args(anno)[0])
    if anno is inspect.Parameter.empty or anno is Any or anno is None:
        return "any"
    if anno is Sample:
        return "sample"
    if anno is InputMeta:
        return "input_meta"
    if anno is TargetMeta:
        return "target_meta"
    if anno is Pair:
        return "pair"
    if anno is tuple or get_origin(anno) is tuple:
        return "pair"
    if anno is dict or get_origin(anno) is dict:
        return "metadata"
    if isinstance(anno, type) and issubclass(anno, Sample):
        return "sample"
    return "any"


def _is_meta_annotation(anno: Any) -> bool:
    """True when an annotation names a metadata dict (``Dict[str, ...]`` / ``dict``)."""
    anno = _unwrap_optional(anno)
    return anno is dict or get_origin(anno) is dict


def _return_contract(anno: Any) -> Tuple[SampleKind, bool]:
    """(produced kind, expands) from a return annotation."""
    anno = _unwrap_optional(anno)
    origin = get_origin(anno)
    if origin in _EXPANDING_ORIGINS or (isinstance(anno, type) and anno in _EXPANDING_ORIGINS):
        args = get_args(anno)
        element = args[0] if args else Any
        return _kind_of(element), True
    return _kind_of(anno), False


# Per-parameter binding vocabulary: which field scope one argument of an UNPACKED op binds.
_PARAM_BINDINGS = ("input", "target", "metadata", "input_meta", "target_meta")
# Positional defaults — the classic AI convention: f(input, target[, metadata]).
_POSITIONAL_DEFAULTS = ("input", "target", "metadata")
_INPUT_PARAM_NAMES = frozenset({"input"})

# Which fields each binding covers, for the grid summary.
_BINDING_FIELDS: Dict[str, frozenset] = {
    "input": frozenset({"i"}),
    "target": frozenset({"t"}),
    "metadata": frozenset({"m"}),
    "input_meta": frozenset({"i", "m"}),
    "target_meta": frozenset({"t", "m"}),
}
_FIELDS_TO_KIND: Dict[frozenset, str] = {
    frozenset({"i", "t", "m"}): "sample",
    frozenset({"i", "t"}): "pair",
    frozenset({"i", "m"}): "input_meta",
    frozenset({"t", "m"}): "target_meta",
    frozenset({"i"}): "input",
    frozenset({"t"}): "target",
    frozenset({"m"}): "metadata",
}


def _param_binding(param: Any, anno: Any, position: int) -> str:
    """One required parameter's field binding: annotation wins, then the name, then position.

    Positional defaults are the classic ``f(input, target, metadata)`` convention, so an
    unannotated/unnamed multi-arg op keeps the old behavior; a view annotation
    (``InputMeta``/``TargetMeta``/``Input``/``Target``/``dict``) or a recognised name
    (``input``/``target``/``metadata``/``meta``) overrides its slot.
    """
    kind = _kind_of(anno)
    if kind in _PARAM_BINDINGS:
        return kind
    if param.name in _INPUT_PARAM_NAMES:
        return "input"
    if param.name in _TARGET_PARAM_NAMES:
        return "target"
    if param.name in _META_PARAM_NAMES:
        return "metadata"
    return _POSITIONAL_DEFAULTS[position]


def _bindings_summary(bindings: Tuple[str, ...]) -> SampleKind:
    """The grid-summary kind of a binding list (the covered fields)."""
    covered: frozenset = frozenset().union(*(_BINDING_FIELDS[b] for b in bindings))
    return _FIELDS_TO_KIND.get(covered, "sample")  # type: ignore[return-value]


def op_contract(op: Any) -> OpContract:
    """The introspected (cached per type) contract of an op — scope, call style, expansion.

    Explicit class attributes win: ``SAMPLE_KIND_IN`` / ``SAMPLE_KIND_OUT`` (a
    :data:`SampleKind`), ``CALL_STYLE`` (a :data:`CallStyle`), and ``EXPANDS`` (bool) —
    the escape hatch for callables introspection can't read. Annotation resolution
    failures degrade to ``any``/packed so an untyped or exotic op behaves exactly as
    today. Arity counts REQUIRED parameters (no default) only, so an op with optional
    extras after its sample argument keeps single-argument semantics.
    """
    cls = type(op)
    cached = _contract_cache.get(cls)
    if cached is not None:
        return _explicit_overrides(op, cached)

    accepts: SampleKind = "any"
    produces: SampleKind = "any"
    expands = False
    style: CallStyle = "packed"
    bindings: Tuple[str, ...] = ()
    call = getattr(cls, "__call__", None)
    if call is not None:
        try:
            signature = inspect.signature(call)
            hints = get_type_hints(call, include_extras=True)
        except Exception:  # noqa: BLE001 - degrade to "any" on ANY introspection failure
            signature, hints = None, {}
        if signature is not None:
            params = [
                p
                for name, p in signature.parameters.items()
                if name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            required = [p for p in params if p.default is inspect.Parameter.empty]
            arity = len(required) if required else min(len(params), 1)
            if arity == 1 and params:
                accepts = _kind_of(hints.get(params[0].name, params[0].annotation))
            elif 2 <= arity <= 3:
                bindings = tuple(_param_binding(p, hints.get(p.name, p.annotation), i) for i, p in enumerate(required))
                accepts, style = _bindings_summary(bindings), "unpacked"
            # arity 0 or > 3: leave "any"/packed — the op is called with the carrier verbatim.
            produces, expands = _return_contract(hints.get("return", inspect.Parameter.empty))

    bindings = bindings if style == "unpacked" else ()
    contract = OpContract(accepts=accepts, produces=produces, expands=expands, style=style, bindings=bindings)
    _contract_cache[cls] = contract
    return _explicit_overrides(op, contract)


def _explicit_overrides(op: Any, base: OpContract) -> OpContract:
    """Apply the ``SAMPLE_KIND_IN``/``SAMPLE_KIND_OUT``/``EXPANDS``/``CALL_STYLE`` escape hatches."""
    kind_in = getattr(op, "SAMPLE_KIND_IN", None)
    kind_out = getattr(op, "SAMPLE_KIND_OUT", None)
    expands = getattr(op, "EXPANDS", None)
    call_style = getattr(op, "CALL_STYLE", None)
    if kind_in is None and kind_out is None and expands is None and call_style is None:
        return base
    return OpContract(
        accepts=kind_in if kind_in in SAMPLE_KINDS else base.accepts,
        produces=kind_out if kind_out in SAMPLE_KINDS else base.produces,
        expands=bool(expands) if expands is not None else base.expands,
        style=call_style if call_style in CALL_STYLES else base.style,
        bindings=base.bindings,
    )
