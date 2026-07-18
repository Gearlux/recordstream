"""Op-kind introspection — what carrier an op accepts/produces, detected from its annotations.

The native multi-type engine (``Flux(native=True)``) lets carriers other than
:class:`~sampleflux.sample.Sample` flow through a pipeline — metadata-free **pairs** like
``(image, label)`` / ``(tensor, mask)`` / ``(tensor, coco_dict)``, or bare **values**.
Ops can process everything: the engine detects each op's contract by INTROSPECTING the
``__call__`` type annotations (``__call__(self, sample: Sample)`` vs
``__call__(self, pair: tuple[np.ndarray, int])`` vs untyped = works-on-anything) and
adapts the carrier per op. Explicit class attributes (``SAMPLE_KIND_IN`` /
``SAMPLE_KIND_OUT`` / ``EXPANDS``) override detection for cases introspection can't see
(C-extension callables, wrappers around raw functions).

The same introspection powers 1→N detection: a ``-> Iterator[Sample]`` /
``-> Iterable[Sample]`` return annotation (or ``EXPANDS = True``) marks an EXPANDING op —
one carrier in, several out — which makes the pipeline iterable-only (see
``Flux.__len__``/``__getitem__``).
"""

import collections.abc
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple, Union, get_args, get_origin, get_type_hints

from sampleflux.sample import Sample

SampleKind = Literal["sample", "pair", "value", "any"]
"""The carrier taxonomy: a full Sample triplet, a metadata-free 2-tuple, a bare value, or anything."""

SAMPLE_KINDS: Tuple[str, ...] = get_args(SampleKind)

__all__ = ["OpContract", "SAMPLE_KINDS", "SampleKind", "classify_carrier", "op_contract"]

_EXPANDING_ORIGINS = (
    list,
    set,
    frozenset,
    collections.abc.Iterable,
    collections.abc.Iterator,
    collections.abc.Generator,
    collections.abc.Sequence,
)


@dataclass(frozen=True)
class OpContract:
    """What an op consumes and produces.

    ``accepts``/``produces`` are :data:`SampleKind` members; ``expands`` marks a 1→N op
    (returns an iterable of carriers instead of one).
    """

    accepts: SampleKind = "any"
    produces: SampleKind = "any"
    expands: bool = False


_ANY_CONTRACT = OpContract()
_contract_cache: Dict[type, OpContract] = {}


def classify_carrier(obj: Any) -> SampleKind:
    """The carrier kind of a runtime object: Sample -> ``sample``, 2-tuple -> ``pair``, else ``value``."""
    if isinstance(obj, Sample):
        return "sample"
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


def _kind_of(anno: Any) -> SampleKind:
    """The carrier kind an annotation names; unknown/absent/Any -> ``any``."""
    anno = _unwrap_optional(anno)
    if anno is inspect.Parameter.empty or anno is Any or anno is None:
        return "any"
    if anno is Sample:
        return "sample"
    if anno is tuple or get_origin(anno) is tuple:
        return "pair"
    if isinstance(anno, type) and issubclass(anno, Sample):
        return "sample"
    return "any"


def _return_contract(anno: Any) -> Tuple[SampleKind, bool]:
    """(produced kind, expands) from a return annotation."""
    anno = _unwrap_optional(anno)
    origin = get_origin(anno)
    if origin in _EXPANDING_ORIGINS or (isinstance(anno, type) and anno in _EXPANDING_ORIGINS):
        args = get_args(anno)
        element = args[0] if args else Any
        return _kind_of(element), True
    return _kind_of(anno), False


def op_contract(op: Any) -> OpContract:
    """The introspected (cached per type) carrier contract of an op.

    Explicit class attributes win: ``SAMPLE_KIND_IN`` / ``SAMPLE_KIND_OUT`` (a
    :data:`SampleKind` string) and ``EXPANDS`` (bool) override whatever the annotations
    say — the escape hatch for callables introspection can't read. Annotation resolution
    failures (lazy imports, unresolvable forward refs) degrade to ``any`` so an untyped or
    exotic op behaves exactly as today.
    """
    cls = type(op)
    cached = _contract_cache.get(cls)
    if cached is not None:
        return _explicit_overrides(op, cached)

    accepts: SampleKind = "any"
    produces: SampleKind = "any"
    expands = False
    call = getattr(cls, "__call__", None)
    if call is not None:
        try:
            signature = inspect.signature(call)
            hints = get_type_hints(call)
        except Exception:  # noqa: BLE001 - degrade to "any" on ANY introspection failure
            signature, hints = None, {}
        if signature is not None:
            params = [p for name, p in signature.parameters.items() if name != "self"]
            if params:
                first = params[0]
                accepts = _kind_of(hints.get(first.name, first.annotation))
            produces, expands = _return_contract(hints.get("return", inspect.Parameter.empty))

    contract = OpContract(accepts=accepts, produces=produces, expands=expands)
    _contract_cache[cls] = contract
    return _explicit_overrides(op, contract)


def _explicit_overrides(op: Any, base: OpContract) -> OpContract:
    """Apply the ``SAMPLE_KIND_IN``/``SAMPLE_KIND_OUT``/``EXPANDS`` class-attr escape hatches."""
    kind_in = getattr(op, "SAMPLE_KIND_IN", None)
    kind_out = getattr(op, "SAMPLE_KIND_OUT", None)
    expands = getattr(op, "EXPANDS", None)
    if kind_in is None and kind_out is None and expands is None:
        return base
    return OpContract(
        accepts=kind_in if kind_in in SAMPLE_KINDS else base.accepts,
        produces=kind_out if kind_out in SAMPLE_KINDS else base.produces,
        expands=bool(expands) if expands is not None else base.expands,
    )
