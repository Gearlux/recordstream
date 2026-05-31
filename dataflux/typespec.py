"""Type-spec system: describe and match the types flowing through a :class:`~dataflux.sample.Sample`.

Two gaps this fills:

1. A ``Sample`` carries no description of *what kind of data* sits in its ``input`` / ``target``.
2. Ops and sources don't declare which input/target types they accept or produce.

The model is a small set of frozen value objects (no base class — see DataFlux "Functional Purity"
mandate; these are values, not data ops) describing one slot:

* :class:`AnyType` — matches everything (the default when nothing is declared).
* :class:`ArrayType` — an N-D array/tensor across frameworks (numpy / torch / tensorflow): optional
  rank, per-axis :class:`Dim` constraints (exact / bounded-range / unbounded), optional dtype
  (concrete ``"float32"`` or a *family* ``"floating"`` / ``"integer"`` / ``"numeric"`` …), optional
  framework set, optional ``semantic`` tag. ``ArrayType.image(...)`` is a convenience for images.
* :class:`PythonType` — a non-array Python value by qualname (``"PIL.Image.Image"``, ``"dict"`` …).
* :class:`UnionType` — any-of.
* :class:`MappingType` / :class:`ListType` — mirror ``datasets`` struct / ``Sequence`` so structured
  targets (e.g. detection ``{boxes, labels}``) get a real type and bridge 1:1 to HF ``Features``.
* :class:`SampleType` — the ``(input, target)`` pair an op/source declares or a sample reports.

**Matching** is asymmetric (covariant): ``consumer.accepts(producer)`` is True iff every concrete
value the producer can emit is acceptable to the consumer. Two flavours share the leaf logic:

* ``accepts`` — *strict*; used by the runtime check where the producer is a concrete inferred type.
* ``compatible`` — *permissive*; used at edit-time (FluxStudio canvas) and for discovery filtering:
  ``Any``/unknown on **either** side ⇒ compatible (honours "if not defined, assume Any"), and an
  unbounded producer axis against a bounded consumer axis is a soft-pass (the runtime check still
  catches an actual out-of-range value).

Prior art borrowed rather than reinvented: the ``None``-dim base case mirrors ``tf.TensorShape``
(``Dim`` adds the bounded-range extension tf/HF/jaxtyping lack); dtype family names align with
``numpy.isdtype`` / the array-api taxonomy; ``ArrayType.parse`` accepts a jaxtyping-style shape
string; and :meth:`SampleType.from_hf_features` / :meth:`SampleType.to_hf_features` bridge to the
``datasets.Features`` we already depend on (used for the concrete per-sample stored type).

Everything is JSON round-trippable (``to_dict`` / :func:`type_from_dict` / :func:`sampletype_from_dict`)
so specs ride the discovery manifest and can be re-implemented by FluxStudio's JS connection-validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataflux.sample import Sample

# A single-slot type spec. Defined as a forward-ref union so leaf classes can annotate it before the
# alias is bound at runtime (annotations are strings under ``from __future__ import annotations``).
TypeSpec = Union["AnyType", "ArrayType", "PythonType", "UnionType", "MappingType", "ListType"]

# Closed enumerations for the small, fixed string sets the type system uses — declared as ``Literal``
# rather than bare ``str`` so authors get a typo-checked value and UIs / the FluxStudio connection-
# validator enumerate the choices straight from the annotation (``typing.get_args(...)``); the
# workspace "prefer closed ``Literal``s over bare strings" mandate. Both are *deliberately closed* —
# extend the Literal when adding real support (e.g. a ``"jax"`` framework), don't widen to ``str``.
# (The dtype enumerations — ``Dtype`` / ``DtypeFamily`` / ``DtypeSpec`` — are defined next to
# ``_DTYPE_FAMILIES`` below, since they share that block's set membership as their source of truth.)
Framework = Literal["numpy", "torch", "tensorflow"]
ImageLayout = Literal["CHW", "HWC"]

C = TypeVar("C")

__all__ = [
    "Dim",
    "AnyType",
    "ArrayType",
    "PythonType",
    "UnionType",
    "MappingType",
    "ListType",
    "SampleType",
    "TypeSpec",
    "Framework",
    "ImageLayout",
    "Dtype",
    "DtypeFamily",
    "DtypeSpec",
    "typed",
    "accepts",
    "compatible",
    "infer_type",
    "infer_sample_type",
    "canonical_dtype",
    "type_to_dict",
    "type_from_dict",
    "sampletype_from_dict",
]


# --------------------------------------------------------------------------------------------------
# dtype canonicalization + families (array-api-aligned names)
# --------------------------------------------------------------------------------------------------

_DTYPE_ALIASES: Dict[str, str] = {
    "double": "float64",
    "single": "float32",
    "half": "float16",
    "bool_": "bool",
    "boolean": "bool",
}

_FLOATING = {"float16", "bfloat16", "float32", "float64"}
_INTEGER = {"int8", "int16", "int32", "int64"}
_UNSIGNED = {"uint8", "uint16", "uint32", "uint64"}
_COMPLEX = {"complex64", "complex128"}
_DTYPE_FAMILIES: Dict[str, FrozenSet[str]] = {
    "floating": frozenset(_FLOATING),
    "integer": frozenset(_INTEGER),
    "unsigned": frozenset(_UNSIGNED),
    "bool": frozenset({"bool"}),
    "complex": frozenset(_COMPLEX),
    "numeric": frozenset(_FLOATING | _INTEGER | _UNSIGNED),
}

#: A concrete dtype name — a closed ``Literal`` (not bare ``str``) so an authored ``ACCEPTS`` /
#: ``PRODUCES`` dtype is typo-checked and UIs / the FluxStudio connection-validator enumerate the
#: choices via ``typing.get_args(Dtype)``. These ARE the union of the family members above (pinned
#: equal in ``tests/test_typespec.py`` so the two can't drift). Authoring uses canonical lowercase
#: names; aliases / casing (``"double"``, ``"FLOAT32"``) and genuinely exotic, platform-dependent
#: dtypes (``float128``, ``complex256``) are *runtime-only* — they reach the field via
#: :func:`canonical_dtype`, the single boundary that normalizes arbitrary input into this domain, and
#: an unmodeled one keeps its own name and simply matches no family.
Dtype = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
#: A relaxed dtype *family* constraint (matches any concrete member). The names are the keys of
#: ``_DTYPE_FAMILIES`` (pinned equal in tests); kept as a ``Literal`` for the same author/UI reasons.
DtypeFamily = Literal["floating", "integer", "unsigned", "bool", "complex", "numeric"]
#: What :attr:`ArrayType.dtype` accepts: a concrete :data:`Dtype` or a relaxed :data:`DtypeFamily`.
DtypeSpec = Union[Dtype, DtypeFamily]


def canonical_dtype(x: Any) -> DtypeSpec:
    """Normalize a dtype (str, numpy dtype/scalar-type, or torch dtype) to a canonical lowercase name.

    Family names (``"floating"``, ``"integer"``, ``"numeric"`` …) pass through unchanged so they can
    be used as relaxed dtype constraints on an :class:`ArrayType`. This is the single boundary where
    arbitrary input (aliases, casing, framework dtype objects, exotic dtypes) crosses into the typed
    :data:`DtypeSpec` domain — hence the closing ``cast``: a genuinely unmodeled dtype keeps its own
    name (and simply matches no family), which is correct even though it lies outside the Literal.
    """
    if isinstance(x, str):
        s = x.lower()
        name = _DTYPE_ALIASES.get(s, s)
    elif isinstance(x, np.dtype):
        name = str(x.name)
    elif isinstance(x, type) and issubclass(x, np.generic):
        name = str(np.dtype(x).name)
    else:
        # Default for any non-numpy value; refined to the bare name for a torch dtype (lazy import —
        # torch is a hard dep but we avoid importing it at module load).
        name = str(x).lower()
        try:
            import torch

            if isinstance(x, torch.dtype):
                name = str(x).replace("torch.", "")
        except ImportError:  # pragma: no cover - torch is a hard dep
            pass
    return cast(DtypeSpec, name)


def _dtype_accepts(consumer: str, producer: str) -> bool:
    """True iff a ``producer`` concrete/family dtype satisfies a ``consumer`` dtype constraint."""
    c = canonical_dtype(consumer)
    p = canonical_dtype(producer)
    if c == p:
        return True
    fam = _DTYPE_FAMILIES.get(c)
    if fam is not None:
        # producer may itself be a (sub)family name or a concrete member
        if p in fam:
            return True
        pfam = _DTYPE_FAMILIES.get(p)
        return pfam is not None and pfam <= fam
    return False


# --------------------------------------------------------------------------------------------------
# Spec value objects
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dim:
    """A single axis-size constraint: the closed interval ``[min, max]`` (``None`` = unbounded).

    ``name`` is informational only (never matched). ``Dim.any()`` is exactly tf's ``None`` dim.
    """

    min: Optional[int] = None
    max: Optional[int] = None
    name: Optional[str] = None

    @classmethod
    def exact(cls, n: int, name: Optional[str] = None) -> "Dim":
        return cls(n, n, name)

    @classmethod
    def any(cls, name: Optional[str] = None) -> "Dim":
        return cls(None, None, name)

    @classmethod
    def range(cls, lo: Optional[int], hi: Optional[int], name: Optional[str] = None) -> "Dim":
        return cls(lo, hi, name)

    def accepts(self, other: "Dim") -> bool:
        """Strict: ``other``'s whole possible interval lies within this interval."""
        lo = self.min if self.min is not None else 0
        o_lo = other.min if other.min is not None else 0
        if o_lo < lo:
            return False
        if self.max is not None and (other.max is None or other.max > self.max):
            return False
        return True

    def compatible(self, other: "Dim") -> bool:
        """Permissive: the two intervals could overlap (an unbounded ``other`` is a soft-pass)."""
        c_lo = self.min if self.min is not None else 0
        o_lo = other.min if other.min is not None else 0
        if self.max is not None and o_lo > self.max:
            return False
        if other.max is not None and other.max < c_lo:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"min": self.min, "max": self.max, "name": self.name}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Dim":
        return cls(d.get("min"), d.get("max"), d.get("name"))


@dataclass(frozen=True)
class AnyType:
    """Matches everything; the default when an op/source/sample declares no type."""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "any"}


@dataclass(frozen=True)
class ArrayType:
    """An N-D array/tensor constraint across numpy / torch / tensorflow.

    All fields are optional and each ``None`` means "unconstrained":

    * ``ndim`` — required rank (derived from ``shape`` when that is given).
    * ``shape`` — per-axis :class:`Dim` tuple.
    * ``dtype`` — a :data:`DtypeSpec`: a concrete :data:`Dtype` (``"float32"``) or a relaxed
      :data:`DtypeFamily` (``"floating"``/``"numeric"`` …). Stored canonicalized via
      :func:`canonical_dtype`, which also accepts aliases / casing / framework dtype objects at runtime.
    * ``frameworks`` — allowed framework set, each a :data:`Framework` (``{"numpy", "torch"}`` …).
    * ``semantic`` — free-form tag (e.g. ``"image"``); display/inference hint, never matched.
    """

    ndim: Optional[int] = None
    shape: Optional[Tuple[Dim, ...]] = None
    dtype: Optional[DtypeSpec] = None
    # Accept any set on construction (ergonomic ``frameworks={"torch"}``); ``__post_init__`` freezes it.
    frameworks: Optional[AbstractSet[Framework]] = None
    semantic: Optional[str] = None

    def __post_init__(self) -> None:
        if self.shape is not None:
            shape = tuple(self.shape)
            object.__setattr__(self, "shape", shape)
            if self.ndim is None:
                object.__setattr__(self, "ndim", len(shape))
            elif self.ndim != len(shape):
                raise ValueError(f"ArrayType: ndim={self.ndim} disagrees with shape length {len(shape)}")
        if self.frameworks is not None and not isinstance(self.frameworks, frozenset):
            object.__setattr__(self, "frameworks", frozenset(self.frameworks))
        if self.dtype is not None:
            object.__setattr__(self, "dtype", canonical_dtype(self.dtype))

    @classmethod
    def image(
        cls,
        layout: ImageLayout = "CHW",
        channels: Union[int, Tuple[int, ...]] = 3,
        dtype: Optional[DtypeSpec] = None,
        framework: Optional[Framework] = None,
    ) -> "ArrayType":
        """Rank-3 image convenience. ``channels`` as a tuple is treated as the inclusive range
        ``[min, max]`` (a pragmatic approximation — pass an exact :class:`Dim` via the constructor
        for anything finer)."""
        if isinstance(channels, int):
            cdim = Dim.exact(channels, "C")
        else:
            cdim = Dim(min(channels), max(channels), "C")
        h, w = Dim.any("H"), Dim.any("W")
        up = layout.upper()
        if up == "CHW":
            shape: Tuple[Dim, ...] = (cdim, h, w)
        elif up == "HWC":
            shape = (h, w, cdim)
        else:
            raise ValueError(f"ArrayType.image: layout must be 'CHW' or 'HWC', got {layout!r}")
        fws = frozenset({framework}) if framework else None
        return cls(ndim=3, shape=shape, dtype=dtype, frameworks=fws, semantic="image")

    @classmethod
    def parse(
        cls,
        spec: str,
        dtype: Optional[DtypeSpec] = None,
        framework: Optional[Framework] = None,
        semantic: Optional[str] = None,
    ) -> "ArrayType":
        """Build from a jaxtyping-style shape string: space-separated axes where a bare int is an
        exact size, ``lo-hi`` a bounded range (open ends allowed, e.g. ``"1-"``), and any other token
        a named free axis. Example: ``ArrayType.parse("3 h w", dtype="float32")``.
        """
        dims: List[Dim] = []
        for tok in spec.split():
            if tok.startswith("*"):
                raise ValueError("ArrayType.parse: variadic '*' axes are not supported in v1")
            if tok.isdigit():
                dims.append(Dim.exact(int(tok)))
            elif "-" in tok[1:] and all(part == "" or part.isdigit() for part in tok.split("-", 1)):
                lo_s, hi_s = tok.split("-", 1)
                dims.append(Dim.range(int(lo_s) if lo_s else None, int(hi_s) if hi_s else None))
            else:
                dims.append(Dim.any(tok))
        fws = frozenset({framework}) if framework else None
        return cls(ndim=len(dims), shape=tuple(dims), dtype=dtype, frameworks=fws, semantic=semantic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "array",
            "ndim": self.ndim,
            "shape": [d.to_dict() for d in self.shape] if self.shape is not None else None,
            "dtype": self.dtype,
            "frameworks": sorted(self.frameworks) if self.frameworks is not None else None,
            "semantic": self.semantic,
        }


@dataclass(frozen=True)
class PythonType:
    """A non-array Python value, matched by exact qualname (e.g. ``"PIL.Image.Image"``, ``"dict"``)."""

    qualname: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "python", "qualname": self.qualname}


@dataclass(frozen=True)
class UnionType:
    """Any-of. Stored as a tuple so the value object stays hashable/immutable."""

    members: Tuple[TypeSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "union", "members": [type_to_dict(m) for m in self.members]}


@dataclass(frozen=True)
class ListType:
    """A homogeneous sequence (mirrors ``datasets.Sequence`` / ``List``)."""

    item: TypeSpec

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "list", "item": type_to_dict(self.item)}


@dataclass(frozen=True)
class MappingType:
    """A struct with named fields (mirrors a nested ``datasets.Features`` / Python ``dict`` schema).

    Fields are stored as a sorted tuple of ``(name, spec)`` pairs so the object stays hashable.
    Build via :meth:`of`.
    """

    fields: Tuple[Tuple[str, TypeSpec], ...]

    def __post_init__(self) -> None:
        items = self.fields.items() if isinstance(self.fields, dict) else self.fields
        object.__setattr__(self, "fields", tuple(sorted(items, key=lambda kv: kv[0])))

    @classmethod
    def of(cls, fields: Dict[str, TypeSpec]) -> "MappingType":
        return cls(tuple(sorted(fields.items(), key=lambda kv: kv[0])))

    def field_map(self) -> Dict[str, TypeSpec]:
        return dict(self.fields)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "mapping", "fields": {k: type_to_dict(v) for k, v in self.fields}}


@dataclass(frozen=True)
class SampleType:
    """The ``(input, target)`` type pair an op/source declares or a sample reports."""

    input: TypeSpec = field(default_factory=AnyType)
    target: TypeSpec = field(default_factory=AnyType)

    def accepts(self, producer: "SampleType") -> bool:
        """Strict: this consumer accepts every value ``producer`` can emit (both slots)."""
        return _accepts(self.input, producer.input, permissive=False) and _accepts(
            self.target, producer.target, permissive=False
        )

    def compatible(self, producer: "SampleType") -> bool:
        """Permissive: edit-time/discovery wiring — ``Any``/unknown on either side passes."""
        return _accepts(self.input, producer.input, permissive=True) and _accepts(
            self.target, producer.target, permissive=True
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "sample", "input": type_to_dict(self.input), "target": type_to_dict(self.target)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SampleType":
        return cls(type_from_dict(d["input"]), type_from_dict(d["target"]))

    # -- datasets.Features bridge (the concrete per-sample stored representation) ------------------

    @classmethod
    def from_hf_features(cls, features: Any, extras: Optional[Dict[str, Any]] = None) -> "SampleType":
        """Build from a ``datasets.Features`` (or its ``.to_dict()`` form) keyed by ``"input"`` /
        ``"target"``, applying the sidecar ``extras`` that Features can't express (framework, ranges,
        or a full ``typespec`` override for Any/Union/family-only slots)."""
        from datasets import Features

        if not isinstance(features, Features):
            features = Features.from_dict(features)
        extras = extras or {}
        return cls(
            input=_slot_from_features(features.get("input"), extras.get("input")),
            target=_slot_from_features(features.get("target"), extras.get("target")),
        )

    def to_hf_features(self) -> Tuple[Any, Dict[str, Any]]:
        """Split into a ``datasets.Features`` (the Feature-expressible part) and an ``extras`` dict
        (framework/range refinements + a ``typespec`` fallback for slots Features can't represent)."""
        from datasets import Features

        feats: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for slot, spec in (("input", self.input), ("target", self.target)):
            feat, slot_extra = _feature_from_typespec(spec)
            if feat is not None:
                feats[slot] = feat
            if slot_extra:
                extras[slot] = slot_extra
        return Features(feats), extras


# --------------------------------------------------------------------------------------------------
# Matching (module-level dispatch — no base class)
# --------------------------------------------------------------------------------------------------


def accepts(consumer: TypeSpec, producer: TypeSpec) -> bool:
    """Strict single-slot match: does ``consumer`` accept every value ``producer`` can emit?"""
    return _accepts(consumer, producer, permissive=False)


def compatible(consumer: TypeSpec, producer: TypeSpec) -> bool:
    """Permissive single-slot match (edit-time / discovery): ``Any``/unknown on either side passes."""
    return _accepts(consumer, producer, permissive=True)


def _accepts(consumer: TypeSpec, producer: TypeSpec, *, permissive: bool) -> bool:
    if isinstance(consumer, AnyType):
        return True
    if isinstance(producer, AnyType):
        return permissive
    if isinstance(producer, UnionType):
        return all(_accepts(consumer, m, permissive=permissive) for m in producer.members)
    if isinstance(consumer, UnionType):
        return any(_accepts(m, producer, permissive=permissive) for m in consumer.members)
    if isinstance(consumer, ArrayType) and isinstance(producer, ArrayType):
        return _array_accepts(consumer, producer, permissive)
    if isinstance(consumer, PythonType) and isinstance(producer, PythonType):
        return consumer.qualname == producer.qualname
    if isinstance(consumer, ListType) and isinstance(producer, ListType):
        return _accepts(consumer.item, producer.item, permissive=permissive)
    if isinstance(consumer, MappingType) and isinstance(producer, MappingType):
        pm = producer.field_map()
        for key, cval in consumer.fields:
            if key not in pm or not _accepts(cval, pm[key], permissive=permissive):
                return False
        return True
    return False


def _array_accepts(consumer: ArrayType, producer: ArrayType, permissive: bool) -> bool:
    if consumer.ndim is not None:
        if producer.ndim is None:
            if not permissive:
                return False
        elif producer.ndim != consumer.ndim:
            return False
    if consumer.shape is not None:
        producer_shape = producer.shape
        if producer_shape is None and producer.ndim is not None and producer.ndim == len(consumer.shape):
            # A shapeless producer of matching rank carries no per-axis info — equivalent to all
            # unbounded dims. This keeps matching symmetric across the Features round-trip, where a
            # 1-D ``ArrayType(shape=None)`` stores as a Sequence and reads back as ``shape=(Dim.any(),)``.
            producer_shape = tuple(Dim.any() for _ in consumer.shape)
        if producer_shape is None:
            if not permissive:
                return False
        elif len(producer_shape) != len(consumer.shape):
            return False
        else:
            for cdim, pdim in zip(consumer.shape, producer_shape):
                ok = cdim.compatible(pdim) if permissive else cdim.accepts(pdim)
                if not ok:
                    return False
    if consumer.dtype is not None:
        if producer.dtype is None:
            if not permissive:
                return False
        elif not _dtype_accepts(consumer.dtype, producer.dtype):
            return False
    if consumer.frameworks is not None:
        if producer.frameworks is None:
            if not permissive:
                return False
        elif permissive:
            if not (producer.frameworks & consumer.frameworks):
                return False
        elif not (producer.frameworks <= consumer.frameworks):
            return False
    return True


# --------------------------------------------------------------------------------------------------
# Inference from live values
# --------------------------------------------------------------------------------------------------


def infer_type(value: Any) -> TypeSpec:
    """Infer a concrete :data:`TypeSpec` from a live value. ``None`` ⇒ :class:`AnyType` (so an unset
    target stays permissive). torch/tf/PIL are imported lazily and guarded."""
    if value is None:
        return AnyType()
    if isinstance(value, np.ndarray):
        return ArrayType(
            ndim=value.ndim,
            shape=tuple(Dim.exact(int(s)) for s in value.shape),
            dtype=canonical_dtype(value.dtype),
            frameworks=frozenset({"numpy"}),
        )
    if isinstance(value, np.generic):
        return ArrayType(ndim=0, shape=(), dtype=canonical_dtype(value.dtype), frameworks=frozenset({"numpy"}))
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return ArrayType(
                ndim=value.dim(),
                shape=tuple(Dim.exact(int(s)) for s in tuple(value.shape)),
                dtype=canonical_dtype(value.dtype),
                frameworks=frozenset({"torch"}),
            )
    except ImportError:  # pragma: no cover - torch is a hard dep
        pass
    try:
        import tensorflow as tf

        if isinstance(value, tf.Tensor):  # pragma: no cover - tensorflow is optional / not installed here
            shape = tuple(Dim.exact(int(s)) if s is not None else Dim.any() for s in value.shape)
            return ArrayType(
                ndim=len(shape),
                shape=shape,
                dtype=canonical_dtype(value.dtype.name),
                frameworks=frozenset({"tensorflow"}),
            )
    except ImportError:
        pass
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            return PythonType("PIL.Image.Image")
    except ImportError:  # pragma: no cover - Pillow optional
        pass
    if isinstance(value, bool):
        return PythonType("bool")
    if isinstance(value, int):
        return PythonType("int")
    if isinstance(value, float):
        return PythonType("float")
    if isinstance(value, str):
        return PythonType("str")
    if isinstance(value, dict):
        return PythonType("dict")
    if isinstance(value, list):
        return PythonType("list")
    if isinstance(value, tuple):
        return PythonType("tuple")
    return PythonType(type(value).__qualname__)


def infer_sample_type(sample: "Sample") -> SampleType:
    """Infer the :class:`SampleType` of a live sample (duck-typed: reads ``.input`` / ``.target``)."""
    return SampleType(input=infer_type(sample.input), target=infer_type(sample.target))


# --------------------------------------------------------------------------------------------------
# JSON (de)serialization
# --------------------------------------------------------------------------------------------------


def type_to_dict(t: TypeSpec) -> Dict[str, Any]:
    return t.to_dict()


def type_from_dict(d: Dict[str, Any]) -> TypeSpec:
    kind = d["kind"]
    if kind == "any":
        return AnyType()
    if kind == "python":
        return PythonType(d["qualname"])
    if kind == "array":
        shape_d = d.get("shape")
        shape = tuple(Dim.from_dict(x) for x in shape_d) if shape_d is not None else None
        fw = d.get("frameworks")
        return ArrayType(
            ndim=d.get("ndim"),
            shape=shape,
            dtype=d.get("dtype"),
            frameworks=frozenset(fw) if fw is not None else None,
            semantic=d.get("semantic"),
        )
    if kind == "union":
        return UnionType(tuple(type_from_dict(m) for m in d["members"]))
    if kind == "list":
        return ListType(type_from_dict(d["item"]))
    if kind == "mapping":
        return MappingType.of({k: type_from_dict(v) for k, v in d["fields"].items()})
    raise ValueError(f"type_from_dict: unknown kind {kind!r}")


def sampletype_from_dict(d: Dict[str, Any]) -> SampleType:
    return SampleType.from_dict(d)


# --------------------------------------------------------------------------------------------------
# datasets.Features bridge helpers
# --------------------------------------------------------------------------------------------------


def _slot_from_features(feat: Any, extra: Optional[Dict[str, Any]]) -> TypeSpec:
    """Reconstruct one slot's :data:`TypeSpec` from its Feature plus sidecar ``extras`` (recursively)."""
    extra = extra or {}
    if "typespec" in extra:
        return type_from_dict(extra["typespec"])
    if feat is None:
        return AnyType()
    return _apply_extras(_typespec_from_feature(feat), extra)


def _apply_extras(spec: TypeSpec, extra: Dict[str, Any]) -> TypeSpec:
    """Layer the sidecar refinements Features can't hold back onto a Feature-derived spec, recursing
    into mapping fields / list items so per-field framework/range info survives the round-trip."""
    if not extra:
        return spec
    if "typespec" in extra:
        return type_from_dict(extra["typespec"])
    if isinstance(spec, ArrayType):
        frameworks = frozenset(extra["frameworks"]) if "frameworks" in extra else spec.frameworks
        shape = spec.shape
        bounds = extra.get("shape_bounds")
        if bounds and shape is not None:
            new_shape = list(shape)
            for idx_s, (lo, hi) in bounds.items():
                new_shape[int(idx_s)] = Dim.range(lo, hi, new_shape[int(idx_s)].name)
            shape = tuple(new_shape)
        return ArrayType(
            ndim=spec.ndim,
            shape=shape,
            dtype=spec.dtype,
            frameworks=frameworks,
            semantic=extra.get("semantic", spec.semantic),
        )
    if isinstance(spec, MappingType) and "fields" in extra:
        sub = extra["fields"]
        return MappingType.of({k: _apply_extras(v, sub.get(k, {})) for k, v in spec.field_map().items()})
    if isinstance(spec, ListType) and "item" in extra:
        return ListType(_apply_extras(spec.item, extra["item"]))
    return spec


def _typespec_from_feature(feat: Any) -> TypeSpec:
    from datasets import Features, Image, Value

    if isinstance(feat, (Features, dict)):
        return MappingType.of({k: _typespec_from_feature(v) for k, v in feat.items()})
    inner = feat[0] if isinstance(feat, list) else getattr(feat, "feature", None)
    is_sequence = isinstance(feat, list) or feat.__class__.__name__ in ("Sequence", "List", "LargeList")
    if inner is not None and is_sequence:
        # A sequence of a numeric scalar is canonically a 1-D array (that's how HF stores a [N] tensor);
        # only a sequence of non-scalars (struct / nested array / string) is a true ListType.
        if isinstance(inner, Value) and inner.dtype not in ("string", "large_string"):
            return ArrayType(ndim=1, shape=(Dim.any(),), dtype=canonical_dtype(inner.dtype))
        return ListType(_typespec_from_feature(inner))
    if isinstance(feat, Image):
        return PythonType("PIL.Image.Image")
    shape = getattr(feat, "shape", None)
    dtype = getattr(feat, "dtype", None)
    if shape is not None and dtype is not None:
        dims = tuple(Dim.any() if s is None else Dim.exact(int(s)) for s in shape)
        return ArrayType(ndim=len(dims), shape=dims, dtype=canonical_dtype(dtype))
    if isinstance(feat, Value):
        if feat.dtype in ("string", "large_string"):
            return PythonType("str")
        return ArrayType(ndim=0, dtype=canonical_dtype(feat.dtype))
    return AnyType()


def _feature_from_typespec(spec: TypeSpec) -> Tuple[Any, Dict[str, Any]]:
    """Return ``(feature_or_None, extras)`` for one slot. ``feature`` is ``None`` when the spec can't
    be a concrete ``Features`` entry (Any/Union/family-dtype/unknown-rank/exotic python type)."""
    from datasets import Array2D, Array3D, Array4D, Array5D, Features, Image, Sequence, Value

    if isinstance(spec, PythonType):
        if spec.qualname == "PIL.Image.Image":
            return Image(), {}
        if spec.qualname == "str":
            return Value("string"), {}
        return None, {"typespec": spec.to_dict()}
    if isinstance(spec, ListType):
        inner_feat, inner_extra = _feature_from_typespec(spec.item)
        if inner_feat is None:
            return None, {"typespec": spec.to_dict()}
        return Sequence(inner_feat), ({"item": inner_extra} if inner_extra else {})
    if isinstance(spec, MappingType):
        feats: Dict[str, Any] = {}
        sub: Dict[str, Any] = {}
        for key, val in spec.fields:
            feat, extra = _feature_from_typespec(val)
            if feat is None:
                return None, {"typespec": spec.to_dict()}
            feats[key] = feat
            if extra:
                sub[key] = extra
        return Features(feats), ({"fields": sub} if sub else {})
    if isinstance(spec, ArrayType):
        if spec.ndim is None or spec.dtype is None or spec.dtype in _DTYPE_FAMILIES:
            return None, {"typespec": spec.to_dict()}
        extras: Dict[str, Any] = {}
        if spec.frameworks is not None:
            extras["frameworks"] = sorted(spec.frameworks)
        if spec.semantic is not None:
            extras["semantic"] = spec.semantic
        if spec.ndim == 0:
            return Value(spec.dtype), extras
        if spec.ndim == 1:
            return Sequence(Value(spec.dtype)), extras
        arrcls = {2: Array2D, 3: Array3D, 4: Array4D, 5: Array5D}.get(spec.ndim)
        if arrcls is None:
            return None, {"typespec": spec.to_dict()}
        shape_list: List[Optional[int]] = []
        bounds: Dict[str, List[Optional[int]]] = {}
        if spec.shape is None:
            shape_list = [None] * spec.ndim  # rank known but no per-axis info -> all dynamic
        else:
            for i, d in enumerate(spec.shape):
                if d.min is not None and d.min == d.max:
                    shape_list.append(int(d.min))
                else:
                    shape_list.append(None)
                    if d.min is not None or d.max is not None:
                        bounds[str(i)] = [d.min, d.max]
        if bounds:
            extras["shape_bounds"] = bounds
        return arrcls(shape=tuple(shape_list), dtype=spec.dtype), extras
    # AnyType / UnionType
    return None, {"typespec": spec.to_dict()}


# --------------------------------------------------------------------------------------------------
# Declaration decorator
# --------------------------------------------------------------------------------------------------


def typed(*, accepts: Optional[SampleType] = None, produces: Optional[SampleType] = None) -> Callable[[C], C]:
    """Class decorator that sets ``ACCEPTS`` / ``PRODUCES`` on an op/source (ergonomic alternative to
    plain class attributes). Validates the arguments are :class:`SampleType` at decoration time."""

    def deco(cls: C) -> C:
        if accepts is not None:
            if not isinstance(accepts, SampleType):
                raise TypeError(f"@typed(accepts=...) expects a SampleType, got {type(accepts).__name__}")
            setattr(cls, "ACCEPTS", accepts)
        if produces is not None:
            if not isinstance(produces, SampleType):
                raise TypeError(f"@typed(produces=...) expects a SampleType, got {type(produces).__name__}")
            setattr(cls, "PRODUCES", produces)
        return cls

    return deco
