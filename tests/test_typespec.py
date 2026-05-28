"""Exhaustive tests for the dataflux type-spec system (matching, inference, JSON, HF bridge)."""

from typing import Any, List, Tuple, cast

import numpy as np
import pytest

from dataflux.core import Flux
from dataflux.sample import FEATURES_KEY, SPEC_KEY, Sample
from dataflux.typespec import (
    AnyType,
    ArrayType,
    Dim,
    ListType,
    MappingType,
    PythonType,
    SampleType,
    TypeSpec,
    UnionType,
    accepts,
    canonical_dtype,
    compatible,
    infer_sample_type,
    infer_type,
    type_from_dict,
    typed,
)


class _Widget:
    """Module-level class so its ``__qualname__`` is the bare name (used by infer_type fallback test)."""


# --------------------------------------------------------------------------------------------------
# Dim
# --------------------------------------------------------------------------------------------------


def test_dim_accepts_strict() -> None:
    assert Dim.range(1, 10).accepts(Dim.exact(5))
    assert Dim.range(1, 10).accepts(Dim.range(2, 9))
    assert not Dim.range(1, 10).accepts(Dim.exact(50))
    assert not Dim.range(1, 10).accepts(Dim.exact(0))
    assert Dim.any().accepts(Dim.exact(999))
    assert Dim.any().accepts(Dim.any())
    # unbounded producer against bounded consumer -> strict reject
    assert not Dim.range(1, 10).accepts(Dim.any())
    # half-open consumer bounds
    assert Dim.range(2, None).accepts(Dim.exact(100))
    assert not Dim.range(2, None).accepts(Dim.exact(1))


def test_dim_compatible_permissive() -> None:
    # unbounded producer against bounded consumer -> soft pass
    assert Dim.range(1, 10).compatible(Dim.any())
    assert Dim.range(1, 10).compatible(Dim.exact(5))
    # provably disjoint -> reject even when permissive
    assert not Dim.range(1, 10).compatible(Dim.exact(50))
    assert not Dim.range(20, 30).compatible(Dim.range(1, 5))


def test_dim_json_roundtrip() -> None:
    d = Dim.range(1, 10, "axis")
    assert Dim.from_dict(d.to_dict()) == d


# --------------------------------------------------------------------------------------------------
# AnyType
# --------------------------------------------------------------------------------------------------


def test_any_type_matches_everything() -> None:
    for producer in (PythonType("dict"), ArrayType(ndim=3), UnionType((AnyType(),))):
        assert accepts(AnyType(), producer)
    # concrete consumer rejects an Any producer (strict) but accepts it permissively
    assert not accepts(ArrayType(ndim=2), AnyType())
    assert compatible(ArrayType(ndim=2), AnyType())


# --------------------------------------------------------------------------------------------------
# ArrayType
# --------------------------------------------------------------------------------------------------


def test_array_ndim() -> None:
    assert accepts(ArrayType(ndim=2), ArrayType(ndim=2))
    assert not accepts(ArrayType(ndim=2), ArrayType(ndim=3))
    assert accepts(ArrayType(ndim=None), ArrayType(ndim=5))  # rank-agnostic consumer
    # unknown producer rank: strict reject, permissive pass
    assert not accepts(ArrayType(ndim=2), ArrayType(ndim=None))
    assert compatible(ArrayType(ndim=2), ArrayType(ndim=None))


def test_array_shape() -> None:
    consumer = ArrayType(shape=(Dim.range(1, 10), Dim.any()))
    assert accepts(consumer, ArrayType(shape=(Dim.exact(5), Dim.exact(7))))
    assert not accepts(consumer, ArrayType(shape=(Dim.exact(50), Dim.exact(7))))
    # producer without shape: strict reject, permissive pass
    assert not accepts(consumer, ArrayType(ndim=2))
    assert compatible(consumer, ArrayType(ndim=2))


def test_array_dtype_families() -> None:
    assert accepts(ArrayType(dtype="floating"), ArrayType(dtype="float32"))
    assert accepts(ArrayType(dtype="numeric"), ArrayType(dtype="int64"))
    assert not accepts(ArrayType(dtype="floating"), ArrayType(dtype="int64"))
    assert accepts(ArrayType(dtype="float32"), ArrayType(dtype="float32"))
    assert not accepts(ArrayType(dtype="float32"), ArrayType(dtype="float64"))
    # family-accepts-subfamily: numeric accepts the whole floating family
    assert accepts(ArrayType(dtype="numeric"), ArrayType(dtype="floating"))
    assert not accepts(ArrayType(dtype="floating"), ArrayType(dtype="numeric"))
    # producer dtype unknown
    assert not accepts(ArrayType(dtype="float32"), ArrayType())
    assert compatible(ArrayType(dtype="float32"), ArrayType())


def test_array_frameworks() -> None:
    assert accepts(ArrayType(frameworks={"numpy", "torch"}), ArrayType(frameworks={"torch"}))
    assert not accepts(ArrayType(frameworks={"torch"}), ArrayType(frameworks={"numpy"}))
    # permissive: overlap is enough
    assert compatible(ArrayType(frameworks={"torch"}), ArrayType(frameworks={"torch", "numpy"}))
    assert not compatible(ArrayType(frameworks={"torch"}), ArrayType(frameworks={"numpy"}))
    # producer framework unknown
    assert not accepts(ArrayType(frameworks={"torch"}), ArrayType())
    assert compatible(ArrayType(frameworks={"torch"}), ArrayType())


def test_array_post_init() -> None:
    a = ArrayType(shape=(Dim.exact(3), Dim.any()))
    assert a.ndim == 2  # derived from shape
    assert isinstance(ArrayType(frameworks={"numpy"}).frameworks, frozenset)  # set coerced to frozenset
    assert ArrayType(dtype="FLOAT32").dtype == "float32"  # normalized
    with pytest.raises(ValueError):
        ArrayType(ndim=3, shape=(Dim.any(),))


def test_array_image_constructor() -> None:
    chw = ArrayType.image("CHW", channels=3, dtype="float32", framework="torch")
    assert chw.ndim == 3 and chw.semantic == "image"
    assert chw.shape is not None and chw.shape[0] == Dim.exact(3, "C")
    hwc = ArrayType.image("HWC", channels=(1, 4))
    assert hwc.shape is not None and hwc.shape[2] == Dim(1, 4, "C")
    with pytest.raises(ValueError):
        ArrayType.image("XYZ")


def test_array_parse() -> None:
    a = ArrayType.parse("3 h w", dtype="float32", framework="torch")
    assert a.shape == (Dim.exact(3), Dim.any("h"), Dim.any("w"))
    assert a.dtype == "float32" and a.frameworks == frozenset({"torch"})
    ranged = ArrayType.parse("1-10 N")
    assert ranged.shape == (Dim.range(1, 10), Dim.any("N"))
    assert ArrayType.parse("2-").shape == (Dim.range(2, None),)
    with pytest.raises(ValueError):
        ArrayType.parse("*batch c h w")


# --------------------------------------------------------------------------------------------------
# PythonType / Union / Mapping / List
# --------------------------------------------------------------------------------------------------


def test_python_type_exact() -> None:
    assert accepts(PythonType("dict"), PythonType("dict"))
    assert not accepts(PythonType("dict"), PythonType("list"))
    # class mismatch
    assert not accepts(PythonType("dict"), ArrayType(ndim=1))
    assert not accepts(ArrayType(ndim=1), PythonType("dict"))


def test_union_variance() -> None:
    consumer = UnionType((ArrayType(dtype="numeric"), PythonType("PIL.Image.Image")))
    assert accepts(consumer, ArrayType(dtype="int64"))
    assert accepts(consumer, PythonType("PIL.Image.Image"))
    assert not accepts(consumer, PythonType("str"))
    # producer union: every branch must be accepted
    prod = UnionType((ArrayType(dtype="int64"), ArrayType(dtype="float32")))
    assert accepts(ArrayType(dtype="numeric"), prod)
    assert not accepts(ArrayType(dtype="floating"), prod)  # int64 branch fails


def test_mapping_and_list() -> None:
    consumer = MappingType.of({"a": ArrayType(ndim=1), "b": PythonType("str")})
    assert accepts(consumer, MappingType.of({"a": ArrayType(ndim=1), "b": PythonType("str"), "c": AnyType()}))
    assert not accepts(consumer, MappingType.of({"a": ArrayType(ndim=1)}))  # missing 'b'
    assert not accepts(consumer, MappingType.of({"a": ArrayType(ndim=2), "b": PythonType("str")}))
    assert accepts(ListType(ArrayType(dtype="floating")), ListType(ArrayType(dtype="float32")))
    assert not accepts(ListType(ArrayType(dtype="floating")), ListType(ArrayType(dtype="int64")))
    assert not accepts(MappingType.of({"a": AnyType()}), ListType(AnyType()))  # class mismatch


# --------------------------------------------------------------------------------------------------
# SampleType
# --------------------------------------------------------------------------------------------------


def test_sample_type_pair() -> None:
    consumer = SampleType(input=ArrayType(ndim=2), target=ArrayType(dtype="int64"))
    assert consumer.accepts(SampleType(input=ArrayType(ndim=2), target=ArrayType(dtype="int64")))
    assert not consumer.accepts(SampleType(input=ArrayType(ndim=3), target=ArrayType(dtype="int64")))
    # default target is Any -> input-only constraints accept any target
    input_only = SampleType(input=ArrayType(ndim=2))
    assert input_only.accepts(SampleType(input=ArrayType(ndim=2), target=PythonType("dict")))
    assert SampleType() == SampleType()  # both AnyType defaults compare equal


# --------------------------------------------------------------------------------------------------
# dtype canonicalization
# --------------------------------------------------------------------------------------------------


def test_canonical_dtype() -> None:
    assert canonical_dtype("Float32") == "float32"
    assert canonical_dtype("double") == "float64"
    assert canonical_dtype("floating") == "floating"  # family passes through
    assert canonical_dtype(np.dtype("int64")) == "int64"
    assert canonical_dtype(np.float32) == "float32"
    import torch

    assert canonical_dtype(torch.float32) == "float32"
    assert canonical_dtype(torch.int64) == "int64"


# --------------------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------------------


def test_infer_type_numpy() -> None:
    spec = infer_type(np.zeros((3, 8, 8), dtype=np.float32))
    assert isinstance(spec, ArrayType)
    assert spec.ndim == 3 and spec.dtype == "float32" and spec.frameworks == frozenset({"numpy"})
    assert spec.shape == (Dim.exact(3), Dim.exact(8), Dim.exact(8))
    scalar = infer_type(np.int64(7))
    assert isinstance(scalar, ArrayType) and scalar.ndim == 0 and scalar.dtype == "int64"


def test_infer_type_torch() -> None:
    import torch

    spec = infer_type(torch.zeros(2, 4, dtype=torch.float64))
    assert isinstance(spec, ArrayType) and spec.frameworks == frozenset({"torch"})
    assert spec.ndim == 2 and spec.dtype == "float64"


def test_infer_type_pil() -> None:
    Image = pytest.importorskip("PIL.Image")
    assert infer_type(Image.new("RGB", (4, 4))) == PythonType("PIL.Image.Image")


def test_infer_type_python_values() -> None:
    assert infer_type(None) == AnyType()
    assert infer_type(True) == PythonType("bool")
    assert infer_type(3) == PythonType("int")
    assert infer_type(2.5) == PythonType("float")
    assert infer_type("x") == PythonType("str")
    assert infer_type({"a": 1}) == PythonType("dict")
    assert infer_type([1, 2]) == PythonType("list")
    assert infer_type((1, 2)) == PythonType("tuple")
    assert infer_type(_Widget()) == PythonType("_Widget")


def test_infer_sample_type() -> None:
    st = infer_sample_type(Sample(input=np.zeros((2, 2), dtype=np.float32), target=np.int64(1)))
    assert isinstance(st.input, ArrayType) and st.input.ndim == 2
    assert isinstance(st.target, ArrayType) and st.target.ndim == 0
    # unset target -> Any
    assert infer_sample_type(Sample(input=np.zeros(2))).target == AnyType()


# --------------------------------------------------------------------------------------------------
# JSON round-trip
# --------------------------------------------------------------------------------------------------


def test_json_roundtrip_all_kinds() -> None:
    specs: List[TypeSpec] = [
        AnyType(),
        ArrayType.image("CHW", 3, "float32", "torch"),
        ArrayType(shape=(Dim.range(1, 10), Dim.any("N")), dtype="numeric", frameworks={"numpy"}),
        PythonType("PIL.Image.Image"),
        UnionType((ArrayType(ndim=1), PythonType("str"), AnyType())),
        MappingType.of({"boxes": ArrayType(ndim=2), "labels": ListType(ArrayType(ndim=0, dtype="int64"))}),
    ]
    for spec in specs:
        assert type_from_dict(spec.to_dict()) == spec
    st = SampleType(input=specs[1], target=specs[5])
    assert SampleType.from_dict(st.to_dict()) == st


def test_type_from_dict_unknown_kind() -> None:
    with pytest.raises(ValueError):
        type_from_dict({"kind": "bogus"})


# --------------------------------------------------------------------------------------------------
# datasets.Features bridge
# --------------------------------------------------------------------------------------------------


def test_hf_features_classification_target() -> None:
    spec = SampleType(
        input=PythonType("PIL.Image.Image"),
        target=ArrayType(ndim=0, dtype="int64", frameworks={"torch"}),
    )
    features, extras = spec.to_hf_features()
    assert set(features.keys()) == {"input", "target"}
    rt = SampleType.from_hf_features(features, extras)
    assert rt.accepts(spec) and spec.accepts(rt)


def test_hf_features_detection_target() -> None:
    spec = SampleType(
        target=MappingType.of(
            {
                "boxes": ArrayType(ndim=2, shape=(Dim.any("N"), Dim.exact(4)), dtype="float32", frameworks={"torch"}),
                "labels": ArrayType(ndim=1, shape=(Dim.any("N"),), dtype="int64", frameworks={"torch"}),
            }
        )
    )
    features, extras = spec.to_hf_features()
    rt = SampleType.from_hf_features(features, extras)
    assert rt.accepts(spec) and spec.accepts(rt)


def test_hf_features_segmentation_target() -> None:
    spec = SampleType(
        input=ArrayType.image("CHW", 3, "float32", "torch"),
        target=ArrayType(ndim=2, shape=(Dim.any("H"), Dim.any("W")), dtype="int64", frameworks={"torch"}),
    )
    features, extras = spec.to_hf_features()
    rt = SampleType.from_hf_features(features, extras)
    assert rt.accepts(spec) and spec.accepts(rt)


def test_hf_features_non_expressible_falls_back_to_typespec() -> None:
    # Any / Union / family-dtype / unknown-rank can't be a concrete Feature -> stored as a typespec blob.
    spec = SampleType(
        input=UnionType((ArrayType(dtype="numeric"), PythonType("PIL.Image.Image"))),
        target=ArrayType(dtype="floating"),  # family dtype, no rank
    )
    features, extras = spec.to_hf_features()
    assert "input" not in features and "target" not in features
    assert "typespec" in extras["input"] and "typespec" in extras["target"]
    rt = SampleType.from_hf_features(features, extras)
    assert rt == spec


def test_hf_features_from_dict_form() -> None:
    spec = SampleType(input=ArrayType(ndim=2, shape=(Dim.any(), Dim.exact(4)), dtype="float32"))
    features, extras = spec.to_hf_features()
    # describe() passes the Features.to_dict() form (a plain dict), not a Features instance
    rt = SampleType.from_hf_features(features.to_dict(), extras)
    assert rt.accepts(spec) and spec.accepts(rt)


def test_hf_features_pythontype_and_string_roundtrip() -> None:
    # PIL <-> Image, str <-> Value("string"), and a non-Feature python type falls back to a typespec blob.
    spec = SampleType(input=PythonType("str"), target=PythonType("dict"))
    features, extras = spec.to_hf_features()
    assert "input" in features and "target" not in features  # str -> Value, dict -> typespec fallback
    assert SampleType.from_hf_features(features, extras) == spec


def test_hf_features_high_rank_and_nested_fallback() -> None:
    # ndim>5 has no ArrayXD feature, and a mapping/list with a non-expressible member falls back whole.
    high_rank = SampleType(input=ArrayType(ndim=6, dtype="float32"))
    f, e = high_rank.to_hf_features()
    assert "input" not in f and SampleType.from_hf_features(f, e) == high_rank

    nested = SampleType(target=MappingType.of({"a": ArrayType(ndim=2, dtype="float32"), "b": AnyType()}))
    f, e = nested.to_hf_features()
    assert "target" not in f and SampleType.from_hf_features(f, e) == nested

    listed = SampleType(target=ListType(AnyType()))
    f, e = listed.to_hf_features()
    assert SampleType.from_hf_features(f, e) == listed


def test_canonical_dtype_fallback() -> None:
    # an object that is neither a str nor a known dtype object falls through to str(x).lower()
    assert canonical_dtype(123) == "123"


# --------------------------------------------------------------------------------------------------
# @typed decorator
# --------------------------------------------------------------------------------------------------


def test_typed_decorator() -> None:
    acc = SampleType(input=PythonType("PIL.Image.Image"))
    prod = SampleType(input=ArrayType.image("CHW", 3, "float32", "torch"))

    @typed(accepts=acc, produces=prod)
    class Op:
        pass

    assert getattr(Op, "ACCEPTS") == acc and getattr(Op, "PRODUCES") == prod

    with pytest.raises(TypeError):

        @typed(accepts=ArrayType(ndim=1))  # type: ignore[arg-type]
        class Bad:
            pass

    with pytest.raises(TypeError):

        @typed(produces="nope")  # type: ignore[arg-type]
        class Bad2:
            pass


# --------------------------------------------------------------------------------------------------
# Sample.describe / with_type
# --------------------------------------------------------------------------------------------------


def test_sample_describe_infers_when_unstored() -> None:
    s = Sample(input=np.zeros((3, 8, 8), dtype=np.float32))
    st = s.describe()
    assert isinstance(st.input, ArrayType) and st.input.ndim == 3 and st.input.frameworks == frozenset({"numpy"})


def test_sample_with_type_and_describe_stored() -> None:
    declared = SampleType(input=ArrayType.image("CHW", 3, "float32", "torch"))
    s = Sample(input=np.zeros((3, 8, 8), dtype=np.float32), metadata={"id": 7})
    typed_s = s.with_type(declared)
    assert FEATURES_KEY in typed_s.metadata and SPEC_KEY in typed_s.metadata
    assert typed_s.metadata["id"] == 7  # pre-existing metadata preserved
    assert s.metadata == {"id": 7} and FEATURES_KEY not in s.metadata  # copy-on-write: original untouched
    rt = typed_s.describe()
    assert rt.accepts(declared) and declared.accepts(rt)


# --------------------------------------------------------------------------------------------------
# Pipeline freshness (maintain-if-present)
# --------------------------------------------------------------------------------------------------


class _ToFloat64Op:
    PRODUCES = SampleType(input=ArrayType(ndim=1, dtype="float64", frameworks={"numpy"}))

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input.astype("float64"))


class _UntypedOp:
    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input + 1)


def _run(sample: Sample, op: Any) -> Sample:
    return cast(Sample, list(Flux(source=[sample], ops=[op]))[0])


def test_pipeline_does_not_stamp_untracked_samples() -> None:
    out = _run(Sample(input=np.array([1, 2, 3])), _ToFloat64Op())
    assert FEATURES_KEY not in out.metadata and SPEC_KEY not in out.metadata


def test_pipeline_refreshes_stored_type_from_produces() -> None:
    stamped = Sample(input=np.array([1, 2, 3])).with_type(
        SampleType(input=ArrayType(ndim=1, dtype="int64", frameworks={"numpy"}))
    )
    out = _run(stamped, _ToFloat64Op())
    assert FEATURES_KEY in out.metadata
    assert out.describe().accepts(_ToFloat64Op.PRODUCES)
    assert _ToFloat64Op.PRODUCES.accepts(out.describe())


def test_pipeline_drops_stored_type_when_op_has_no_produces() -> None:
    stamped = Sample(input=np.array([1, 2, 3])).with_type(
        SampleType(input=ArrayType(ndim=1, dtype="int64", frameworks={"numpy"}))
    )
    out = _run(stamped, _UntypedOp())
    assert FEATURES_KEY not in out.metadata and SPEC_KEY not in out.metadata
    # describe() falls back to inference -> still correct, just not "stored"
    assert isinstance(out.describe().input, ArrayType)


# --------------------------------------------------------------------------------------------------
# Op annotation conformance: declared PRODUCES must accept the real inferred output, and declared
# ACCEPTS must accept the real input (proves the static specs match runtime reality).
# --------------------------------------------------------------------------------------------------


def test_dataflux_op_spec_conformance() -> None:
    import dataflux.ops.numpy as N
    import dataflux.ops.torch as T

    rgb = (np.random.rand(3, 8, 8) * 255).astype(np.float32)
    cases: List[Tuple[Any, Sample]] = [
        (N.StandardizeOp(mean=0.5, std=0.5), Sample(input=rgb.copy())),
        (N.RescaleOp(in_min=0, in_max=255), Sample(input=rgb.copy())),
        (N.ClipPercentilesOp(), Sample(input=rgb.copy())),
        (N.ReplaceNonFiniteOp(), Sample(input=rgb.copy())),
        (N.ThresholdOp(value=0.5), Sample(input=rgb.copy())),
        (N.ConnectedComponentsOp(), Sample(input=(np.random.rand(8, 8) > 0.5))),
        (T.RescaleOp(in_min=0, in_max=255), Sample(input=__import__("torch").rand(3, 8, 8) * 255)),
        (T.StandardizeOp(mean=0.5, std=0.5), Sample(input=__import__("torch").rand(3, 8, 8))),
    ]
    for op, sample in cases:
        name = type(op).__module__ + "." + type(op).__name__
        assert op.ACCEPTS.accepts(infer_sample_type(sample)), f"{name}: ACCEPTS rejects its real input"
        out = op(sample)
        assert op.PRODUCES.accepts(infer_sample_type(out)), f"{name}: PRODUCES rejects its real output"
        # specs are JSON round-trippable
        assert SampleType.from_dict(op.PRODUCES.to_dict()) == op.PRODUCES
