"""
SampleFlux: Modular, functional data pipelines.

The TYPED-BAG model (``TypedSample`` + typed items + type-dispatched ``Transform``\\ s) is THE
data model — import its surface from here (``from sampleflux import TypedSample, Image, ...``);
the internal module layout is transitional. The legacy ``Sample`` triple surface below it is
being migrated out and will be deleted once every consumer has flipped.
"""

# --- the typed-bag surface (THE data model; frozen — consumers import ONLY from here) -----
from sampleflux.bag import (
    ROLES,
    EncodedField,
    EncodedItem,
    FunctionTransform,
    Image,
    Label,
    Mask,
    NDArrayItem,
    Pipeline,
    Regions,
    Role,
    Transform,
    TypedSample,
    as_transform,
    coerce_transform,
    decode_item,
    decode_sample,
    dispatch,
    encode_item,
    encode_sample,
    get_item_type,
    is_item,
    item_data,
    item_type_names,
    item_types,
    primary,
    register_adapter,
    register_io,
    register_item,
    register_kernel,
    with_data,
)

# --- shared infrastructure (carrier-agnostic) ----------------------------------------------
from sampleflux.collate import collate, get_collate, register_collate
from sampleflux.context import Context
from sampleflux.core import Flux, JointFlux, WrappedOp
from sampleflux.flow import FlowGraph, from_ops, to_ops

# --- LEGACY surface (the Sample triple era — dies with the purge stage) --------------------
from sampleflux.kinds import INPUT, TARGET, Input, OpContract, SampleKind, Target, classify_carrier, op_contract
from sampleflux.labels import LabelMap
from sampleflux.ops import RescaleOp, StandardizeOp, ToTensorOp
from sampleflux.projection import ProjectionField, SupportsProjection, iter_inputs, iter_targets, num_classes, project
from sampleflux.sample import InputMeta, Pair, Sample, TargetMeta
from sampleflux.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource, SplitName
from sampleflux.typespec import (
    AnyType,
    ArrayType,
    Dim,
    Dtype,
    DtypeFamily,
    DtypeSpec,
    Framework,
    ListType,
    MappingType,
    PythonType,
    SampleType,
    UnionType,
    infer_field_types,
    infer_sample_type,
    infer_type,
    typed,
)

__all__ = [
    # ---- typed-bag surface (THE data model) ----
    "TypedSample",
    "Role",
    "ROLES",
    "primary",
    "NDArrayItem",
    "Image",
    "Mask",
    "Regions",
    "Label",
    "register_item",
    "item_types",
    "item_type_names",
    "get_item_type",
    "is_item",
    "item_data",
    "with_data",
    "Transform",
    "Pipeline",
    "FunctionTransform",
    "as_transform",
    "register_adapter",
    "coerce_transform",
    "dispatch",
    "register_kernel",
    "EncodedItem",
    "EncodedField",
    "register_io",
    "encode_item",
    "decode_item",
    "encode_sample",
    "decode_sample",
    "infer_field_types",
    # ---- shared infrastructure ----
    "Context",
    "Flux",
    "JointFlux",
    "FlowGraph",
    "from_ops",
    "to_ops",
    "collate",
    "get_collate",
    "register_collate",
    "LabelMap",
    # ---- legacy surface (dies with the purge stage) ----
    "AnyType",
    "ArrayType",
    "ConcatSource",
    "DatasetSplit",
    "Dim",
    "INPUT",
    "Input",
    "InputMeta",
    "OpContract",
    "Pair",
    "TARGET",
    "Target",
    "TargetMeta",
    "SampleKind",
    "classify_carrier",
    "op_contract",
    "Dtype",
    "DtypeFamily",
    "DtypeSpec",
    "Framework",
    "HuggingFaceSource",
    "ListType",
    "MappingType",
    "ProjectionField",
    "PythonType",
    "RangeSource",
    "RescaleOp",
    "Sample",
    "SampleType",
    "SplitName",
    "StandardizeOp",
    "SupportsProjection",
    "ToTensorOp",
    "UnionType",
    "WrappedOp",
    "infer_sample_type",
    "infer_type",
    "iter_inputs",
    "iter_targets",
    "num_classes",
    "project",
    "typed",
]
