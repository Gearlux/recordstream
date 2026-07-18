"""
SampleFlux: Modular, functional data pipelines.
"""

from sampleflux.collate import collate, get_collate, register_collate
from sampleflux.context import Context
from sampleflux.core import Flux, JointFlux, WrappedOp
from sampleflux.flow import FlowGraph, from_ops, to_ops
from sampleflux.kinds import OpContract, SampleKind, classify_carrier, op_contract
from sampleflux.labels import LabelMap
from sampleflux.ops import RescaleOp, StandardizeOp, ToTensorOp
from sampleflux.paired import AnnotationJoinSource, AnnotationStore
from sampleflux.projection import ProjectionField, SupportsProjection, iter_inputs, iter_targets, num_classes, project
from sampleflux.sample import Sample
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
    infer_sample_type,
    infer_type,
    typed,
)

__all__ = [
    "AnnotationJoinSource",
    "AnnotationStore",
    "AnyType",
    "ArrayType",
    "ConcatSource",
    "Context",
    "DatasetSplit",
    "Dim",
    "FlowGraph",
    "OpContract",
    "SampleKind",
    "classify_carrier",
    "collate",
    "from_ops",
    "get_collate",
    "op_contract",
    "register_collate",
    "to_ops",
    "Dtype",
    "DtypeFamily",
    "DtypeSpec",
    "Flux",
    "Framework",
    "HuggingFaceSource",
    "JointFlux",
    "LabelMap",
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
