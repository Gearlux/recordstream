"""
DataFlux: Modular, functional data pipelines.
"""

from dataflux.core import Flux, JointFlux, WrappedOp
from dataflux.ops import RescaleOp, StandardizeOp, ToTensorOp
from dataflux.paired import AnnotationJoinSource, AnnotationStore
from dataflux.projection import ProjectionField, SupportsProjection, iter_inputs, iter_targets, num_classes, project
from dataflux.sample import Sample
from dataflux.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource, SplitName
from dataflux.typespec import (
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
    "DatasetSplit",
    "Dim",
    "Dtype",
    "DtypeFamily",
    "DtypeSpec",
    "Flux",
    "Framework",
    "HuggingFaceSource",
    "JointFlux",
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
