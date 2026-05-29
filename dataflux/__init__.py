"""
DataFlux: Modular, functional data pipelines.
"""

from dataflux.core import Flux, JointFlux, WrappedOp
from dataflux.ops import RescaleOp, StandardizeOp, ToTensorOp
from dataflux.paired import PairedSource
from dataflux.projection import SupportsProjection, iter_inputs, iter_targets, num_classes, project
from dataflux.sample import Sample
from dataflux.sources import DatasetSplit, HuggingFaceSource
from dataflux.typespec import (
    AnyType,
    ArrayType,
    Dim,
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
    "AnyType",
    "ArrayType",
    "DatasetSplit",
    "Dim",
    "Flux",
    "HuggingFaceSource",
    "JointFlux",
    "ListType",
    "MappingType",
    "PairedSource",
    "PythonType",
    "RescaleOp",
    "Sample",
    "SampleType",
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
