"""
SampleFlux: Modular, functional data pipelines.

The data model is the TYPED BAG: a :class:`Sample` is a named bag of typed items (each
owning its metadata), ``input`` / ``target`` are ROLE TAGS on fields, and transforms
dispatch on item TYPE. Import the whole surface from the package top level
(``from sampleflux import Sample, Image, Transform, primary, ...``); the internal module
layout (``sampleflux.bag.*``) is transitional and may be promoted to the package root.
"""

# --- the typed-bag data model + transforms + item codec ------------------------------------
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
    Sample,
    Transform,
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

# --- shared infrastructure -----------------------------------------------------------------
from sampleflux.collate import collate, get_collate, register_collate, registered_collates, typed_collate
from sampleflux.context import Context
from sampleflux.core import FilterOp, Flux, JointFlux, WrappedOp
from sampleflux.flow import FlowGraph, from_ops, to_ops
from sampleflux.labels import LabelMap
from sampleflux.processing import DatasetProcessor
from sampleflux.projection import ProjectionField, SupportsProjection, iter_inputs, iter_targets, num_classes, project
from sampleflux.runnable import (
    ProgressCallback,
    ProgressReporting,
    TorchRunner,
    entrypoint,
    entrypoint_tasks,
    runnable_entrypoints,
)
from sampleflux.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource, SplitName
from sampleflux.workflow import AllOf, AnyOf, Conditional, Not, PathExists, Sequence, Switch

__all__ = [
    # ---- typed-bag data model ----
    "Sample",
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
    # ---- shared infrastructure ----
    "Context",
    "Flux",
    "JointFlux",
    "FilterOp",
    "WrappedOp",
    "FlowGraph",
    "from_ops",
    "to_ops",
    "collate",
    "get_collate",
    "register_collate",
    "registered_collates",
    "typed_collate",
    "LabelMap",
    # ---- sources ----
    "HuggingFaceSource",
    "DatasetSplit",
    "RangeSource",
    "ConcatSource",
    "SplitName",
    # ---- projection ----
    "ProjectionField",
    "SupportsProjection",
    "iter_inputs",
    "iter_targets",
    "num_classes",
    "project",
    # ---- runnable protocol + orchestration ----
    "TorchRunner",
    "ProgressReporting",
    "ProgressCallback",
    "entrypoint",
    "entrypoint_tasks",
    "runnable_entrypoints",
    "DatasetProcessor",
    "Sequence",
    "Conditional",
    "Switch",
    "PathExists",
    "Not",
    "AllOf",
    "AnyOf",
]
