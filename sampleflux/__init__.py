"""
SampleFlux: Modular, functional data pipelines.

The data model is the RECORD: a sample is a plain ``dict`` of typed values (each value
owning its metadata — an ``Image`` its layout, a ``Label`` its classes), and ops dispatch
on value TYPE (the torchvision-v2 model). Bare albumentations / torchvision ``transforms.v2``
transforms drop into any ops list AS-IS — the engine invokes each op family natively
(``sampleflux.core._apply_op``). Import the whole surface from the package top level
(``from sampleflux import Record, Image, Transform, Pipeline, ...``).
"""

# --- shared infrastructure -----------------------------------------------------------------
from sampleflux.collate import collate, collate_records, get_collate, register_collate, registered_collates
from sampleflux.context import Context
from sampleflux.core import FilterOp, Flux, JointFlux, WrappedOp

# --- the record data model + transforms + item codec ----------------------------------------
from sampleflux.dispatch import dispatch, register_kernel, registered_kernels
from sampleflux.flow import FlowGraph, from_ops, to_ops
from sampleflux.io import EncodedField, EncodedItem, decode_item, decode_record, encode_item, encode_record, register_io
from sampleflux.items import (
    Image,
    Label,
    Mask,
    NDArrayItem,
    Record,
    Regions,
    get_item_type,
    is_item,
    item_data,
    item_type_names,
    item_types,
    register_item,
    with_data,
)
from sampleflux.labels import LabelMap
from sampleflux.processing import DatasetProcessor
from sampleflux.projection import SupportsProjection, iter_key, num_classes, project
from sampleflux.runnable import (
    ProgressCallback,
    ProgressReporting,
    TorchRunner,
    entrypoint,
    entrypoint_tasks,
    runnable_entrypoints,
)
from sampleflux.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource, SplitName
from sampleflux.transform import FunctionTransform, Pipeline, Transform, as_transform
from sampleflux.workflow import AllOf, AnyOf, Conditional, Not, PathExists, Sequence, Switch

__all__ = [
    # ---- record data model ----
    "Record",
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
    "dispatch",
    "register_kernel",
    "registered_kernels",
    "EncodedItem",
    "EncodedField",
    "register_io",
    "encode_item",
    "decode_item",
    "encode_record",
    "decode_record",
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
    "collate_records",
    "get_collate",
    "register_collate",
    "registered_collates",
    "LabelMap",
    # ---- sources ----
    "HuggingFaceSource",
    "DatasetSplit",
    "RangeSource",
    "ConcatSource",
    "SplitName",
    # ---- projection ----
    "SupportsProjection",
    "iter_key",
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
