"""
RecordStream: Modular, functional data pipelines.

The data model is the RECORD: a record is a plain ``dict`` of typed values (each value
owning its metadata — an ``Image`` its layout, a ``Label`` its classes), and ops dispatch
on value TYPE (the torchvision-v2 model). Bare albumentations / torchvision ``transforms.v2``
transforms drop into any ops list AS-IS — the engine invokes each op family natively
(``recordstream.core._apply_op``). Import the whole surface from the package top level
(``from recordstream import Record, Image, Transform, Pipeline, ...``).
"""

# --- shared infrastructure -----------------------------------------------------------------
from recordstream.batch import batch_metadata, batch_tensor, batch_values, multi_hot
from recordstream.collate import collate, collate_records, get_collate, register_collate, registered_collates
from recordstream.context import Context
from recordstream.core import FilterOp, JointStream, Stream, WrappedOp, register_op_family, registered_op_families

# --- the record data model + transforms + item codec ----------------------------------------
from recordstream.dispatch import dispatch, register_kernel, registered_kernels
from recordstream.flow import FlowGraph, from_ops, to_ops
from recordstream.io import (
    EncodedField,
    EncodedItem,
    decode_item,
    decode_record,
    encode_item,
    encode_record,
    register_io,
)
from recordstream.items import (
    Image,
    Label,
    Mask,
    MultiLabel,
    NDArrayItem,
    Record,
    Regions,
    get_item_type,
    is_class_id,
    is_item,
    item_data,
    item_type_names,
    item_types,
    register_item,
    with_data,
)
from recordstream.labels import LabelMap
from recordstream.processing import DatasetProcessor
from recordstream.projection import SupportsProjection, iter_key, num_classes, project
from recordstream.runnable import (
    ProgressCallback,
    ProgressReporting,
    TorchRunner,
    entrypoint,
    entrypoint_tasks,
    runnable_entrypoints,
)
from recordstream.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource, SplitName
from recordstream.transform import FunctionTransform, Pipeline, Transform, as_transform
from recordstream.workflow import AllOf, AnyOf, Conditional, Not, PathExists, Sequence, Switch

__all__ = [
    # ---- record data model ----
    "Record",
    "NDArrayItem",
    "Image",
    "Mask",
    "Regions",
    "Label",
    "MultiLabel",
    "is_class_id",
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
    "Stream",
    "JointStream",
    "FilterOp",
    "WrappedOp",
    "register_op_family",
    "registered_op_families",
    "FlowGraph",
    "from_ops",
    "to_ops",
    "collate",
    "batch_metadata",
    "batch_tensor",
    "batch_values",
    "multi_hot",
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
