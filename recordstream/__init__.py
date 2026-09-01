"""
RecordStream: Modular, functional data pipelines.

The data model is the RECORD: a record is a plain ``dict`` of typed values (each value
owning its metadata — an ``Image`` its layout, a ``Label`` its classes), and ops dispatch
on value TYPE (the torchvision-v2 model). Bare albumentations / torchvision ``transforms.v2``
transforms drop into any ops list AS-IS — the engine invokes each op family natively
(``recordstream.core.families._apply_op``). Import the whole surface from the package top level
(``from recordstream import Record, Image, Transform, Pipeline, ...``).
"""

# --- shared infrastructure -----------------------------------------------------------------
from recordstream.batch import (
    batch_boxes,
    batch_metadata,
    batch_tensor,
    batch_values,
    multi_hot,
    per_record_predictions,
)
from recordstream.collate import (
    collate,
    collate_list,
    collate_records,
    get_collate,
    register_collate,
    registered_collates,
)
from recordstream.core import (
    FilterOp,
    JointStream,
    RecordSource,
    Stream,
    WrappedOp,
    ensure_materialized,
    ensure_record_dataset,
    prepare_record_dataset,
    register_op_family,
    registered_op_families,
)

# --- the record data model + transforms + item codec ----------------------------------------
from recordstream.dispatch import dispatch, register_kernel, registered_kernels
from recordstream.flow import FlowGraph
from recordstream.formats import FORMAT_GROUP, FileFormat, file_formats, sibling
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
    Boxes,
    Image,
    Label,
    Mask,
    MultiLabel,
    NDArrayItem,
    Record,
    get_item_type,
    is_class_id,
    is_item,
    item_data,
    item_type_names,
    item_types,
    item_value,
    register_item,
    resolve_entry,
    resolve_item,
    with_data,
)
from recordstream.labels import LabelMap, class_counts, inverse_frequency_weights
from recordstream.outputs import (
    ClassificationOutput,
    DetectionOutput,
    DetectionPredictions,
    RestorationOutput,
    SegmentationOutput,
    classification_output,
    restoration_output,
    segmentation_output,
)
from recordstream.predictions import ClassificationPredictionsSink, PredictionsSink
from recordstream.processing import DatasetProcessor
from recordstream.projection import (
    SupportsProjection,
    class_names,
    first_value,
    iter_key,
    num_classes,
    num_mask_classes,
    project,
)
from recordstream.runnable import (
    ProgressCallback,
    ProgressReporting,
    RunnableTask,
    TorchRunner,
    entrypoint,
    entrypoint_tasks,
    run_entrypoint,
    runnable_entrypoints,
)
from recordstream.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource, SplitName
from recordstream.transform import FunctionTransform, Pipeline, Transform, as_transform
from recordstream.uri import SupportsDatasetIdentity, dataset_uri, dataset_uris, dataset_url
from recordstream.workflow import AllOf, AnyOf, Conditional, Not, PathExists, Sequence, Switch

__all__ = [
    # ---- record data model ----
    "Record",
    "NDArrayItem",
    "Image",
    "Mask",
    "Boxes",
    "Label",
    "MultiLabel",
    "is_class_id",
    "register_item",
    "item_types",
    "item_type_names",
    "get_item_type",
    "is_item",
    "item_data",
    "item_value",
    "resolve_entry",
    "resolve_item",
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
    "Stream",
    "JointStream",
    "RecordSource",
    "ensure_materialized",
    "ensure_record_dataset",
    "prepare_record_dataset",
    "FilterOp",
    "WrappedOp",
    "register_op_family",
    "registered_op_families",
    "FlowGraph",
    "collate",
    "batch_metadata",
    "batch_boxes",
    "batch_tensor",
    "batch_values",
    "multi_hot",
    "per_record_predictions",
    "collate_list",
    "collate_records",
    "get_collate",
    "register_collate",
    "registered_collates",
    "LabelMap",
    "class_counts",
    "inverse_frequency_weights",
    # ---- prediction contracts + sinks ----
    "ClassificationOutput",
    "DetectionOutput",
    "DetectionPredictions",
    "RestorationOutput",
    "SegmentationOutput",
    "classification_output",
    "restoration_output",
    "segmentation_output",
    "PredictionsSink",
    "ClassificationPredictionsSink",
    # ---- sources ----
    "HuggingFaceSource",
    "DatasetSplit",
    "RangeSource",
    "ConcatSource",
    "SplitName",
    # ---- file formats ----
    "FORMAT_GROUP",
    "FileFormat",
    "file_formats",
    "sibling",
    # ---- dataset identity ----
    "SupportsDatasetIdentity",
    "dataset_uri",
    "dataset_uris",
    "dataset_url",
    # ---- projection ----
    "SupportsProjection",
    "first_value",
    "iter_key",
    "class_names",
    "num_classes",
    "num_mask_classes",
    "project",
    # ---- runnable protocol + orchestration ----
    "TorchRunner",
    "ProgressReporting",
    "ProgressCallback",
    "RunnableTask",
    "entrypoint",
    "entrypoint_tasks",
    "run_entrypoint",
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
