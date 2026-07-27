# mypy: disable-error-code="attr-defined,union-attr"
"""Discovery-category coverage for recordstream ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
MCP tool and, downstream, the visual-editor form-spec picker (``get_node_form_spec``).
A class silently losing its category empties the relevant picker, so the tags are
pinned here as a regression gate.
"""

from confluid.registry import get_registry

from recordstream import Pipeline
from recordstream.core import FilterOp, JointStream, Stream, WrappedOp
from recordstream.ops.configure import ConfigureOp
from recordstream.ops.context import Apply, Capture, Drop, MergeFields, Save, Use
from recordstream.ops.debug import PrintRecordOp
from recordstream.ops.enable import Enable
from recordstream.ops.formula import FormulaOp
from recordstream.ops.image import ConvertToImage
from recordstream.ops.numpy import ConnectedComponents, Threshold
from recordstream.ops.parallel import Parallel
from recordstream.ops.random_apply import RandomApply
from recordstream.ops.sink import RecordSinkOp
from recordstream.ops.structure import CopyField, DropField, RenameField, SelectFields
from recordstream.ops.target import CocoToTorchVisionDetection, DecodeTarget, EncodeTarget, MasksToDetectionBoxes
from recordstream.ops.torch import ToTensor
from recordstream.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource
from recordstream.storage.directory import DirectorySink
from recordstream.storage.hdf5 import HDF5Sink, HDF5Source
from recordstream.storage.zarr import ZarrBatchSink, ZarrGroupSink


def test_engine_classes_tagged() -> None:
    assert Stream.__confluid_category__ == "engine"
    assert JointStream.__confluid_category__ == "engine"


def test_raw_callable_wrappers_uncategorised() -> None:
    assert getattr(FilterOp, "__confluid_category__", None) is None
    assert getattr(WrappedOp, "__confluid_category__", None) is None
    assert FilterOp.__confluid_configurable__ is True
    assert WrappedOp.__confluid_configurable__ is True


def test_source_classes_tagged() -> None:
    assert HuggingFaceSource.__confluid_category__ == "source"
    assert DatasetSplit.__confluid_category__ == "source"
    assert RangeSource.__confluid_category__ == "source"
    assert ConcatSource.__confluid_category__ == "source"


def test_op_classes_tagged() -> None:
    for cls in (
        Threshold,
        ConnectedComponents,
        ToTensor,
        ConvertToImage,
        Enable,
        Pipeline,
        Parallel,
        RandomApply,
        RecordSinkOp,
        EncodeTarget,
        DecodeTarget,
        CocoToTorchVisionDetection,
        MasksToDetectionBoxes,
        ConfigureOp,
        FormulaOp,
        RenameField,
        DropField,
        CopyField,
        SelectFields,
        Save,
        Use,
        Drop,
        Apply,
        Capture,
        MergeFields,
        PrintRecordOp,
    ):
        assert cls.__confluid_category__ == "op", cls.__name__


def test_random_apply_random_tagged() -> None:
    assert RandomApply.__confluid_random__ is True


def test_storage_sink_classes_tagged() -> None:
    assert HDF5Sink.__confluid_category__ == "sink"
    assert ZarrGroupSink.__confluid_category__ == "sink"
    assert ZarrBatchSink.__confluid_category__ == "sink"
    assert DirectorySink.__confluid_category__ == "sink"
    assert getattr(HDF5Source, "__confluid_category__", None) is None


def test_op_group_tags() -> None:
    assert Threshold.__confluid_group__ == "numpy"
    assert ConnectedComponents.__confluid_group__ == "numpy"
    assert ToTensor.__confluid_group__ == "torch"
    assert ConvertToImage.__confluid_group__ == "image"
    assert SelectFields.__confluid_group__ == "structure"
    assert PrintRecordOp.__confluid_group__ == "debug"
    assert EncodeTarget.__confluid_group__ == "structure"
    assert DecodeTarget.__confluid_group__ == "structure"
    assert CocoToTorchVisionDetection.__confluid_group__ == "structure"
    assert MasksToDetectionBoxes.__confluid_group__ == "structure"
    for ctx_op in (Save, Use, Drop, Apply, Capture, MergeFields):
        assert ctx_op.__confluid_group__ == "structure", ctx_op.__name__
    assert Parallel.__confluid_group__ == "compose"
    assert Enable.__confluid_group__ == "compose"
    assert Pipeline.__confluid_group__ == "compose"
    assert RandomApply.__confluid_group__ == "compose"
    assert ConfigureOp.__confluid_group__ == "compose"
    assert FormulaOp.__confluid_group__ == "compose"
    assert RecordSinkOp.__confluid_group__ == "sink"


def test_categories_enumerable_via_registry() -> None:
    registry = get_registry()
    assert {"Stream", "JointStream"} <= registry.list_classes(category="engine")
    assert "DatasetSplit" not in registry.list_classes(category="engine")
    assert not ({"FilterOp", "WrappedOp"} & registry.list_classes(category="engine"))
    assert {"HuggingFaceSource", "DatasetSplit", "RangeSource", "ConcatSource"} <= registry.list_classes(
        category="source"
    )
    assert {
        "Threshold",
        "ConnectedComponents",
        "ToTensor",
        "ConvertToImage",
        "Enable",
        "Pipeline",
        "RecordSinkOp",
        "EncodeTarget",
        "DecodeTarget",
        "CocoToTorchVisionDetection",
        "MasksToDetectionBoxes",
    } <= registry.list_classes(category="op")
    assert {"HDF5Sink", "ZarrGroupSink", "ZarrBatchSink", "DirectorySink"} <= registry.list_classes(category="sink")
    assert "RecordSinkOp" not in registry.list_classes(category="sink")


def test_groups_enumerable_via_registry() -> None:
    registry = get_registry()
    assert {"Threshold", "ConnectedComponents"} <= registry.list_classes(group="numpy")
    assert {"ToTensor"} <= registry.list_classes(group="torch")
    assert {"ConvertToImage"} <= registry.list_classes(group="image")
    assert {"Parallel", "Enable", "Pipeline", "RandomApply", "ConfigureOp", "FormulaOp"} <= registry.list_classes(
        group="compose"
    )
    assert {"RecordSinkOp"} <= registry.list_classes(group="sink")
    assert {
        "EncodeTarget",
        "DecodeTarget",
        "CocoToTorchVisionDetection",
        "MasksToDetectionBoxes",
        "SelectFields",
        "Save",
        "Use",
        "MergeFields",
    } <= registry.list_classes(group="structure")
    assert "Pipeline" in registry.list_classes(category="op", group="compose")
