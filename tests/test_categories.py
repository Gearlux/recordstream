# mypy: disable-error-code="attr-defined,union-attr"
"""Discovery-category coverage for sampleflux ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
MCP tool and, downstream, the visual-editor form-spec picker (``get_node_form_spec``).
A class silently losing its category empties the relevant picker, so the tags are
pinned here as a regression gate.
"""

from confluid.registry import get_registry

from sampleflux.core import FilterOp, Flux, JointFlux, WrappedOp
from sampleflux.ops.albumentations import AlbumentationsOp
from sampleflux.ops.configure import ConfigureOp
from sampleflux.ops.debug import PrintSampleOp
from sampleflux.ops.enable import Enable
from sampleflux.ops.formula import FormulaOp
from sampleflux.ops.image import ConvertToImage
from sampleflux.ops.numpy import ConnectedComponents, Threshold
from sampleflux.ops.parallel import Parallel
from sampleflux.ops.sink import SampleSinkOp
from sampleflux.ops.structure import CopyField, DropField, RenameField, SelectFields, SetRole
from sampleflux.ops.target import (
    CocoToTorchVisionDetection,
    DecodeTarget,
    EncodeTarget,
    MasksToDetectionBoxes,
    MetadataToTarget,
)
from sampleflux.ops.torch import ToTensor
from sampleflux.ops.torchvision import TorchvisionTransformOp
from sampleflux.ops.transform_chain import TransformChain
from sampleflux.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource
from sampleflux.storage.directory import DirectorySink
from sampleflux.storage.hdf5 import HDF5Sink, HDF5Source
from sampleflux.storage.zarr import ZarrBatchSink, ZarrGroupSink


def test_engine_classes_tagged() -> None:
    assert Flux.__confluid_category__ == "engine"
    assert JointFlux.__confluid_category__ == "engine"


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
        TransformChain,
        Parallel,
        SampleSinkOp,
        MetadataToTarget,
        EncodeTarget,
        DecodeTarget,
        CocoToTorchVisionDetection,
        MasksToDetectionBoxes,
        ConfigureOp,
        FormulaOp,
        AlbumentationsOp,
        TorchvisionTransformOp,
        SetRole,
        RenameField,
        DropField,
        CopyField,
        SelectFields,
        PrintSampleOp,
    ):
        assert cls.__confluid_category__ == "op", cls.__name__


def test_augmentation_adapters_random_tagged() -> None:
    assert AlbumentationsOp.__confluid_random__ is True
    assert TorchvisionTransformOp.__confluid_random__ is True


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
    assert SetRole.__confluid_group__ == "structure"
    assert SelectFields.__confluid_group__ == "structure"
    assert PrintSampleOp.__confluid_group__ == "debug"
    assert MetadataToTarget.__confluid_group__ == "structure"
    assert EncodeTarget.__confluid_group__ == "structure"
    assert DecodeTarget.__confluid_group__ == "structure"
    assert CocoToTorchVisionDetection.__confluid_group__ == "structure"
    assert MasksToDetectionBoxes.__confluid_group__ == "structure"
    assert Parallel.__confluid_group__ == "compose"
    assert Enable.__confluid_group__ == "compose"
    assert TransformChain.__confluid_group__ == "compose"
    assert ConfigureOp.__confluid_group__ == "compose"
    assert FormulaOp.__confluid_group__ == "compose"
    assert SampleSinkOp.__confluid_group__ == "sink"
    assert AlbumentationsOp.__confluid_group__ == "augment"
    assert TorchvisionTransformOp.__confluid_group__ == "augment"


def test_categories_enumerable_via_registry() -> None:
    registry = get_registry()
    assert {"Flux", "JointFlux"} <= registry.list_classes(category="engine")
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
        "SampleSinkOp",
        "MetadataToTarget",
        "EncodeTarget",
        "DecodeTarget",
        "CocoToTorchVisionDetection",
        "MasksToDetectionBoxes",
        "TransformChain",
    } <= registry.list_classes(category="op")
    assert {"HDF5Sink", "ZarrGroupSink", "ZarrBatchSink", "DirectorySink"} <= registry.list_classes(category="sink")
    assert "SampleSinkOp" not in registry.list_classes(category="sink")


def test_groups_enumerable_via_registry() -> None:
    registry = get_registry()
    assert {"Threshold", "ConnectedComponents"} <= registry.list_classes(group="numpy")
    assert {"ToTensor"} <= registry.list_classes(group="torch")
    assert {"ConvertToImage"} <= registry.list_classes(group="image")
    assert {"Parallel", "Enable", "TransformChain"} <= registry.list_classes(group="compose")
    assert {"SampleSinkOp"} <= registry.list_classes(group="sink")
    assert {
        "MetadataToTarget",
        "EncodeTarget",
        "DecodeTarget",
        "CocoToTorchVisionDetection",
        "MasksToDetectionBoxes",
        "SetRole",
        "SelectFields",
    } <= registry.list_classes(group="structure")
    assert {"AlbumentationsOp", "TorchvisionTransformOp"} <= registry.list_classes(group="augment")
    assert "TransformChain" in registry.list_classes(category="op", group="compose")
