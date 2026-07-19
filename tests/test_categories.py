# mypy: disable-error-code="attr-defined"
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
from sampleflux.ops.copy import CopyInputOp
from sampleflux.ops.debug import PrintSampleOp
from sampleflux.ops.enable import Enable
from sampleflux.ops.formula import FormulaOp
from sampleflux.ops.image import ConvertToImageOp, NormalizeToUint8Op
from sampleflux.ops.metadata import DropMetadataOp
from sampleflux.ops.numpy import RescaleOp, StandardizeOp, ThresholdOp
from sampleflux.ops.parallel import Parallel
from sampleflux.ops.sink import SampleSinkOp
from sampleflux.ops.stash import StashTargetOp, UnstashTargetOp
from sampleflux.ops.target import (
    CocoToTorchVisionDetectionOp,
    DecodeTargetOp,
    EncodeTargetOp,
    MasksToDetectionBoxesOp,
    MetadataToTargetOp,
)
from sampleflux.ops.torch import ToTensorOp
from sampleflux.ops.torchvision import TorchvisionTransformOp
from sampleflux.ops.transform_chain import TransformChain
from sampleflux.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource
from sampleflux.storage.directory import DirectorySink
from sampleflux.storage.hdf5 import HDF5Sink, HDF5Source
from sampleflux.storage.zarr import ZarrBatchSink, ZarrGroupSink


def test_engine_classes_tagged() -> None:
    """The generic, task-agnostic *engines* — composition primitives that compose sources + ops.

    ``Flux`` / ``JointFlux`` carry ``category="engine"``. They (and ``DatasetSplit``, now a
    ``source``) are canvas-composable in FluxStudio: its allowlist now includes ``engine`` and the
    source-typed constructor params (``source`` / ``fluxes`` / ``ops``) render as wired sockets."""
    assert Flux.__confluid_category__ == "engine"
    assert JointFlux.__confluid_category__ == "engine"


def test_raw_callable_wrappers_uncategorised() -> None:
    """``FilterOp`` / ``WrappedOp`` wrap a *raw Python callable*, so they are neither an ``op``
    (nothing to wire) nor an ``engine`` — they carry NO category (bare ``@configurable``) and are
    excluded from FluxStudio by being uncategorised, like a module-level helper function. So even
    once ``engine`` is added to the allowlist these wrappers stay out (correct — they're not buildable)."""
    assert getattr(FilterOp, "__confluid_category__", None) is None
    assert getattr(WrappedOp, "__confluid_category__", None) is None
    # Still registered/configurable, just untagged.
    assert FilterOp.__confluid_configurable__ is True
    assert WrappedOp.__confluid_configurable__ is True


def test_source_classes_tagged() -> None:
    """``HuggingFaceSource`` is a concrete data *source* (it loads a dataset).

    ``DatasetSplit`` / ``RangeSource`` / ``ConcatSource`` are also ``source``s: they yield
    ``Sample``s and are wired into a trainer's ``source:`` slot, each exposing a derived *view*
    of other source(s) (split / contiguous slice / concatenation) — they apply no ops, so they
    are sources, not engines."""
    assert HuggingFaceSource.__confluid_category__ == "source"
    assert DatasetSplit.__confluid_category__ == "source"
    assert RangeSource.__confluid_category__ == "source"
    assert ConcatSource.__confluid_category__ == "source"


def test_op_classes_tagged() -> None:
    """Concrete ``Sample → Sample`` ops carry ``category="op"`` (the FluxStudio op-node allowlist)."""
    assert RescaleOp.__confluid_category__ == "op"
    assert StandardizeOp.__confluid_category__ == "op"
    assert ThresholdOp.__confluid_category__ == "op"
    assert Enable.__confluid_category__ == "op"
    assert TransformChain.__confluid_category__ == "op"
    assert SampleSinkOp.__confluid_category__ == "op"
    assert MetadataToTargetOp.__confluid_category__ == "op"
    assert EncodeTargetOp.__confluid_category__ == "op"
    assert DecodeTargetOp.__confluid_category__ == "op"
    assert CocoToTorchVisionDetectionOp.__confluid_category__ == "op"
    assert MasksToDetectionBoxesOp.__confluid_category__ == "op"
    assert ConfigureOp.__confluid_category__ == "op"
    assert FormulaOp.__confluid_category__ == "op"
    assert AlbumentationsOp.__confluid_category__ == "op"
    assert TorchvisionTransformOp.__confluid_category__ == "op"


def test_augmentation_adapters_random_tagged() -> None:
    """The library-augmentation adapters are stochastic (the wrapped library draws its own
    random parameters per call), so they carry ``random=True`` — the confluid mark UIs use
    to inject cache-busting (e.g. FluxStudio's ``IS_CHANGED``)."""
    assert AlbumentationsOp.__confluid_random__ is True
    assert TorchvisionTransformOp.__confluid_random__ is True


def test_storage_sink_classes_tagged() -> None:
    """The SampleFlux storage SINKS carry ``category="sink"`` so FluxStudio surfaces them as
    ``DatasetProcessor`` sink nodes (``SAMPLEFLUX_OBJECT:sink``). Their matching SOURCES stay
    UNcategorised — they read a sink's layout back via YAML ``!class:``, they are not canvas nodes.
    (``SampleSinkOp`` is the op-FORM sink, ``category="op"`` — a different thing, asserted above.)"""
    assert HDF5Sink.__confluid_category__ == "sink"
    assert ZarrGroupSink.__confluid_category__ == "sink"
    assert ZarrBatchSink.__confluid_category__ == "sink"
    assert DirectorySink.__confluid_category__ == "sink"
    # The matching source is NOT tagged, so the positive allowlist surfaces only the sink half.
    assert getattr(HDF5Source, "__confluid_category__", None) is None


def test_op_group_tags() -> None:
    """Ops carry a path-like ``group`` (FluxStudio palette nesting: Taidal/SampleFlux/Op/<group>).

    Presentation-only — orthogonal to the category that gates discovery. A renamed/dropped group
    re-files the node in the palette but never hides it; pinned so the taxonomy is a regression gate."""
    assert RescaleOp.__confluid_group__ == "numpy"
    assert StandardizeOp.__confluid_group__ == "numpy"
    assert ThresholdOp.__confluid_group__ == "numpy"
    assert ToTensorOp.__confluid_group__ == "torch"
    assert CopyInputOp.__confluid_group__ == "structure"
    assert DropMetadataOp.__confluid_group__ == "structure"
    assert PrintSampleOp.__confluid_group__ == "debug"
    assert StashTargetOp.__confluid_group__ == "structure"
    assert UnstashTargetOp.__confluid_group__ == "structure"
    assert MetadataToTargetOp.__confluid_group__ == "structure"
    assert EncodeTargetOp.__confluid_group__ == "structure"
    assert DecodeTargetOp.__confluid_group__ == "structure"
    assert CocoToTorchVisionDetectionOp.__confluid_group__ == "structure"
    assert MasksToDetectionBoxesOp.__confluid_group__ == "structure"
    assert Parallel.__confluid_group__ == "compose"
    assert Enable.__confluid_group__ == "compose"
    assert TransformChain.__confluid_group__ == "compose"
    assert ConfigureOp.__confluid_group__ == "compose"
    assert FormulaOp.__confluid_group__ == "compose"
    assert ConvertToImageOp.__confluid_group__ == "image"
    assert NormalizeToUint8Op.__confluid_group__ == "image"
    assert SampleSinkOp.__confluid_group__ == "sink"
    assert AlbumentationsOp.__confluid_group__ == "augment"
    assert TorchvisionTransformOp.__confluid_group__ == "augment"


def test_categories_enumerable_via_registry() -> None:
    """Importing the classes registers them; the category index must surface them.

    The navigaitor picker queries ``list_classes(category=...)``, so the index —
    not just the class attribute — has to carry the tag.
    """
    registry = get_registry()
    assert {"Flux", "JointFlux"} <= registry.list_classes(category="engine")
    # DatasetSplit is a source now, not an engine.
    assert "DatasetSplit" not in registry.list_classes(category="engine")
    # FilterOp / WrappedOp are uncategorised, so they appear in NO category index.
    assert not ({"FilterOp", "WrappedOp"} & registry.list_classes(category="engine"))
    assert {"HuggingFaceSource", "DatasetSplit", "RangeSource", "ConcatSource"} <= registry.list_classes(
        category="source"
    )
    assert {
        "RescaleOp",
        "StandardizeOp",
        "ThresholdOp",
        "Enable",
        "SampleSinkOp",
        "MetadataToTargetOp",
        "EncodeTargetOp",
        "DecodeTargetOp",
        "CocoToTorchVisionDetectionOp",
        "MasksToDetectionBoxesOp",
        "TransformChain",
    } <= registry.list_classes(category="op")
    # The storage sinks surface under the NEW "sink" category index (FluxStudio's allowlist + the
    # navigaitor sink picker). SampleSinkOp is category="op", so it is NOT here.
    assert {"HDF5Sink", "ZarrGroupSink", "ZarrBatchSink", "DirectorySink"} <= registry.list_classes(category="sink")
    assert "SampleSinkOp" not in registry.list_classes(category="sink")


def test_groups_enumerable_via_registry() -> None:
    """The registry's group index must surface the tagged ops (``list_classes(group=...)``)."""
    registry = get_registry()
    assert {
        "RescaleOp",
        "StandardizeOp",
        "ThresholdOp",
    } <= registry.list_classes(group="numpy")
    # The FFT ops exist in BOTH framework groups (a numpy + a torch variant under the one name,
    # exactly like RescaleOp/StandardizeOp), so they surface under the torch group too.
    assert {"ToTensorOp"} <= registry.list_classes(group="torch")
    assert {"ConvertToImageOp", "NormalizeToUint8Op"} <= registry.list_classes(group="image")
    assert {"Parallel", "Enable", "TransformChain"} <= registry.list_classes(group="compose")
    assert {"SampleSinkOp"} <= registry.list_classes(group="sink")
    assert {
        "MetadataToTargetOp",
        "EncodeTargetOp",
        "DecodeTargetOp",
        "CocoToTorchVisionDetectionOp",
        "MasksToDetectionBoxesOp",
    } <= registry.list_classes(group="structure")
    assert {"AlbumentationsOp", "TorchvisionTransformOp"} <= registry.list_classes(group="augment")
    # group × category intersect, like task × role.
    assert "TransformChain" in registry.list_classes(category="op", group="compose")
