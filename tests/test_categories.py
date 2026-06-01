# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for dataflux ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
MCP tool and, downstream, the visual-editor form-spec picker (``get_node_form_spec``).
A class silently losing its category empties the relevant picker, so the tags are
pinned here as a regression gate.
"""

from confluid.registry import get_registry

from dataflux.core import FilterOp, Flux, JointFlux, WrappedOp
from dataflux.ops.copy import CopyInputOp
from dataflux.ops.enable import Enable
from dataflux.ops.image import ConvertToImageOp, NormalizeToUint8Op
from dataflux.ops.numpy import RescaleOp, StandardizeOp, ThresholdOp
from dataflux.ops.parallel import Parallel
from dataflux.ops.sink import SampleSinkOp
from dataflux.ops.target import DecodeTargetOp, EncodeTargetOp, MetadataToTargetOp
from dataflux.ops.tee import Tee
from dataflux.ops.torch import ToTensorOp
from dataflux.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource


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
    assert Tee.__confluid_category__ == "op"
    assert Enable.__confluid_category__ == "op"
    assert SampleSinkOp.__confluid_category__ == "op"
    assert MetadataToTargetOp.__confluid_category__ == "op"
    assert EncodeTargetOp.__confluid_category__ == "op"
    assert DecodeTargetOp.__confluid_category__ == "op"


def test_op_group_tags() -> None:
    """Ops carry a path-like ``group`` (FluxStudio palette nesting: Taidal/DataFlux/Op/<group>).

    Presentation-only — orthogonal to the category that gates discovery. A renamed/dropped group
    re-files the node in the palette but never hides it; pinned so the taxonomy is a regression gate."""
    assert RescaleOp.__confluid_group__ == "numpy"
    assert StandardizeOp.__confluid_group__ == "numpy"
    assert ThresholdOp.__confluid_group__ == "numpy"
    assert ToTensorOp.__confluid_group__ == "torch"
    assert CopyInputOp.__confluid_group__ == "structure"
    assert MetadataToTargetOp.__confluid_group__ == "structure"
    assert EncodeTargetOp.__confluid_group__ == "structure"
    assert DecodeTargetOp.__confluid_group__ == "structure"
    assert Tee.__confluid_group__ == "compose"
    assert Parallel.__confluid_group__ == "compose"
    assert Enable.__confluid_group__ == "compose"
    assert ConvertToImageOp.__confluid_group__ == "image"
    assert NormalizeToUint8Op.__confluid_group__ == "image"
    assert SampleSinkOp.__confluid_group__ == "sink"


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
        "Tee",
        "Enable",
        "SampleSinkOp",
        "MetadataToTargetOp",
        "EncodeTargetOp",
        "DecodeTargetOp",
    } <= registry.list_classes(category="op")


def test_groups_enumerable_via_registry() -> None:
    """The registry's group index must surface the tagged ops (``list_classes(group=...)``)."""
    registry = get_registry()
    assert {"RescaleOp", "StandardizeOp", "ThresholdOp"} <= registry.list_classes(group="numpy")
    assert {"ConvertToImageOp", "NormalizeToUint8Op"} <= registry.list_classes(group="image")
    assert {"Tee", "Parallel", "Enable"} <= registry.list_classes(group="compose")
    assert {"SampleSinkOp"} <= registry.list_classes(group="sink")
    assert {"MetadataToTargetOp", "EncodeTargetOp", "DecodeTargetOp"} <= registry.list_classes(group="structure")
    # group × category intersect, like task × role.
    assert "Tee" in registry.list_classes(category="op", group="compose")
