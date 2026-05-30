# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for dataflux ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
MCP tool and, downstream, the visual-editor form-spec picker (``get_node_form_spec``).
A class silently losing its category empties the relevant picker, so the tags are
pinned here as a regression gate.
"""

from confluid.registry import get_registry

from dataflux.core import FilterOp, Flux, JointFlux, WrappedOp
from dataflux.ops.numpy import RescaleOp, StandardizeOp, ThresholdOp
from dataflux.ops.tee import Tee
from dataflux.sources import DatasetSplit, HuggingFaceSource


def test_engine_classes_tagged() -> None:
    """The generic, task-agnostic *engines* / composition primitives — NOT GUI-buildable nodes.

    ``Flux`` / ``JointFlux`` / ``DatasetSplit`` compose sources + ops; ``FilterOp`` / ``WrappedOp``
    wrap a raw Python callable. All carry ``category="engine"`` so FluxStudio's positive
    op/source/dataset allowlist excludes them (you wire Source → Op instead)."""
    assert Flux.__confluid_category__ == "engine"
    assert JointFlux.__confluid_category__ == "engine"
    assert DatasetSplit.__confluid_category__ == "engine"
    assert FilterOp.__confluid_category__ == "engine"
    assert WrappedOp.__confluid_category__ == "engine"


def test_source_classes_tagged() -> None:
    """``HuggingFaceSource`` is a concrete data *source* (it loads a dataset)."""
    assert HuggingFaceSource.__confluid_category__ == "source"


def test_op_classes_tagged() -> None:
    """Concrete ``Sample → Sample`` ops carry ``category="op"`` (the FluxStudio op-node allowlist)."""
    assert RescaleOp.__confluid_category__ == "op"
    assert StandardizeOp.__confluid_category__ == "op"
    assert ThresholdOp.__confluid_category__ == "op"
    assert Tee.__confluid_category__ == "op"


def test_categories_enumerable_via_registry() -> None:
    """Importing the classes registers them; the category index must surface them.

    The navigaitor picker queries ``list_classes(category=...)``, so the index —
    not just the class attribute — has to carry the tag.
    """
    registry = get_registry()
    assert {"Flux", "JointFlux", "DatasetSplit", "FilterOp", "WrappedOp"} <= registry.list_classes(category="engine")
    assert {"HuggingFaceSource"} <= registry.list_classes(category="source")
    assert {"RescaleOp", "StandardizeOp", "ThresholdOp", "Tee"} <= registry.list_classes(category="op")
