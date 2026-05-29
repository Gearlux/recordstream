# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for dataflux ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
MCP tool and, downstream, the visual-editor form-spec picker (``get_node_form_spec``).
A class silently losing its category empties the relevant picker, so the tags are
pinned here as a regression gate.
"""

from confluid.registry import get_registry

from dataflux.core import FilterOp, Flux, JointFlux, WrappedOp
from dataflux.sources import DatasetSplit, HuggingFaceSource


def test_engine_classes_tagged() -> None:
    """``Flux`` / ``JointFlux`` / ``DatasetSplit`` are the generic, task-agnostic *engines*
    (containers that compose sources + ops), NOT datasets. They are excluded from FluxStudio's
    node palette (you wire Source -> Op instead) and are not offered in navigaitor dataset slots."""
    assert Flux.__confluid_category__ == "engine"
    assert JointFlux.__confluid_category__ == "engine"
    assert DatasetSplit.__confluid_category__ == "engine"


def test_source_classes_tagged() -> None:
    """``HuggingFaceSource`` is a concrete data *source* (it loads a dataset)."""
    assert HuggingFaceSource.__confluid_category__ == "source"


def test_op_classes_tagged() -> None:
    """``FilterOp`` / ``WrappedOp`` are pipeline ops."""
    assert FilterOp.__confluid_category__ == "op"
    assert WrappedOp.__confluid_category__ == "op"


def test_categories_enumerable_via_registry() -> None:
    """Importing the classes registers them; the category index must surface them.

    The navigaitor picker queries ``list_classes(category=...)``, so the index —
    not just the class attribute — has to carry the tag.
    """
    registry = get_registry()
    assert {"Flux", "JointFlux", "DatasetSplit"} <= registry.list_classes(category="engine")
    assert {"HuggingFaceSource"} <= registry.list_classes(category="source")
    assert {"FilterOp", "WrappedOp"} <= registry.list_classes(category="op")
