"""Layout invariants of the engine PACKAGES — `sources`, `core`, `flow`.

Each was one oversized module and is now one module per cohesive unit (architecture §11/§12).
That split made three things load-bearing that a casual tidy-up would undo, and every one of
them fails SILENTLY in production:

* the SUBMODULE path is the canonical ``!class:`` spelling, because ``cls.__module__`` is what
  every generator writes — so nobody may pin ``__module__`` back to the package (that also
  breaks ``confluid.registry.key_for``, whose miss only surfaces once a namesake registers);
* ``__init__.py``'s ``__all__`` is what a visual editor's node bridge walks, because
  ``discovery.scan_module`` filters on ``__module__`` and now sees nothing in a package;
* the IMPORT DIRECTION inside ``core`` / ``flow`` — ``core`` is the bottom of the op-facing
  layer (architecture §5), and a top-level import pointing the wrong way closes a cycle.
"""

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any, List, Set

import pytest
from confluid.pydantic_export import _qualname
from confluid.registry import get_registry, resolve_class

import recordstream.core as core_pkg
import recordstream.flow as flow_pkg
import recordstream.sources as sources_pkg
from recordstream.core import FilterOp, JointStream, Stream, WrappedOp
from recordstream.discovery import scan_module
from recordstream.flow import FlowGraph
from recordstream.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource

#: Every public engine class and the module it must be DEFINED in (not merely re-exported from).
CLASS_MODULES = {
    HuggingFaceSource: "recordstream.sources.huggingface",
    DatasetSplit: "recordstream.sources.split",
    RangeSource: "recordstream.sources.range",
    ConcatSource: "recordstream.sources.concat",
    Stream: "recordstream.core.stream",
    JointStream: "recordstream.core.stream",
    FilterOp: "recordstream.core.wrappers",
    WrappedOp: "recordstream.core.wrappers",
    FlowGraph: "recordstream.flow.graph",
}

#: package -> the submodules whose @configurable classes it must re-export.
PACKAGES = {
    sources_pkg: ["huggingface", "split", "range", "concat"],
    core_pkg: ["families", "mapstyle", "wrappers", "stream"],
    flow_pkg: ["steps", "parse", "execute", "graph"],
}


def _module_level_imports(module_name: str) -> Set[str]:
    """The modules ``module_name`` imports at MODULE level (body-local imports excluded).

    Body-local imports are the sanctioned seam for the one direction that must stay lazy, so
    only top-level statements count here.
    """
    source = Path(inspect.getsourcefile(importlib.import_module(module_name)) or "").read_text()
    imported: Set[str] = set()
    for node in ast.parse(source).body:  # top level only — never recurse into function bodies
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


@pytest.mark.parametrize("cls,module", list(CLASS_MODULES.items()), ids=lambda v: getattr(v, "__name__", v))
def test_each_class_lives_in_its_declared_module(cls: type, module: str) -> None:
    """``__module__`` reports the defining submodule — never the package."""
    assert cls.__module__ == module
    # Pinning __module__ back to the package would make the assert above pass by lying;
    # getsource is what catches that (it searches the named module's file for the definition).
    assert inspect.getsource(cls).lstrip().startswith("@configurable")


@pytest.mark.parametrize("cls,module", list(CLASS_MODULES.items()), ids=lambda v: getattr(v, "__name__", v))
def test_the_canonical_class_path_is_the_submodule_one(cls: type, module: str) -> None:
    """``_qualname`` is what a generated config / form-spec / enrichment key spells out."""
    assert _qualname(cls) == f"{module}.{cls.__name__}"
    assert resolve_class(_qualname(cls)) is cls


@pytest.mark.parametrize("cls", list(CLASS_MODULES), ids=lambda v: v.__name__)
def test_the_package_re_export_still_resolves(cls: type) -> None:
    """An older hand-written config spelling the PACKAGE path keeps loading."""
    package = cls.__module__.rsplit(".", 1)[0]
    assert resolve_class(f"{package}.{cls.__name__}") is cls


@pytest.mark.parametrize("cls", list(CLASS_MODULES), ids=lambda v: v.__name__)
def test_pinning_module_would_break_the_registry_lookup(cls: type) -> None:
    """``key_for`` re-derives ``module.qualname``; a rewritten ``__module__`` misses the entry.

    This is the failure mode that ruled out keeping the old paths by pinning ``__module__`` —
    it is silent, surfacing only as an ambiguous ``!class:`` tag once a namesake registers.
    """
    assert get_registry().key_for(cls) is not None


@pytest.mark.parametrize("package,submodules", list(PACKAGES.items()), ids=lambda v: getattr(v, "__name__", ""))
def test_every_configurable_is_in_all(package: Any, submodules: List[str]) -> None:
    """``__all__`` is the ONLY pass that surfaces these as palette nodes — see the module docstring."""
    exported = getattr(package, "__all__", [])
    for name in submodules:
        module_path = f"{package.__name__}.{name}"
        module = importlib.import_module(module_path)
        for attr, member in vars(module).items():
            if attr.startswith("_") or not isinstance(member, type):
                continue
            if getattr(member, "__confluid_configurable__", False) and member.__module__ == module_path:
                assert attr in exported, f"{module_path}.{attr} is @configurable but not in __all__"


@pytest.mark.parametrize("package", list(PACKAGES), ids=lambda v: getattr(v, "__name__", ""))
def test_scan_module_no_longer_sees_the_package(package: Any) -> None:
    """The reason ``__all__`` matters: the ``__module__`` filter finds nothing in a package."""
    assert scan_module(package.__name__) == []


def test_the_engine_internals_stay_importable_from_the_package() -> None:
    """``ops/`` reaches the op dispatch as ``from recordstream.core import _apply_op``.

    The private re-exports in ``core/__init__.py`` are a deliberate cross-module surface, not
    stray imports — dropping one breaks every composing op at run time, not at import.
    """
    for name in ("_apply_op", "_op_expands", "_expand", "_extra_op_families", "_sync_op_families", "_worker_task"):
        assert hasattr(core_pkg, name), f"recordstream.core.{name} is the engine's internal surface"
    # A mutable list re-exported by IDENTITY — this is what lets the suite's registry fixture
    # restore the real families with `core._OP_FAMILIES[:] = snapshot`.
    from recordstream.core.families import _OP_FAMILIES

    assert core_pkg._OP_FAMILIES is _OP_FAMILIES


def test_core_families_is_the_bottom_of_the_op_facing_layer() -> None:
    """``core.families`` imports NOTHING from its siblings — architecture §5's whole argument.

    Every composing op imports ``_apply_op`` from here; a top-level import back into the engine
    (or into ``flow``) would close the cycle that record exists to prevent.
    """
    imported = _module_level_imports("recordstream.core.families")
    offenders = [m for m in imported if m.startswith(("recordstream.core.", "recordstream.flow", "recordstream.ops"))]
    assert offenders == [], f"core.families must not import {offenders} at module level"


def test_flow_reaches_core_only_through_the_dispatch_layer() -> None:
    """``flow`` may import ``core.families``; importing ``core.stream`` would invert the layering.

    ``core.stream`` reaches ``flow`` through BODY-LOCAL imports precisely so this direction can
    stay a module-level one.
    """
    for name in ("steps", "parse", "execute", "graph"):
        imported = _module_level_imports(f"recordstream.flow.{name}")
        assert "recordstream.core.stream" not in imported, f"flow.{name} must not import core.stream at module level"
        assert "recordstream.core" not in imported, f"flow.{name} must reach the dispatch via core.families"


def test_core_stream_defers_its_flow_imports() -> None:
    """The one direction that MUST stay lazy: ``core.stream`` -> ``flow`` is body-local only."""
    imported = _module_level_imports("recordstream.core.stream")
    offenders = [m for m in imported if m.startswith("recordstream.flow")]
    assert offenders == [], f"core.stream must import flow inside function bodies, not {offenders}"
