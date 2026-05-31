"""Passive introspection for DataFlux callables.

A round-trip bridge between live Python callables (sources, ops, plain
functions) and JSON-serializable schemas, so downstream tools can discover and
wire pipeline pieces without any hand-written tool definitions (the workspace's
"Passive Introspection" mandate). Two halves:

* **Serialization** (callable <-> string): :func:`get_callable_path` turns a
  callable into an importable ``"module:qualname"`` key and
  :func:`resolve_callable` imports it back. This is how a pipeline step is
  referenced in a Confluid manifest and resurrected later for reproducibility.
* **Discovery** (callable -> JSON schema): :func:`introspect_callable` reflects
  a single callable into a schema (signature + docstring + the ``ACCEPTS`` /
  ``PRODUCES`` typespec contract), and :func:`scan_module` does the same for
  every callable *defined in* a module. FluxStudio reads these to auto-generate
  ComfyUI nodes and their property panels; navigaitor builds its MCP form-spec
  from the same data.
"""

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast


def get_callable_path(func: Callable) -> str:
    """
    Convert a function/class into an importable string path (``"module:qualname"``).
    Avoids '__main__' by resolving the script's filename, so the path stays
    re-importable from another process.

    Use: the serialization key that lets a discovered callable be referenced in
    a Confluid manifest and rehydrated later via :func:`resolve_callable`.
    """
    if not callable(func):
        raise TypeError(f"Object {func} is not callable")

    module_name = getattr(func, "__module__", None)
    name = getattr(func, "__qualname__", getattr(func, "__name__", None))

    if module_name == "__main__":
        try:
            main_module = sys.modules["__main__"]
            main_file = getattr(main_module, "__file__", None)
            if main_file:
                file_path = Path(main_file).resolve()
                module_name = file_path.name
        except (AttributeError, KeyError):
            pass  # pragma: no cover

    if module_name is None or name is None:
        raise ValueError(f"Could not determine path for {func}")

    return f"{module_name}:{name}"


def resolve_callable(path: Union[str, Callable]) -> Callable:
    """Resolve an importable string path back into a callable (inverse of
    :func:`get_callable_path`). Handles a normal module import, a ``.py`` file
    path loaded via ``spec_from_file_location``, and a ``.py``-suffix fallback;
    an already-callable argument is returned unchanged.

    Use: rehydrating a serialized pipeline — turning the stored ``"module:func"``
    string back into the live op/source.
    """
    if callable(path):
        return path

    if not isinstance(path, str) or ":" not in path:
        raise ValueError(f"Invalid callable path format: {path}. Expected 'module:function'")

    mod_name, func_name = path.split(":", 1)

    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        if mod_name.endswith(".py") and os.path.exists(mod_name):
            spec = importlib.util.spec_from_file_location("dynamic_mod", mod_name)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                raise ImportError(f"Could not load script {mod_name}")
        else:
            try:
                mod = importlib.import_module(mod_name.replace(".py", ""))
            except ImportError:
                msg = f"Cannot resolve '{mod_name}' for path '{path}'"
                raise ImportError(msg)

    try:
        parts = func_name.split(".")
        func: Any = mod
        for part in parts:
            func = getattr(func, part)
        return cast(Callable, func)
    except AttributeError as e:
        raise AttributeError(f"Module '{mod_name}' has no attribute '{func_name}': {e}")


def introspect_callable(func: Callable) -> Dict[str, Any]:
    """
    Build a JSON-serializable schema for a callable by reflecting over its
    signature: ``path``, ``name``, ``doc``, per-parameter info (``name`` /
    ``type`` / ``default`` / ``required``, skipping ``self`` / ``cls`` /
    ``*args`` / ``**kwargs``), plus the declared ``ACCEPTS`` / ``PRODUCES``
    typespec contract when present.

    Use: FluxStudio reads this to render a node and its property-panel widgets,
    and it feeds navigaitor's MCP form-spec.
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return {}

    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls", "args", "kwargs"):
            continue

        param_info = {
            "name": name,
            "type": (str(param.annotation) if param.annotation is not inspect.Parameter.empty else "Any"),
            "default": (str(param.default) if param.default is not inspect.Parameter.empty else None),
            "required": param.default is inspect.Parameter.empty,
        }
        params.append(param_info)

    return {
        "path": get_callable_path(func),
        "name": getattr(func, "__name__", str(func)),
        "doc": func.__doc__.strip() if func.__doc__ else "",
        "parameters": params,
        "accepts": _spec_dict(getattr(func, "ACCEPTS", None)),
        "produces": _spec_dict(getattr(func, "PRODUCES", None)),
    }


def _spec_dict(spec: Any) -> Optional[Dict[str, Any]]:
    """JSON-serialize a declared ``ACCEPTS`` / ``PRODUCES`` (a :class:`~dataflux.typespec.SampleType`),
    or ``None`` when undeclared. Duck-typed so ``discovery`` needn't import ``typespec``."""
    to_dict = getattr(spec, "to_dict", None)
    return cast(Dict[str, Any], to_dict()) if callable(to_dict) else None


def scan_module(path_or_name: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Scan a module (by import name) or a ``.py`` script (by path) and return an
    :func:`introspect_callable` schema for every callable *defined in* that
    module — names merely imported into it are filtered out by checking
    ``member.__module__ == mod_name``.

    Use: the entry point for whole-module discovery — FluxStudio's ``bridge``
    calls this to auto-generate one node per source/op, fulfilling the
    "never require manual tool definitions" mandate.
    """
    is_py = isinstance(path_or_name, Path) or (isinstance(path_or_name, str) and path_or_name.endswith(".py"))

    if is_py:
        # Load as a file-based module
        mod_path = Path(path_or_name).resolve()
        if not mod_path.exists():
            return []
        # Use the filename stem so it matches the expected module name
        mod_name = mod_path.stem
        spec = importlib.util.spec_from_file_location(mod_name, str(mod_path))
        if not spec or not spec.loader:
            return []
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        # Load as a standard import
        try:
            mod = importlib.import_module(str(path_or_name))
            mod_name = mod.__name__
        except ImportError:
            return []

    schemas = []
    for name, member in inspect.getmembers(mod):
        # We only want callables defined IN this module (not imported ones)
        if callable(member):
            member_mod = getattr(member, "__module__", None)
            if member_mod == mod_name:
                schema = introspect_callable(member)
                if schema:
                    schemas.append(schema)

    return schemas
