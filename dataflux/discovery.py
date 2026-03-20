import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union


def get_callable_path(func: Callable) -> str:
    """
    Convert a function/class into an importable string path.
    Avoids '__main__' by resolving the script's filename.
    """
    if not callable(func):
        raise TypeError(f"Object {func} is not callable")

    module_name = getattr(func, "__module__", None)
    name = getattr(func, "__qualname__", getattr(func, "__name__", None))

    if module_name == "__main__":
        try:
            main_module = sys.modules["__main__"]
            file_path = Path(main_module.__file__).resolve()  # type: ignore
            if file_path.parent == Path.cwd():
                module_name = file_path.name
            else:
                module_name = file_path.name
        except (AttributeError, KeyError):
            pass

    if module_name is None or name is None:
        raise ValueError(f"Could not determine path for {func}")

    return f"{module_name}:{name}"


def resolve_callable(path: Union[str, Callable]) -> Callable:
    """Resolve an importable string path back into a callable."""
    if callable(path):
        return path

    if not isinstance(path, str) or ":" not in path:
        return path  # type: ignore

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
        func = mod
        for part in parts:
            func = getattr(func, part)
        return func  # type: ignore
    except AttributeError as e:
        raise AttributeError(f"Module '{mod_name}' has no attribute '{func_name}': {e}")


def introspect_callable(func: Callable) -> Dict[str, Any]:
    """
    Build a JSON-serializable schema for a callable.
    Used by FluxStudio to render nodes and property panels.
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
            "type": str(param.annotation) if param.annotation is not inspect.Parameter.empty else "Any",
            "default": str(param.default) if param.default is not inspect.Parameter.empty else None,
            "required": param.default is inspect.Parameter.empty,
        }
        params.append(param_info)

    return {
        "path": get_callable_path(func),
        "name": getattr(func, "__name__", str(func)),
        "doc": func.__doc__.strip() if func.__doc__ else "",
        "parameters": params,
    }


def scan_module(path_or_name: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Scan a module or script file and return schemas for locally callables.
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
