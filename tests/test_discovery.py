import sys
from pathlib import Path
from typing import Any

import pytest

from dataflux.discovery import (
    get_callable_path,
    introspect_callable,
    resolve_callable,
    scan_module,
)


def sample_func(a: int, b: str = "default") -> str:
    """Sample docstring."""
    return f"{a}-{b}"


class SampleClass:
    def __init__(self, x: float) -> None:
        self.x = x

    def method(self, y: int) -> float:
        return self.x + y


def test_get_callable_path() -> None:
    path = get_callable_path(sample_func)
    assert "test_discovery:sample_func" in path

    path_cls = get_callable_path(SampleClass)
    assert "test_discovery:SampleClass" in path_cls


def test_get_callable_path_errors() -> None:
    with pytest.raises(TypeError):
        get_callable_path("not a callable")  # type: ignore

    # lambda in some environments might fail to resolve module
    # But usually it has __module__ == '__main__' or similar.
    # Let's try to mock determine path failure
    pass


def test_resolve_callable_errors() -> None:
    # Test ImportError
    with pytest.raises(ImportError):
        resolve_callable("nonexistent_mod:func")

    # Test AttributeError
    with pytest.raises(AttributeError):
        resolve_callable("dataflux.discovery:nonexistent_func")


def test_introspect_errors() -> None:
    # Introspect something that isn't a callable should return {}
    assert introspect_callable(123) == {}  # type: ignore


def test_scan_module_errors() -> None:
    # Nonexistent module
    assert scan_module("nonexistent_package_123") == []

    # Nonexistent file
    assert scan_module(Path("nonexistent_file.py")) == []


def local_func() -> None:
    pass


def test_get_callable_path_main(monkeypatch: Any) -> None:
    # Mock __main__
    import types

    m = types.ModuleType("__main__")
    m.__file__ = "myscript.py"
    monkeypatch.setitem(sys.modules, "__main__", m)

    # We need to set the module of our function to __main__
    # And it must be at top level to avoid .<locals>.
    orig_mod = local_func.__module__
    monkeypatch.setattr(local_func, "__module__", "__main__")

    try:
        path = get_callable_path(local_func)
        assert "myscript.py:local_func" in path
    finally:
        local_func.__module__ = orig_mod


def test_resolve_callable_direct() -> None:
    # Test non-string non-callable returns self
    assert resolve_callable(123) == 123  # type: ignore
    path = get_callable_path(sample_func)
    resolved = resolve_callable(path)
    assert resolved == sample_func

    # Test nested resolution
    path_method = f"{SampleClass.__module__}:SampleClass.method"
    # Note: resolving a method on a class (unbound)
    resolved_method = resolve_callable(path_method)
    assert resolved_method == SampleClass.method

    # Test already a callable
    assert resolve_callable(sample_func) == sample_func

    # Test non-string/no-colon
    assert resolve_callable("simple_string") == "simple_string"


def test_introspect_callable() -> None:
    schema = introspect_callable(sample_func)
    assert schema["name"] == "sample_func"
    assert schema["doc"] == "Sample docstring."
    assert len(schema["parameters"]) == 2

    p0 = schema["parameters"][0]
    assert p0["name"] == "a"
    assert "int" in p0["type"]
    assert p0["required"] is True

    p1 = schema["parameters"][1]
    assert p1["name"] == "b"
    assert p1["default"] == "default"
    assert p1["required"] is False


def test_resolve_callable_script(tmp_path: Path) -> None:
    # hits lines 52-57 (loading from .py file)
    script_path = tmp_path / "dynamic.py"
    script_path.write_text("def my_dynamic_func(): return 42")

    path = f"{script_path}:my_dynamic_func"
    resolved = resolve_callable(path)
    assert resolved() == 42


def test_introspect_signature_error() -> None:
    # hits line 88
    # inspect.signature fails on some builtins
    assert introspect_callable(iter) == {}


def test_scan_module(tmp_path: Path) -> None:
    # Create a dummy script
    script_path = tmp_path / "my_script.py"
    script_path.write_text("""
def func_in_script(x: int) -> int:
    return x * 2

class ClassInScript:
    pass
""")

    schemas = scan_module(script_path)
    names = [s["name"] for s in schemas]
    assert "func_in_script" in names
    assert "ClassInScript" in names

    # Standard module scan
    schemas_self = scan_module("dataflux.discovery")
    names_self = [s["name"] for s in schemas_self]
    assert "scan_module" in names_self
    assert "get_callable_path" in names_self
