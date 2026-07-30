"""torch is an EXTRA — the engine's core is numpy.

The claim is not "recordstream avoids torch" (`ToTensor` and the output builders need it); it is
that **importing recordstream imports no ML framework**, so a Keras-only, TensorFlow-only or
plain-numpy consumer does not install ~2GB it never calls.

The load-bearing test here is :func:`test_importing_recordstream_works_with_torch_blocked`, which
proves it in a SUBPROCESS with torch made unimportable. The rest guard the three mechanisms that
make that true, each of which was a real coupling before: the `Dataset` base class, the eager
`ToTensor` export, and the module-level `isinstance(x, torch.Tensor)` checks.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from recordstream import Stream
from recordstream._compat import is_torch_tensor
from recordstream.core import MapStyle

# --------------------------------------------------------------------------- #
# The whole claim, executed
# --------------------------------------------------------------------------- #

_IMPORT_WITH_TORCH_BLOCKED = textwrap.dedent(
    """
    import sys

    class _NoTorch:
        \"\"\"A meta-path finder that makes torch unimportable, simulating an install without it.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is not installed (blocked by this test)")
            return None

    sys.meta_path.insert(0, _NoTorch())

    import recordstream
    from recordstream import Stream, LabelMap, collate_records, multi_hot, batch_values

    # Not just the package: the surfaces a non-torch backend actually uses must work too.
    stream = Stream(source=[{"class": 1}, {"class": 0}])
    assert len(stream) == 2
    assert [r["class"] for r in stream] == [1, 0]

    leaked = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
    print("LEAKED:" + ",".join(leaked) if leaked else "CLEAN")
    """
)


def test_importing_recordstream_works_with_torch_blocked(tmp_path: object) -> None:
    """`import recordstream` must not need torch, and must not import it as a side effect.

    Run in a subprocess because this process has already imported torch — the check is only
    meaningful in an interpreter where it was never available. `cwd` is a temp dir so the import
    resolves to the INSTALLED package, not a same-named source directory (PEP 420 shadowing).
    """
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITH_TORCH_BLOCKED],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )

    assert result.returncode == 0, f"importing recordstream without torch failed:\n{result.stderr}"
    assert "CLEAN" in result.stdout, f"torch was imported anyway: {result.stdout.strip()}"


# --------------------------------------------------------------------------- #
# 1. The `Dataset` base — dropped, because a DataLoader never needed it
# --------------------------------------------------------------------------- #


def test_stream_does_not_inherit_torchs_dataset() -> None:
    """The regression guard: re-adding the base would silently make torch mandatory again.

    Nothing in the workspace does `isinstance(x, Dataset)` or subclasses `Stream`, so the base
    bought nothing but the import.
    """
    import torch.utils.data

    assert not issubclass(Stream, torch.utils.data.Dataset)


def test_stream_is_map_style_which_is_all_a_dataloader_needs() -> None:
    """`DataLoader` duck-types its argument — `__len__` + `__getitem__` IS the contract."""
    stream = Stream(source=[{"class": i} for i in range(4)])

    assert isinstance(stream, MapStyle)
    assert len(stream) == 4 and stream[2]["class"] == 2


def test_a_dataloader_actually_accepts_a_stream() -> None:
    """The duck-typing claim, executed rather than asserted about.

    `cast(Any, ...)` at the call site is the accepted cost: torch's STUB still declares
    `Dataset[T]`, so type checkers reject the runtime-valid call.
    """
    from typing import Any, cast

    import torch.utils.data

    stream = Stream(source=[{"class": i} for i in range(4)])
    loader = torch.utils.data.DataLoader(cast(Any, stream), batch_size=2)

    assert len(list(loader)) == 2


# --------------------------------------------------------------------------- #
# 2. `ToTensor` — reachable, but never eagerly imported
# --------------------------------------------------------------------------- #


def test_to_tensor_is_not_imported_eagerly_but_is_still_reachable() -> None:
    import recordstream.ops as ops

    assert "ToTensor" in ops._OPTIONAL_OPS, "the lazy export must stay registered"
    assert ops.ToTensor is not None
    assert "ToTensor" in dir(ops), "TAB completion / dir() must still advertise it"


def test_a_missing_optional_op_names_the_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """The error an operator without the extra sees — not a traceback from three libraries down."""
    import importlib

    import recordstream.ops as ops

    def _fail(name: str) -> None:
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(importlib, "import_module", _fail)

    with pytest.raises(ImportError, match=r"recordstream\[torch\]"):
        ops.__getattr__("ToTensor")


def test_an_unknown_attribute_still_raises_attribute_error() -> None:
    """The lazy `__getattr__` must not turn every typo into an ImportError."""
    import recordstream.ops as ops

    with pytest.raises(AttributeError):
        ops.__getattr__("NoSuchOp")


# --------------------------------------------------------------------------- #
# 3. Recognising a tensor without importing torch
# --------------------------------------------------------------------------- #


def test_is_torch_tensor_recognizes_a_real_tensor() -> None:
    import torch

    assert is_torch_tensor(torch.zeros(3))


@pytest.mark.parametrize("value", [np.zeros(3), [1, 2, 3], "not a tensor", None, 7])
def test_is_torch_tensor_rejects_everything_else(value: object) -> None:
    assert not is_torch_tensor(value)


def test_is_torch_tensor_is_false_when_torch_was_never_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mechanism: a torch tensor cannot exist unless torch is loaded, so `sys.modules` is EXACT.

    With torch hidden the answer must be False without any attempt to import it — which is what
    makes this usable at module level in a package that does not depend on torch.
    """
    import torch

    monkeypatch.delitem(sys.modules, "torch")
    tensor = torch.zeros(3)  # a real tensor, while the module is hidden

    assert not is_torch_tensor(tensor)


# --------------------------------------------------------------------------- #
# 4. The packaging half — the code above is only true if the metadata agrees
# --------------------------------------------------------------------------- #


def test_pyproject_declares_torch_as_an_extra_not_a_dependency() -> None:
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]

    required = [d for d in project["dependencies"] if d.split(">")[0].split("=")[0].strip() == "torch"]
    assert not required, f"torch must not be a hard dependency, found: {required}"
    assert any("torch" in d for d in project["optional-dependencies"]["torch"])


# --------------------------------------------------------------------------- #
# 5. keras is an extra too — and DISCOVERY must not pull it either
# --------------------------------------------------------------------------- #

_SCAN_WITHOUT_KERAS = textwrap.dedent(
    """
    import sys

    import recordstream
    from recordstream.discovery import scan_module

    scan_module("recordstream")          # what a GUI bridge / MCP bootstrap does
    scan_module("recordstream.ops")

    leaked = sorted(m for m in sys.modules if m == "keras" or m.startswith("keras."))
    print("LEAKED:" + ",".join(leaked) if leaked else "CLEAN")
    """
)


def test_neither_importing_nor_scanning_recordstream_imports_keras(tmp_path: object) -> None:
    """Why `RecordSequence` lives in `recordstream.keras` and NOT at the package root.

    `inspect.getmembers` — inside `scan_module`, and in the GUI bridges — getattrs every name a
    module advertises, so a PEP 562 lazy root export (the `ops.ToTensor` pattern) would import
    keras on every discovery scan, including on installs that never asked for it. Run in a
    subprocess because this one has already imported keras via the sequence tests.
    """
    result = subprocess.run(
        [sys.executable, "-c", _SCAN_WITHOUT_KERAS],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )

    assert result.returncode == 0, f"scanning recordstream failed:\n{result.stderr}"
    assert "CLEAN" in result.stdout, f"keras was imported by a discovery scan: {result.stdout.strip()}"


def test_pyproject_declares_keras_as_an_extra_naming_no_compute_engine() -> None:
    """Keras 3 is an API, not a runtime: the extra must not drag torch/TF/jax in — the consumer's
    own extra picks one, and `recordstream.keras` defaults to whichever is installed."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]

    assert any("keras" in d for d in extras["keras"])
    engines = [d for d in extras["keras"] if d.split(">")[0].split("=")[0].strip() in ("torch", "tensorflow", "jax")]
    assert not engines, f"the keras extra must name no compute engine, found: {engines}"
