"""Tests for ``Flux.from_ops_yaml`` — attach an exported ops-only YAML to a source.

The YAML shape is what ``fluxstudio.export`` emits: a ``{ops: [!class:...()]}`` document.
The key behaviour under test is that the helper **materializes** the deferred ``!class:``
markers (which ``confluid.load`` leaves un-flowed when nested under a mapping key) before
attaching them, so iteration sees live callables rather than ``Instance`` markers.
"""

from pathlib import Path

import torch

from dataflux import Flux, Sample
from dataflux.ops.torch import RescaleOp  # noqa: F401 - import registers the @configurable for !class: resolution

OPS_YAML = """ops:
- !class:dataflux.ops.torch.RescaleOp()
  in_min: 0.0
  in_max: 255.0
- !class:dataflux.ops.torch.RescaleOp()
  in_min: 0.0
  in_max: 1.0
  out_max: 10.0
"""


def test_from_ops_yaml_materializes_and_attaches(tmp_path: Path) -> None:
    path = tmp_path / "ops.yaml"
    path.write_text(OPS_YAML)
    src = [Sample(input=torch.tensor([0.0, 255.0]), target=None, metadata={})]

    flux = Flux.from_ops_yaml(str(path), source=src)

    # Materialized to live ops — NOT deferred Instance markers (which iteration would reject).
    assert [type(o).__name__ for o in flux.ops] == ["RescaleOp", "RescaleOp"]
    out = list(flux)[0].input
    assert torch.allclose(out, torch.tensor([0.0, 10.0]))


def test_from_ops_yaml_without_ops_key_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("other: 1\n")

    flux = Flux.from_ops_yaml(str(path), source=[])

    assert flux.ops == []
