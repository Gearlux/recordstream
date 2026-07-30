"""Guard: every node-facing recordstream Source/Op documents all its constructor params.

These classes surface in visual editors (as widget tooltips) and MCP form-specs (as
pydantic ``Field(description=...)``) purely from their docstring ``Args:`` block — see
``confluid.parse_param_docs``. A param that loses its doc silently loses its
tooltip/description, so this pins the coverage.
"""

import inspect
from typing import List

import pytest
from confluid import parse_param_docs  # type: ignore[import-not-found]

from recordstream import Pipeline, Transform
from recordstream.core import FilterOp, JointStream, Stream, WrappedOp
from recordstream.ops.configure import ConfigureOp
from recordstream.ops.debug import PrintRecordOp
from recordstream.ops.enable import Enable
from recordstream.ops.formula import FormulaOp
from recordstream.ops.image import ConvertToImage
from recordstream.ops.numpy import ConnectedComponents, Threshold
from recordstream.ops.parallel import Parallel
from recordstream.ops.random_apply import RandomApply
from recordstream.ops.structure import CopyField, DropField, RenameField, SelectFields
from recordstream.ops.target import CocoToTorchVisionDetection, DecodeTarget, EncodeTarget, MasksToDetectionBoxes
from recordstream.ops.torch import ToTensor
from recordstream.sources import HuggingFaceSource

_NODE_CLASSES = [
    HuggingFaceSource,
    Stream,
    JointStream,
    FilterOp,
    WrappedOp,
    Transform,
    Pipeline,
    Threshold,
    ConnectedComponents,
    ConvertToImage,
    ToTensor,
    EncodeTarget,
    DecodeTarget,
    CocoToTorchVisionDetection,
    MasksToDetectionBoxes,
    RenameField,
    DropField,
    CopyField,
    SelectFields,
    ConfigureOp,
    FormulaOp,
    Enable,
    Parallel,
    RandomApply,
    PrintRecordOp,
]


def _constructor_params(cls: type) -> List[str]:
    # signature(cls) is the constructor signature (no ``self``), and it survives
    # confluid's @configurable __init__ wrapping (verified against real classes).
    sig = inspect.signature(cls)
    return [
        name
        for name, p in sig.parameters.items()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


@pytest.mark.parametrize("cls", _NODE_CLASSES, ids=lambda c: c.__name__)
def test_all_constructor_params_documented(cls: type) -> None:
    docs = parse_param_docs(cls)
    missing = [p for p in _constructor_params(cls) if not docs.get(p)]
    assert not missing, f"{cls.__name__} is missing Args docs for: {missing}"
