"""Guard: every node-facing sampleflux Source/Op documents all its constructor params.

These classes surface in FluxStudio (as widget tooltips) and navigaitor (as
pydantic ``Field(description=...)`` in the form-spec) purely from their docstring
``Args:`` block — see ``confluid.parse_param_docs``. A param that loses its doc
silently loses its tooltip/description, so this pins the coverage.
"""

import inspect
from typing import List

import pytest
from confluid import parse_param_docs  # type: ignore[import-not-found]

from sampleflux.core import FilterOp, Flux, JointFlux, WrappedOp
from sampleflux.ops.albumentations import AlbumentationsOp
from sampleflux.ops.numpy import ConnectedComponentsOp, StandardizeOp, ThresholdOp
from sampleflux.ops.target import DecodeTargetOp, EncodeTargetOp, MetadataToTargetOp
from sampleflux.ops.torch import StandardizeOp as TorchStandardizeOp
from sampleflux.ops.torch import ToTensorOp
from sampleflux.ops.torchvision import TorchvisionTransformOp
from sampleflux.ops.transform_chain import TransformChain
from sampleflux.sources import HuggingFaceSource

_NODE_CLASSES = [
    HuggingFaceSource,
    Flux,
    JointFlux,
    FilterOp,
    WrappedOp,
    StandardizeOp,
    ThresholdOp,
    ConnectedComponentsOp,
    ToTensorOp,
    TorchStandardizeOp,
    MetadataToTargetOp,
    EncodeTargetOp,
    DecodeTargetOp,
    TransformChain,
    AlbumentationsOp,
    TorchvisionTransformOp,
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
