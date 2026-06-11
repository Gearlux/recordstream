"""
DataFlux operations.

Submodules:
    - dataflux.ops.numpy: RescaleOp, StandardizeOp, ClipPercentilesOp,
      ReplaceNonFiniteOp, ThresholdOp, ConnectedComponentsOp, SqueezeOp,
      UnsqueezeOp (ndarray)
    - dataflux.ops.torch: RescaleOp, StandardizeOp, ToTensorOp, SqueezeOp,
      UnsqueezeOp (tensor)
    - dataflux.ops.tee: Tee (fan-out branching)
    - dataflux.ops.parallel: Parallel (worker-pool sub-pipeline)
    - dataflux.ops.enable: Enable (toggle an op-list via one named CLI flag)
    - dataflux.ops.random_apply: RandomApply (gate any op behind a Bernoulli flip)
    - dataflux.ops.configure: ConfigureOp (per-sample parameter injection — the helios Configure pattern)
    - dataflux.ops.formula: FormulaOp (math formula over sample.input — the Math node's op form)
    - dataflux.ops.sink: SampleSinkOp (adapt a DataSink as a pass-through op)
    - dataflux.ops.transform_chain: TransformChain (sequential op-chain grouping)
    - dataflux.ops.copy: CopySampleOp, CopyInputOp, CopyTargetOp, CopyMetadataOp
    - dataflux.ops.swap: SwapInputTargetOp
    - dataflux.ops.stash: StashInputOp, UnstashInputOp, StashTargetOp, UnstashTargetOp
    - dataflux.ops.target: MetadataToTargetOp, EncodeTargetOp, DecodeTargetOp (target field)

Flat imports default to torch variants for the data ops; flow / copy /
swap / stash / target utilities are field-agnostic.
"""

from dataflux.ops.configure import ConfigureOp
from dataflux.ops.copy import CopyInputOp, CopyMetadataOp, CopySampleOp, CopyTargetOp
from dataflux.ops.enable import Enable
from dataflux.ops.formula import FormulaOp
from dataflux.ops.parallel import Parallel
from dataflux.ops.random_apply import RandomApply
from dataflux.ops.sink import SampleSinkOp
from dataflux.ops.stash import StashInputOp, StashTargetOp, UnstashInputOp, UnstashTargetOp
from dataflux.ops.swap import SwapInputTargetOp
from dataflux.ops.target import (
    CocoToTorchVisionDetectionOp,
    DecodeTargetOp,
    EncodeTargetOp,
    MasksToDetectionBoxesOp,
    MetadataToTargetOp,
)
from dataflux.ops.tee import Tee
from dataflux.ops.torch import RescaleOp, SqueezeOp, StandardizeOp, ToTensorOp, UnsqueezeOp
from dataflux.ops.transform_chain import TransformChain

__all__ = [
    "ConfigureOp",
    "CopyInputOp",
    "CopyMetadataOp",
    "CopySampleOp",
    "CopyTargetOp",
    "DecodeTargetOp",
    "Enable",
    "FormulaOp",
    "EncodeTargetOp",
    "MetadataToTargetOp",
    "CocoToTorchVisionDetectionOp",
    "MasksToDetectionBoxesOp",
    "Parallel",
    "RandomApply",
    "RescaleOp",
    "SampleSinkOp",
    "SqueezeOp",
    "StandardizeOp",
    "StashInputOp",
    "StashTargetOp",
    "SwapInputTargetOp",
    "Tee",
    "TransformChain",
    "ToTensorOp",
    "UnstashInputOp",
    "UnstashTargetOp",
    "UnsqueezeOp",
]
