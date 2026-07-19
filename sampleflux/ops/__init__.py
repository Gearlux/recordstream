"""
SampleFlux operations.

Submodules:
    - sampleflux.ops.numpy: RescaleOp, StandardizeOp, ClipPercentilesOp,
      ReplaceNonFiniteOp, ThresholdOp, ConnectedComponentsOp, SqueezeOp,
      UnsqueezeOp, MinOp, MaxOp, MedianOp, PercentileOp, StatsOp (ndarray)
    - sampleflux.ops.torch: RescaleOp, StandardizeOp, ToTensorOp, SqueezeOp,
      UnsqueezeOp (tensor)
    - sampleflux.ops.parallel: Parallel (worker-pool sub-pipeline)
    - sampleflux.ops.enable: Enable (toggle an op-list via one named CLI flag)
    - sampleflux.ops.random_apply: RandomApply (gate any op behind a Bernoulli flip)
    - sampleflux.ops.configure: ConfigureOp (per-sample parameter injection — the helios Configure pattern)
    - sampleflux.ops.formula: FormulaOp (math formula over sample.input — the Math node's op form)
    - sampleflux.ops.sink: SampleSinkOp (adapt a DataSink as a pass-through op)
    - sampleflux.ops.transform_chain: TransformChain (sequential op-chain grouping)
    - sampleflux.ops.context: Save, Use, Drop, Apply, Capture, Mix (per-sample Context
      graph plane — the flat-list building blocks a branchy flow: document lowers to)
    - sampleflux.ops.copy: CopySampleOp, CopyInputOp, CopyTargetOp, CopyMetadataOp
    - sampleflux.ops.swap: SwapInputTargetOp
    - sampleflux.ops.stash: StashInputOp, UnstashInputOp, StashTargetOp, UnstashTargetOp
      (metadata-bus snapshots — only for crossing a Parallel boundary or persisting
      a snapshot into a sink; graph wiring uses sampleflux.ops.context)
    - sampleflux.ops.target: MetadataToTargetOp, EncodeTargetOp, DecodeTargetOp (target field)

Flat imports default to torch variants for the data ops; flow / copy /
swap / stash / target utilities are field-agnostic.
"""

from sampleflux.ops.configure import ConfigureOp
from sampleflux.ops.context import Apply, Capture, Drop, Mix, Save, Use
from sampleflux.ops.copy import CopyInputOp, CopyMetadataOp, CopySampleOp, CopyTargetOp
from sampleflux.ops.enable import Enable
from sampleflux.ops.formula import FormulaOp
from sampleflux.ops.parallel import Parallel
from sampleflux.ops.random_apply import RandomApply
from sampleflux.ops.sink import SampleSinkOp
from sampleflux.ops.stash import StashInputOp, StashTargetOp, UnstashInputOp, UnstashTargetOp
from sampleflux.ops.swap import SwapInputTargetOp
from sampleflux.ops.target import (
    CocoToTorchVisionDetectionOp,
    DecodeTargetOp,
    EncodeTargetOp,
    MasksToDetectionBoxesOp,
    MetadataToTargetOp,
)
from sampleflux.ops.torch import RescaleOp, SqueezeOp, StandardizeOp, ToTensorOp, UnsqueezeOp
from sampleflux.ops.transform_chain import TransformChain

__all__ = [
    "ConfigureOp",
    "CopyInputOp",
    "CopyMetadataOp",
    "CopySampleOp",
    "Apply",
    "Capture",
    "CopyTargetOp",
    "DecodeTargetOp",
    "Drop",
    "Enable",
    "Mix",
    "Save",
    "Use",
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
    "TransformChain",
    "ToTensorOp",
    "UnstashInputOp",
    "UnstashTargetOp",
    "UnsqueezeOp",
]
