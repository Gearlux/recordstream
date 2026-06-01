"""
DataFlux operations.

Submodules:
    - dataflux.ops.numpy: RescaleOp, StandardizeOp, ClipPercentilesOp,
      ReplaceNonFiniteOp, ThresholdOp, ConnectedComponentsOp (ndarray)
    - dataflux.ops.torch: RescaleOp, StandardizeOp, ToTensorOp (tensor)
    - dataflux.ops.tee: Tee (fan-out branching)
    - dataflux.ops.parallel: Parallel (worker-pool sub-pipeline)
    - dataflux.ops.enable: Enable (toggle an op-list via one named CLI flag)
    - dataflux.ops.sink: SampleSinkOp (adapt a DataSink as a pass-through op)
    - dataflux.ops.copy: CopySampleOp, CopyInputOp, CopyTargetOp, CopyMetadataOp
    - dataflux.ops.swap: SwapInputTargetOp
    - dataflux.ops.stash: StashInputOp, UnstashInputOp
    - dataflux.ops.target: MetadataToTargetOp, EncodeTargetOp, DecodeTargetOp (target field)

Flat imports default to torch variants for the data ops; flow / copy /
swap / stash / target utilities are field-agnostic.
"""

from dataflux.ops.copy import CopyInputOp, CopyMetadataOp, CopySampleOp, CopyTargetOp
from dataflux.ops.enable import Enable
from dataflux.ops.parallel import Parallel
from dataflux.ops.sink import SampleSinkOp
from dataflux.ops.stash import StashInputOp, UnstashInputOp
from dataflux.ops.swap import SwapInputTargetOp
from dataflux.ops.target import DecodeTargetOp, EncodeTargetOp, MetadataToTargetOp
from dataflux.ops.tee import Tee
from dataflux.ops.torch import RescaleOp, StandardizeOp, ToTensorOp

__all__ = [
    "CopyInputOp",
    "CopyMetadataOp",
    "CopySampleOp",
    "CopyTargetOp",
    "DecodeTargetOp",
    "Enable",
    "EncodeTargetOp",
    "MetadataToTargetOp",
    "Parallel",
    "RescaleOp",
    "SampleSinkOp",
    "StandardizeOp",
    "StashInputOp",
    "SwapInputTargetOp",
    "Tee",
    "ToTensorOp",
    "UnstashInputOp",
]
