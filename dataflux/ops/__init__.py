"""
DataFlux operations.

Submodules:
    - dataflux.ops.numpy: RescaleOp, StandardizeOp, ClipPercentilesOp,
      ReplaceNonFiniteOp, ThresholdOp, ConnectedComponentsOp (ndarray)
    - dataflux.ops.torch: RescaleOp, StandardizeOp, ToTensorOp (tensor)
    - dataflux.ops.tee: Tee (fan-out branching)
    - dataflux.ops.copy: CopySampleOp, CopyInputOp, CopyTargetOp, CopyMetadataOp
    - dataflux.ops.swap: SwapInputTargetOp
    - dataflux.ops.stash: StashInputOp, UnstashInputOp

Flat imports default to torch variants for the data ops; flow / copy /
swap / stash utilities are field-agnostic.
"""

from dataflux.ops.copy import CopyInputOp, CopyMetadataOp, CopySampleOp, CopyTargetOp
from dataflux.ops.stash import StashInputOp, UnstashInputOp
from dataflux.ops.swap import SwapInputTargetOp
from dataflux.ops.tee import Tee
from dataflux.ops.torch import RescaleOp, StandardizeOp, ToTensorOp

__all__ = [
    "CopyInputOp",
    "CopyMetadataOp",
    "CopySampleOp",
    "CopyTargetOp",
    "RescaleOp",
    "StandardizeOp",
    "StashInputOp",
    "SwapInputTargetOp",
    "Tee",
    "ToTensorOp",
    "UnstashInputOp",
]
