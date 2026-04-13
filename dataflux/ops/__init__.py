"""
DataFlux operations.

Submodules:
    - dataflux.ops.numpy: NormalizeOp, StandardizeOp (ndarray)
    - dataflux.ops.torch: NormalizeOp, StandardizeOp, ToTensorOp (tensor)

Flat imports default to torch variants for backward compatibility.
"""

from dataflux.ops.torch import NormalizeOp, StandardizeOp, ToTensorOp

__all__ = ["NormalizeOp", "StandardizeOp", "ToTensorOp"]
