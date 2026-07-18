"""
SampleFlux operations.

Submodules:
    - sampleflux.ops.numpy: RescaleOp, StandardizeOp, ClipPercentilesOp,
      ReplaceNonFiniteOp, ThresholdOp, ConnectedComponentsOp, SqueezeOp,
      UnsqueezeOp, FourierOp, InverseFourierOp, FftShiftOp, IfftShiftOp,
      WindowOp, SpectrumScalingOp (ndarray)
    - sampleflux.ops.torch: RescaleOp, StandardizeOp, ToTensorOp, SqueezeOp,
      UnsqueezeOp, FourierOp, InverseFourierOp, FftShiftOp, IfftShiftOp,
      WindowOp, SpectrumScalingOp (tensor)
    - sampleflux.windows: get_window / scale_spectrum + the WindowName /
      SpectrumScaling Literals — the window + unit-scaling math the FFT ops share
    - sampleflux.ops.tee: Tee (fan-out branching)
    - sampleflux.ops.parallel: Parallel (worker-pool sub-pipeline)
    - sampleflux.ops.enable: Enable (toggle an op-list via one named CLI flag)
    - sampleflux.ops.random_apply: RandomApply (gate any op behind a Bernoulli flip)
    - sampleflux.ops.configure: ConfigureOp (per-sample parameter injection — the helios Configure pattern)
    - sampleflux.ops.capture: CaptureOutputOp (record an op's @output value into metadata)
    - sampleflux.ops.formula: FormulaOp (math formula over sample.input — the Math node's op form)
    - sampleflux.ops.sink: SampleSinkOp (adapt a DataSink as a pass-through op)
    - sampleflux.ops.transform_chain: TransformChain (sequential op-chain grouping)
    - sampleflux.ops.context: Save, Use, Drop, Apply, Capture, Mix (per-sample Context
      graph plane — the flat-list building blocks a branchy flow: document lowers to)
    - sampleflux.ops.copy: CopySampleOp, CopyInputOp, CopyTargetOp, CopyMetadataOp
    - sampleflux.ops.swap: SwapInputTargetOp
    - sampleflux.ops.stash: StashInputOp, UnstashInputOp, StashTargetOp, UnstashTargetOp
    - sampleflux.ops.target: MetadataToTargetOp, EncodeTargetOp, DecodeTargetOp (target field)

Flat imports default to torch variants for the data ops; flow / copy /
swap / stash / target utilities are field-agnostic.
"""

from sampleflux.ops.capture import CaptureOutputOp
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
from sampleflux.ops.tee import Tee
from sampleflux.ops.torch import (
    FftShiftOp,
    FourierOp,
    IfftShiftOp,
    InverseFourierOp,
    RescaleOp,
    SpectrumScalingOp,
    SqueezeOp,
    StandardizeOp,
    ToTensorOp,
    UnsqueezeOp,
    WindowOp,
)
from sampleflux.ops.transform_chain import TransformChain

__all__ = [
    "CaptureOutputOp",
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
    "FftShiftOp",
    "FormulaOp",
    "FourierOp",
    "IfftShiftOp",
    "InverseFourierOp",
    "EncodeTargetOp",
    "MetadataToTargetOp",
    "CocoToTorchVisionDetectionOp",
    "MasksToDetectionBoxesOp",
    "Parallel",
    "RandomApply",
    "RescaleOp",
    "SampleSinkOp",
    "SpectrumScalingOp",
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
    "WindowOp",
]
