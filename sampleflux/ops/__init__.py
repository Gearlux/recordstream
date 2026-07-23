"""
SampleFlux operations (typed-bag :class:`~sampleflux.Sample` transforms).

Submodules:
    - sampleflux.ops.numpy: Threshold, ConnectedComponents (+ threshold_array /
      connected_component_bboxes / resolve_expression helpers)
    - sampleflux.ops.torch: ToTensor (+ to_tensor helper)
    - sampleflux.ops.image: ConvertToImage (+ value_to_image / normalize_to_uint8 …)
    - sampleflux.ops.target: MetadataToTarget, EncodeTarget, DecodeTarget,
      CocoToTorchVisionDetection, MasksToDetectionBoxes
    - sampleflux.ops.structure: SetRole, RenameField, DropField, CopyField, SelectFields
    - sampleflux.ops.parallel: Parallel (worker-pool sub-pipeline)
    - sampleflux.ops.enable: Enable (toggle an op-list via one named CLI flag)
    - sampleflux.ops.random_apply: RandomApply (gate any op behind a Bernoulli flip)
    - sampleflux.ops.configure: ConfigureOp (per-sample parameter injection)
    - sampleflux.ops.formula: FormulaOp (math formula over the primary input)
    - sampleflux.ops.sink: SampleSinkOp (adapt a DataSink as a pass-through op)
    - sampleflux.ops.transform_chain: TransformChain (sequential op-chain grouping)
    - sampleflux.ops.context: Save, Use, Drop, Apply, Capture, MergeFields (the per-sample
      Context graph plane — the flat-list building blocks a branchy flow: document lowers to)
    - sampleflux.ops.debug: PrintSampleOp (per-sample summary probe)
"""

from sampleflux.ops.configure import ConfigureOp
from sampleflux.ops.context import Apply, Capture, Drop, MergeFields, Save, Use
from sampleflux.ops.debug import PrintSampleOp
from sampleflux.ops.enable import Enable
from sampleflux.ops.formula import FormulaOp
from sampleflux.ops.image import ConvertToImage
from sampleflux.ops.numpy import ConnectedComponents, Threshold
from sampleflux.ops.parallel import Parallel
from sampleflux.ops.random_apply import RandomApply
from sampleflux.ops.sink import SampleSinkOp
from sampleflux.ops.structure import CopyField, DropField, RenameField, SelectFields, SetRole
from sampleflux.ops.target import (
    CocoToTorchVisionDetection,
    DecodeTarget,
    EncodeTarget,
    MasksToDetectionBoxes,
    MetadataToTarget,
)
from sampleflux.ops.torch import ToTensor
from sampleflux.ops.transform_chain import TransformChain

__all__ = [
    "Apply",
    "Capture",
    "CocoToTorchVisionDetection",
    "ConfigureOp",
    "ConnectedComponents",
    "ConvertToImage",
    "CopyField",
    "DecodeTarget",
    "Drop",
    "DropField",
    "Enable",
    "EncodeTarget",
    "FormulaOp",
    "MasksToDetectionBoxes",
    "MergeFields",
    "MetadataToTarget",
    "Parallel",
    "PrintSampleOp",
    "RandomApply",
    "RenameField",
    "Save",
    "SampleSinkOp",
    "SelectFields",
    "SetRole",
    "Threshold",
    "ToTensor",
    "TransformChain",
    "Use",
]
