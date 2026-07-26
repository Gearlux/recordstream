"""
SampleFlux operations (record-dict ops).

Submodules:
    - sampleflux.ops.numpy: Threshold, ConnectedComponents (+ threshold_array /
      connected_component_bboxes / resolve_expression helpers)
    - sampleflux.ops.torch: ToTensor (+ to_tensor helper)
    - sampleflux.ops.image: ConvertToImage (+ value_to_image / normalize_to_uint8 …)
    - sampleflux.ops.target: EncodeTarget, DecodeTarget,
      CocoToTorchVisionDetection, MasksToDetectionBoxes
    - sampleflux.ops.structure: RenameField, DropField, CopyField, SelectFields
    - sampleflux.ops.parallel: Parallel (worker-pool sub-pipeline)
    - sampleflux.ops.enable: Enable (toggle an op-list via one named CLI flag)
    - sampleflux.ops.random_apply: RandomApply (gate any op behind a Bernoulli flip)
    - sampleflux.ops.configure: ConfigureOp (per-record parameter injection)
    - sampleflux.ops.formula: FormulaOp (math formula over one record entry)
    - sampleflux.ops.sink: RecordSinkOp (adapt a DataSink as a pass-through op)
    - sampleflux.ops.context: Save, Use, Drop, Apply, Capture, MergeFields (the per-record
      Context graph plane — the flat-list building blocks a branchy flow: document lowers to)
    - sampleflux.ops.debug: PrintSampleOp (per-record summary probe)

The sequential composer ``Pipeline`` lives in :mod:`sampleflux.transform` (package-root
export) — one list mixing native ops with bare albumentations / torchvision-v2 transforms.
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
from sampleflux.ops.sink import RecordSinkOp
from sampleflux.ops.structure import CopyField, DropField, RenameField, SelectFields
from sampleflux.ops.target import CocoToTorchVisionDetection, DecodeTarget, EncodeTarget, MasksToDetectionBoxes
from sampleflux.ops.torch import ToTensor

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
    "Parallel",
    "PrintSampleOp",
    "RandomApply",
    "RenameField",
    "Save",
    "RecordSinkOp",
    "SelectFields",
    "Threshold",
    "ToTensor",
    "Use",
]
