"""
RecordStream operations (record-dict ops).

Submodules:
    - recordstream.ops.numpy: Threshold, ConnectedComponents (+ threshold_array /
      connected_component_bboxes / resolve_expression helpers)
    - recordstream.ops.torch: ToTensor (+ to_tensor helper)
    - recordstream.ops.image: ConvertToImage (+ value_to_image / normalize_to_uint8 …)
    - recordstream.ops.target: EncodeTarget, DecodeTarget,
      CocoToTorchVisionDetection, MasksToDetectionBoxes
    - recordstream.ops.structure: RenameField, DropField, CopyField, SelectFields
    - recordstream.ops.parallel: Parallel (worker-pool sub-pipeline)
    - recordstream.ops.enable: Enable (toggle an op-list via one named CLI flag)
    - recordstream.ops.random_apply: RandomApply (gate any op behind a Bernoulli flip)
    - recordstream.ops.configure: ConfigureOp (per-record parameter injection)
    - recordstream.ops.formula: FormulaOp (math formula over one record entry)
    - recordstream.ops.sink: RecordSinkOp (adapt a DataSink as a pass-through op)
    - recordstream.ops.debug: PrintRecordOp (per-record summary probe)

The sequential composer ``Pipeline`` lives in :mod:`recordstream.transform` (package-root
export) — one list mixing native ops with bare albumentations / torchvision-v2 transforms.
"""

from recordstream.ops.configure import ConfigureOp
from recordstream.ops.debug import PrintRecordOp
from recordstream.ops.enable import Enable
from recordstream.ops.formula import FormulaOp
from recordstream.ops.image import ConvertToImage
from recordstream.ops.numpy import ConnectedComponents, Threshold
from recordstream.ops.parallel import Parallel
from recordstream.ops.random_apply import RandomApply
from recordstream.ops.sink import RecordSinkOp
from recordstream.ops.structure import CopyField, DropField, RenameField, SelectFields
from recordstream.ops.target import CocoToTorchVisionDetection, DecodeTarget, EncodeTarget, MasksToDetectionBoxes
from recordstream.ops.torch import ToTensor

__all__ = [
    "CocoToTorchVisionDetection",
    "ConfigureOp",
    "ConnectedComponents",
    "ConvertToImage",
    "CopyField",
    "DecodeTarget",
    "DropField",
    "Enable",
    "EncodeTarget",
    "FormulaOp",
    "MasksToDetectionBoxes",
    "Parallel",
    "PrintRecordOp",
    "RandomApply",
    "RenameField",
    "RecordSinkOp",
    "SelectFields",
    "Threshold",
    "ToTensor",
]
