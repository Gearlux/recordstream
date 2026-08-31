"""
RecordStream operations (record-dict ops).

Submodules:
    - recordstream.ops.numpy: Threshold, ConnectedComponents (+ threshold_array /
      connected_component_boxes / resolve_expression helpers)
    - recordstream.ops.torch: ToTensor (+ to_tensor helper)
    - recordstream.ops.image: ConvertToImage, ConvertToMask (+ value_to_image / normalize_to_uint8 …)
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
    - recordstream.ops.contract: RecordContract (pass-through interface contract at a pipeline boundary),
      ClassNamesOutput (the dataset-level class vocabulary a graph declares as an output)

The sequential composer ``Pipeline`` lives in :mod:`recordstream.transform` (package-root
export) — one list mixing native ops with bare albumentations / torchvision-v2 transforms.
"""

import importlib
from typing import Any, Dict, List, Tuple

from recordstream.ops.configure import ConfigureOp
from recordstream.ops.contract import ClassNamesOutput, ClassNamesScan, ContractError, RecordContract
from recordstream.ops.debug import PrintRecordOp
from recordstream.ops.enable import Enable
from recordstream.ops.formula import FormulaOp
from recordstream.ops.image import ConvertToImage, ConvertToMask
from recordstream.ops.numpy import ConnectedComponents, Threshold
from recordstream.ops.parallel import Parallel
from recordstream.ops.random_apply import RandomApply
from recordstream.ops.sink import RecordSinkOp
from recordstream.ops.structure import CopyField, DropField, RenameField, SelectFields
from recordstream.ops.target import CocoToTorchVisionDetection, DecodeTarget, EncodeTarget, MasksToDetectionBoxes

__all__ = [
    "CocoToTorchVisionDetection",
    "ConfigureOp",
    "ConnectedComponents",
    "ContractError",
    "ConvertToImage",
    "ConvertToMask",
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
    "ClassNamesOutput",
    "ClassNamesScan",
    "RecordContract",
    "RenameField",
    "RecordSinkOp",
    "SelectFields",
    "Threshold",
    "ToTensor",
]


#: Ops whose module needs an optional framework — resolved on first attribute access (PEP 562)
#: so ``import recordstream`` never imports one. ``ToTensor`` is the only such op today: its
#: module is torch by definition, and eagerly re-exporting it here was the single line that made
#: torch a hard dependency of the whole engine.
_OPTIONAL_OPS: Dict[str, Tuple[str, str]] = {"ToTensor": ("recordstream.ops.torch", "torch")}


def __getattr__(name: str) -> Any:
    """Resolve an optional-framework op on first use, or name the extra that provides it."""
    entry = _OPTIONAL_OPS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, extra = entry
    try:
        return getattr(importlib.import_module(module_path), name)
    except ImportError as exc:
        raise ImportError(
            f"recordstream.ops.{name} needs the {extra!r} extra: pip install 'recordstream[{extra}]'\n"
            f"(underlying import error: {exc})"
        ) from exc


def __dir__() -> List[str]:
    """Advertise the lazily-resolved names to ``dir()`` / tab-completion."""
    return sorted({*globals(), *_OPTIONAL_OPS})
