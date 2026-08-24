"""Run a task model over records — inference as ONE op in a pipeline.

This is what lets a graph (or any config) CARRY its inference: a source, its
preprocessing, and a ``ModelPredict`` op wired to a project's model wrapper compose
into one runnable pipeline — a viewer executes it per record and reads the stamped
fields back as layers, and the same document runs offline with ``recordstream run``.
The op lives here (not in a viewer or a project package) because it is fully generic:
the model is any callable ``model(batch)``, duck-typed — no torch import of its own.
"""

from typing import Any, Dict, Literal, get_args

import numpy as np
from confluid import configurable
from loggair import get_logger

from recordstream.items import Boxes, Label

logger = get_logger(__name__)

#: What a model's output means — how it is stamped back onto the record.
PredictKind = Literal["classification", "detection", "segmentation", "restoration"]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)  # type: ignore[no-any-return]


def _batch(x: Any) -> Any:
    return x[None] if hasattr(x, "shape") else np.asarray(x)[None]


def _first(out: Any) -> Any:
    """The first element of a batched output (a sequence, a batched array, or a mapping of batched arrays)."""
    if isinstance(out, (list, tuple)):
        return out[0]
    if isinstance(out, dict):
        return {k: (v[0] if hasattr(v, "__len__") and len(v) else v) for k, v in out.items()}
    if hasattr(out, "shape") and len(out.shape) > 0:
        return out[0]
    return out


def _fields(out: Any, *names: str) -> Any:
    """``out[name]`` / ``out.name`` for the first name present."""
    for name in names:
        if isinstance(out, dict) and name in out:
            return out[name]
        if hasattr(out, name):
            return getattr(out, name)
    raise ValueError(f"model output carries none of {names} (got {type(out).__name__})")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=-1, keepdims=True), dtype=np.float64)


def _image_hw(value: Any) -> "tuple[int, int]":
    """The (H, W) of an image field — HWC, HW, or CHW (1/3 channels first)."""
    arr = _to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        return int(arr.shape[1]), int(arr.shape[2])
    return int(arr.shape[0]), int(arr.shape[1])


@configurable(category="op")
class ModelPredict:
    """Run a task model on ONE record and stamp its prediction as a record field.

    The model is any callable ``model(batch)`` — typically a project's checkpointed
    wrapper. Its real work (build the network, load the checkpoint) happens in its
    ``solidify()``, called lazily on the first record, so constructing this op is free.
    What gets stamped follows the field conventions a viewer reads back as layers:

    - ``classification``: ``output`` = the predicted class as a :class:`~recordstream.items.Label`
      (output shape ``[1, C]`` probs/logits, or a mapping/object with ``probs``/``logits``);
    - ``detection``: ``output`` = a :class:`~recordstream.items.Boxes` (per-image
      ``boxes``/``scores``/``labels``, pixel xyxy);
    - ``segmentation``: ``<output>_mask`` = an int class mask ``[H, W]`` (argmax over
      ``[C, H, W]`` logits when needed);
    - ``restoration``: ``output`` = the restored image array (channels-first is moved last).

    Args:
        model: The callable model (a checkpointed wrapper; solidified on first use).
        kind: What the model's output means — ``classification`` / ``detection`` /
            ``segmentation`` / ``restoration``.
        key: Record field fed to the model.
        output: Record field stamped with the prediction (``segmentation`` stamps
            ``<output>_mask``). Keep the ``predict`` prefix — that is what marks a
            field as a prediction downstream.
        device: Where a torch model runs (``cpu`` / ``cuda`` / ``mps``); ignored otherwise.
    """

    def __init__(
        self,
        model: Any = None,  # any callable model wrapper — naming a real type would force a torch-shaped import
        kind: PredictKind = "classification",
        key: str = "image",
        output: str = "predict",
        device: str = "cpu",
    ) -> None:
        if kind not in get_args(PredictKind):
            raise ValueError(f"ModelPredict kind must be one of {get_args(PredictKind)}, got {kind!r}")
        self.model = model
        self.kind = kind
        self.key = key
        self.output = output
        self.device = device
        self._ready: Any = None

    def _model(self) -> Any:
        if self._ready is None:
            model = self.model
            if model is None:
                raise ValueError("ModelPredict needs 'model' (a callable model wrapper)")
            if hasattr(model, "solidify") and callable(model.solidify):
                built = model.solidify()
                model = built if built is not None else model
            if hasattr(model, "eval") and callable(model.eval):
                model.eval()
            if self.device and hasattr(model, "to") and callable(model.to):
                model.to(self.device)
            self._ready = model
        return self._ready

    def _call(self, batch: Any) -> Any:
        model = self._model()
        try:
            import torch  # noqa: F401

            with torch.no_grad():
                if hasattr(batch, "to") and self.device:
                    batch = batch.to(self.device)
                return model(batch)
        except ImportError:
            return model(batch)

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if self.key not in record:
            raise ValueError(f"ModelPredict: record has no field {self.key!r} (fields: {sorted(record)})")
        out = self._call(_batch(record[self.key]))
        if self.kind == "classification":
            return {**record, self.output: self._classification(out)}
        if self.kind == "detection":
            return {**record, self.output: self._detection(out, record[self.key])}
        if self.kind == "segmentation":
            return {**record, f"{self.output}_mask": self._segmentation(out)}
        return {**record, self.output: self._restoration(out)}

    def _classification(self, out: Any) -> Label:
        values = out
        if isinstance(out, dict) or (not hasattr(out, "shape") and hasattr(out, "probs")):
            values = _fields(out, "probs", "logits")
        scores = _to_numpy(_first(values)).astype(np.float64).reshape(-1)
        if scores.min() < 0.0 or scores.sum() > 1.0001:
            scores = _softmax(scores)
        return Label(int(np.argmax(scores)))

    def _detection(self, out: Any, image: Any) -> Boxes:
        first = _first(out)
        boxes = _to_numpy(_fields(first, "boxes")).reshape(-1, 4)
        scores = _to_numpy(_fields(first, "scores")).reshape(-1) if _has(first, "scores") else np.ones(len(boxes))
        labels = (
            _to_numpy(_fields(first, "labels")).reshape(-1).astype(int)
            if _has(first, "labels")
            else np.zeros(len(boxes), dtype=int)
        )
        return Boxes(boxes=boxes.tolist(), labels=labels.tolist(), scores=scores.tolist(), canvas=_image_hw(image))

    def _segmentation(self, out: Any) -> np.ndarray:
        values = out
        if isinstance(out, dict) or (not hasattr(out, "shape") and (hasattr(out, "mask") or hasattr(out, "logits"))):
            values = _fields(out, "mask", "logits", "probs")
        arr = _to_numpy(_first(values))
        return np.argmax(arr, axis=0).astype(np.int64) if arr.ndim == 3 else arr.astype(np.int64)

    def _restoration(self, out: Any) -> np.ndarray:
        values = out
        if isinstance(out, dict) or (not hasattr(out, "shape") and hasattr(out, "image")):
            values = _fields(out, "image")
        arr = _to_numpy(_first(values))
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        return arr


def _has(out: Any, name: str) -> bool:
    return (isinstance(out, dict) and name in out) or hasattr(out, name)


__all__ = ["ModelPredict", "PredictKind"]
