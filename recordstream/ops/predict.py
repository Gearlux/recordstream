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

#: Where a torch model runs. ``auto`` picks the best AVAILABLE device (cuda > mps > cpu);
#: naming one that this machine does not have is an error listing what it does.
DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def available_devices() -> "list[str]":
    """The torch devices THIS machine can run — always ``cpu``, plus what probes true.

    Imports torch to probe, so call it only on a path that already involves a torch model
    (the op resolves its device lazily, on the first record, and only for a model that has
    a ``.to``).
    """
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass
    return devices


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)  # type: ignore[no-any-return]


def _nth(out: Any, index: int) -> Any:
    """Element ``index`` of a batched output (a sequence, a batched array, or a mapping of batched arrays)."""
    if isinstance(out, (list, tuple)):
        return out[index]
    if isinstance(out, dict):
        return {k: (v[index] if hasattr(v, "__len__") and len(v) > index else v) for k, v in out.items()}
    if hasattr(out, "shape") and len(out.shape) > 0:
        return out[index]
    return out


def _first(out: Any) -> Any:
    return _nth(out, 0)


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
            ``<output>_mask``).
        device: Where a torch model runs — ``auto`` (default) picks the best available
            (cuda > mps > cpu); naming an absent one raises listing this machine's
            devices. Ignored for a model without a ``.to``.
        score: Classification only — record field the CONFIDENCE (max probability) is
            ADDED under, as a ``Label``. ``""`` disables it. Declared AFTER ``device``
            on purpose: canvas widget values are POSITIONAL, so a new parameter is
            APPENDED — inserting one mid-signature shifts every later widget on a saved
            canvas converted by a not-yet-restarted editor (measured: ``device: score``).
        batch_size: How many records a batching RUNNER may hand :meth:`predict_batch` per
            forward. ``0`` (default) = no opinion — the runner's own default decides;
            set it on the node to pin this model's limit. Appended last (see ``score``).
    """

    def __init__(
        self,
        model: Any = None,  # any callable model wrapper — naming a real type would force a torch-shaped import
        kind: PredictKind = "classification",
        key: str = "image",
        output: str = "predict",
        device: DeviceName = "auto",
        score: str = "score",
        batch_size: int = 0,
    ) -> None:
        if kind not in get_args(PredictKind):
            raise ValueError(f"ModelPredict kind must be one of {get_args(PredictKind)}, got {kind!r}")
        if device not in get_args(DeviceName):
            raise ValueError(f"ModelPredict device must be one of {get_args(DeviceName)}, got {device!r}")
        self.model = model
        self.kind = kind
        self.key = key
        self.output = output
        self.score = score
        self.device = device
        self.batch_size = int(batch_size)
        self._ready: Any = None
        self._resolved_device: str = ""

    def _resolve_device(self) -> str:
        """The concrete device for THIS machine — resolved once, on the first torch model."""
        if not self._resolved_device:
            devices = available_devices()
            if self.device == "auto":
                self._resolved_device = "cuda" if "cuda" in devices else ("mps" if "mps" in devices else "cpu")
            elif self.device in devices:
                self._resolved_device = self.device
            else:
                raise ValueError(
                    f"ModelPredict: device {self.device!r} is not available on this machine — "
                    f"available: {', '.join(devices)}"
                )
        return self._resolved_device

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
            if hasattr(model, "to") and callable(model.to):
                model.to(self._resolve_device())
            self._ready = model
        return self._ready

    def _call(self, batch: Any) -> Any:
        model = self._model()
        try:
            import torch  # noqa: F401

            with torch.no_grad():
                if hasattr(batch, "to") and self._resolved_device:
                    batch = batch.to(self._resolved_device)
                return model(batch)
        except ImportError:
            return model(batch)

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict_batch([record])[0]

    def predict_batch(self, records: "list[Dict[str, Any]]") -> "list[Dict[str, Any]]":
        """ONE forward over a chunk of records, each stamped with its own row of the output.

        The batching runner's entry point — and :meth:`__call__` is a batch of one, so the
        per-record and batched paths cannot drift. The chunk SIZE is the caller's business
        (see ``batch_size``); this method batches whatever it is handed.
        """
        if not records:
            return []
        for record in records:
            if self.key not in record:
                raise ValueError(f"ModelPredict: record has no field {self.key!r} (fields: {sorted(record)})")
        out = self._call(self._stack([record[self.key] for record in records]))
        stamped: "list[Dict[str, Any]]" = []
        for index, record in enumerate(records):
            if self.kind == "classification":
                label, confidence = self._classification(out, index)
                one = {**record, self.output: label}
                if self.score:
                    one[self.score] = confidence
            elif self.kind == "detection":
                one = {**record, self.output: self._detection(out, record[self.key], index)}
            elif self.kind == "segmentation":
                one = {**record, f"{self.output}_mask": self._segmentation(out, index)}
            else:
                one = {**record, self.output: self._restoration(out, index)}
            stamped.append(one)
        return stamped

    def _stack(self, values: "list[Any]") -> Any:
        """One batch from the records' inputs — ragged shapes fail naming the fix."""
        try:
            if hasattr(values[0], "detach"):
                import torch

                return torch.stack(list(values))
            return np.stack([np.asarray(value) for value in values])
        except (RuntimeError, ValueError) as exc:
            shapes = sorted({tuple(np.asarray(_to_numpy(value)).shape) for value in values})
            raise ValueError(
                f"ModelPredict: the records under {self.key!r} do not stack into one batch "
                f"(shapes: {shapes}) — resize before the model so every record arrives in ONE shape"
            ) from exc

    def _classification(self, out: Any, index: int = 0) -> "tuple[Label, Label]":
        values = out
        if isinstance(out, dict) or (not hasattr(out, "shape") and hasattr(out, "probs")):
            values = _fields(out, "probs", "logits")
        scores = _to_numpy(_nth(values, index)).astype(np.float64).reshape(-1)
        if scores.min() < 0.0 or scores.sum() > 1.0001:
            scores = _softmax(scores)
        index = int(np.argmax(scores))
        return Label(index), Label(float(scores[index]))

    def _detection(self, out: Any, image: Any, index: int = 0) -> Boxes:
        first = _nth(out, index)
        boxes = _to_numpy(_fields(first, "boxes")).reshape(-1, 4)
        scores = _to_numpy(_fields(first, "scores")).reshape(-1) if _has(first, "scores") else np.ones(len(boxes))
        labels = (
            _to_numpy(_fields(first, "labels")).reshape(-1).astype(int)
            if _has(first, "labels")
            else np.zeros(len(boxes), dtype=int)
        )
        return Boxes(boxes=boxes.tolist(), labels=labels.tolist(), scores=scores.tolist(), canvas=_image_hw(image))

    def _segmentation(self, out: Any, index: int = 0) -> np.ndarray:
        values = out
        if isinstance(out, dict) or (not hasattr(out, "shape") and (hasattr(out, "mask") or hasattr(out, "logits"))):
            values = _fields(out, "mask", "logits", "probs")
        arr = _to_numpy(_nth(values, index))
        return np.argmax(arr, axis=0).astype(np.int64) if arr.ndim == 3 else arr.astype(np.int64)

    def _restoration(self, out: Any, index: int = 0) -> np.ndarray:
        values = out
        if isinstance(out, dict) or (not hasattr(out, "shape") and hasattr(out, "image")):
            values = _fields(out, "image")
        arr = _to_numpy(_nth(values, index))
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        return arr


def _has(out: Any, name: str) -> bool:
    return (isinstance(out, dict) and name in out) or hasattr(out, name)


__all__ = ["DeviceName", "ModelPredict", "PredictKind", "available_devices"]
