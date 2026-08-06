"""Prediction-output contracts — what a model hands to a sink, metric, or visualizer.

Every model's eval-mode ``forward`` should return one of these — or a task-specific superset.
The keys *are* the documentation: downstream metrics, sinks, visualizers, and evaluators
inspect known keys instead of guessing whether a tensor is logits, probabilities, or argmax'd
class ids. That guess is not hypothetical: two independently-written detector wrappers agree
that ``boxes`` is xyxy in absolute pixels only because :class:`DetectionOutput` says so.

**Generic in the array type.** Each contract is a generic ``TypedDict`` parameterized by the
array type it carries, so the SAME contract describes a torch run, a numpy/JAX run, or a
TensorFlow one::

    ClassificationOutput[Tensor]      # a torch model
    ClassificationOutput[np.ndarray]  # a numpy / TF / JAX model

The parameter is optional — a bare ``ClassificationOutput`` means "whatever array type".

**Why here.** These describe the boundary between a model and whatever consumes its output,
and the consumer this package ships is :mod:`recordstream.predictions` — the sink that reads
``probs`` / ``class_idx`` by name. A contract owned by a different package than its reader is
how "it lives there because that other thing lives there" starts.

The BUILDERS (bottom of this module) are necessarily per-framework — ``softmax`` and ``argmax``
are library calls, not type declarations — and are torch, like the rest of this package's tensor
surface. A backend on another framework adds its own builders beside these; it does NOT redefine
the contracts.
"""

from typing import TYPE_CHECKING, Generic, List, TypedDict, TypeVar

if TYPE_CHECKING:  # torch is imported inside the BUILDERS — the contracts are typing-only
    from torch import Tensor

#: The array type a contract carries — ``torch.Tensor``, ``np.ndarray``, a TF/JAX array.
ArrayT = TypeVar("ArrayT")

__all__ = [
    "ArrayT",
    "ClassificationOutput",
    "DetectionOutput",
    "DetectionPredictions",
    "RestorationOutput",
    "SegmentationOutput",
    "classification_output",
    "restoration_output",
    "segmentation_output",
]


class ClassificationOutput(TypedDict, Generic[ArrayT]):
    """Per-record classification prediction.

    Keys:
        logits: ``[B, C]`` float — raw pre-softmax scores.
        probs: ``[B, C]`` float — softmax over the last dim.
        class_idx: ``[B]`` int64 — argmax of ``logits`` along the last dim.
    """

    logits: ArrayT
    probs: ArrayT
    class_idx: ArrayT


class DetectionOutput(TypedDict, Generic[ArrayT]):
    """Per-image object detection prediction.

    Keys:
        boxes: ``[N, 4]`` float32 — xyxy in **absolute pixels**.
        scores: ``[N]`` float32 — confidence in ``[0, 1]``.
        labels: ``[N]`` int64 — class ids. Class 0 is conventionally reserved
            for background in torchvision-style detectors.
    """

    boxes: ArrayT
    scores: ArrayT
    labels: ArrayT


class SegmentationOutput(TypedDict, Generic[ArrayT]):
    """Per-pixel segmentation prediction.

    Keys:
        logits: ``[B, C, H, W]`` float — raw pre-softmax scores.
        probs: ``[B, C, H, W]`` float — softmax over the channel dim.
        mask: ``[B, H, W]`` int64 — argmax across channels.
    """

    logits: ArrayT
    probs: ArrayT
    mask: ArrayT


class RestorationOutput(TypedDict, Generic[ArrayT]):
    """Per-record image-to-image restoration prediction.

    Keys:
        image: ``[B, C, H, W]`` float — the restored image, in the SAME value range as the
            model's input (``[0, 1]`` for a pipeline that ends in ``ToTensor(normalize=True)``).

    **One key, and that is the whole point of the contract.** The other tasks here carry three
    because their raw output needs interpreting — ``logits`` are not ``probs`` are not
    ``class_idx``. A restoration model emits the answer directly, so what a consumer needs told
    is precisely that: the array under ``image`` is a picture in the input's units, NOT logits
    to be softmaxed and NOT a residual to be added back. A sink that writes a PNG, a PSNR metric
    and a viewer all read it the same way because this says so.

    The residual (what the model removed) is deliberately absent: it is ``input - image``, and a
    contract key that a consumer can compute is a second place for the two to disagree.
    """

    image: ArrayT


#: A detector's per-image results — one :class:`DetectionOutput` per image in the batch.
DetectionPredictions = List[DetectionOutput]


def classification_output(logits: "Tensor") -> "ClassificationOutput[Tensor]":
    """Build a full :class:`ClassificationOutput` from raw ``[B, C]`` logits.

    Example::

        def predict_step(self, batch, batch_idx):
            return classification_output(self(x))     # what a predictions sink reads
    """
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=-1)
    class_idx = torch.argmax(logits, dim=-1).to(torch.int64)
    return ClassificationOutput(logits=logits, probs=probs, class_idx=class_idx)


def segmentation_output(logits: "Tensor") -> "SegmentationOutput[Tensor]":
    """Build a full :class:`SegmentationOutput` from raw ``[B, C, H, W]`` logits."""
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=1)
    mask = torch.argmax(logits, dim=1).to(torch.int64)
    return SegmentationOutput(logits=logits, probs=probs, mask=mask)


def restoration_output(image: "Tensor") -> "RestorationOutput[Tensor]":
    """Build a :class:`RestorationOutput` from a restored ``[B, C, H, W]`` image.

    It transforms nothing — unlike its two siblings, which softmax and argmax — because a
    restoration model's output already IS the contract's one key. The builder exists anyway so
    every backend spells the wrapping identically and a reader of ``predict_step`` sees the same
    shape of line in all four tasks::

        def predict_step(self, batch, batch_idx):
            return restoration_output(self(x))    # what a predictions sink reads

    Clamping is deliberately NOT done here: whether a restored image may leave ``[0, 1]`` is the
    model's business (a residual denoiser can legitimately overshoot, and a metric computed on
    clamped values is a different number), so a run that wants it wires it where it decides that.
    """
    return RestorationOutput(image=image)


# Detection deliberately has NO builder: boxes come from the detector's own interface, so the
# dict is built inline at the call site rather than invented from nothing here.
