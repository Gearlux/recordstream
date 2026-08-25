"""Tests for :mod:`recordstream.outputs` — the typed prediction-output contracts + torch builders."""

import torch

from recordstream.outputs import (
    ClassificationOutput,
    DetectionOutput,
    RestorationOutput,
    SegmentationOutput,
    classification_output,
    restoration_output,
    segmentation_output,
)


def test_classification_output_shapes_and_dtypes() -> None:
    logits = torch.randn(4, 3)
    out = classification_output(logits)

    # TypedDict values are plain dict entries at runtime.
    assert set(out.keys()) == {"logits", "probs", "class_idx"}
    assert out["logits"] is logits
    assert out["probs"].shape == (4, 3)
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(4), atol=1e-5)
    assert out["class_idx"].dtype == torch.int64
    assert out["class_idx"].shape == (4,)
    assert torch.equal(out["class_idx"], logits.argmax(dim=-1).to(torch.int64))


def test_segmentation_output_per_pixel_argmax() -> None:
    logits = torch.randn(2, 5, 8, 8)
    out = segmentation_output(logits)

    assert set(out.keys()) == {"logits", "probs", "mask"}
    assert out["probs"].shape == (2, 5, 8, 8)
    # Softmax over channel dim.
    assert torch.allclose(out["probs"].sum(dim=1), torch.ones(2, 8, 8), atol=1e-5)
    assert out["mask"].dtype == torch.int64
    assert out["mask"].shape == (2, 8, 8)
    assert torch.equal(out["mask"], logits.argmax(dim=1).to(torch.int64))


def test_restoration_output_carries_the_image_unchanged() -> None:
    """The builder transforms NOTHING — a restored image already IS the contract's one key."""
    image = torch.rand(2, 3, 8, 8)
    out = restoration_output(image)

    assert set(out.keys()) == {"image"}
    assert out["image"] is image
    assert out["image"].shape == (2, 3, 8, 8)


def test_restoration_output_does_not_clamp_the_range() -> None:
    """A residual denoiser can legitimately overshoot [0, 1], and clamping would change the score.

    Whether the output is clipped is the RUN's decision (a metric on clamped values is a
    different number), so the contract carries what the model produced.
    """
    out = restoration_output(torch.tensor([[[[-0.25, 1.5]]]]))

    assert float(out["image"].min()) == -0.25
    assert float(out["image"].max()) == 1.5


def test_restoration_output_has_no_residual_key() -> None:
    """``input - image`` is derivable, and a derivable key is a second place for the two to disagree."""
    assert "residual" not in restoration_output(torch.zeros(1, 3, 4, 4))


def test_restoration_output_is_typed_dict_instance() -> None:
    out: RestorationOutput = restoration_output(torch.zeros(1, 3, 4, 4))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"image"}


def test_detection_output_typed_dict_construction() -> None:
    # DetectionOutput is constructed directly at call sites; verify the
    # TypedDict's runtime behavior matches a plain dict.
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    scores = torch.tensor([0.9], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    out = DetectionOutput(boxes=boxes, scores=scores, labels=labels)
    assert out["boxes"].dtype == torch.float32
    assert out["scores"].dtype == torch.float32
    assert out["labels"].dtype == torch.int64
    assert set(out.keys()) == {"boxes", "scores", "labels"}


def test_classification_output_is_typed_dict_instance() -> None:
    out: ClassificationOutput = classification_output(torch.zeros(1, 2))
    # TypedDicts are plain dicts at runtime.
    assert isinstance(out, dict)
    assert "logits" in out and "probs" in out and "class_idx" in out


def test_segmentation_output_is_typed_dict_instance() -> None:
    out: SegmentationOutput = segmentation_output(torch.zeros(1, 2, 3, 3))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"logits", "probs", "mask"}


# --------------------------------------------------------------------------- #
# The contracts stay generic in the array type — only the BUILDERS are torch
# --------------------------------------------------------------------------- #


def test_a_contract_parameterizes_over_the_array_type() -> None:
    """The same contract describes a torch run and a numpy/TF/JAX one.

    recordstream hard-depends on torch (a `Stream` IS a `torch.utils.data.Dataset`), so the
    contracts and their torch builders share one module — but the contracts themselves are
    typing-only and generic, so a backend on another framework declares its output with the
    SAME types and adds its own builders beside these.
    """
    import numpy as np

    numpy_out: ClassificationOutput[np.ndarray] = {
        "logits": np.zeros(2),
        "probs": np.zeros(2),
        "class_idx": np.zeros(2),
    }
    torch_out: ClassificationOutput[torch.Tensor] = classification_output(torch.zeros(1, 2))

    assert isinstance(numpy_out["probs"], np.ndarray)
    assert isinstance(torch_out["probs"], torch.Tensor)


def test_detection_has_no_builder_on_purpose() -> None:
    """Boxes come from a detector's own interface; there is nothing to derive from logits."""
    import recordstream.outputs as outputs

    assert not hasattr(outputs, "detection_output")


def test_the_package_root_exports_the_contracts_and_builders() -> None:
    import recordstream

    for name in (
        "ClassificationOutput",
        "DetectionOutput",
        "DetectionPredictions",
        "RestorationOutput",
        "SegmentationOutput",
        "classification_output",
        "restoration_output",
        "segmentation_output",
    ):
        assert name in recordstream.__all__ and hasattr(recordstream, name)
