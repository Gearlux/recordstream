"""Tests for sampleflux.ops: torch and numpy variants."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from sampleflux.ops import (
    ConfigureOp,
    CopyInputOp,
    CopyMetadataOp,
    CopySampleOp,
    CopyTargetOp,
    FormulaOp,
    RescaleOp,
    SqueezeOp,
    StandardizeOp,
    StashInputOp,
    StashTargetOp,
    SwapInputTargetOp,
    ToTensorOp,
    UnsqueezeOp,
    UnstashInputOp,
    UnstashTargetOp,
)
from sampleflux.ops import numpy as np_ops
from sampleflux.ops.numpy import MaxOp, ThresholdOp
from sampleflux.sample import Sample

# ---------------------------------------------------------------------------
# ToTensorOp
# ---------------------------------------------------------------------------


class TestToTensorOp:
    """Tests for ToTensorOp."""

    def test_pil_image_with_normalize(self) -> None:
        img = Image.fromarray(np.full((28, 28), 128, dtype=np.uint8))
        result = ToTensorOp(normalize=True)(Sample(input=img))
        assert isinstance(result.input, torch.Tensor)
        assert result.input.dtype == torch.float32
        assert result.input.max() <= 1.0

    def test_pil_image_without_normalize(self) -> None:
        img = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))
        result = ToTensorOp(normalize=False)(Sample(input=img))
        assert isinstance(result.input, torch.Tensor)
        assert result.input.dtype == torch.uint8
        assert result.input.max() == 200

    def test_2d_array_adds_channel_dim(self) -> None:
        arr = np.zeros((28, 28), dtype=np.uint8)
        result = ToTensorOp(normalize=False)(Sample(input=arr))
        assert result.input.shape == (1, 28, 28)

    def test_3d_array_transposes(self) -> None:
        arr = np.zeros((28, 28, 3), dtype=np.uint8)
        result = ToTensorOp(normalize=False)(Sample(input=arr))
        assert result.input.shape == (3, 28, 28)

    def test_normalize_float_above_one(self) -> None:
        arr = np.array([[128.0, 255.0]], dtype=np.float32)
        result = ToTensorOp(normalize=True)(Sample(input=arr))
        assert result.input.max() == 1.0

    def test_non_array_passthrough(self) -> None:
        t = torch.tensor([1.0, 2.0])
        result = ToTensorOp(normalize=False)(Sample(input=t))
        assert torch.equal(result.input, t)

    def test_preserves_target_and_metadata(self) -> None:
        arr = np.zeros((28, 28), dtype=np.uint8)
        result = ToTensorOp()(Sample(input=arr, target=5, metadata={"k": "v"}))
        assert result.target == 5
        assert result.meta == {"k": "v"}

    def test_mode_rgb_forces_three_channels_from_mixed_pil_modes(self) -> None:
        # A mixed-mode image dataset (RGBA / grayscale / palette) → uniform 3-channel
        # RGB for a fixed-channel model (e.g. torchvision Faster R-CNN's 3-ch normalize).
        op = ToTensorOp(mode="RGB")
        for mode, arr in (
            ("RGBA", np.zeros((8, 8, 4), dtype=np.uint8)),
            ("L", np.zeros((8, 8), dtype=np.uint8)),
            ("RGB", np.zeros((8, 8, 3), dtype=np.uint8)),
        ):
            out = op(Sample(input=Image.fromarray(arr, mode=mode)))
            assert out.input.shape == (3, 8, 8), f"{mode} → {tuple(out.input.shape)}"

    def test_mode_none_leaves_channels_as_is(self) -> None:
        # Default mode=None arrays the image verbatim — RGBA stays 4-channel.
        rgba = Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8), mode="RGBA")
        assert ToTensorOp()(Sample(input=rgba)).input.shape == (4, 8, 8)


# ---------------------------------------------------------------------------
# Torch RescaleOp
# ---------------------------------------------------------------------------


class TestRescaleOp:
    """Tests for torch RescaleOp."""

    def test_default_output_range(self) -> None:
        tensor = torch.tensor([0.0, 128.0, 255.0])
        result = RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=tensor))
        assert result.input[0] == 0.0
        assert abs(result.input[1] - 128.0 / 255.0) < 1e-6
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_custom_input_range(self) -> None:
        tensor = torch.tensor([10.0, 55.0, 100.0])
        result = RescaleOp(in_min=10.0, in_max=100.0)(Sample(input=tensor))
        assert abs(result.input[0] - 0.0) < 1e-6
        assert abs(result.input[1] - 0.5) < 1e-6
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_custom_output_range(self) -> None:
        tensor = torch.tensor([0.0, 0.5, 1.0])
        result = RescaleOp(in_min=0.0, in_max=1.0, out_min=10.0, out_max=20.0)(Sample(input=tensor))
        assert abs(result.input[0] - 10.0) < 1e-6
        assert abs(result.input[1] - 15.0) < 1e-6
        assert abs(result.input[2] - 20.0) < 1e-6

    def test_clip_true_clamps(self) -> None:
        tensor = torch.tensor([-50.0, 0.0, 128.0, 300.0])
        result = RescaleOp(in_min=0.0, in_max=255.0, clip=True)(Sample(input=tensor))
        assert result.input[0] == 0.0
        assert result.input[3] == 1.0

    def test_clip_false_extrapolates(self) -> None:
        tensor = torch.tensor([-255.0, 510.0])
        result = RescaleOp(in_min=0.0, in_max=255.0, clip=False)(Sample(input=tensor))
        assert abs(result.input[0] - (-1.0)) < 1e-6
        assert abs(result.input[1] - 2.0) < 1e-6

    def test_uint8_converts_to_float(self) -> None:
        tensor = torch.tensor([0, 128, 255], dtype=torch.uint8)
        result = RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=tensor))
        assert result.input.dtype == torch.float32
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_preserves_float64(self) -> None:
        tensor = torch.tensor([0.0, 255.0], dtype=torch.float64)
        result = RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=tensor))
        assert result.input.dtype == torch.float64

    def test_preserves_target_and_metadata(self) -> None:
        tensor = torch.tensor([128.0])
        result = RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=tensor, target=7, metadata={"key": "val"}))
        assert result.target == 7
        assert result.meta == {"key": "val"}

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="RescaleOp expects a torch.Tensor"):
            RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=np.array([1, 2, 3])))

    def test_validation_rejects_bad_input_range(self) -> None:
        op = RescaleOp(in_min=10.0, in_max=10.0)  # lazy: construction succeeds
        with pytest.raises(ValueError, match="require in_min < in_max"):
            op(Sample(input=torch.zeros(2)))

    def test_validation_rejects_bad_output_range(self) -> None:
        op = RescaleOp(in_min=0.0, in_max=1.0, out_min=5.0, out_max=5.0)
        with pytest.raises(ValueError, match="require out_min < out_max"):
            op(Sample(input=torch.zeros(2)))

    def test_pipeline_to_tensor_then_rescale(self) -> None:
        """Integration: ToTensorOp(normalize=False) -> RescaleOp()."""
        img = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))
        sample = Sample(input=img)
        sample = ToTensorOp(normalize=False)(sample)
        sample = RescaleOp(in_min=0.0, in_max=255.0)(sample)
        assert sample.input.dtype == torch.float32
        assert abs(sample.input.max().item() - 200.0 / 255.0) < 1e-6


# ---------------------------------------------------------------------------
# Torch StandardizeOp
# ---------------------------------------------------------------------------


class TestStandardizeOp:
    """Tests for torch StandardizeOp."""

    def test_scalar_mean_and_std(self) -> None:
        tensor = torch.tensor([2.0, 4.0, 6.0])
        result = StandardizeOp(mean=4.0, std=2.0)(Sample(input=tensor))
        assert abs(result.input[0] - (-1.0)) < 1e-6
        assert abs(result.input[1] - 0.0) < 1e-6
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_per_channel_mean_and_std(self) -> None:
        tensor = torch.ones(3, 2, 2)
        tensor[0] *= 10.0
        tensor[1] *= 20.0
        tensor[2] *= 30.0
        result = StandardizeOp(mean=[10.0, 20.0, 30.0], std=[1.0, 1.0, 1.0])(Sample(input=tensor))
        assert torch.allclose(result.input, torch.zeros(3, 2, 2))

    def test_uint8_converts_to_float(self) -> None:
        tensor = torch.tensor([100, 200], dtype=torch.uint8)
        result = StandardizeOp(mean=150.0, std=50.0)(Sample(input=tensor))
        assert result.input.dtype == torch.float32
        assert abs(result.input[0] - (-1.0)) < 1e-6
        assert abs(result.input[1] - 1.0) < 1e-6

    def test_preserves_float64(self) -> None:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = StandardizeOp(mean=0.0, std=1.0)(Sample(input=tensor))
        assert result.input.dtype == torch.float64

    def test_preserves_target_and_metadata(self) -> None:
        tensor = torch.tensor([5.0])
        result = StandardizeOp(mean=0.0, std=1.0)(Sample(input=tensor, target=3, metadata={"a": 1}))
        assert result.target == 3
        assert result.meta == {"a": 1}

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="StandardizeOp expects a torch.Tensor"):
            StandardizeOp(mean=0.0, std=1.0)(Sample(input=[1, 2, 3]))

    def test_1d_per_channel(self) -> None:
        tensor = torch.tensor([10.0])
        result = StandardizeOp(mean=[10.0], std=[5.0])(Sample(input=tensor))
        assert abs(result.input[0] - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Numpy StandardizeOp
# ---------------------------------------------------------------------------


class TestNpStandardizeOp:
    """Tests for numpy StandardizeOp."""

    def test_scalar_mean_and_std(self) -> None:
        arr = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        result = np_ops.StandardizeOp(mean=4.0, std=2.0)(Sample(input=arr))
        assert abs(result.input[0] - (-1.0)) < 1e-6
        assert abs(result.input[1] - 0.0) < 1e-6
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_per_channel_mean_and_std(self) -> None:
        arr = np.ones((3, 2, 2), dtype=np.float32)
        arr[0] *= 10.0
        arr[1] *= 20.0
        arr[2] *= 30.0
        result = np_ops.StandardizeOp(mean=[10.0, 20.0, 30.0], std=[1.0, 1.0, 1.0])(Sample(input=arr))
        assert np.allclose(result.input, np.zeros((3, 2, 2)))

    def test_uint8_converts_to_float32(self) -> None:
        arr = np.array([100, 200], dtype=np.uint8)
        result = np_ops.StandardizeOp(mean=150.0, std=50.0)(Sample(input=arr))
        assert result.input.dtype == np.float32
        assert abs(result.input[0] - (-1.0)) < 1e-6
        assert abs(result.input[1] - 1.0) < 1e-6

    def test_preserves_float64(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = np_ops.StandardizeOp(mean=0.0, std=1.0)(Sample(input=arr))
        assert result.input.dtype == np.float64

    def test_preserves_target_and_metadata(self) -> None:
        arr = np.array([5.0], dtype=np.float32)
        result = np_ops.StandardizeOp(mean=0.0, std=1.0)(Sample(input=arr, target=3, metadata={"a": 1}))
        assert result.target == 3
        assert result.meta == {"a": 1}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="StandardizeOp expects an np.ndarray"):
            np_ops.StandardizeOp(mean=0.0, std=1.0)(Sample(input=[1, 2, 3]))

    def test_pil_image_input(self) -> None:
        img = Image.fromarray(np.full((28, 28), 150, dtype=np.uint8))
        result = np_ops.StandardizeOp(mean=150.0, std=50.0)(Sample(input=img))
        assert isinstance(result.input, np.ndarray)
        assert np.allclose(result.input, 0.0)

    def test_1d_per_channel(self) -> None:
        arr = np.array([10.0], dtype=np.float32)
        result = np_ops.StandardizeOp(mean=[10.0], std=[5.0])(Sample(input=arr))
        assert abs(result.input[0] - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Numpy ClipPercentilesOp
# ---------------------------------------------------------------------------


class TestClipPercentilesOp:
    def test_happy_path_no_outliers(self) -> None:
        arr = np.linspace(-50.0, -10.0, 1000).reshape(20, 50)
        out = np_ops.ClipPercentilesOp(low=2, high=98)(Sample(input=arr)).input
        assert out.min() == pytest.approx(float(np.percentile(arr, 2)))
        assert out.max() == pytest.approx(float(np.percentile(arr, 98)))

    def test_ignores_inf_and_nan(self) -> None:
        arr = np.linspace(-50.0, -10.0, 100).reshape(10, 10).copy()
        arr[0, 0] = np.inf
        arr[0, 1] = -np.inf
        arr[0, 2] = np.nan
        finite = arr[np.isfinite(arr)]
        expected_lo = float(np.percentile(finite, 2))
        expected_hi = float(np.percentile(finite, 98))
        out = np_ops.ClipPercentilesOp(low=2, high=98)(Sample(input=arr)).input
        assert out[0, 0] == pytest.approx(expected_hi)
        assert out[0, 1] == pytest.approx(expected_lo)
        assert np.isnan(out[0, 2])

    def test_all_non_finite_passes_through(self) -> None:
        arr = np.full((4, 4), np.nan)
        sample = Sample(input=arr)
        out = np_ops.ClipPercentilesOp()(sample)
        assert out is sample

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ClipPercentilesOp expects an np.ndarray"):
            np_ops.ClipPercentilesOp()(Sample(input=torch.tensor([1.0])))

    @pytest.mark.parametrize("low,high", [(50, 50), (60, 50), (-1, 50), (50, 101)])
    def test_validation_rejects_bad_bounds(self, low: float, high: float) -> None:
        op = np_ops.ClipPercentilesOp(low=low, high=high)  # lazy: construction succeeds
        with pytest.raises(ValueError, match="ClipPercentilesOp: require"):
            op(Sample(input=np.array([1.0, 2.0, 3.0])))


# ---------------------------------------------------------------------------
# Numpy RescaleOp
# ---------------------------------------------------------------------------


class TestNpRescaleOp:
    def test_default_output_range(self) -> None:
        arr = np.array([[-80.0, -50.0, -20.0]])
        out = np_ops.RescaleOp(in_min=-80.0, in_max=-20.0)(Sample(input=arr)).input
        np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]])

    def test_custom_output_range(self) -> None:
        arr = np.array([[-80.0, -50.0, -20.0]])
        out = np_ops.RescaleOp(in_min=-80.0, in_max=-20.0, out_min=10.0, out_max=20.0)(Sample(input=arr)).input
        np.testing.assert_allclose(out, [[10.0, 15.0, 20.0]])

    def test_clip_true_clamps(self) -> None:
        arr = np.array([[-100.0, -50.0, 0.0]])
        out = np_ops.RescaleOp(in_min=-80.0, in_max=-20.0, clip=True)(Sample(input=arr)).input
        np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]])

    def test_clip_false_extrapolates(self) -> None:
        arr = np.array([[-100.0, -50.0, 0.0]])
        out = np_ops.RescaleOp(in_min=-80.0, in_max=-20.0, clip=False)(Sample(input=arr)).input
        assert out[0, 0] == pytest.approx(-1.0 / 3.0)
        assert out[0, 1] == pytest.approx(0.5)
        assert out[0, 2] == pytest.approx(4.0 / 3.0)

    def test_uint8_converts_to_float32(self) -> None:
        arr = np.array([0, 128, 255], dtype=np.uint8)
        out = np_ops.RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=arr)).input
        assert out.dtype == np.float32
        assert abs(out[2] - 1.0) < 1e-6

    def test_preserves_float64(self) -> None:
        arr = np.array([0.0, 255.0], dtype=np.float64)
        out = np_ops.RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=arr)).input
        assert out.dtype == np.float64

    def test_pil_image_input(self) -> None:
        img = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))
        out = np_ops.RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=img)).input
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert abs(out.max() - 200.0 / 255.0) < 1e-6

    def test_preserves_target_and_metadata(self) -> None:
        arr = np.array([128.0], dtype=np.float32)
        result = np_ops.RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=arr, target=7, metadata={"key": "val"}))
        assert result.target == 7
        assert result.meta == {"key": "val"}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="RescaleOp expects an np.ndarray"):
            np_ops.RescaleOp(in_min=0.0, in_max=1.0)(Sample(input=[1.0, 2.0]))

    def test_validation_rejects_bad_input_range(self) -> None:
        op = np_ops.RescaleOp(in_min=10.0, in_max=10.0)  # lazy: construction succeeds
        with pytest.raises(ValueError, match="require in_min < in_max"):
            op(Sample(input=np.zeros(2)))

    def test_validation_rejects_bad_output_range(self) -> None:
        op = np_ops.RescaleOp(in_min=0.0, in_max=1.0, out_min=5.0, out_max=5.0)
        with pytest.raises(ValueError, match="require out_min < out_max"):
            op(Sample(input=np.zeros(2)))

    def test_pipeline_rescale_then_to_tensor(self) -> None:
        """Integration: numpy RescaleOp -> ToTensorOp(normalize=False)."""
        img = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))
        sample = Sample(input=img)
        sample = np_ops.RescaleOp(in_min=0.0, in_max=255.0)(sample)
        sample = ToTensorOp(normalize=False)(sample)
        assert isinstance(sample.input, torch.Tensor)
        assert sample.input.dtype == torch.float32
        assert abs(sample.input.max().item() - 200.0 / 255.0) < 1e-6


# ---------------------------------------------------------------------------
# Numpy ReplaceNonFiniteOp
# ---------------------------------------------------------------------------


class TestReplaceNonFiniteOp:
    def test_numeric_value(self) -> None:
        arr = np.array([[1.0, np.inf, 2.0], [-np.inf, np.nan, 3.0]])
        out = np_ops.ReplaceNonFiniteOp(value=-99.0)(Sample(input=arr)).input
        assert out.tolist() == [[1.0, -99.0, 2.0], [-99.0, -99.0, 3.0]]

    def test_min_replacement(self) -> None:
        arr = np.array([[1.0, np.inf, 2.0], [-np.inf, np.nan, 3.0]])
        out = np_ops.ReplaceNonFiniteOp(value="min")(Sample(input=arr)).input
        assert out.tolist() == [[1.0, 1.0, 2.0], [1.0, 1.0, 3.0]]

    def test_max_replacement(self) -> None:
        arr = np.array([[1.0, np.inf, 2.0], [-np.inf, np.nan, 3.0]])
        out = np_ops.ReplaceNonFiniteOp(value="max")(Sample(input=arr)).input
        assert out.tolist() == [[1.0, 3.0, 2.0], [3.0, 3.0, 3.0]]

    def test_already_finite_passes_through(self) -> None:
        arr = np.array([[1.0, 2.0, 3.0]])
        sample = Sample(input=arr)
        out = np_ops.ReplaceNonFiniteOp(value="min")(sample)
        assert out is sample

    def test_all_non_finite_passes_through(self) -> None:
        arr = np.full((3, 3), np.nan)
        sample = Sample(input=arr)
        out = np_ops.ReplaceNonFiniteOp(value="min")(sample)
        assert out is sample

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ReplaceNonFiniteOp expects an np.ndarray"):
            np_ops.ReplaceNonFiniteOp()(Sample(input=torch.tensor([1.0])))

    def test_validation_rejects_unknown_string(self) -> None:
        op = np_ops.ReplaceNonFiniteOp(value="median")  # lazy: construction succeeds
        with pytest.raises(ValueError, match="value string must be 'min' or 'max'"):
            op(Sample(input=np.array([1.0, np.inf])))


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Copy* ops
# ---------------------------------------------------------------------------


class TestCopyOps:
    def test_copy_sample_deepcopies_all_fields(self) -> None:
        meta = {"k": [1, 2, 3]}
        sample = Sample(input=np.array([1.0, 2.0]), target=[10], metadata=meta)
        out = CopySampleOp()(sample)
        assert out.input is not sample.input
        assert out.target is not sample.target
        assert out.meta is not sample.meta
        assert out.meta["k"] is not sample.meta["k"]

    def test_copy_input_only_copies_input(self) -> None:
        sample = Sample(input=np.array([1.0]), target=[5], metadata={"k": "v"})
        out = CopyInputOp()(sample)
        assert out.input is not sample.input
        assert out.target is sample.target
        assert out.meta is sample.meta

    def test_copy_target_only_copies_target(self) -> None:
        sample = Sample(input=[1, 2], target=[10, 20], metadata={})
        out = CopyTargetOp()(sample)
        assert out.target is not sample.target
        assert out.input is sample.input

    def test_copy_metadata_breaks_aliasing(self) -> None:
        meta = {"k": [1]}
        sample = Sample(input=None, target=None, metadata=meta)
        out = CopyMetadataOp()(sample)
        out.meta["k"].append(2)
        assert meta["k"] == [1]


# ---------------------------------------------------------------------------
# SwapInputTargetOp
# ---------------------------------------------------------------------------


class TestSwapInputTargetOp:
    def test_swaps_input_and_target(self) -> None:
        sample = Sample(input=1, target=2, metadata={"k": "v"})
        out = SwapInputTargetOp()(sample)
        assert out.input == 2
        assert out.target == 1
        assert out.meta == {"k": "v"}


# ---------------------------------------------------------------------------
# StashInputOp / UnstashInputOp
# ---------------------------------------------------------------------------


class TestStashUnstash:
    def test_stash_aliases_by_default(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=arr, target=None, metadata={})
        out = StashInputOp(key="snap")(sample)
        assert out.meta["snap"] is arr
        assert out.input is arr

    def test_stash_with_copy_deepcopies(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=arr, target=None, metadata={})
        out = StashInputOp(key="snap", copy=True)(sample)
        assert out.meta["snap"] is not arr
        np.testing.assert_array_equal(out.meta["snap"], arr)

    def test_unstash_default_copies_to_isolate_branches(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=None, target=None, metadata={"snap": arr})
        out = UnstashInputOp(key="snap")(sample)
        assert out.input is not arr
        np.testing.assert_array_equal(out.input, arr)

    def test_unstash_no_copy_aliases(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=None, target=None, metadata={"snap": arr})
        out = UnstashInputOp(key="snap", copy=False)(sample)
        assert out.input is arr

    def test_two_unstashes_with_in_place_mutation_dont_corrupt(self) -> None:
        """Default copy=True prevents branch-A's in-place write from leaking into branch-B.

        A multi-unstash of the SAME key needs remove=False on the NON-final unstashes so the
        snapshot survives (the compiler emits exactly this for a fan-out); the LAST unstash
        cleans it up.
        """
        arr = np.array([1.0, 2.0, 3.0])
        sample = Sample(input=None, target=None, metadata={"snap": arr})
        a = UnstashInputOp(key="snap", remove=False)(sample)  # keep the key for branch B
        a.input.fill(99.0)  # in-place mutation on branch A's restored array
        b = UnstashInputOp(key="snap")(sample)  # final unstash → removes the key
        np.testing.assert_array_equal(b.input, [1.0, 2.0, 3.0])
        assert "snap" not in sample.meta  # cleaned up by the final unstash

    def test_unstash_removes_key_by_default(self) -> None:
        sample = Sample(input=None, target=None, metadata={"snap": np.array([1.0, 2.0]), "keep": 1})
        out = UnstashInputOp(key="snap")(sample)
        np.testing.assert_array_equal(out.input, [1.0, 2.0])
        assert "snap" not in out.meta  # removed by default
        assert out.meta["keep"] == 1  # other keys untouched

    def test_unstash_keeps_key_when_remove_false(self) -> None:
        sample = Sample(input=None, target=None, metadata={"snap": np.array([1.0])})
        out = UnstashInputOp(key="snap", remove=False)(sample)
        assert "snap" in out.meta


# ---------------------------------------------------------------------------
# StashTargetOp / UnstashTargetOp
# ---------------------------------------------------------------------------


class TestStashUnstashTarget:
    def test_stash_target_aliases_by_default(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=None, target=arr, metadata={})
        out = StashTargetOp(key="snap")(sample)
        assert out.meta["snap"] is arr
        assert out.target is arr

    def test_stash_target_with_copy_deepcopies(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=None, target=arr, metadata={})
        out = StashTargetOp(key="snap", copy=True)(sample)
        assert out.meta["snap"] is not arr
        np.testing.assert_array_equal(out.meta["snap"], arr)

    def test_unstash_target_default_copies_to_isolate_branches(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=None, target=None, metadata={"snap": arr})
        out = UnstashTargetOp(key="snap")(sample)
        assert out.target is not arr
        np.testing.assert_array_equal(out.target, arr)

    def test_unstash_target_no_copy_aliases(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=None, target=None, metadata={"snap": arr})
        out = UnstashTargetOp(key="snap", copy=False)(sample)
        assert out.target is arr

    def test_unstash_target_missing_key_raises_lazily(self) -> None:
        sample = Sample(input=None, target=None, metadata={})
        with pytest.raises(KeyError):
            UnstashTargetOp(key="nope")(sample)

    def test_stash_restore_round_trip_preserves_fork_target(self) -> None:
        """The DAG→sequential pattern: snapshot at a fork, restore after a branch replaced it."""
        sample = Sample(input=None, target="fork-target", metadata={})
        stashed = StashTargetOp(key="fork")(sample)
        branched = stashed._replace(target="branch-target")
        restored = UnstashTargetOp(key="fork")(branched)
        assert restored.target == "fork-target"
        assert "fork" not in restored.meta  # removed by default after restore

    def test_unstash_target_keeps_key_when_remove_false(self) -> None:
        sample = Sample(input=None, target=None, metadata={"snap": np.array([1.0])})
        out = UnstashTargetOp(key="snap", remove=False)(sample)
        assert "snap" in out.meta


# ---------------------------------------------------------------------------
# FormulaOp
# ---------------------------------------------------------------------------


class TestFormulaOp:
    def test_evaluates_formula_over_input(self) -> None:
        sample = Sample(input=10.0, target=None, metadata={})
        out = FormulaOp(formula="a * 0.2")(sample)
        assert out.input == 2.0

    def test_custom_var_binding(self) -> None:
        sample = Sample(input=9.0, target=None, metadata={})
        out = FormulaOp(formula="sqrt(b)", var="b")(sample)
        assert out.input == 3.0

    def test_math_namespace_and_helpers(self) -> None:
        sample = Sample(input=-4.2, target=None, metadata={})
        out = FormulaOp(formula="round(abs(a))")(sample)
        assert out.input == 4

    def test_no_builtins_in_namespace(self) -> None:
        sample = Sample(input=1.0, target=None, metadata={})
        with pytest.raises(ValueError, match="failed"):
            FormulaOp(formula="__import__('os').getcwd()")(sample)

    def test_bad_formula_raises_value_error(self) -> None:
        sample = Sample(input=1.0, target=None, metadata={})
        with pytest.raises(ValueError, match="failed"):
            FormulaOp(formula="a +")(sample)

    def test_empty_formula_raises_lazily(self) -> None:
        sample = Sample(input=1.0, target=None, metadata={})
        with pytest.raises(ValueError, match="non-empty"):
            FormulaOp(formula="  ")(sample)

    def test_default_is_identity(self) -> None:
        sample = Sample(input=7.5, target=None, metadata={})
        assert FormulaOp()(sample).input == 7.5


# ---------------------------------------------------------------------------
# ConfigureOp (the helios Configure pattern)
# ---------------------------------------------------------------------------


class TestConfigureOp:
    def test_computes_injects_and_applies(self) -> None:
        """compute-chain value → metadata + target attribute → target applied to the ORIGINAL sample."""
        sample = Sample(input=np.array([1.0, 5.0, 3.0]), target=None, metadata={})
        op = ConfigureOp(
            ops=[MaxOp()],
            target=ThresholdOp(low_op=">="),
            param="low_level",
        )
        out = op(sample)
        assert out is not None
        target = op.target
        assert isinstance(target, ThresholdOp) and target.low_level == 5.0  # injected per sample
        assert out.meta["low_level"] == 5.0  # traceability: the value rides metadata too
        np.testing.assert_array_equal(out.input, [False, True, False])  # threshold on the ORIGINAL array

    def test_empty_compute_chain_uses_incoming_input(self) -> None:
        class _Target:
            def __init__(self) -> None:
                self.level: object = None

            def __call__(self, s: Sample) -> Sample:
                return s

        target = _Target()
        sample = Sample(input=7.5, target=None, metadata={})
        out = ConfigureOp(target=target, param="level")(sample)
        assert out is not None
        assert target.level == 7.5  # no compute chain → the incoming input IS the value
        assert out.meta["level"] == 7.5

    def test_key_overrides_metadata_key(self) -> None:
        sample = Sample(input=np.array([2.0]), target=None, metadata={})
        op = ConfigureOp(ops=[MaxOp()], target=ThresholdOp(low_op=">="), param="low_level", key="thr")
        out = op(sample)
        assert out is not None and out.meta["thr"] == 2.0
        assert "low_level" not in out.meta

    def test_missing_target_or_param_raise_lazily(self) -> None:
        sample = Sample(input=np.array([1.0]), target=None, metadata={})
        with pytest.raises(ValueError, match="'target' op is required"):
            ConfigureOp(param="x")(sample)
        with pytest.raises(ValueError, match="'param'"):
            ConfigureOp(target=ThresholdOp())(sample)

    def test_compute_chain_filtering_drops_sample(self) -> None:
        """A compute op returning None propagates the drop (FilterOp semantics)."""
        sample = Sample(input=np.array([1.0]), target=None, metadata={})
        op = ConfigureOp(ops=[lambda s: None], target=ThresholdOp(), param="low_level")
        assert op(sample) is None

    def test_fluid_markers_flow_lazily(self) -> None:
        """!class: markers in ops/target are flowed at first call (YAML-built ConfigureOp)."""
        from confluid.fluid import Class

        sample = Sample(input=np.array([1.0, 4.0]), target=None, metadata={})
        op = ConfigureOp(
            ops=[Class(MaxOp)],
            target=Class(ThresholdOp, low_op=">="),
            param="low_level",
        )
        out = op(sample)
        assert out is not None
        assert isinstance(op.target, ThresholdOp) and op.target.low_level == 4.0
        np.testing.assert_array_equal(out.input, [False, True])


# ---------------------------------------------------------------------------
# numpy.resolve_expression
# ---------------------------------------------------------------------------


class TestResolveExpression:
    def test_no_substitution_returns_verbatim(self) -> None:
        sample = Sample(input=None, target=None, metadata={})
        assert np_ops.resolve_expression("hello", sample) == "hello"
        assert np_ops.resolve_expression("5.5", sample) == "5.5"

    def test_metadata_substitution(self) -> None:
        sample = Sample(input=None, target=None, metadata={"snr": -30.5, "drone": "yz"})
        assert np_ops.resolve_expression("{snr}", sample) == "-30.5"
        assert np_ops.resolve_expression("-{snr}", sample) == "--30.5"
        assert np_ops.resolve_expression("{drone}", sample) == "yz"

    def test_env_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REF_SNR", "12.5")
        sample = Sample(input=None, target=None, metadata={})
        assert np_ops.resolve_expression("$REF_SNR", sample) == "12.5"
        assert np_ops.resolve_expression("-$REF_SNR", sample) == "-12.5"

    def test_missing_metadata_key_raises(self) -> None:
        sample = Sample(input=None, target=None, metadata={})
        with pytest.raises(KeyError, match="metadata key 'nope' missing"):
            np_ops.resolve_expression("{nope}", sample)

    def test_missing_env_var_raises(self) -> None:
        os.environ.pop("SAMPLEFLUX_TEST_NOPE", None)
        sample = Sample(input=None, target=None, metadata={})
        with pytest.raises(KeyError, match="environment variable 'SAMPLEFLUX_TEST_NOPE'"):
            np_ops.resolve_expression("$SAMPLEFLUX_TEST_NOPE", sample)


# ---------------------------------------------------------------------------
# ThresholdOp
# ---------------------------------------------------------------------------


class TestThresholdOp:
    def test_numeric_low_level(self) -> None:
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        out = np_ops.ThresholdOp(low_level=1.5)(Sample(input=arr, metadata={}))
        np.testing.assert_array_equal(out.input, [False, False, True, True])
        assert out.meta["threshold_low"] == 1.5
        assert "threshold_high" not in out.meta

    def test_numeric_high_level(self) -> None:
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        out = np_ops.ThresholdOp(high_level=1.5)(Sample(input=arr, metadata={}))
        np.testing.assert_array_equal(out.input, [True, True, False, False])
        assert out.meta["threshold_high"] == 1.5
        assert "threshold_low" not in out.meta

    def test_band_low_and_high(self) -> None:
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = np_ops.ThresholdOp(low_level=1.0, high_level=3.0)(Sample(input=arr, metadata={}))
        # strictly between 1.0 and 3.0 (default open interval: > and <)
        np.testing.assert_array_equal(out.input, [False, False, True, False, False])
        assert out.meta["threshold_low"] == 1.0
        assert out.meta["threshold_high"] == 3.0

    def test_low_level_inclusive(self) -> None:
        arr = np.array([0.0, 1.0, 2.0])
        # ">" excludes the boundary; ">=" includes it.
        strict = np_ops.ThresholdOp(low_level=1.0)(Sample(input=arr, metadata={}))
        np.testing.assert_array_equal(strict.input, [False, False, True])
        inclusive = np_ops.ThresholdOp(low_level=1.0, low_op=">=")(Sample(input=arr, metadata={}))
        np.testing.assert_array_equal(inclusive.input, [False, True, True])

    def test_high_level_inclusive(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        # "<" excludes the boundary; "<=" includes it.
        strict = np_ops.ThresholdOp(high_level=2.0)(Sample(input=arr, metadata={}))
        np.testing.assert_array_equal(strict.input, [True, False, False])
        inclusive = np_ops.ThresholdOp(high_level=2.0, high_op="<=")(Sample(input=arr, metadata={}))
        np.testing.assert_array_equal(inclusive.input, [True, True, False])

    def test_closed_band(self) -> None:
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = np_ops.ThresholdOp(low_level=1.0, high_level=3.0, low_op=">=", high_op="<=")(
            Sample(input=arr, metadata={})
        )
        # closed interval [1.0, 3.0]: both boundaries kept
        np.testing.assert_array_equal(out.input, [False, True, True, True, False])

    def test_invalid_operator_rejected(self) -> None:
        # ``low_op`` is a closed ``Literal[">", ">="]`` — confluid's pydantic
        # validation rejects anything else before the body runs.
        from pydantic import ValidationError

        with pytest.raises((ValueError, ValidationError)):
            np_ops.ThresholdOp(low_level=1.0, low_op=">>")  # type: ignore[arg-type]

    def test_string_numeric(self) -> None:
        arr = np.array([0.0, 1.0, 2.0])
        out = np_ops.ThresholdOp(low_level="1.5")(Sample(input=arr))
        np.testing.assert_array_equal(out.input, [False, False, True])

    def test_metadata_lookup(self) -> None:
        arr = np.array([-50.0, -30.0, -10.0])
        sample = Sample(input=arr, target=None, metadata={"reference_snr_level": -25.0})
        out = np_ops.ThresholdOp(low_level="{reference_snr_level}")(sample)
        np.testing.assert_array_equal(out.input, [False, False, True])
        assert out.meta["threshold_low"] == -25.0

    def test_metadata_lookup_with_negation(self) -> None:
        arr = np.array([-50.0, -30.0, -10.0])
        sample = Sample(input=arr, target=None, metadata={"reference_snr_level": 30.0})
        out = np_ops.ThresholdOp(low_level="-{reference_snr_level}")(sample)
        np.testing.assert_array_equal(out.input, [False, False, True])
        assert out.meta["threshold_low"] == -30.0

    def test_env_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAMPLEFLUX_TEST_THRESHOLD", "1.0")
        arr = np.array([0.0, 1.0, 2.0])
        out = np_ops.ThresholdOp(low_level="$SAMPLEFLUX_TEST_THRESHOLD")(Sample(input=arr))
        np.testing.assert_array_equal(out.input, [False, False, True])

    def test_high_level_expression(self) -> None:
        arr = np.array([-50.0, -30.0, -10.0])
        sample = Sample(input=arr, target=None, metadata={"ceiling": -20.0})
        out = np_ops.ThresholdOp(high_level="{ceiling}")(sample)
        np.testing.assert_array_equal(out.input, [True, True, False])
        assert out.meta["threshold_high"] == -20.0

    def test_raises_when_no_bounds(self) -> None:
        op = np_ops.ThresholdOp()  # lazy: construction succeeds (zero-arg)
        with pytest.raises(ValueError, match="at least one of 'low_level' / 'high_level'"):
            op(Sample(input=np.zeros(3)))

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ThresholdOp expects an np.ndarray"):
            np_ops.ThresholdOp(low_level=0.0)(Sample(input=[1.0, 2.0]))

    def test_raises_on_non_numeric_resolution(self) -> None:
        sample = Sample(input=np.array([0.0]), target=None, metadata={"drone": "yz"})
        with pytest.raises(ValueError, match="not a number"):
            np_ops.ThresholdOp(low_level="{drone}")(sample)

    def test_raises_on_bad_value_type(self) -> None:
        # Confluid's ``@configurable`` validates kwargs against the
        # auto-generated pydantic schema before the body runs. ``low_level`` is
        # typed as ``float | int | str | None``, so a list is rejected at the
        # validation layer first; the body's hand-rolled ``TypeError``
        # remains as a safety net.
        from pydantic import ValidationError

        with pytest.raises((TypeError, ValidationError)):
            np_ops.ThresholdOp(low_level=[1, 2])(Sample(input=np.array([0.0])))  # type: ignore[arg-type]

    def test_numpy_scalar_and_zero_d_array_bounds_accepted(self) -> None:
        # A value chain (MaxOp → FormulaOp → ConfigureOp) injects a NumPy scalar / 0-d array into a
        # bound via setattr, bypassing the pydantic ctor. np.float64 SUBCLASSES Python float (so it
        # slipped through the old `isinstance(bound, (int, float))`), but np.float32 does NOT —
        # _resolve must accept anything float() accepts. Live regression for a float32 spectrogram:
        # "ThresholdOp bounds must be a number or expression string; got float32".
        arr = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        for bound in (np.float32(2.0), np.array(2.0)):
            op = np_ops.ThresholdOp(low_op=">=")
            op.low_level = bound  # type: ignore[assignment]  # post-construction injection (ConfigureOp does this)
            out = op(Sample(input=arr))
            np.testing.assert_array_equal(out.input, [False, False, True])
            assert out.meta["threshold_low"] == 2.0


def test_threshold_comparison_maps_match_literals() -> None:
    # The operator-dispatch dicts must stay in lockstep with their closed
    # Literals (one source of truth) — a new operator added to the Literal but
    # not the map (or vice versa) is a bug this pins.
    from typing import get_args

    assert set(np_ops._LOW_COMPARISONS) == set(get_args(np_ops.LowComparison))
    assert set(np_ops._HIGH_COMPARISONS) == set(get_args(np_ops.HighComparison))


# ---------------------------------------------------------------------------
# ConnectedComponentsOp
# ---------------------------------------------------------------------------


class TestConnectedComponentsOp:
    def test_two_separate_blobs(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[1:3, 1:3] = True  # 2x2 blob at (1,1)
        mask[6:9, 6:9] = True  # 3x3 blob at (6,6)
        out = np_ops.ConnectedComponentsOp(min_area_bins=1, connectivity=4)(Sample(input=mask))
        assert sorted(out.input) == [(1, 2, 1, 2), (6, 8, 6, 8)]

    def test_min_area_drops_small_components(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True  # area 1
        mask[5:7, 5:7] = True  # area 4
        out = np_ops.ConnectedComponentsOp(min_area_bins=2, connectivity=4)(Sample(input=mask))
        assert out.input == [(5, 6, 5, 6)]

    def test_connectivity_4_keeps_diagonals_separate(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        mask[2, 2] = True
        out = np_ops.ConnectedComponentsOp(min_area_bins=1, connectivity=4)(Sample(input=mask))
        assert len(out.input) == 3

    def test_connectivity_8_merges_diagonals(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        mask[2, 2] = True
        out = np_ops.ConnectedComponentsOp(min_area_bins=1, connectivity=8)(Sample(input=mask))
        assert len(out.input) == 1
        assert out.input[0] == (0, 2, 0, 2)

    def test_empty_mask_returns_empty_list(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        out = np_ops.ConnectedComponentsOp(connectivity=4)(Sample(input=mask))
        assert out.input == []

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ConnectedComponentsOp expects an np.ndarray"):
            np_ops.ConnectedComponentsOp()(Sample(input=[[True, False]]))

    def test_raises_on_non_2d(self) -> None:
        with pytest.raises(ValueError, match="expects a 2-D mask"):
            np_ops.ConnectedComponentsOp()(Sample(input=np.array([True, False])))

    def test_validation_rejects_bad_min_area(self) -> None:
        op = np_ops.ConnectedComponentsOp(min_area_bins=0)  # lazy: construction succeeds
        with pytest.raises(ValueError, match="min_area_bins must be >= 1"):
            op(Sample(input=np.zeros((2, 2), dtype=bool)))

    def test_validation_rejects_bad_connectivity(self) -> None:
        op = np_ops.ConnectedComponentsOp(connectivity=6)
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            op(Sample(input=np.zeros((2, 2), dtype=bool)))


# ---------------------------------------------------------------------------
# Torch SqueezeOp
# ---------------------------------------------------------------------------


class TestTorchSqueezeOp:
    """Tests for torch SqueezeOp."""

    def test_squeeze_all_size1_dims(self) -> None:
        tensor = torch.zeros(1, 3, 1, 4)
        result = SqueezeOp()(Sample(input=tensor))
        assert result.input.shape == (3, 4)

    def test_squeeze_specific_dim(self) -> None:
        tensor = torch.zeros(1, 3, 4)
        result = SqueezeOp(dim=0)(Sample(input=tensor))
        assert result.input.shape == (3, 4)

    def test_squeeze_non_unit_dim_is_noop(self) -> None:
        # torch.squeeze leaves non-size-1 dims unchanged
        tensor = torch.zeros(2, 3)
        result = SqueezeOp(dim=0)(Sample(input=tensor))
        assert result.input.shape == (2, 3)

    def test_preserves_target_and_metadata(self) -> None:
        tensor = torch.zeros(1, 4)
        result = SqueezeOp()(Sample(input=tensor, target=7, metadata={"k": "v"}))
        assert result.target == 7
        assert result.meta == {"k": "v"}

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="SqueezeOp expects a torch.Tensor"):
            SqueezeOp()(Sample(input=np.zeros((1, 3))))

    def test_zero_arg_construction(self) -> None:
        op = SqueezeOp()
        assert op.dim is None

    def test_negative_dim(self) -> None:
        tensor = torch.zeros(3, 1)
        result = SqueezeOp(dim=-1)(Sample(input=tensor))
        assert result.input.shape == (3,)


# ---------------------------------------------------------------------------
# Torch UnsqueezeOp
# ---------------------------------------------------------------------------


class TestTorchUnsqueezeOp:
    """Tests for torch UnsqueezeOp."""

    def test_unsqueeze_at_dim0(self) -> None:
        tensor = torch.zeros(3, 4)
        result = UnsqueezeOp(dim=0)(Sample(input=tensor))
        assert result.input.shape == (1, 3, 4)

    def test_unsqueeze_at_dim1(self) -> None:
        tensor = torch.zeros(3, 4)
        result = UnsqueezeOp(dim=1)(Sample(input=tensor))
        assert result.input.shape == (3, 1, 4)

    def test_unsqueeze_at_last_dim(self) -> None:
        tensor = torch.zeros(3, 4)
        result = UnsqueezeOp(dim=-1)(Sample(input=tensor))
        assert result.input.shape == (3, 4, 1)

    def test_default_dim_is_zero(self) -> None:
        tensor = torch.zeros(5)
        result = UnsqueezeOp()(Sample(input=tensor))
        assert result.input.shape == (1, 5)

    def test_preserves_target_and_metadata(self) -> None:
        tensor = torch.zeros(4)
        result = UnsqueezeOp()(Sample(input=tensor, target=2, metadata={"x": 1}))
        assert result.target == 2
        assert result.meta == {"x": 1}

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="UnsqueezeOp expects a torch.Tensor"):
            UnsqueezeOp()(Sample(input=np.zeros(3)))

    def test_roundtrip_squeeze_unsqueeze(self) -> None:
        tensor = torch.zeros(3, 4)
        squeezed = UnsqueezeOp(dim=0)(Sample(input=tensor))
        restored = SqueezeOp(dim=0)(squeezed)
        assert restored.input.shape == tensor.shape


# ---------------------------------------------------------------------------
# Numpy SqueezeOp
# ---------------------------------------------------------------------------


class TestNumpySqueezeOp:
    """Tests for numpy SqueezeOp."""

    def test_squeeze_all_size1_axes(self) -> None:
        arr = np.zeros((1, 3, 1, 4))
        result = np_ops.SqueezeOp()(Sample(input=arr))
        assert result.input.shape == (3, 4)

    def test_squeeze_specific_axis(self) -> None:
        arr = np.zeros((1, 3, 4))
        result = np_ops.SqueezeOp(axis=0)(Sample(input=arr))
        assert result.input.shape == (3, 4)

    def test_squeeze_non_unit_axis_raises(self) -> None:
        arr = np.zeros((2, 3))
        with pytest.raises(ValueError):
            np_ops.SqueezeOp(axis=0)(Sample(input=arr))

    def test_preserves_target_and_metadata(self) -> None:
        arr = np.zeros((1, 4))
        result = np_ops.SqueezeOp()(Sample(input=arr, target=7, metadata={"k": "v"}))
        assert result.target == 7
        assert result.meta == {"k": "v"}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="SqueezeOp expects an np.ndarray"):
            np_ops.SqueezeOp()(Sample(input=torch.zeros(1, 3)))

    def test_zero_arg_construction(self) -> None:
        op = np_ops.SqueezeOp()
        assert op.axis is None

    def test_negative_axis(self) -> None:
        arr = np.zeros((3, 1))
        result = np_ops.SqueezeOp(axis=-1)(Sample(input=arr))
        assert result.input.shape == (3,)


# ---------------------------------------------------------------------------
# Numpy UnsqueezeOp
# ---------------------------------------------------------------------------


class TestNumpyUnsqueezeOp:
    """Tests for numpy UnsqueezeOp."""

    def test_unsqueeze_at_axis0(self) -> None:
        arr = np.zeros((3, 4))
        result = np_ops.UnsqueezeOp(axis=0)(Sample(input=arr))
        assert result.input.shape == (1, 3, 4)

    def test_unsqueeze_at_axis1(self) -> None:
        arr = np.zeros((3, 4))
        result = np_ops.UnsqueezeOp(axis=1)(Sample(input=arr))
        assert result.input.shape == (3, 1, 4)

    def test_unsqueeze_at_last_axis(self) -> None:
        arr = np.zeros((3, 4))
        result = np_ops.UnsqueezeOp(axis=-1)(Sample(input=arr))
        assert result.input.shape == (3, 4, 1)

    def test_default_axis_is_zero(self) -> None:
        arr = np.zeros(5)
        result = np_ops.UnsqueezeOp()(Sample(input=arr))
        assert result.input.shape == (1, 5)

    def test_preserves_target_and_metadata(self) -> None:
        arr = np.zeros(4)
        result = np_ops.UnsqueezeOp()(Sample(input=arr, target=2, metadata={"x": 1}))
        assert result.target == 2
        assert result.meta == {"x": 1}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="UnsqueezeOp expects an np.ndarray"):
            np_ops.UnsqueezeOp()(Sample(input=torch.zeros(3)))

    def test_roundtrip_squeeze_unsqueeze(self) -> None:
        arr = np.zeros((3, 4))
        unsqueezed = np_ops.UnsqueezeOp(axis=0)(Sample(input=arr))
        restored = np_ops.SqueezeOp(axis=0)(unsqueezed)
        assert restored.input.shape == arr.shape


# ---------------------------------------------------------------------------
# DropMetadataOp
# ---------------------------------------------------------------------------
class TestDropMetadataOp:
    def test_literal_exclude_drops_exact_keys(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        # A pattern with no wildcards is an EXACT key match; a missing key is ignored.
        sample = Sample(input=np.zeros(2), target=None, metadata={"keep": 1, "drop_me": 2, "also": 3})
        out = DropMetadataOp(exclude=["drop_me", "also", "nope"])(sample)
        assert out.meta == {"keep": 1}

    def test_glob_star_drops_all_matching(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        meta = {"real": 1, "__taidal_stash_456:input": [1j], "__taidal_stash_456:target": [2j]}
        out = DropMetadataOp(exclude=["__taidal_stash*"])(Sample(input=np.zeros(2), metadata=meta))
        assert out.meta == {"real": 1}

    def test_glob_mid_wildcard_is_specific(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        # `__taidal_stash_456:*input` drops ONLY node 456's input stash — keeps its target and
        # other nodes' inputs.
        meta = {
            "__taidal_stash_456:input": 1,
            "__taidal_stash_456:target": 2,
            "__taidal_stash_99:input": 3,
        }
        out = DropMetadataOp(exclude=["__taidal_stash_456:*input"])(Sample(input=np.zeros(2), metadata=meta))
        assert out.meta == {"__taidal_stash_456:target": 2, "__taidal_stash_99:input": 3}

    def test_multiple_exclude_patterns_any_match(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        meta = {"a": 1, "b": 2, "__t_x": 3, "__t_y": 4}
        out = DropMetadataOp(exclude=["a", "__t_*"])(Sample(input=np.zeros(2), metadata=meta))
        assert out.meta == {"b": 2}

    def test_question_mark_and_set_globs(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        meta = {"img0": 1, "img1": 2, "imgX": 3, "image": 4}
        out = DropMetadataOp(exclude=["img[0-9]"])(Sample(input=np.zeros(2), metadata=meta))
        assert out.meta == {"imgX": 3, "image": 4}  # only single-digit img0/img1 dropped

    def test_matching_is_case_sensitive(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        out = DropMetadataOp(exclude=["key"])(Sample(input=np.zeros(2), metadata={"Key": 1, "key": 2}))
        assert out.meta == {"Key": 1}

    def test_include_protects_keys_from_exclude(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        # include WINS: drop every stash key EXCEPT node 456's (carved out by include).
        meta = {
            "real": 1,
            "__taidal_stash_456:input": 2,
            "__taidal_stash_456:target": 3,
            "__taidal_stash_99:input": 4,
        }
        out = DropMetadataOp(exclude=["__taidal_stash*"], include=["__taidal_stash_456:*"])(
            Sample(input=np.zeros(2), metadata=meta)
        )
        assert out.meta == {"real": 1, "__taidal_stash_456:input": 2, "__taidal_stash_456:target": 3}

    def test_include_without_exclude_drops_nothing(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        meta = {"a": 1, "b": 2}
        out = DropMetadataOp(include=["a"])(Sample(input=np.zeros(2), metadata=meta))
        assert out.meta == {"a": 1, "b": 2}  # include only protects against exclude

    def test_zero_arg_is_identity_metadata(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        meta = {"a": 1, "b": 2}
        out = DropMetadataOp()(Sample(input=np.zeros(2), metadata=meta))
        assert out.meta == {"a": 1, "b": 2}

    def test_copy_on_write_does_not_mutate_original(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        original = {"a": 1, "drop": 2}
        out = DropMetadataOp(exclude=["drop"])(Sample(input=np.zeros(2), metadata=original))
        assert original == {"a": 1, "drop": 2}  # untouched
        assert out.meta == {"a": 1}

    def test_input_and_target_untouched(self) -> None:
        from sampleflux.ops.metadata import DropMetadataOp

        arr = np.arange(3)
        out = DropMetadataOp(exclude=["x"])(Sample(input=arr, target=7, metadata={"x": 1, "y": 2}))
        np.testing.assert_array_equal(out.input, arr)
        assert out.target == 7


# ---------------------------------------------------------------------------
# PrintSampleOp
# ---------------------------------------------------------------------------
class TestPrintSampleOp:
    def test_returns_sample_unchanged(self) -> None:
        from sampleflux.ops.debug import PrintSampleOp

        sample = Sample(input=np.zeros(3), target=1, metadata={"a": 1})
        out = PrintSampleOp(to_console=False)(sample)
        assert out is sample

    def test_prints_to_console(self, capsys: pytest.CaptureFixture) -> None:
        from sampleflux.ops.debug import PrintSampleOp

        PrintSampleOp(label="probe")(Sample(input=np.zeros((2, 3)), target=None, metadata={"k": 1}))
        captured = capsys.readouterr().out
        assert "[probe #0]" in captured
        assert "shape=(2, 3)" in captured  # input summary
        assert "'k'" in captured  # metadata key

    def test_summarizes_large_array_metadata_without_dumping(self, capsys: pytest.CaptureFixture) -> None:
        from sampleflux.ops.debug import PrintSampleOp

        big = np.arange(100000, dtype=np.complex64)  # would flood / not be reprable in full
        PrintSampleOp(label="p")(Sample(input=np.zeros(2), metadata={"iq": big}))
        out = capsys.readouterr().out
        assert "shape=(100000,)" in out and "complex64" in out
        assert "..." in out and "50000" not in out  # values elided, not dumped in full

    def test_prints_small_array_values(self, capsys: pytest.CaptureFixture) -> None:
        from sampleflux.ops.debug import PrintSampleOp

        PrintSampleOp(label="p")(Sample(input=np.array([1, 2, 3]), target=None, metadata={}))
        out = capsys.readouterr().out
        assert "values=[1, 2, 3]" in out  # actual values shown for a small array

    def test_limit_caps_emissions_but_passes_all(self, capsys: pytest.CaptureFixture) -> None:
        from sampleflux.ops.debug import PrintSampleOp

        op = PrintSampleOp(label="p", limit=2)
        for _ in range(5):
            assert op(Sample(input=np.zeros(1), metadata={})) is not None  # all pass through
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[p #")]
        assert len(lines) == 2  # only the first 2 printed

    def test_to_console_false_is_silent_on_stdout(self, capsys: pytest.CaptureFixture) -> None:
        from sampleflux.ops.debug import PrintSampleOp

        PrintSampleOp(to_console=False)(Sample(input=np.zeros(1), metadata={}))
        assert capsys.readouterr().out == ""
