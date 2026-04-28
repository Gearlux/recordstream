"""Tests for dataflux.ops: torch and numpy variants."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from dataflux.ops import (
    CopyInputOp,
    CopyMetadataOp,
    CopySampleOp,
    CopyTargetOp,
    RescaleOp,
    StandardizeOp,
    StashInputOp,
    SwapInputTargetOp,
    Tee,
    ToTensorOp,
    UnstashInputOp,
)
from dataflux.ops import numpy as np_ops
from dataflux.sample import Sample

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
        assert result.metadata == {"k": "v"}


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
        assert result.metadata == {"key": "val"}

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="RescaleOp expects a torch.Tensor"):
            RescaleOp(in_min=0.0, in_max=255.0)(Sample(input=np.array([1, 2, 3])))

    def test_validation_rejects_bad_input_range(self) -> None:
        with pytest.raises(ValueError, match="require in_min < in_max"):
            RescaleOp(in_min=10.0, in_max=10.0)

    def test_validation_rejects_bad_output_range(self) -> None:
        with pytest.raises(ValueError, match="require out_min < out_max"):
            RescaleOp(in_min=0.0, in_max=1.0, out_min=5.0, out_max=5.0)

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
        assert result.metadata == {"a": 1}

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
        assert result.metadata == {"a": 1}

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
        with pytest.raises(ValueError, match="ClipPercentilesOp: require"):
            np_ops.ClipPercentilesOp(low=low, high=high)


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
        assert result.metadata == {"key": "val"}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="RescaleOp expects an np.ndarray"):
            np_ops.RescaleOp(in_min=0.0, in_max=1.0)(Sample(input=[1.0, 2.0]))

    def test_validation_rejects_bad_input_range(self) -> None:
        with pytest.raises(ValueError, match="require in_min < in_max"):
            np_ops.RescaleOp(in_min=10.0, in_max=10.0)

    def test_validation_rejects_bad_output_range(self) -> None:
        with pytest.raises(ValueError, match="require out_min < out_max"):
            np_ops.RescaleOp(in_min=0.0, in_max=1.0, out_min=5.0, out_max=5.0)

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
        with pytest.raises(ValueError, match="value string must be 'min' or 'max'"):
            np_ops.ReplaceNonFiniteOp(value="median")


# ---------------------------------------------------------------------------
# Tee
# ---------------------------------------------------------------------------


class TestTee:
    def test_two_branches_share_metadata(self) -> None:
        sample = Sample(input=np.array([1.0]), target=None, metadata={})

        def writer_a(s: Sample) -> Sample:
            s.metadata["a"] = 1
            return s

        def writer_b(s: Sample) -> Sample:
            assert s.metadata["a"] == 1  # branch A's write is visible
            s.metadata["b"] = 2
            return s

        out = Tee(branches=[[writer_a], [writer_b]])(sample)
        assert out is not None
        assert out.metadata == {"a": 1, "b": 2}

    def test_branches_run_sequentially(self) -> None:
        from typing import Callable

        order: list[str] = []

        def make(tag: str) -> Callable[[Sample], Sample]:
            def op(s: Sample) -> Sample:
                order.append(tag)
                return s

            return op

        Tee(branches=[[make("A1"), make("A2")], [make("B1"), make("B2")]])(Sample(input=None))
        assert order == ["A1", "A2", "B1", "B2"]

    def test_none_propagates(self) -> None:
        def filter_out(s: Sample) -> None:
            return None

        def should_not_run(s: Sample) -> Sample:
            raise AssertionError("downstream branch must not run after None")

        out = Tee(branches=[[filter_out], [should_not_run]])(Sample(input=1))
        assert out is None

    def test_input_mutations_flow_into_next_branch(self) -> None:
        def to_zero(s: Sample) -> Sample:
            return s._replace(input=0)

        def must_see_zero(s: Sample) -> Sample:
            assert s.input == 0
            return s._replace(input=99)

        out = Tee(branches=[[to_zero], [must_see_zero]])(Sample(input=42))
        assert out is not None
        assert out.input == 99


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
        assert out.metadata is not sample.metadata
        assert out.metadata["k"] is not sample.metadata["k"]

    def test_copy_input_only_copies_input(self) -> None:
        sample = Sample(input=np.array([1.0]), target=[5], metadata={"k": "v"})
        out = CopyInputOp()(sample)
        assert out.input is not sample.input
        assert out.target is sample.target
        assert out.metadata is sample.metadata

    def test_copy_target_only_copies_target(self) -> None:
        sample = Sample(input=[1, 2], target=[10, 20], metadata={})
        out = CopyTargetOp()(sample)
        assert out.target is not sample.target
        assert out.input is sample.input

    def test_copy_metadata_breaks_aliasing(self) -> None:
        meta = {"k": [1]}
        sample = Sample(input=None, target=None, metadata=meta)
        out = CopyMetadataOp()(sample)
        out.metadata["k"].append(2)
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
        assert out.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# StashInputOp / UnstashInputOp
# ---------------------------------------------------------------------------


class TestStashUnstash:
    def test_stash_aliases_by_default(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=arr, target=None, metadata={})
        out = StashInputOp(key="snap")(sample)
        assert out.metadata["snap"] is arr
        assert out.input is arr

    def test_stash_with_copy_deepcopies(self) -> None:
        arr = np.array([1.0, 2.0])
        sample = Sample(input=arr, target=None, metadata={})
        out = StashInputOp(key="snap", copy=True)(sample)
        assert out.metadata["snap"] is not arr
        np.testing.assert_array_equal(out.metadata["snap"], arr)

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
        """Default copy=True prevents branch-A's in-place write from leaking into branch-B."""
        arr = np.array([1.0, 2.0, 3.0])
        sample = Sample(input=None, target=None, metadata={"snap": arr})
        a = UnstashInputOp(key="snap")(sample)
        a.input.fill(99.0)  # in-place mutation on branch A's restored array
        b = UnstashInputOp(key="snap")(sample)
        np.testing.assert_array_equal(b.input, [1.0, 2.0, 3.0])


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
        os.environ.pop("DATAFLUX_TEST_NOPE", None)
        sample = Sample(input=None, target=None, metadata={})
        with pytest.raises(KeyError, match="environment variable 'DATAFLUX_TEST_NOPE'"):
            np_ops.resolve_expression("$DATAFLUX_TEST_NOPE", sample)


# ---------------------------------------------------------------------------
# ThresholdOp
# ---------------------------------------------------------------------------


class TestThresholdOp:
    def test_numeric_value(self) -> None:
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        out = np_ops.ThresholdOp(value=1.5)(Sample(input=arr))
        np.testing.assert_array_equal(out.input, [False, False, True, True])
        assert out.metadata["threshold"] == 1.5

    def test_string_numeric(self) -> None:
        arr = np.array([0.0, 1.0, 2.0])
        out = np_ops.ThresholdOp(value="1.5")(Sample(input=arr))
        np.testing.assert_array_equal(out.input, [False, False, True])

    def test_metadata_lookup(self) -> None:
        arr = np.array([-50.0, -30.0, -10.0])
        sample = Sample(input=arr, target=None, metadata={"reference_snr_level": -25.0})
        out = np_ops.ThresholdOp(value="{reference_snr_level}")(sample)
        np.testing.assert_array_equal(out.input, [False, False, True])
        assert out.metadata["threshold"] == -25.0

    def test_metadata_lookup_with_negation(self) -> None:
        arr = np.array([-50.0, -30.0, -10.0])
        sample = Sample(input=arr, target=None, metadata={"reference_snr_level": 30.0})
        out = np_ops.ThresholdOp(value="-{reference_snr_level}")(sample)
        np.testing.assert_array_equal(out.input, [False, False, True])
        assert out.metadata["threshold"] == -30.0

    def test_env_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAFLUX_TEST_THRESHOLD", "1.0")
        arr = np.array([0.0, 1.0, 2.0])
        out = np_ops.ThresholdOp(value="$DATAFLUX_TEST_THRESHOLD")(Sample(input=arr))
        np.testing.assert_array_equal(out.input, [False, False, True])

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="ThresholdOp expects an np.ndarray"):
            np_ops.ThresholdOp(value=0.0)(Sample(input=[1.0, 2.0]))

    def test_raises_on_non_numeric_resolution(self) -> None:
        sample = Sample(input=np.array([0.0]), target=None, metadata={"drone": "yz"})
        with pytest.raises(ValueError, match="not a number"):
            np_ops.ThresholdOp(value="{drone}")(sample)

    def test_raises_on_bad_value_type(self) -> None:
        with pytest.raises(TypeError, match="must be a number or expression string"):
            np_ops.ThresholdOp(value=[1, 2])(Sample(input=np.array([0.0])))  # type: ignore[arg-type]


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
        with pytest.raises(ValueError, match="min_area_bins must be >= 1"):
            np_ops.ConnectedComponentsOp(min_area_bins=0)

    def test_validation_rejects_bad_connectivity(self) -> None:
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            np_ops.ConnectedComponentsOp(connectivity=6)
