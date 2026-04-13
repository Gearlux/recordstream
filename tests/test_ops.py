"""Tests for dataflux.ops: torch and numpy variants."""

import numpy as np
import pytest
import torch
from PIL import Image

from dataflux.ops import NormalizeOp, StandardizeOp, ToTensorOp
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
# Torch NormalizeOp
# ---------------------------------------------------------------------------


class TestNormalizeOp:
    """Tests for torch NormalizeOp."""

    def test_default_range(self) -> None:
        tensor = torch.tensor([0.0, 128.0, 255.0])
        result = NormalizeOp()(Sample(input=tensor))
        assert result.input[0] == 0.0
        assert abs(result.input[1] - 128.0 / 255.0) < 1e-6
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_custom_range(self) -> None:
        tensor = torch.tensor([10.0, 55.0, 100.0])
        result = NormalizeOp(min_value=10.0, max_value=100.0)(Sample(input=tensor))
        assert abs(result.input[0] - 0.0) < 1e-6
        assert abs(result.input[1] - 0.5) < 1e-6
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_uint8_converts_to_float(self) -> None:
        tensor = torch.tensor([0, 128, 255], dtype=torch.uint8)
        result = NormalizeOp()(Sample(input=tensor))
        assert result.input.dtype == torch.float32
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_preserves_float64(self) -> None:
        tensor = torch.tensor([0.0, 255.0], dtype=torch.float64)
        result = NormalizeOp()(Sample(input=tensor))
        assert result.input.dtype == torch.float64

    def test_preserves_target_and_metadata(self) -> None:
        tensor = torch.tensor([128.0])
        result = NormalizeOp()(Sample(input=tensor, target=7, metadata={"key": "val"}))
        assert result.target == 7
        assert result.metadata == {"key": "val"}

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="NormalizeOp expects a torch.Tensor"):
            NormalizeOp()(Sample(input=np.array([1, 2, 3])))

    def test_pipeline_to_tensor_then_normalize(self) -> None:
        """Integration: ToTensorOp(normalize=False) -> NormalizeOp()."""
        img = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))
        sample = Sample(input=img)
        sample = ToTensorOp(normalize=False)(sample)
        sample = NormalizeOp(min_value=0.0, max_value=255.0)(sample)
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
# Numpy NormalizeOp
# ---------------------------------------------------------------------------


class TestNpNormalizeOp:
    """Tests for numpy NormalizeOp."""

    def test_default_range(self) -> None:
        arr = np.array([0.0, 128.0, 255.0], dtype=np.float32)
        result = np_ops.NormalizeOp()(Sample(input=arr))
        assert result.input[0] == 0.0
        assert abs(result.input[1] - 128.0 / 255.0) < 1e-6
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_custom_range(self) -> None:
        arr = np.array([10.0, 55.0, 100.0], dtype=np.float32)
        result = np_ops.NormalizeOp(min_value=10.0, max_value=100.0)(Sample(input=arr))
        assert abs(result.input[0] - 0.0) < 1e-6
        assert abs(result.input[1] - 0.5) < 1e-6
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_uint8_converts_to_float32(self) -> None:
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = np_ops.NormalizeOp()(Sample(input=arr))
        assert result.input.dtype == np.float32
        assert abs(result.input[2] - 1.0) < 1e-6

    def test_preserves_float64(self) -> None:
        arr = np.array([0.0, 255.0], dtype=np.float64)
        result = np_ops.NormalizeOp()(Sample(input=arr))
        assert result.input.dtype == np.float64

    def test_pil_image_input(self) -> None:
        img = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))
        result = np_ops.NormalizeOp()(Sample(input=img))
        assert isinstance(result.input, np.ndarray)
        assert result.input.dtype == np.float32
        assert abs(result.input.max() - 200.0 / 255.0) < 1e-6

    def test_preserves_target_and_metadata(self) -> None:
        arr = np.array([128.0], dtype=np.float32)
        result = np_ops.NormalizeOp()(Sample(input=arr, target=7, metadata={"key": "val"}))
        assert result.target == 7
        assert result.metadata == {"key": "val"}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="NormalizeOp expects an np.ndarray"):
            np_ops.NormalizeOp()(Sample(input=torch.tensor([1.0])))

    def test_pipeline_normalize_then_to_tensor(self) -> None:
        """Integration: NpNormalizeOp() -> ToTensorOp(normalize=False)."""
        img = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))
        sample = Sample(input=img)
        sample = np_ops.NormalizeOp(min_value=0.0, max_value=255.0)(sample)
        sample = ToTensorOp(normalize=False)(sample)
        assert isinstance(sample.input, torch.Tensor)
        assert sample.input.dtype == torch.float32
        assert abs(sample.input.max().item() - 200.0 / 255.0) < 1e-6


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
