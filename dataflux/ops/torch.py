from typing import Sequence, Union

import numpy as np
import torch
from confluid import configurable

from dataflux.sample import Sample


@configurable
class ToTensorOp:
    """
    Converts input (PIL Image, NumPy array, etc.) to a Torch Tensor.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def __call__(self, sample: Sample) -> Sample:
        img = sample.input

        # Handle PIL / PngImageFile
        if hasattr(img, "convert"):
            # Ensure grayscale or RGB as needed, but for generic we just array it
            img = np.array(img)

        # Convert to Tensor
        if isinstance(img, np.ndarray):
            # Standard Vision format: [H, W, C] -> [C, H, W]
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)
            elif img.ndim == 2:
                img = img[np.newaxis, :]

            tensor = torch.from_numpy(img)
        else:
            tensor = torch.as_tensor(img)

        # Normalize 0-255 to 0-1
        if self.normalize and tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif self.normalize and tensor.max() > 1.0:
            # Fallback for floats that are still in 0-255 range
            tensor = tensor / 255.0

        return sample._replace(input=tensor)


@configurable
class RescaleOp:
    """Affine rescale a torch.Tensor from ``[in_min, in_max]`` to ``[out_min, out_max]``.

    The default ``out_min=0.0`` / ``out_max=1.0`` covers the common
    ``[0, 255] -> [0, 1]`` image-normalization case. Integer dtypes are
    promoted to ``float32`` (``float64`` is preserved).

    Args:
        in_min: Lower edge of the input range. Required.
        in_max: Upper edge of the input range, must be ``> in_min``. Required.
        out_min: Lower edge of the output range. Default ``0.0``.
        out_max: Upper edge of the output range, must be ``> out_min``. Default ``1.0``.
        clip: When True (default), clamp values outside ``[in_min, in_max]``
            before rescaling. When False, extrapolate linearly.
    """

    def __init__(
        self,
        in_min: float,
        in_max: float,
        out_min: float = 0.0,
        out_max: float = 1.0,
        clip: bool = True,
    ) -> None:
        if not (in_min < in_max):
            raise ValueError(f"RescaleOp: require in_min < in_max; got in_min={in_min}, in_max={in_max}")
        if not (out_min < out_max):
            raise ValueError(f"RescaleOp: require out_min < out_max; got out_min={out_min}, out_max={out_max}")
        self.in_min = float(in_min)
        self.in_max = float(in_max)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.clip = bool(clip)

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"RescaleOp expects a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dtype != torch.float32 and tensor.dtype != torch.float64:
            tensor = tensor.float()
        src = tensor.clamp(self.in_min, self.in_max) if self.clip else tensor
        scaled = (src - self.in_min) / (self.in_max - self.in_min)
        out = scaled * (self.out_max - self.out_min) + self.out_min
        return sample._replace(input=out)


@configurable
class StandardizeOp:
    """
    Standardizes tensor values with given mean and standard deviation.

    Formula: output = (input - mean) / std

    mean/std can be a single float (applied uniformly) or a sequence of
    per-channel values that broadcasts over [C, H, W] format.
    """

    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"StandardizeOp expects a torch.Tensor, got {type(tensor).__name__}")

        if tensor.dtype != torch.float32 and tensor.dtype != torch.float64:
            tensor = tensor.float()

        mean_t = torch.tensor(
            [self.mean] if isinstance(self.mean, (int, float)) else self.mean, dtype=tensor.dtype, device=tensor.device
        )
        std_t = torch.tensor(
            [self.std] if isinstance(self.std, (int, float)) else self.std, dtype=tensor.dtype, device=tensor.device
        )

        # Reshape to [C, 1, 1, ...] for broadcasting over [C, H, W]
        mean_t = mean_t.view(-1, *([1] * (tensor.ndim - 1)))
        std_t = std_t.view(-1, *([1] * (tensor.ndim - 1)))

        tensor = (tensor - mean_t) / std_t

        return sample._replace(input=tensor)
