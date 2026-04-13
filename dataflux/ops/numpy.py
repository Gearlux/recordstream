from typing import Sequence, Union

import numpy as np
from confluid import configurable

from dataflux.sample import Sample


@configurable
class NormalizeOp:
    """
    Scales ndarray values from [min_value, max_value] to [0, 1].

    Formula: output = (input - min_value) / (max_value - min_value)

    Handles PIL images by converting to ndarray first.
    """

    def __init__(self, min_value: float = 0.0, max_value: float = 255.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, sample: Sample) -> Sample:
        arr = sample.input

        # Handle PIL / PngImageFile
        if hasattr(arr, "convert"):
            arr = np.array(arr)

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"NormalizeOp expects an np.ndarray, got {type(arr).__name__}")

        if arr.dtype == np.float64:
            arr = arr.astype(np.float64)
        else:
            arr = arr.astype(np.float32)

        arr = (arr - self.min_value) / (self.max_value - self.min_value)

        return sample._replace(input=arr)


@configurable
class StandardizeOp:
    """
    Standardizes ndarray values with given mean and standard deviation.

    Formula: output = (input - mean) / std

    mean/std can be a single float (applied uniformly) or a sequence of
    per-channel values that broadcasts over [C, H, W] format.

    Handles PIL images by converting to ndarray first.
    """

    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Sample) -> Sample:
        arr = sample.input

        # Handle PIL / PngImageFile
        if hasattr(arr, "convert"):
            arr = np.array(arr)

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"StandardizeOp expects an np.ndarray, got {type(arr).__name__}")

        if arr.dtype == np.float64:
            arr = arr.astype(np.float64)
        else:
            arr = arr.astype(np.float32)

        if isinstance(self.mean, (int, float)):
            mean_a = np.array([self.mean], dtype=arr.dtype)
        else:
            mean_a = np.array(self.mean, dtype=arr.dtype)

        if isinstance(self.std, (int, float)):
            std_a = np.array([self.std], dtype=arr.dtype)
        else:
            std_a = np.array(self.std, dtype=arr.dtype)

        # Reshape to [C, 1, 1, ...] for broadcasting over [C, H, W]
        mean_a = mean_a.reshape(-1, *([1] * (arr.ndim - 1)))
        std_a = std_a.reshape(-1, *([1] * (arr.ndim - 1)))

        arr = (arr - mean_a) / std_a

        return sample._replace(input=arr)
