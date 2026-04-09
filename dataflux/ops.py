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
