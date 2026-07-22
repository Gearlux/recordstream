from typing import Optional, Sequence, Union

import numpy as np
import torch
from confluid import configurable

from sampleflux.bag.items import Image as ImageItem
from sampleflux.bag.items import NDArrayItem, item_data
from sampleflux.bag.sample import TypedSample
from sampleflux.bag.transform import Transform
from sampleflux.sample import Sample
from sampleflux.typespec import ArrayType, PythonType, SampleType, UnionType

_TORCH = ArrayType(frameworks={"torch"})
_TORCH_FLOAT = ArrayType(dtype="floating", frameworks={"torch"})


@configurable(category="op", group="torch")
class ToTensorOp:
    """
    Converts input (PIL Image, NumPy array, etc.) to a Torch Tensor.

    Args:
        normalize: When ``True``, scale integer pixel inputs into the ``[0, 1]`` float range during conversion.
        mode: Optional PIL mode to convert to (e.g. "RGB" forces 3 channels); None (default) arrays as-is.
    """

    ACCEPTS = SampleType(input=UnionType((PythonType("PIL.Image.Image"), ArrayType(frameworks={"numpy"}))))
    PRODUCES = SampleType(input=_TORCH)

    def __init__(self, normalize: bool = True, mode: Optional[str] = None):
        self.normalize = normalize
        self.mode = mode

    def __call__(self, sample: Sample) -> Sample:
        img = sample.input

        # Handle PIL / PngImageFile
        if hasattr(img, "convert"):
            # Optionally coerce the PIL mode (e.g. "RGB") so a mixed-mode dataset
            # (RGBA / grayscale / palette samples) yields a uniform channel count.
            if self.mode is not None:
                img = img.convert(self.mode)
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


@configurable(category="op", group="torch")
class RescaleOp:
    """Affine rescale a torch.Tensor from ``[in_min, in_max]`` to ``[out_min, out_max]``.

    The default ``out_min=0.0`` / ``out_max=1.0`` covers the common
    ``[0, 255] -> [0, 1]`` image-normalization case. Integer dtypes are
    promoted to ``float32`` (``float64`` is preserved).

    Args:
        in_min: Lower edge of the input range. Default ``0.0``.
        in_max: Upper edge of the input range, must be ``> in_min``. Default ``1.0``.
        out_min: Lower edge of the output range. Default ``0.0``.
        out_max: Upper edge of the output range, must be ``> out_min``. Default ``1.0``.
        clip: When True (default), clamp values outside ``[in_min, in_max]``
            before rescaling. When False, extrapolate linearly.
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH_FLOAT)

    def __init__(
        self,
        in_min: float = 0.0,
        in_max: float = 1.0,
        out_min: float = 0.0,
        out_max: float = 1.0,
        clip: bool = True,
    ) -> None:
        # Lazy / zero-arg: store config only; the bound relationships are validated lazily in __call__.
        self.in_min = float(in_min)
        self.in_max = float(in_max)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.clip = bool(clip)

    def __call__(self, sample: Sample) -> Sample:
        if not (self.in_min < self.in_max):
            raise ValueError(f"RescaleOp: require in_min < in_max; got in_min={self.in_min}, in_max={self.in_max}")
        if not (self.out_min < self.out_max):
            raise ValueError(
                f"RescaleOp: require out_min < out_max; got out_min={self.out_min}, out_max={self.out_max}"
            )
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"RescaleOp expects a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dtype != torch.float32 and tensor.dtype != torch.float64:
            tensor = tensor.float()
        src = tensor.clamp(self.in_min, self.in_max) if self.clip else tensor
        scaled = (src - self.in_min) / (self.in_max - self.in_min)
        out = scaled * (self.out_max - self.out_min) + self.out_min
        return sample._replace(input=out)


@configurable(category="op", group="torch")
class SqueezeOp:
    """Remove size-1 dimensions from a ``torch.Tensor``.

    Args:
        dim: Axis index to remove. When ``None`` (default), all size-1 dimensions are removed.
            When specified, the dimension must have size 1; otherwise the tensor is returned unchanged
            (matching ``torch.squeeze`` semantics).
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH)

    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"SqueezeOp expects a torch.Tensor, got {type(tensor).__name__}")
        out = torch.squeeze(tensor) if self.dim is None else torch.squeeze(tensor, self.dim)
        return sample._replace(input=out)


@configurable(category="op", group="torch")
class UnsqueezeOp:
    """Insert a size-1 dimension at the specified position in a ``torch.Tensor``.

    Args:
        dim: Axis index at which the new dimension is inserted. Default ``0``.
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH)

    def __init__(self, dim: int = 0) -> None:
        self.dim = dim

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"UnsqueezeOp expects a torch.Tensor, got {type(tensor).__name__}")
        return sample._replace(input=torch.unsqueeze(tensor, self.dim))


@configurable(category="op", group="torch")
class StandardizeOp:
    """
    Standardizes tensor values with given mean and standard deviation.

    Formula: output = (input - mean) / std

    mean/std can be a single float (applied uniformly) or a sequence of
    per-channel values that broadcasts over [C, H, W] format.

    Args:
        mean: Mean to subtract — a single float (uniform) or a per-channel sequence broadcasting over [C, H, W].
        std: Standard deviation to divide by — a single float (uniform) or a per-channel sequence.
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH_FLOAT)

    def __init__(self, mean: Union[float, Sequence[float]] = 0.0, std: Union[float, Sequence[float]] = 1.0):
        # Lazy / zero-arg: store config only. The defaults (mean 0, std 1) are an identity standardize.
        self.mean = mean
        self.std = std

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"StandardizeOp expects a torch.Tensor, got {type(tensor).__name__}")

        if tensor.dtype != torch.float32 and tensor.dtype != torch.float64:
            tensor = tensor.float()

        mean_t = torch.tensor(
            [self.mean] if isinstance(self.mean, (int, float)) else self.mean,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        std_t = torch.tensor(
            [self.std] if isinstance(self.std, (int, float)) else self.std,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        # Reshape to [C, 1, 1, ...] for broadcasting over [C, H, W]
        mean_t = mean_t.view(-1, *([1] * (tensor.ndim - 1)))
        std_t = std_t.view(-1, *([1] * (tensor.ndim - 1)))

        tensor = (tensor - mean_t) / std_t

        return sample._replace(input=tensor)


@configurable(category="op", group="torch")
class ToTensor(Transform):
    """Typed twin of :class:`ToTensorOp` — an array-bearing field → a CHW-float ``Image`` item.

    The typed-bag counterpart of :class:`ToTensorOp`: it reads the payload of an array-bearing
    field (blank ``field`` picks the first array/PIL-bearing item — typically the
    :class:`~sampleflux.Image` a :class:`~sampleflux.ops.image.ConvertToImage` produced), runs the
    SAME HWC→CHW transpose + ``normalize`` conversion (this twin REUSES the legacy op verbatim on a
    shim ``Sample``, so the numbers are identical), and writes a CHW-layout :class:`~sampleflux.Image`
    back. By default it REPLACES the resolved field in place (``output`` blank), so the field's role
    is preserved — the model's working image tensor stays the ``input`` it already was; set
    ``output`` to write a NEW field (tagged ``input``) instead. Any other field passes through
    untouched.

    IMPORTANT — payload dtype. A :class:`~sampleflux.NDArrayItem` (which ``Image`` is) coerces its
    payload through ``np.asarray`` on construction, so it CANNOT hold a live ``torch.Tensor``: the
    stored payload is a CHW ``float32`` **numpy** array whose values are byte-identical to the legacy
    ``ToTensorOp`` tensor (``legacy.input.numpy()``). The typed collate (``typed_collate``) stacks
    these field payloads with ``np.stack`` into a batched CHW-float array; the numpy→``torch.Tensor``
    conversion happens at the collate / model boundary (exactly as for any numpy-backed dataset). A
    torch-``Tensor``-subclass item that would let a field carry a live tensor is the documented
    follow-up (see ``sampleflux.bag.items`` — "torch payloads ride in wrapper items in the PoC").

    Args:
        normalize: When ``True`` (default), scale integer pixel inputs into the ``[0, 1]`` float range.
        mode: Optional PIL mode to convert a PIL payload to (e.g. ``"RGB"`` forces 3 channels); ``None`` = as-is.
        field: Name of the source field to tensorize; blank (default) picks the first array/PIL-bearing item.
        output: Field the CHW ``Image`` is written to; blank (default) replaces the source field in place
            (role preserved). A non-blank name writes a new field tagged ``input``.
    """

    handles = (NDArrayItem,)
    consumes = (NDArrayItem,)
    produces = (ImageItem,)

    def __init__(
        self,
        normalize: bool = True,
        mode: Optional[str] = None,
        field: str = "",
        output: str = "",
    ) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.mode = mode
        self.field = field
        self.output = output

    def _find_field(self, sample: TypedSample) -> str:
        """Resolve the KEY of the field to tensorize (``self.field`` or the first array/PIL item)."""
        if self.field:
            if self.field not in sample.keys():
                raise ValueError(f"ToTensor: field {self.field!r} not in sample (fields: {list(sample.keys())})")
            return self.field
        for key, item in sample.items():
            data = item_data(item)
            if isinstance(data, np.ndarray) or hasattr(data, "convert"):
                return key
        raise ValueError(f"ToTensor: no array-bearing field in sample (fields: {list(sample.keys())})")

    def __call__(self, sample: TypedSample) -> TypedSample:
        key = self._find_field(sample)
        data = item_data(sample[key])
        # Reuse the legacy op's conversion VERBATIM on a shim Sample so the CHW / normalization
        # values are identical; NDArrayItem then coerces the tensor to a CHW float32 numpy payload.
        tensor = ToTensorOp(self.normalize, self.mode)(Sample(input=data, target=None, metadata={})).input
        arr = tensor.detach().cpu().numpy()
        out_key = self.output or key
        out = sample.replace_field(out_key, ImageItem(arr, layout="CHW"))
        if self.output:
            out = out.set_role(out_key, "input")
        return out
