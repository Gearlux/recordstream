from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from confluid import configurable

from sampleflux.sample import Sample
from sampleflux.typespec import ArrayType, PythonType, SampleType, UnionType
from sampleflux.windows import (
    WINDOW_SUM_KEY,
    WINDOW_SUMSQ_KEY,
    SpectrumScaling,
    WindowName,
    get_window,
    window_metadata,
    window_sums,
)

_TORCH = ArrayType(frameworks={"torch"})
_TORCH_FLOAT = ArrayType(dtype="floating", frameworks={"torch"})
# Spectrum-scaling ops emit complex (none/amplitude) OR real (power/density).
_TORCH_COMPLEX_OR_FLOAT = UnionType(
    (ArrayType(dtype="complex", frameworks={"torch"}), ArrayType(dtype="floating", frameworks={"torch"}))
)


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


def _apply_window(tensor: torch.Tensor, window: np.ndarray, dim: int) -> torch.Tensor:
    """Multiply ``tensor`` by the 1-D numpy ``window`` broadcast along ``dim`` (dtype/device-preserving)."""
    if tensor.is_complex():
        real_dtype = tensor.real.dtype
    elif tensor.is_floating_point():
        real_dtype = tensor.dtype
    else:
        real_dtype = torch.float32
    w = torch.as_tensor(window, dtype=real_dtype, device=tensor.device)
    moved = torch.movedim(tensor, dim, -1)
    return torch.movedim(moved * w, -1, dim)


def _fold_one_sided(spectrum: torch.Tensor, dim: int) -> torch.Tensor:
    """Fold a two-sided spectrum (natural order, DC at index 0) to one-sided (mirror of windows.fold_one_sided)."""
    moved = torch.movedim(spectrum, dim, -1)
    n = moved.shape[-1]
    out = moved[..., : n // 2 + 1].clone()
    if n % 2 == 0:
        out[..., 1:-1] = out[..., 1:-1] * 2  # exclude DC and Nyquist
    else:
        out[..., 1:] = out[..., 1:] * 2
    return torch.movedim(out, -1, dim)


def _scale_spectrum(
    spectrum: torch.Tensor,
    scaling: SpectrumScaling,
    *,
    s1: float,
    s2: float,
    sample_rate: Optional[float],
    one_sided: bool,
    dim: int,
) -> torch.Tensor:
    """Torch mirror of :func:`sampleflux.windows.scale_spectrum` (assumes ``norm="backward"``)."""
    if scaling == "none":
        out = spectrum
    elif scaling == "amplitude":
        out = spectrum / s1
    elif scaling == "power":
        out = spectrum.abs().square() / (s1 * s1)
    elif scaling == "density":
        fs = float(sample_rate) if (sample_rate is not None and sample_rate > 0) else 1.0
        out = spectrum.abs().square() / (fs * s2)
    else:
        raise ValueError(f"unknown scaling {scaling!r}")
    return _fold_one_sided(out, dim) if one_sided else out


def _resolve_fft_sample_rate(sample: Sample, explicit: Optional[float]) -> Optional[float]:
    """Resolve the density rate: explicit → ``metadata['samplerate']`` → ``None`` (mirrors the numpy op)."""
    if explicit is not None:
        return explicit
    if sample.is_batched:
        return None
    raw = sample.meta.get("samplerate")
    return float(raw) if raw else None


# FFT normalization mode — see the numpy ``FourierOp`` for the rationale. Exactly the three strings
# ``torch.fft.fft`` accepts for its ``norm=`` argument; a closed ``Literal`` so GUIs enumerate the
# choice via ``typing.get_args(...)``.
FourierNorm = Literal["backward", "ortho", "forward"]


@configurable(category="op", group="torch")
class FourierOp:
    """Compute the 1-D discrete Fourier transform of ``sample.input`` (``torch.fft.fft``).

    Accepts real **and** complex tensors; the output is always complex — ``complex64`` for
    integer / ``float32`` / ``complex64`` input, ``complex128`` for ``float64`` / ``complex128``.
    Half-precision (``float16`` / ``bfloat16``) tensors are promoted to ``float32`` first because
    ``torch.fft.fft`` does not support them; every other dtype (including integer and bool) is handled
    natively (integers auto-promote to ``complex64``). This is the **1-D** transform
    (``torch.fft.fft``), not the 2-D / N-D one: for an N-D tensor it runs along a single ``dim``
    (default the last), so a ``[B, N]`` batch transforms per row. :class:`InverseFourierOp` is the
    inverse; set ``shift=True`` to center the zero-frequency bin (the standalone :class:`FftShiftOp`
    does the same independently).

    **Windowing & units.** Mirrors the numpy ``FourierOp``: ``window`` applies a
    :func:`sampleflux.windows.get_window` taper before the transform (default ``"boxcar"`` = none) and
    stashes the window correction; ``scaling`` returns the spectrum in real units — ``"amplitude"``
    (V), ``"power"`` (V²) or ``"density"`` (V²/Hz, via ``sample_rate``). ``scaling="none"`` (default)
    leaves the raw complex spectrum. Calibrated ``scaling`` requires ``norm="backward"`` (any other
    ``norm`` with ``scaling != "none"`` raises).

    Args:
        n: Output length along ``dim`` — zero-pad/truncate to ``n`` points. ``None`` (default) uses the input length.
        dim: Dimension to transform over. Default ``-1`` (the last dim — the natural choice for a 1-D signal).
        norm: Normalization — ``"backward"`` (default, unscaled forward), ``"ortho"`` (1/sqrt(n) both ways),
            or ``"forward"`` (1/n on the forward transform). Calibrated ``scaling`` requires ``"backward"``.
        shift: When True, ``fftshift`` along ``dim`` after transforming (centers the zero bin). Default False.
        window: Taper applied before the FFT — a ``WindowName`` (``"boxcar"`` default = no taper).
        window_param: Kaiser ``β`` (def 8.6) / Tukey ``α`` (def 0.5) / Gaussian ``σ`` std (required); else ignored.
        periodic: ``True`` (default) = DFT-even window (correct for FFT analysis); ``False`` = symmetric.
        scaling: Units — ``"none"`` (complex, default), ``"amplitude"`` V, ``"power"`` V², ``"density"`` V²/Hz.
        sample_rate: Hz, for ``"density"``. ``None`` reads ``metadata["samplerate"]``, else ``1.0`` (normalized).
        one_sided: Fold to a one-sided spectrum (real signals). Default ``False``; exclusive with ``shift``.
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH_COMPLEX_OR_FLOAT)

    def __init__(
        self,
        n: Optional[int] = None,
        dim: int = -1,
        norm: FourierNorm = "backward",
        shift: bool = False,
        window: WindowName = "boxcar",
        window_param: Optional[float] = None,
        periodic: bool = True,
        scaling: SpectrumScaling = "none",
        sample_rate: Optional[float] = None,
        one_sided: bool = False,
    ) -> None:
        # Lazy / zero-arg: store config only. ``n`` and window params are validated lazily in __call__.
        self.n = n
        self.dim = dim
        self.norm = norm
        self.shift = bool(shift)
        self.window: WindowName = window
        self.window_param = window_param
        self.periodic = bool(periodic)
        self.scaling: SpectrumScaling = scaling
        self.sample_rate = sample_rate
        self.one_sided = bool(one_sided)

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"FourierOp expects a torch.Tensor, got {type(tensor).__name__}")
        if self.scaling != "none" and self.norm != "backward":
            raise ValueError(
                f"FourierOp: calibrated scaling={self.scaling!r} requires norm='backward' "
                f"(the unscaled transform); got norm={self.norm!r}"
            )
        if self.shift and self.one_sided:
            raise ValueError("FourierOp: shift and one_sided are mutually exclusive (one_sided is a half-spectrum)")
        # torch.fft.fft rejects half precision; promote to float32. Integer/bool/float/complex are
        # all accepted natively (integers auto-promote to complex64), so leave them untouched.
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        window = None
        signal = tensor
        if self.window != "boxcar":
            window = get_window(
                self.window, tensor.shape[self.dim], window_param=self.window_param, periodic=self.periodic
            )
            signal = _apply_window(tensor, window, self.dim)
        out = torch.fft.fft(signal, n=self.n, dim=self.dim, norm=self.norm)
        if self.scaling != "none":
            s1, s2 = window_sums(window) if window is not None else (float(tensor.shape[self.dim]),) * 2
            out = _scale_spectrum(
                out,
                self.scaling,
                s1=s1,
                s2=s2,
                sample_rate=_resolve_fft_sample_rate(sample, self.sample_rate),
                one_sided=self.one_sided,
                dim=self.dim,
            )
        if self.shift:
            out = torch.fft.fftshift(out, dim=self.dim)
        if window is not None and not sample.is_batched:
            new_meta = dict(sample.meta)
            new_meta.update(window_metadata(self.window, window))
            return sample._replace(input=out, metadata=new_meta)
        return sample._replace(input=out)


@configurable(category="op", group="torch")
class InverseFourierOp:
    """Compute the 1-D inverse discrete Fourier transform of ``sample.input`` (``torch.fft.ifft``).

    The sibling of :class:`FourierOp`: it maps a spectrum back to the time domain. The output is
    always complex (``torch.fft.ifft`` always returns complex; take ``.real`` downstream if the
    original signal was real). Half-precision (``float16``/``bfloat16``) tensors are promoted to
    ``float32`` first (``torch.fft.ifft`` rejects them); other dtypes are handled natively.
    ``InverseFourierOp(norm=…)`` must use the **same** ``norm`` as the forward transform to
    round-trip. With ``shift=True`` an ``ifftshift`` is applied to the input **before** inverting,
    exactly undoing a prior ``FourierOp(shift=True)`` (the correct pairing even for odd-length dims).

    Args:
        n: Output length along ``dim`` — zero-pad/truncate to ``n`` points. ``None`` (default) uses the input length.
        dim: Dimension to transform over. Default ``-1`` (the last dim — the natural choice for a 1-D signal).
        norm: Normalization — must match the forward transform: ``"backward"`` (default), ``"ortho"``, or ``"forward"``.
        shift: When True, ``ifftshift`` along ``dim`` before inverting (undoes a prior ``fftshift``). Default False.
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=ArrayType(dtype="complex", frameworks={"torch"}))

    def __init__(
        self, n: Optional[int] = None, dim: int = -1, norm: FourierNorm = "backward", shift: bool = False
    ) -> None:
        # Lazy / zero-arg: store config only. ``n`` (if set) is validated lazily by torch in __call__.
        self.n = n
        self.dim = dim
        self.norm = norm
        self.shift = bool(shift)

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"InverseFourierOp expects a torch.Tensor, got {type(tensor).__name__}")
        # torch.fft.ifft rejects half precision; promote to float32 (mirrors FourierOp).
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        if self.shift:
            tensor = torch.fft.ifftshift(tensor, dim=self.dim)
        out = torch.fft.ifft(tensor, n=self.n, dim=self.dim, norm=self.norm)
        return sample._replace(input=out)


@configurable(category="op", group="torch")
class FftShiftOp:
    """Shift the zero-frequency component to the center of the spectrum (``torch.fft.fftshift``).

    A pure bin-rearrangement — no FFT is computed, so it is dtype- AND shape-preserving and works
    on **any** tensor (real, complex, or integer; half precision included). Chain it after
    :class:`FourierOp` to center a spectrum for display (the ``FourierOp(shift=True)`` flag is the
    one-node convenience), or use it standalone to center an already-computed spectrum such as a 2-D
    spectrogram. :class:`IfftShiftOp` is its exact inverse (they differ only for odd-length dims).

    Args:
        dim: Dimension to shift. Default ``-1`` (last dim, matches :class:`FourierOp`); ``None`` shifts every dim.
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH)

    def __init__(self, dim: Optional[int] = -1) -> None:
        self.dim = dim

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"FftShiftOp expects a torch.Tensor, got {type(tensor).__name__}")
        return sample._replace(input=torch.fft.fftshift(tensor, dim=self.dim))


@configurable(category="op", group="torch")
class IfftShiftOp:
    """Undo an :class:`FftShiftOp` — move the center frequency back to index 0 (``torch.fft.ifftshift``).

    The exact inverse of :class:`FftShiftOp` (the two coincide for even-length dims but differ for
    odd-length ones, which is why both exist). Like its sibling it is a pure, dtype- and
    shape-preserving rearrangement that accepts any tensor. Apply it before :class:`InverseFourierOp`
    to recover the natural FFT bin order (``InverseFourierOp(shift=True)`` folds it in).

    Args:
        dim: Dimension to shift. Default ``-1`` (last dim, matches :class:`InverseFourierOp`); ``None`` = all dims.
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH)

    def __init__(self, dim: Optional[int] = -1) -> None:
        self.dim = dim

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"IfftShiftOp expects a torch.Tensor, got {type(tensor).__name__}")
        return sample._replace(input=torch.fft.ifftshift(tensor, dim=self.dim))


@configurable(category="op", group="torch")
class WindowOp:
    """Apply a window taper to ``sample.input`` and record the unit-scaling correction (torch mirror).

    The tensor counterpart of :class:`sampleflux.ops.numpy.WindowOp`: multiplies the signal by a
    :func:`sampleflux.windows.get_window` taper (broadcast along ``dim``) and stashes the window
    correction (``window`` / ``window_sum`` ``S1`` / ``window_sum_sq`` ``S2`` / ``window_enbw_bins`` /
    ``window_coherent_gain``) into ``sample.metadata`` for a later :class:`SpectrumScalingOp`.
    dtype/device-preserving — real stays real, complex stays complex.

    Args:
        window: Which taper — a ``WindowName`` (default ``"hann"``; ``"boxcar"`` is the rectangular identity).
        window_param: Kaiser ``β`` (def 8.6) / Tukey ``α`` (def 0.5) / Gaussian ``σ`` std (required); else ignored.
        periodic: ``True`` (default) = DFT-even window (correct for FFT analysis); ``False`` = symmetric.
        dim: Dimension the window is applied along. Default ``-1`` (the last dim — the 1-D signal).
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH)

    def __init__(
        self,
        window: WindowName = "hann",
        window_param: Optional[float] = None,
        periodic: bool = True,
        dim: int = -1,
    ) -> None:
        # Lazy / zero-arg: store config only; window params validated lazily by get_window.
        self.window: WindowName = window
        self.window_param = window_param
        self.periodic = bool(periodic)
        self.dim = dim

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"WindowOp expects a torch.Tensor, got {type(tensor).__name__}")
        window = get_window(self.window, tensor.shape[self.dim], window_param=self.window_param, periodic=self.periodic)
        out = _apply_window(tensor, window, self.dim)
        if sample.is_batched:
            return sample._replace(input=out)
        new_meta = dict(sample.meta)
        new_meta.update(window_metadata(self.window, window))
        return sample._replace(input=out, metadata=new_meta)


@configurable(category="op", group="torch")
class SpectrumScalingOp:
    """Scale a (complex) FFT spectrum to physical units using the window correction (torch mirror).

    The tensor counterpart of :class:`sampleflux.ops.numpy.SpectrumScalingOp`: amplitude (V) / power
    (V²) / density (V²/Hz), dividing out the window ``S1``/``S2`` read from the ``window_*`` metadata
    (rectangular ``S1=S2=N`` if absent). Assumes the spectrum came from the unscaled forward transform
    (``norm="backward"``). Output is complex for ``"none"``/``"amplitude"``, real for
    ``"power"``/``"density"``.

    Args:
        scaling: Units — ``"none"``, ``"amplitude"`` V, ``"power"`` V² (default), ``"density"`` V²/Hz.
        sample_rate: Hz, for ``"density"``. ``None`` (default) reads ``metadata["samplerate"]``, else ``1.0``.
        one_sided: Fold to one-sided (real-signal convention: keep 0…N/2, double interior bins). Default ``False``.
        dim: Spectrum dimension. Default ``-1``.
    """

    ACCEPTS = SampleType(input=_TORCH)
    PRODUCES = SampleType(input=_TORCH_COMPLEX_OR_FLOAT)

    def __init__(
        self,
        scaling: SpectrumScaling = "power",
        sample_rate: Optional[float] = None,
        one_sided: bool = False,
        dim: int = -1,
    ) -> None:
        # Lazy / zero-arg: store config only.
        self.scaling: SpectrumScaling = scaling
        self.sample_rate = sample_rate
        self.one_sided = bool(one_sided)
        self.dim = dim

    def __call__(self, sample: Sample) -> Sample:
        tensor = sample.input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"SpectrumScalingOp expects a torch.Tensor, got {type(tensor).__name__}")
        n = tensor.shape[self.dim]
        if sample.is_batched:
            s1 = s2 = float(n)
        else:
            meta = sample.meta
            raw_s1, raw_s2 = meta.get(WINDOW_SUM_KEY), meta.get(WINDOW_SUMSQ_KEY)
            s1, s2 = (
                (float(raw_s1), float(raw_s2)) if raw_s1 is not None and raw_s2 is not None else (float(n), float(n))
            )
        out = _scale_spectrum(
            tensor,
            self.scaling,
            s1=s1,
            s2=s2,
            sample_rate=_resolve_fft_sample_rate(sample, self.sample_rate),
            one_sided=self.one_sided,
            dim=self.dim,
        )
        return sample._replace(input=out)
