import os
import re
from typing import List, Sequence, Tuple, Union

import numpy as np
from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)


_EXPR_PATTERN = re.compile(r"\{(\w+)\}|\$(\w+)")


def resolve_expression(value: str, sample: Sample) -> str:
    """Substitute ``{key}`` from ``sample.metadata`` and ``$NAME`` from ``os.environ``.

    Returns the substituted string verbatim — the caller is responsible for
    any further casting (e.g. ``float(...)`` for a numeric expression).

    Examples:
        ``"5.5"``                → ``"5.5"`` (no substitution)
        ``"{reference_snr_level}"`` → ``str(metadata["reference_snr_level"])``
        ``"-{reference_snr_level}"`` → ``"-<value>"`` (sign passes through to ``float()``)
        ``"$REF_SNR"``           → ``os.environ["REF_SNR"]``

    Raises:
        KeyError: A referenced metadata key or environment variable is missing.
    """

    def _repl(match: "re.Match[str]") -> str:
        meta_key = match.group(1)
        env_name = match.group(2)
        if meta_key is not None:
            if meta_key not in sample.metadata:
                raise KeyError(
                    f"resolve_expression: metadata key {meta_key!r} missing in {value!r}; "
                    f"available keys: {sorted(sample.metadata)}"
                )
            return str(sample.metadata[meta_key])
        assert env_name is not None
        if env_name not in os.environ:
            raise KeyError(f"resolve_expression: environment variable {env_name!r} missing in {value!r}")
        return os.environ[env_name]

    return _EXPR_PATTERN.sub(_repl, value)


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


def _require_ndarray(sample: Sample, op_name: str) -> np.ndarray:
    arr = sample.input
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{op_name} expects an np.ndarray on sample.input, got {type(arr).__name__}")
    return arr


@configurable
class ClipPercentilesOp:
    """Clip ``sample.input`` to ``[p_low, p_high]`` percentiles of finite values.

    Percentiles are computed over only finite entries — ``inf`` / ``-inf`` /
    ``nan`` are excluded from the percentile estimate. ``np.clip`` then maps
    ``+inf`` to the upper bound and ``-inf`` to the lower bound; ``nan``
    survives unchanged. Chain :class:`ReplaceNonFiniteOp` upstream if remaining
    ``nan`` values matter.

    Args:
        low: Lower percentile in ``[0, 100)``. Default ``2.0``.
        high: Upper percentile in ``(0, 100]``, must be ``> low``. Default ``98.0``.
    """

    def __init__(self, low: float = 2.0, high: float = 98.0) -> None:
        if not (0.0 <= low < high <= 100.0):
            raise ValueError(f"ClipPercentilesOp: require 0 <= low < high <= 100; got low={low}, high={high}")
        self.low = float(low)
        self.high = float(high)

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "ClipPercentilesOp")
        finite = np.isfinite(arr)
        if not finite.any():
            logger.warning("ClipPercentilesOp: input is entirely non-finite; passing through")
            return sample
        lo = float(np.percentile(arr[finite], self.low))
        hi = float(np.percentile(arr[finite], self.high))
        return sample._replace(input=np.clip(arr, lo, hi))


@configurable
class RescaleOp:
    """Affine rescale ``sample.input`` from ``[in_min, in_max]`` to ``[out_min, out_max]``.

    The default ``out_min=0.0`` / ``out_max=1.0`` covers the common
    ``[0, 255] -> [0, 1]`` image-normalization case. PIL inputs are
    converted to ndarray; integer dtypes are promoted to ``float32``
    (``float64`` is preserved).

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
        arr = sample.input
        if hasattr(arr, "convert"):
            arr = np.array(arr)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"RescaleOp expects an np.ndarray, got {type(arr).__name__}")
        arr = arr.astype(np.float64 if arr.dtype == np.float64 else np.float32)
        src = np.clip(arr, self.in_min, self.in_max) if self.clip else arr
        scaled = (src - self.in_min) / (self.in_max - self.in_min)
        out = scaled * (self.out_max - self.out_min) + self.out_min
        return sample._replace(input=out)


@configurable
class ReplaceNonFiniteOp:
    """Replace ``inf`` / ``-inf`` / ``nan`` entries in ``sample.input``.

    Args:
        value: Replacement specifier. Either:

            * a ``float`` / ``int`` — literal replacement value;
            * the string ``"min"`` — replace with the array's finite min;
            * the string ``"max"`` — replace with the array's finite max.

            Default ``"min"``.
    """

    def __init__(self, value: Union[float, int, str] = "min") -> None:
        if isinstance(value, str) and value not in ("min", "max"):
            raise ValueError(f"ReplaceNonFiniteOp: value string must be 'min' or 'max'; got {value!r}")
        self.value = value

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "ReplaceNonFiniteOp")
        non_finite = ~np.isfinite(arr)
        if not non_finite.any():
            return sample
        if isinstance(self.value, str):
            finite = ~non_finite
            if not finite.any():
                logger.warning("ReplaceNonFiniteOp: array is entirely non-finite; passing through")
                return sample
            finite_values = arr[finite]
            repl = float(finite_values.min() if self.value == "min" else finite_values.max())
        else:
            repl = float(self.value)
        return sample._replace(input=np.where(non_finite, repl, arr))


@configurable
class ThresholdOp:
    """Threshold ``sample.input`` (ndarray) into a boolean mask: ``input > value``.

    ``value`` is either a numeric literal or a string expression resolved via
    :func:`resolve_expression` against ``sample.metadata`` and ``os.environ``:

    * ``5.5`` or ``"5.5"``                — fixed threshold
    * ``"{reference_snr_level}"``         — looks up ``metadata["reference_snr_level"]``
    * ``"-{reference_snr_level}"``        — negated lookup (the leading ``-`` is
                                             carried through ``float(...)`` after substitution)
    * ``"$REF_SNR"`` / ``"-$REF_SNR"``    — environment-variable lookup

    Records the resolved threshold under ``metadata["threshold"]`` for traceability.
    """

    def __init__(self, value: Union[float, int, str] = 0.0) -> None:
        self.value = value

    def _resolve(self, sample: Sample) -> float:
        if isinstance(self.value, (int, float)):
            return float(self.value)
        if not isinstance(self.value, str):
            raise TypeError(f"ThresholdOp.value must be a number or expression string; got {type(self.value).__name__}")
        resolved = resolve_expression(self.value, sample)
        try:
            return float(resolved)
        except ValueError as exc:
            raise ValueError(
                f"ThresholdOp: expression {self.value!r} resolved to {resolved!r}, " f"which is not a number"
            ) from exc

    def __call__(self, sample: Sample) -> Sample:
        arr = sample.input
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"ThresholdOp expects an np.ndarray on sample.input, got {type(arr).__name__}")
        threshold = self._resolve(sample)
        sample.metadata["threshold"] = threshold
        return sample._replace(input=arr > threshold)


@configurable
class ConnectedComponentsOp:
    """Label connected ``True`` regions of a boolean mask into bin-bbox tuples.

    Reads ``sample.input`` as a 2-D boolean ndarray; writes ``sample.input`` as
    a list of ``(row_min, row_max, col_min, col_max)`` integer tuples (inclusive
    bounds). Components smaller than ``min_area_bins`` are dropped.

    ``connectivity`` selects the neighborhood:

    * ``4`` — orthogonal neighbors only (N/S/E/W); diagonally touching
      components stay separate.
    * ``8`` — orthogonal + diagonal neighbors; diagonally touching
      components merge.

    Requires ``scipy`` (install via ``pip install data-flux[vision]``).
    """

    def __init__(self, min_area_bins: int = 1, connectivity: int = 4) -> None:
        if min_area_bins < 1:
            raise ValueError(f"min_area_bins must be >= 1; got {min_area_bins!r}")
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8; got {connectivity!r}")
        self.min_area_bins = int(min_area_bins)
        self.connectivity = int(connectivity)

    def __call__(self, sample: Sample) -> Sample:
        try:
            from scipy.ndimage import find_objects, generate_binary_structure, label
        except ImportError as exc:
            raise ImportError(
                "ConnectedComponentsOp requires scipy. "
                "Install with `pip install data-flux[vision]` or add scipy to your environment."
            ) from exc

        mask = sample.input
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"ConnectedComponentsOp expects an np.ndarray on sample.input, got {type(mask).__name__}")
        if mask.ndim != 2:
            raise ValueError(f"ConnectedComponentsOp expects a 2-D mask; got shape {mask.shape}")

        structure = generate_binary_structure(2, 1 if self.connectivity == 4 else 2)
        labels, n_components = label(mask, structure=structure)
        bboxes: List[Tuple[int, int, int, int]] = []
        if n_components > 0:
            for idx, sl in enumerate(find_objects(labels), start=1):
                if sl is None:
                    continue
                row_slice, col_slice = sl
                area = int((labels[row_slice, col_slice] == idx).sum())
                if area < self.min_area_bins:
                    continue
                bboxes.append(
                    (
                        int(row_slice.start),
                        int(row_slice.stop) - 1,
                        int(col_slice.start),
                        int(col_slice.stop) - 1,
                    )
                )
        return sample._replace(input=bboxes)
