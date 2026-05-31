import operator
import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample
from dataflux.typespec import ArrayType, PythonType, SampleType, UnionType

# Common shorthands for the numpy ops' declared types.
_NDARRAY = ArrayType(frameworks={"numpy"})
_NUMERIC_OR_PIL = UnionType((ArrayType(dtype="numeric", frameworks={"numpy"}), PythonType("PIL.Image.Image")))

logger = get_logger(__name__)


_EXPR_PATTERN = re.compile(r"\{(\w+)\}|\$(\w+)")


def resolve_expression(value: str, sample: Sample) -> str:
    """Substitute ``{key}`` from ``sample.metadata`` and ``$NAME`` from ``os.environ``.

    Returns the substituted string verbatim — the caller is responsible for
    any further casting (e.g. ``float(...)`` for a numeric expression).

    Args:
        value: Expression string with ``{meta_key}`` and/or ``$ENV_VAR`` placeholders
            (a plain literal returns unchanged).
        sample: The Sample whose ``metadata`` supplies the ``{key}`` substitutions.

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


@configurable(category="op", group="numpy")
class StandardizeOp:
    """
    Standardizes ndarray values with given mean and standard deviation.

    Formula: output = (input - mean) / std

    mean/std can be a single float (applied uniformly) or a sequence of
    per-channel values that broadcasts over [C, H, W] format.

    Handles PIL images by converting to ndarray first.

    Args:
        mean: Mean to subtract — a single float (uniform) or a per-channel sequence broadcasting over [C, H, W].
        std: Standard deviation to divide by — a single float (uniform) or a per-channel sequence.
    """

    ACCEPTS = SampleType(input=_NUMERIC_OR_PIL)
    PRODUCES = SampleType(input=ArrayType(dtype="floating", frameworks={"numpy"}))

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


@configurable(category="op", group="numpy")
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

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

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


@configurable(category="op", group="numpy")
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

    ACCEPTS = SampleType(input=_NUMERIC_OR_PIL)
    PRODUCES = SampleType(input=ArrayType(dtype="floating", frameworks={"numpy"}))

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


@configurable(category="op", group="numpy")
class ReplaceNonFiniteOp:
    """Replace ``inf`` / ``-inf`` / ``nan`` entries in ``sample.input``.

    Args:
        value: Replacement specifier. Either:

            * a ``float`` / ``int`` — literal replacement value;
            * the string ``"min"`` — replace with the array's finite min;
            * the string ``"max"`` — replace with the array's finite max.

            Default ``"min"``.
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

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


# ThresholdOp comparison selectors. Closed ``Literal``s (workspace "prefer closed
# Literals over bare strings" mandate) so FluxStudio / navigaitor render the choice
# as a dropdown and the allowed operators stay machine-introspectable via
# ``typing.get_args(...)``. Two distinct types because the lower bound only sensibly
# uses ``>`` / ``>=`` and the upper bound only ``<`` / ``<=``.
LowComparison = Literal[">", ">="]
HighComparison = Literal["<", "<="]

# Operator dispatch. The dict keys are the single runtime source of truth's
# consumers — ``tests/test_ops.py`` pins ``set(_LOW_COMPARISONS) == get_args(LowComparison)``
# (and likewise for high) so the map can never drift from the Literal.
_LOW_COMPARISONS: Dict[str, Callable[[Any, float], Any]] = {">": operator.gt, ">=": operator.ge}
_HIGH_COMPARISONS: Dict[str, Callable[[Any, float], Any]] = {"<": operator.lt, "<=": operator.le}


@configurable(category="op", group="numpy")
class ThresholdOp:
    """Threshold ``sample.input`` (ndarray) into a boolean mask using one or both bounds.

    Which mask is produced depends on *which* bounds are set (presence-driven), and the
    comparison applied for each is selected by ``low_op`` / ``high_op``:

    * only ``low_level``  → ``input <low_op> low_level``    (values above the floor)
    * only ``high_level`` → ``input <high_op> high_level``  (values below the ceiling)
    * both                → both conditions AND-ed together (band-pass)

    ``low_op`` is ``">"`` (strict, the default) or ``">="`` (inclusive); ``high_op`` is
    ``"<"`` (strict, the default) or ``"<="`` (inclusive). So the defaults yield the OPEN
    interval ``low_level < input < high_level``, while ``low_op=">="`` + ``high_op="<="``
    yield the CLOSED interval ``low_level <= input <= high_level``.

    At least one of ``low_level`` / ``high_level`` MUST be provided; passing
    neither raises ``ValueError`` at construction.

    Each bound is either a numeric literal or a string expression resolved via
    :func:`resolve_expression` against ``sample.metadata`` and ``os.environ``:

    * ``5.5`` or ``"5.5"``                — fixed bound
    * ``"{reference_snr_level}"``         — looks up ``metadata["reference_snr_level"]``
    * ``"-{reference_snr_level}"``        — negated lookup (the leading ``-`` is
                                             carried through ``float(...)`` after substitution)
    * ``"$REF_SNR"`` / ``"-$REF_SNR"``    — environment-variable lookup

    Records each resolved bound that was applied under ``metadata["threshold_low"]``
    / ``metadata["threshold_high"]`` for traceability.

    Args:
        low_level: Lower bound (numeric literal or expression) compared with ``low_op`` when set;
            ``None`` disables the lower bound.
        high_level: Upper bound (numeric literal or expression) compared with ``high_op`` when set;
            ``None`` disables the upper bound.
        low_op: Lower-bound comparison — ``">"`` (strict, default) or ``">="`` (inclusive).
        high_op: Upper-bound comparison — ``"<"`` (strict, default) or ``"<="`` (inclusive).
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=ArrayType(dtype="bool", frameworks={"numpy"}))

    def __init__(
        self,
        low_level: Optional[Union[float, int, str]] = None,
        high_level: Optional[Union[float, int, str]] = None,
        low_op: LowComparison = ">",
        high_op: HighComparison = "<",
    ) -> None:
        if low_level is None and high_level is None:
            raise ValueError("ThresholdOp requires at least one of 'low_level' / 'high_level'")
        self.low_level = low_level
        self.high_level = high_level
        self.low_op = low_op
        self.high_op = high_op

    def _resolve(self, bound: Union[float, int, str], sample: Sample) -> float:
        if isinstance(bound, (int, float)):
            return float(bound)
        if not isinstance(bound, str):
            raise TypeError(f"ThresholdOp bounds must be a number or expression string; got {type(bound).__name__}")
        resolved = resolve_expression(bound, sample)
        try:
            return float(resolved)
        except ValueError as exc:
            raise ValueError(
                f"ThresholdOp: expression {bound!r} resolved to {resolved!r}, " f"which is not a number"
            ) from exc

    def __call__(self, sample: Sample) -> Sample:
        arr = sample.input
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"ThresholdOp expects an np.ndarray on sample.input, got {type(arr).__name__}")
        mask: Optional[np.ndarray] = None
        if self.low_level is not None:
            low = self._resolve(self.low_level, sample)
            sample.metadata["threshold_low"] = low
            mask = _LOW_COMPARISONS[self.low_op](arr, low)
        if self.high_level is not None:
            high = self._resolve(self.high_level, sample)
            sample.metadata["threshold_high"] = high
            below = _HIGH_COMPARISONS[self.high_op](arr, high)
            mask = below if mask is None else (mask & below)
        assert mask is not None  # guaranteed by the __init__ presence check
        return sample._replace(input=mask)


@configurable(category="op", group="numpy")
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

    Args:
        min_area_bins: Minimum component area in bins; smaller connected regions are dropped (``>= 1``).
        connectivity: Pixel neighborhood — ``4`` (orthogonal only) or ``8`` (orthogonal + diagonal).
    """

    ACCEPTS = SampleType(input=ArrayType(ndim=2, dtype="bool", frameworks={"numpy"}))
    PRODUCES = SampleType(input=PythonType("list"))

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
