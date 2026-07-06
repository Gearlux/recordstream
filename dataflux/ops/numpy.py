import operator
import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from confluid import configurable
from loggair import get_logger

from dataflux.sample import Sample
from dataflux.typespec import ArrayType, PythonType, SampleType, UnionType
from dataflux.windows import (
    WINDOW_SUM_KEY,
    WINDOW_SUMSQ_KEY,
    SpectrumScaling,
    WindowName,
    get_window,
    scale_spectrum,
    window_metadata,
    window_sums,
)

# Common shorthands for the numpy ops' declared types.
_NDARRAY = ArrayType(frameworks={"numpy"})
_NUMERIC_OR_PIL = UnionType((ArrayType(dtype="numeric", frameworks={"numpy"}), PythonType("PIL.Image.Image")))
# Spectrum-scaling ops emit complex (none/amplitude) OR real (power/density) — a permissive union.
_COMPLEX_OR_FLOAT = UnionType(
    (ArrayType(dtype="complex", frameworks={"numpy"}), ArrayType(dtype="floating", frameworks={"numpy"}))
)

logger = get_logger(__name__)


_EXPR_PATTERN = re.compile(r"\{(\w+)\}|\$(\w+)")


def resolve_expression(value: str, sample: Sample) -> str:
    """Substitute ``{key}`` from ``sample.meta`` and ``$NAME`` from ``os.environ``.

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
            if meta_key not in sample.meta:
                raise KeyError(
                    f"resolve_expression: metadata key {meta_key!r} missing in {value!r}; "
                    f"available keys: {sorted(sample.meta)}"
                )
            return str(sample.meta[meta_key])
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

    def __init__(self, mean: Union[float, Sequence[float]] = 0.0, std: Union[float, Sequence[float]] = 1.0):
        # Lazy / zero-arg: store config only. The defaults (mean 0, std 1) are an identity standardize.
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
class SqueezeOp:
    """Remove size-1 axes from an ``np.ndarray``.

    Args:
        axis: Axis index to remove. When ``None`` (default), all size-1 axes are removed.
            When specified, the axis must have size 1 (numpy raises ``ValueError`` otherwise).
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(self, axis: Optional[int] = None) -> None:
        self.axis = axis

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "SqueezeOp")
        out = np.squeeze(arr) if self.axis is None else np.squeeze(arr, axis=self.axis)
        return sample._replace(input=out)


@configurable(category="op", group="numpy")
class UnsqueezeOp:
    """Insert a size-1 axis at the specified position in an ``np.ndarray``.

    Args:
        axis: Axis index at which the new dimension is inserted. Default ``0``.
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(self, axis: int = 0) -> None:
        self.axis = axis

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "UnsqueezeOp")
        return sample._replace(input=np.expand_dims(arr, axis=self.axis))


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
        # Lazy / zero-arg: store config only; the bound relationship is validated lazily in __call__.
        self.low = float(low)
        self.high = float(high)

    def __call__(self, sample: Sample) -> Sample:
        if not (0.0 <= self.low < self.high <= 100.0):
            raise ValueError(f"ClipPercentilesOp: require 0 <= low < high <= 100; got low={self.low}, high={self.high}")
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
        in_min: Lower edge of the input range. Default ``0.0``.
        in_max: Upper edge of the input range, must be ``> in_min``. Default ``1.0``.
        out_min: Lower edge of the output range. Default ``0.0``.
        out_max: Upper edge of the output range, must be ``> out_min``. Default ``1.0``.
        clip: When True (default), clamp values outside ``[in_min, in_max]``
            before rescaling. When False, extrapolate linearly.
    """

    ACCEPTS = SampleType(input=_NUMERIC_OR_PIL)
    PRODUCES = SampleType(input=ArrayType(dtype="floating", frameworks={"numpy"}))

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
        # Lazy / zero-arg: store config only; the 'min'/'max' string is validated lazily in __call__.
        self.value = value

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "ReplaceNonFiniteOp")
        non_finite = ~np.isfinite(arr)
        if not non_finite.any():
            return sample
        if isinstance(self.value, str):
            if self.value not in ("min", "max"):
                raise ValueError(f"ReplaceNonFiniteOp: value string must be 'min' or 'max'; got {self.value!r}")
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
    neither raises ``ValueError`` when the op is applied (the zero-arg default is
    deferred-valid so the op stays constructible, per the lazy-init convention).

    Each bound is either a numeric literal or a string expression resolved via
    :func:`resolve_expression` against ``sample.meta`` and ``os.environ``:

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
        # Lazy / zero-arg: store config only; the "at least one bound" requirement is validated
        # lazily in __call__ so the op stays constructible with no arguments.
        self.low_level = low_level
        self.high_level = high_level
        self.low_op = low_op
        self.high_op = high_op

    def _resolve(self, bound: Optional[Union[float, int, str]], sample: Sample) -> float:
        if bound is None:
            raise ValueError("ThresholdOp._resolve called with None — bound was not filtered by __call__")
        if isinstance(bound, str):
            resolved = resolve_expression(bound, sample)
            try:
                return float(resolved)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ThresholdOp: expression {bound!r} resolved to {resolved!r}, which is not a number"
                ) from exc
        # Any non-string numeric: a Python int/float, a NumPy scalar (e.g. the float32 a value-chain
        # MaxOp → FormulaOp → ConfigureOp injects into low_level per sample), or a 0-d array — anything
        # float() accepts. A list / multi-D array / complex value fails float() and raises the TypeError.
        try:
            return float(bound)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"ThresholdOp bounds must be a number or expression string; got {type(bound).__name__}"
            ) from exc

    def __call__(self, sample: Sample) -> Sample:
        arr = sample.input
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"ThresholdOp expects an np.ndarray on sample.input, got {type(arr).__name__}")

        low_level = self.low_level
        high_level = self.high_level
        # Treat empty string (blank STRING widget left unset) as None ("disabled").
        if isinstance(low_level, str) and low_level.strip() == "":
            low_level = None
        if isinstance(high_level, str) and high_level.strip() == "":
            high_level = None

        mask: Optional[np.ndarray] = None
        if low_level is not None:
            low = self._resolve(low_level, sample)
            if np.isnan(low):
                logger.warning(
                    f"ThresholdOp: resolved low_level is NaN; no values will be above the threshold. "
                    f"Expression was {self.low_level!r} resolved to {low!r}"
                )
            else:
                sample.meta["threshold_low"] = low
                mask = _LOW_COMPARISONS[self.low_op](arr, low)
        if high_level is not None:
            high = self._resolve(high_level, sample)
            if np.isnan(high):
                logger.warning(
                    f"ThresholdOp: resolved high_level is NaN; no values will be below the threshold. "
                    f"Expression was {self.high_level!r} resolved to {high!r}"
                )
            else:
                sample.meta["threshold_high"] = high
                below = _HIGH_COMPARISONS[self.high_op](arr, high)
                mask = below if mask is None else (mask & below)
        if mask is None:
            raise ValueError("ThresholdOp requires at least one of 'low_level' / 'high_level'")
        return sample._replace(input=mask)


def connected_component_bboxes(
    mask: np.ndarray, min_area_bins: int = 1, connectivity: int = 4
) -> List[Tuple[int, int, int, int]]:
    """Label connected ``True`` regions of a 2-D bool mask → ``(row_min, row_max, col_min, col_max)`` inclusive tuples.

    Components smaller than ``min_area_bins`` are dropped. ``connectivity`` is ``4``
    (orthogonal neighbors) or ``8`` (orthogonal + diagonal). This is the shared scipy
    core behind :class:`ConnectedComponentsOp` (signal-domain bin bboxes on ``input``)
    AND :class:`dataflux.ops.target.MasksToDetectionBoxesOp` (its ``connected=True``
    mode, which lifts the tuples to xyxy-pixel detection boxes). Requires ``scipy``
    (``pip install data-flux[vision]``).
    """
    if min_area_bins < 1:
        raise ValueError(f"min_area_bins must be >= 1; got {min_area_bins!r}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8; got {connectivity!r}")
    try:
        from scipy.ndimage import find_objects, generate_binary_structure, label
    except ImportError as exc:
        raise ImportError(
            "connected-components labeling requires scipy. "
            "Install with `pip install data-flux[vision]` or add scipy to your environment."
        ) from exc

    structure = generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, n_components = label(mask, structure=structure)
    bboxes: List[Tuple[int, int, int, int]] = []
    if n_components > 0:
        for idx, sl in enumerate(find_objects(labels), start=1):
            if sl is None:
                continue
            row_slice, col_slice = sl
            area = int((labels[row_slice, col_slice] == idx).sum())
            if area < min_area_bins:
                continue
            bboxes.append(
                (
                    int(row_slice.start),
                    int(row_slice.stop) - 1,
                    int(col_slice.start),
                    int(col_slice.stop) - 1,
                )
            )
    return bboxes


@configurable(category="op", group="numpy")
class MinOp:
    """Reduce ``sample.input`` to its minimum value, ignoring NaN.

    Args:
        axis: Axis along which to compute the minimum. ``None`` (default) reduces over all axes.
        keepdims: When ``True``, the reduced axes are retained with size 1 (default ``False``).
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(self, axis: Optional[int] = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = bool(keepdims)

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "MinOp")
        return sample._replace(input=np.nanmin(arr, axis=self.axis, keepdims=self.keepdims))


@configurable(category="op", group="numpy")
class MaxOp:
    """Reduce ``sample.input`` to its maximum value, ignoring NaN.

    Args:
        axis: Axis along which to compute the maximum. ``None`` (default) reduces over all axes.
        keepdims: When ``True``, the reduced axes are retained with size 1 (default ``False``).
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(self, axis: Optional[int] = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = bool(keepdims)

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "MaxOp")
        return sample._replace(input=np.nanmax(arr, axis=self.axis, keepdims=self.keepdims))


@configurable(category="op", group="numpy")
class MedianOp:
    """Reduce ``sample.input`` to its median value, ignoring NaN.

    Args:
        axis: Axis along which to compute the median. ``None`` (default) reduces over all axes.
        keepdims: When ``True``, the reduced axes are retained with size 1 (default ``False``).
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(self, axis: Optional[int] = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = bool(keepdims)

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "MedianOp")
        return sample._replace(input=np.nanmedian(arr, axis=self.axis, keepdims=self.keepdims))


@configurable(category="op", group="numpy")
class PercentileOp:
    """Reduce ``sample.input`` to a 2-element array ``[p_low, p_high]``, ignoring NaN.

    Output shape when ``axis=None``: ``(2,)`` scalar pair. When ``axis=k``:
    ``(2, …)`` stacked along a new leading dimension.

    Args:
        low: Lower percentile in ``[0, 100]``. Default ``5.0``.
        high: Upper percentile in ``[0, 100]``, should be ``> low``. Default ``95.0``.
        axis: Axis along which to compute the percentiles. ``None`` (default) reduces over all axes.
        keepdims: When ``True``, the reduced axes are retained with size 1 (default ``False``).
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(
        self,
        low: float = 5.0,
        high: float = 95.0,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> None:
        self.low = float(low)
        self.high = float(high)
        self.axis = axis
        self.keepdims = bool(keepdims)

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "PercentileOp")
        p_low = np.nanpercentile(arr, self.low, axis=self.axis, keepdims=self.keepdims)
        p_high = np.nanpercentile(arr, self.high, axis=self.axis, keepdims=self.keepdims)
        return sample._replace(input=np.stack([p_low, p_high]))


@configurable(category="op", group="numpy")
class StatsOp:
    """Compute summary statistics of ``sample.input`` and record them in metadata; input is passed through unchanged.

    Writes five scalar float keys to ``sample.metadata``: ``{prefix}min``,
    ``{prefix}max``, ``{prefix}median``, ``{prefix}p_low``, ``{prefix}p_high``.
    NaN values are excluded from all computations.

    Chain anywhere in a pipeline without disrupting the data flow — useful for
    inspecting distribution properties during development or for downstream
    normalisation decisions.

    Args:
        low: Lower percentile bound (0–100). Default ``5.0``.
        high: Upper percentile bound (0–100). Default ``95.0``.
        prefix: Optional string prepended to every metadata key, e.g. ``"input_"`` to
            distinguish multiple ``StatsOp`` invocations in one pipeline.
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(self, low: float = 5.0, high: float = 95.0, prefix: str = "") -> None:
        self.low = float(low)
        self.high = float(high)
        self.prefix = prefix

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "StatsOp")
        p = self.prefix
        meta = dict(sample.meta)
        meta[f"{p}min"] = float(np.nanmin(arr))
        meta[f"{p}max"] = float(np.nanmax(arr))
        meta[f"{p}median"] = float(np.nanmedian(arr))
        meta[f"{p}p_low"] = float(np.nanpercentile(arr, self.low))
        meta[f"{p}p_high"] = float(np.nanpercentile(arr, self.high))
        return sample._replace(metadata=meta)


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
        # Lazy / zero-arg: store config only; bounds are validated lazily in __call__.
        self.min_area_bins = int(min_area_bins)
        self.connectivity = int(connectivity)

    def __call__(self, sample: Sample) -> Sample:
        mask = sample.input
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"ConnectedComponentsOp expects an np.ndarray on sample.input, got {type(mask).__name__}")
        if mask.ndim != 2:
            raise ValueError(f"ConnectedComponentsOp expects a 2-D mask; got shape {mask.shape}")
        # Shared scipy core (also used by dataflux.ops.target.MasksToDetectionBoxesOp);
        # validates min_area_bins / connectivity and raises the scipy ImportError.
        bboxes = connected_component_bboxes(mask, self.min_area_bins, self.connectivity)
        return sample._replace(input=bboxes)


def _apply_window(arr: np.ndarray, window: np.ndarray, axis: int) -> np.ndarray:
    """Multiply ``arr`` by the 1-D ``window`` broadcast along ``axis`` (dtype-preserving).

    The window is cast to the real dtype matching ``arr`` so a ``complex64`` / ``float32`` signal
    keeps its precision (a raw ``float64`` window would otherwise upcast it).
    """
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
        window = window.astype(arr.real.dtype, copy=False)
    moved = np.swapaxes(arr, axis, -1)
    return np.swapaxes(moved * window, axis, -1)


def _resolve_fft_sample_rate(sample: Sample, explicit: Optional[float]) -> Optional[float]:
    """Resolve the density-scaling sample rate: explicit arg → ``metadata['samplerate']`` → ``None``."""
    if explicit is not None:
        return explicit
    if sample.is_batched:
        return None
    raw = sample.meta.get("samplerate")
    return float(raw) if raw else None


# FFT normalization mode. Closed ``Literal`` (workspace "prefer closed Literals over bare strings"
# mandate) so FluxStudio / navigaitor render the choice as a dropdown and the allowed modes stay
# machine-introspectable via ``typing.get_args(...)``. These three strings are EXACTLY what
# ``numpy.fft.fft`` accepts for its ``norm=`` argument (the same set ``torch.fft.fft`` uses), passed
# straight through with no parallel runtime tuple to drift.
FourierNorm = Literal["backward", "ortho", "forward"]


@configurable(category="op", group="numpy")
class FourierOp:
    """Compute the 1-D discrete Fourier transform of ``sample.input`` (``numpy.fft.fft``).

    Accepts real **and** complex arrays; the raw output is always complex — ``complex64`` for
    ``float32``/``complex64`` input, ``complex128`` for ``float64``/integer/``complex128`` input
    (numpy's promotion rule). This is the **1-D** transform (``numpy.fft.fft``), not the 2-D / N-D
    one: for an N-D array it runs along a single ``axis`` (default the last), so a ``[B, N]`` batch
    of signals transforms per row. :class:`InverseFourierOp` is the inverse; set ``shift=True`` to
    center the zero-frequency bin (the standalone :class:`FftShiftOp` does the same independently).

    **Windowing & units.** ``window`` applies a :func:`dataflux.windows.get_window` taper before the
    transform (default ``"boxcar"`` = no taper = unchanged behaviour) and stashes the window
    correction into the metadata; ``scaling`` then returns the spectrum in real units —
    ``"amplitude"`` (V), ``"power"`` (V²) or ``"density"`` (V²/Hz, using ``sample_rate``) — dividing
    out the window's coherent gain / noise bandwidth. ``scaling="none"`` (default) leaves the raw
    complex spectrum. The one-node ``FourierOp(window="hann", scaling="density", sample_rate=…)`` is
    equivalent to the explicit chain ``WindowOp(window="hann") → FourierOp() →
    SpectrumScalingOp(scaling="density", sample_rate=…)``. Calibrated ``scaling`` requires the
    unscaled transform (``norm="backward"``); any other ``norm`` with ``scaling != "none"`` raises.

    Args:
        n: Output length along ``axis`` — zero-pad/truncate to ``n`` points. ``None`` (default) uses the input length.
        axis: Axis to transform over. Default ``-1`` (the last axis — the natural choice for a 1-D signal).
        norm: Normalization — ``"backward"`` (default, unscaled forward), ``"ortho"`` (1/sqrt(n) both ways),
            or ``"forward"`` (1/n on the forward transform). Calibrated ``scaling`` requires ``"backward"``.
        shift: When True, ``fftshift`` along ``axis`` after transforming (centers the zero bin). Default False.
        window: Taper applied before the FFT — a ``WindowName`` (``"boxcar"`` default = no taper).
        window_param: Kaiser ``β`` (def 8.6) / Tukey ``α`` (def 0.5) / Gaussian ``σ`` std (required); else ignored.
        periodic: ``True`` (default) = DFT-even window (correct for FFT analysis); ``False`` = symmetric.
        scaling: Units — ``"none"`` (complex, default), ``"amplitude"`` V, ``"power"`` V², ``"density"`` V²/Hz.
        sample_rate: Hz, for ``"density"``. ``None`` reads ``metadata["samplerate"]``, else ``1.0`` (normalized).
        one_sided: Fold to a one-sided spectrum (real signals). Default ``False``; exclusive with ``shift``.
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_COMPLEX_OR_FLOAT)

    def __init__(
        self,
        n: Optional[int] = None,
        axis: int = -1,
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
        self.axis = axis
        self.norm = norm
        self.shift = bool(shift)
        self.window: WindowName = window
        self.window_param = window_param
        self.periodic = bool(periodic)
        self.scaling: SpectrumScaling = scaling
        self.sample_rate = sample_rate
        self.one_sided = bool(one_sided)

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "FourierOp")
        if self.scaling != "none" and self.norm != "backward":
            raise ValueError(
                f"FourierOp: calibrated scaling={self.scaling!r} requires norm='backward' "
                f"(the unscaled transform); got norm={self.norm!r}"
            )
        if self.shift and self.one_sided:
            raise ValueError("FourierOp: shift and one_sided are mutually exclusive (one_sided is a half-spectrum)")
        window = None
        signal = arr
        if self.window != "boxcar":
            window = get_window(
                self.window, arr.shape[self.axis], window_param=self.window_param, periodic=self.periodic
            )
            signal = _apply_window(arr, window, self.axis)
        out = np.fft.fft(signal, n=self.n, axis=self.axis, norm=self.norm)
        if self.scaling != "none":
            s1, s2 = window_sums(window) if window is not None else (float(arr.shape[self.axis]),) * 2
            out = scale_spectrum(
                out,
                self.scaling,
                s1=s1,
                s2=s2,
                sample_rate=_resolve_fft_sample_rate(sample, self.sample_rate),
                one_sided=self.one_sided,
                axis=self.axis,
            )
        if self.shift:
            out = np.fft.fftshift(out, axes=self.axis)
        if window is not None and not sample.is_batched:
            new_meta = dict(sample.meta)
            new_meta.update(window_metadata(self.window, window))
            return sample._replace(input=out, metadata=new_meta)
        return sample._replace(input=out)


@configurable(category="op", group="numpy")
class InverseFourierOp:
    """Compute the 1-D inverse discrete Fourier transform of ``sample.input`` (``numpy.fft.ifft``).

    The sibling of :class:`FourierOp`: it maps a spectrum back to the time domain. The output is
    always complex (``numpy.fft.ifft`` always returns complex; take ``.real`` downstream if the
    original signal was real). ``InverseFourierOp(norm=…)`` must use the **same** ``norm`` as the
    forward transform to round-trip. With ``shift=True`` an ``ifftshift`` is applied to the input
    **before** the inverse transform, exactly undoing a prior ``FourierOp(shift=True)`` (the correct
    pairing even for odd-length axes).

    Args:
        n: Output length along ``axis`` — zero-pad/truncate to ``n`` points. ``None`` (default) uses the input length.
        axis: Axis to transform over. Default ``-1`` (the last axis — the natural choice for a 1-D signal).
        norm: Normalization — must match the forward transform: ``"backward"`` (default), ``"ortho"``, or ``"forward"``.
        shift: When True, ``ifftshift`` along ``axis`` before inverting (undoes a prior ``fftshift``). Default False.
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=ArrayType(dtype="complex", frameworks={"numpy"}))

    def __init__(
        self, n: Optional[int] = None, axis: int = -1, norm: FourierNorm = "backward", shift: bool = False
    ) -> None:
        # Lazy / zero-arg: store config only. ``n`` (if set) is validated lazily by numpy in __call__.
        self.n = n
        self.axis = axis
        self.norm = norm
        self.shift = bool(shift)

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "InverseFourierOp")
        if self.shift:
            arr = np.fft.ifftshift(arr, axes=self.axis)
        out = np.fft.ifft(arr, n=self.n, axis=self.axis, norm=self.norm)
        return sample._replace(input=out)


@configurable(category="op", group="numpy")
class FftShiftOp:
    """Shift the zero-frequency component to the center of the spectrum (``numpy.fft.fftshift``).

    A pure bin-rearrangement — no FFT is computed, so it is dtype- AND shape-preserving and works
    on **any** array (real, complex, or integer). Chain it after :class:`FourierOp` to center a
    spectrum for display (the ``FourierOp(shift=True)`` flag is the one-node convenience), or use it
    standalone to center an already-computed spectrum such as a 2-D spectrogram. :class:`IfftShiftOp`
    is its exact inverse (they differ only for odd-length axes).

    Args:
        axis: Axis to shift. Default ``-1`` (last axis, matches :class:`FourierOp`); ``None`` shifts every axis.
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(self, axis: Optional[int] = -1) -> None:
        self.axis = axis

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "FftShiftOp")
        return sample._replace(input=np.fft.fftshift(arr, axes=self.axis))


@configurable(category="op", group="numpy")
class IfftShiftOp:
    """Undo an :class:`FftShiftOp` — move the center frequency back to index 0 (``numpy.fft.ifftshift``).

    The exact inverse of :class:`FftShiftOp` (the two coincide for even-length axes but differ for
    odd-length ones, which is why both exist). Like its sibling it is a pure, dtype- and
    shape-preserving rearrangement that accepts any array. Apply it before :class:`InverseFourierOp`
    to recover the natural FFT bin order (``InverseFourierOp(shift=True)`` folds it in).

    Args:
        axis: Axis to shift. Default ``-1`` (last axis, matches :class:`InverseFourierOp`); ``None`` shifts every axis.
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(self, axis: Optional[int] = -1) -> None:
        self.axis = axis

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "IfftShiftOp")
        return sample._replace(input=np.fft.ifftshift(arr, axes=self.axis))


@configurable(category="op", group="numpy")
class WindowOp:
    """Apply a window taper to ``sample.input`` and record the unit-scaling correction.

    Multiplies the signal by a :func:`dataflux.windows.get_window` taper (broadcast along ``axis``)
    — the standard first step of spectral analysis, controlling FFT spectral leakage — and stashes
    the window's correction factors into ``sample.metadata`` (``window`` / ``window_sum`` ``S1`` /
    ``window_sum_sq`` ``S2`` / ``window_enbw_bins`` / ``window_coherent_gain``) so a later
    :class:`SpectrumScalingOp` can divide them out and return the spectrum in real units. Chain
    ``WindowOp → FourierOp → SpectrumScalingOp``, or fold all three into one node via
    ``FourierOp(window=…, scaling=…)``. Shape-preserving; real input stays real, complex stays
    complex (the taper is cast to the input's real dtype so precision is preserved).

    Args:
        window: Which taper — a ``WindowName`` (default ``"hann"``; ``"boxcar"`` is the rectangular identity).
        window_param: Kaiser ``β`` (def 8.6) / Tukey ``α`` (def 0.5) / Gaussian ``σ`` std (required); else ignored.
        periodic: ``True`` (default) = DFT-even window (correct for FFT analysis); ``False`` = symmetric.
        axis: Axis the window is applied along. Default ``-1`` (the last axis — the 1-D signal).
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_NDARRAY)

    def __init__(
        self,
        window: WindowName = "hann",
        window_param: Optional[float] = None,
        periodic: bool = True,
        axis: int = -1,
    ) -> None:
        # Lazy / zero-arg: store config only; window params are validated lazily by get_window.
        self.window: WindowName = window
        self.window_param = window_param
        self.periodic = bool(periodic)
        self.axis = axis

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "WindowOp")
        window = get_window(self.window, arr.shape[self.axis], window_param=self.window_param, periodic=self.periodic)
        out = _apply_window(arr, window, self.axis)
        if sample.is_batched:
            return sample._replace(input=out)
        new_meta = dict(sample.meta)
        new_meta.update(window_metadata(self.window, window))
        return sample._replace(input=out, metadata=new_meta)


@configurable(category="op", group="numpy")
class SpectrumScalingOp:
    """Scale a (complex) FFT spectrum to physical units using the window correction.

    The calibration half of the FFT chain: turns the raw :class:`FourierOp` output into an amplitude
    (V), power (V²) or power-spectral-density (V²/Hz) spectrum, dividing out the window's coherent
    gain ``S1`` / noise bandwidth ``S2`` — read from the ``window_*`` metadata stashed by
    :class:`WindowOp` or ``FourierOp(window=…)``; if absent it assumes a rectangular/boxcar window
    (``S1=S2=N``). Assumes the spectrum came from the **unscaled** forward transform
    (``norm="backward"``, the FourierOp default). Output dtype follows the mode — complex for
    ``"none"``/``"amplitude"`` (phase preserved), real for ``"power"``/``"density"``.

    Args:
        scaling: Units — ``"none"`` (unchanged), ``"amplitude"`` V, ``"power"`` V² (default), ``"density"`` V²/Hz.
        sample_rate: Hz, for ``"density"``. ``None`` (default) reads ``metadata["samplerate"]``, else ``1.0``.
        one_sided: Fold to one-sided (real-signal convention: keep 0…N/2, double interior bins). Default ``False``.
        axis: Spectrum axis. Default ``-1``.
    """

    ACCEPTS = SampleType(input=_NDARRAY)
    PRODUCES = SampleType(input=_COMPLEX_OR_FLOAT)

    def __init__(
        self,
        scaling: SpectrumScaling = "power",
        sample_rate: Optional[float] = None,
        one_sided: bool = False,
        axis: int = -1,
    ) -> None:
        # Lazy / zero-arg: store config only.
        self.scaling: SpectrumScaling = scaling
        self.sample_rate = sample_rate
        self.one_sided = bool(one_sided)
        self.axis = axis

    def __call__(self, sample: Sample) -> Sample:
        arr = _require_ndarray(sample, "SpectrumScalingOp")
        n = arr.shape[self.axis]
        if sample.is_batched:
            s1 = s2 = float(n)
        else:
            meta = sample.meta
            raw_s1, raw_s2 = meta.get(WINDOW_SUM_KEY), meta.get(WINDOW_SUMSQ_KEY)
            s1, s2 = (
                (float(raw_s1), float(raw_s2)) if raw_s1 is not None and raw_s2 is not None else (float(n), float(n))
            )
        fs = _resolve_fft_sample_rate(sample, self.sample_rate)
        if self.scaling == "density" and not fs:
            logger.debug("SpectrumScalingOp: no sample_rate for density; using normalized frequency (Fs=1.0)")
        out = scale_spectrum(arr, self.scaling, s1=s1, s2=s2, sample_rate=fs, one_sided=self.one_sided, axis=self.axis)
        return sample._replace(input=out)
