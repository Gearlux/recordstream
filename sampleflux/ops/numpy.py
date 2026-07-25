import operator
import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from confluid import configurable
from loggair import get_logger

from sampleflux.items import Mask, NDArrayItem, Record, Regions, item_data
from sampleflux.transform import Transform

logger = get_logger(__name__)


_EXPR_PATTERN = re.compile(r"\{(\w+)\}|\$(\w+)")


def resolve_expression(value: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """Substitute ``{key}`` from ``meta`` and ``$NAME`` from ``os.environ``.

    Returns the substituted string verbatim — the caller is responsible for any further
    casting (e.g. ``float(...)`` for a numeric expression). In the record model an item
    owns its own metadata (there is no shared metadata dict), so ``meta`` is usually empty and
    only literals / ``$ENV`` expressions resolve; a ``{key}`` bound then raises ``KeyError``.

    Args:
        value: Expression string with ``{meta_key}`` and/or ``$ENV_VAR`` placeholders.
        meta: Metadata dict supplying the ``{key}`` substitutions (defaults to empty).

    Raises:
        KeyError: A referenced metadata key or environment variable is missing.
    """
    meta = meta or {}

    def _repl(match: "re.Match[str]") -> str:
        meta_key = match.group(1)
        env_name = match.group(2)
        if meta_key is not None:
            if meta_key not in meta:
                raise KeyError(
                    f"resolve_expression: metadata key {meta_key!r} missing in {value!r}; "
                    f"available keys: {sorted(meta)}"
                )
            return str(meta[meta_key])
        assert env_name is not None
        if env_name not in os.environ:
            raise KeyError(f"resolve_expression: environment variable {env_name!r} missing in {value!r}")
        return os.environ[env_name]

    return _EXPR_PATTERN.sub(_repl, value)


# Threshold comparison selectors. Closed ``Literal``s so GUIs / schema generators render the
# choice as a dropdown and the allowed operators stay machine-introspectable via
# ``typing.get_args(...)``. Two distinct types because the lower bound only sensibly uses
# ``>`` / ``>=`` and the upper bound only ``<`` / ``<=``.
LowComparison = Literal[">", ">="]
HighComparison = Literal["<", "<="]

_LOW_COMPARISONS: Dict[str, Callable[[Any, float], Any]] = {">": operator.gt, ">=": operator.ge}
_HIGH_COMPARISONS: Dict[str, Callable[[Any, float], Any]] = {"<": operator.lt, "<=": operator.le}


def _resolve_bound(bound: Union[float, int, str], meta: Optional[Dict[str, Any]]) -> float:
    """Resolve a threshold bound (literal / numeric / ``resolve_expression`` string) to a float."""
    if isinstance(bound, str):
        resolved = resolve_expression(bound, meta)
        try:
            return float(resolved)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"threshold: expression {bound!r} resolved to {resolved!r}, which is not a number"
            ) from exc
    try:
        return float(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"threshold bounds must be a number or expression string; got {type(bound).__name__}") from exc


def threshold_array(
    arr: np.ndarray,
    low_level: Optional[Union[float, int, str]] = None,
    high_level: Optional[Union[float, int, str]] = None,
    low_op: LowComparison = ">",
    high_op: HighComparison = "<",
    meta: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Threshold ``arr`` into a boolean mask using one or both bounds.

    * only ``low_level``  → ``arr <low_op> low_level``    (values above the floor)
    * only ``high_level`` → ``arr <high_op> high_level``  (values below the ceiling)
    * both                → both conditions AND-ed together (band-pass)

    At least one of ``low_level`` / ``high_level`` MUST be provided.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"threshold_array expects an np.ndarray, got {type(arr).__name__}")
    if isinstance(low_level, str) and low_level.strip() == "":
        low_level = None
    if isinstance(high_level, str) and high_level.strip() == "":
        high_level = None

    mask: Optional[np.ndarray] = None
    if low_level is not None:
        low = _resolve_bound(low_level, meta)
        if np.isnan(low):
            logger.warning(f"threshold_array: resolved low_level is NaN ({low_level!r}); no values pass the floor.")
        else:
            mask = _LOW_COMPARISONS[low_op](arr, low)
    if high_level is not None:
        high = _resolve_bound(high_level, meta)
        if np.isnan(high):
            logger.warning(f"threshold_array: resolved high_level is NaN ({high_level!r}); no values pass the ceiling.")
        else:
            below = _HIGH_COMPARISONS[high_op](arr, high)
            mask = below if mask is None else (mask & below)
    if mask is None:
        raise ValueError("threshold_array requires at least one of 'low_level' / 'high_level'")
    return mask


@configurable(category="op", group="numpy")
class Threshold(Transform):
    """An array-bearing field → a boolean ``Mask`` item.

    Reads the array at ``field`` (blank = the first array-bearing item in the record) and thresholds
    it into a boolean mask with the bound / comparison / expression math (:func:`threshold_array`),
    writing a :class:`~sampleflux.Mask` item under ``output`` (a threshold mask is an
    intermediate that a later op — e.g. :class:`ConnectedComponents` — consumes). Any other key
    passes through untouched.

    Each bound is a numeric literal or a ``resolve_expression`` string — ``5.5`` / ``"5.5"``
    (literal) or ``"$REF_SNR"`` (environment variable). NOTE: ``{meta_key}`` expressions have no
    metadata source in the record model, so only literals and ``$ENV`` resolve here.

    Args:
        low_level: Lower bound (numeric literal or ``$ENV`` expression) compared with ``low_op`` when set;
            ``None`` disables the lower bound.
        high_level: Upper bound (numeric literal or ``$ENV`` expression) compared with ``high_op`` when set;
            ``None`` disables the upper bound.
        low_op: Lower-bound comparison — ``">"`` (strict, default) or ``">="`` (inclusive).
        high_op: Upper-bound comparison — ``"<"`` (strict, default) or ``"<="`` (inclusive).
        field: Name of the array field to threshold; blank (default) picks the first array-bearing item.
        output: Name of the key the boolean ``Mask`` item is written to (added if new).
    """

    handles = (NDArrayItem,)
    consumes = (NDArrayItem,)
    produces = (Mask,)

    def __init__(
        self,
        low_level: Optional[Union[float, int, str]] = None,
        high_level: Optional[Union[float, int, str]] = None,
        low_op: LowComparison = ">",
        high_op: HighComparison = "<",
        field: str = "",
        output: str = "mask",
    ) -> None:
        super().__init__()
        self.low_level = low_level
        self.high_level = high_level
        self.low_op = low_op
        self.high_op = high_op
        self.field = field
        self.output = output

    def _find_array(self, record: Record) -> np.ndarray:
        """Resolve the array to threshold (``self.field`` or the first array-bearing item)."""
        if self.field:
            if self.field not in record:
                raise ValueError(f"Threshold: field {self.field!r} not in record (keys: {list(record)})")
            data = item_data(record[self.field])
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Threshold: field {self.field!r} payload is {type(data).__name__}, expected an array")
            return data
        for _key, item in record.items():
            data = item_data(item)
            if isinstance(data, np.ndarray):
                return data
        raise ValueError(f"Threshold: no array-bearing field in record (keys: {list(record)})")

    def __call__(self, record: Record) -> Record:
        arr = self._find_array(record)
        mask = threshold_array(arr, self.low_level, self.high_level, self.low_op, self.high_op)
        return {**record, self.output: Mask(mask)}


def connected_component_bboxes(
    mask: np.ndarray, min_area_bins: int = 1, connectivity: int = 4
) -> List[Tuple[int, int, int, int]]:
    """Label connected ``True`` regions of a 2-D bool mask → ``(row_min, row_max, col_min, col_max)`` inclusive tuples.

    Components smaller than ``min_area_bins`` are dropped. ``connectivity`` is ``4``
    (orthogonal neighbors) or ``8`` (orthogonal + diagonal). Shared by :class:`ConnectedComponents`
    AND :func:`sampleflux.ops.target.masks_to_detection` (its ``connected=True`` mode). Requires
    ``scipy`` (``pip install sampleflux[vision]``).
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
            "Install with `pip install sampleflux[vision]` or add scipy to your environment."
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
class ConnectedComponents(Transform):
    """A boolean ``Mask`` → a ``Regions`` item.

    Reads the :class:`~sampleflux.Mask` at ``field`` (blank = the first ``Mask`` in the record, else the
    first array-bearing item) as a 2-D boolean array and labels its connected ``True`` regions into
    ``(row_min, row_max, col_min, col_max)`` inclusive bin-box tuples via
    :func:`connected_component_bboxes`, writing them as a :class:`~sampleflux.Regions` item under
    ``output`` (RAW detections, not model predictions). Any other key passes through.

    Components smaller than ``min_area_bins`` are dropped; ``connectivity`` selects the 4- or
    8-neighborhood. Requires ``scipy`` (``pip install sampleflux[vision]``).

    Args:
        min_area_bins: Minimum component area in bins; smaller connected regions are dropped (``>= 1``).
        connectivity: Pixel neighborhood — ``4`` (orthogonal only) or ``8`` (orthogonal + diagonal).
        field: Name of the ``Mask`` field to label; blank (default) picks the first ``Mask`` (else first array).
        output: Name of the key the ``Regions`` item is written to (added if new).
    """

    handles = (Mask,)
    consumes = (Mask,)
    produces = (Regions,)

    def __init__(
        self,
        min_area_bins: int = 1,
        connectivity: int = 4,
        field: str = "",
        output: str = "boxes",
    ) -> None:
        super().__init__()
        self.min_area_bins = int(min_area_bins)
        self.connectivity = int(connectivity)
        self.field = field
        self.output = output

    def _find_mask(self, record: Record) -> np.ndarray:
        """Resolve the mask to label (``self.field``, else the first ``Mask``, else the first array)."""
        if self.field:
            if self.field not in record:
                raise ValueError(f"ConnectedComponents: field {self.field!r} not in record (keys: {list(record)})")
            data = item_data(record[self.field])
        else:
            data = None
            for _key, item in record.items():
                if isinstance(item, Mask):
                    data = item_data(item)
                    break
            if data is None:
                for _key, item in record.items():
                    payload = item_data(item)
                    if isinstance(payload, np.ndarray):
                        data = payload
                        break
            if data is None:
                raise ValueError(
                    f"ConnectedComponents: no Mask or array-bearing field in record (keys: {list(record)})"
                )
        if not isinstance(data, np.ndarray):
            raise TypeError(f"ConnectedComponents expects an np.ndarray mask, got {type(data).__name__}")
        if data.ndim != 2:
            raise ValueError(f"ConnectedComponents expects a 2-D mask; got shape {data.shape}")
        return data

    def __call__(self, record: Record) -> Record:
        mask = self._find_mask(record)
        bboxes = connected_component_bboxes(mask, self.min_area_bins, self.connectivity)
        return {**record, self.output: Regions(boxes=list(bboxes))}


__all__ = [
    "resolve_expression",
    "threshold_array",
    "connected_component_bboxes",
    "LowComparison",
    "HighComparison",
    "Threshold",
    "ConnectedComponents",
]
