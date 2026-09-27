import operator
import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, get_args

import numpy as np
from confluid import configurable
from loggair import get_logger

from recordstream._compat import is_torch_tensor
from recordstream.items import Boxes, Image, Mask, NDArrayItem, Record, item_data, with_data
from recordstream.transform import Transform

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
    writing a :class:`~recordstream.Mask` item under ``output`` (a threshold mask is an
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


def connected_component_boxes(
    mask: np.ndarray, min_area_bins: int = 1, connectivity: int = 4
) -> List[Tuple[int, int, int, int]]:
    """Label connected ``True`` regions of a 2-D bool mask → HALF-OPEN xyxy ``(x0, y0, x1, y1)`` tuples.

    The pixel-box convention every :class:`~recordstream.Boxes` producer emits: x = column,
    y = row, far edges EXCLUSIVE — ``mask[y0:y1, x0:x1]`` covers the component exactly.
    (Renamed from ``connected_component_bboxes``, which returned INCLUSIVE
    ``(row_min, row_max, col_min, col_max)`` tuples — the rename makes a stale caller fail
    loudly instead of silently mis-reading axes.)

    Components smaller than ``min_area_bins`` are dropped. ``connectivity`` is ``4``
    (orthogonal neighbors) or ``8`` (orthogonal + diagonal). Shared by :class:`ConnectedComponents`
    AND :func:`recordstream.ops.target.masks_to_detection` (its ``connected=True`` mode). Requires
    ``scipy`` (``pip install recordstream[vision]``).
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
            "Install with `pip install recordstream[vision]` or add scipy to your environment."
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
                    int(col_slice.start),
                    int(row_slice.start),
                    int(col_slice.stop),
                    int(row_slice.stop),
                )
            )
    return bboxes


@configurable(category="op", group="numpy")
class ConnectedComponents(Transform):
    """A boolean ``Mask`` → a ``Boxes`` item.

    Reads the :class:`~recordstream.Mask` at ``field`` (blank = the first ``Mask`` in the record, else the
    first array-bearing item) as a 2-D boolean array and labels its connected ``True`` regions into
    HALF-OPEN pixel xyxy ``(x0, y0, x1, y1)`` tuples via :func:`connected_component_boxes`,
    writing them as a :class:`~recordstream.Boxes` item under ``output`` with ``canvas`` set to
    the mask's shape (RAW detections, not model predictions). Any other key passes through.

    Components smaller than ``min_area_bins`` are dropped; ``connectivity`` selects the 4- or
    8-neighborhood. Requires ``scipy`` (``pip install recordstream[vision]``).

    Args:
        min_area_bins: Minimum component area in bins; smaller connected regions are dropped (``>= 1``).
        connectivity: Pixel neighborhood — ``4`` (orthogonal only) or ``8`` (orthogonal + diagonal).
        field: Name of the ``Mask`` field to label; blank (default) picks the first ``Mask`` (else first array).
        output: Name of the key the ``Boxes`` item is written to (added if new).
    """

    handles = (Mask,)
    consumes = (Mask,)
    produces = (Boxes,)

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
        boxes = connected_component_boxes(mask, self.min_area_bins, self.connectivity)
        # canvas is filled in even for an EMPTY box set — a frame check that silently skips
        # exactly the records with nothing to check reports a clean bill for the wrong reason.
        return {**record, self.output: Boxes(boxes=list(boxes), canvas=(mask.shape[0], mask.shape[1]))}


def entries_to_change(op_name: str, record: Record, field: Optional[str]) -> List[str]:
    """The record keys an array op changes — ONE rule for ``Scale`` / ``ToType`` / ``ConvertMode``.

    A blank ``field`` means every :class:`~recordstream.Image`, and deliberately NOTHING else: a
    ``Mask`` beside it holds class ids that must stay integers, and scaling one silently turns
    every id into a fraction. A named ``field`` is the explicit way to reach any other entry — a
    mask, a spectrogram, a signal's payload.
    """
    if field:
        if field not in record:
            raise ValueError(f"{op_name}: field {field!r} not in record (keys: {list(record)})")
        return [field]
    return [key for key, value in record.items() if isinstance(value, Image)]


def _payload(op_name: str, key: str, value: Any) -> np.ndarray:
    """The array an entry carries, or a refusal naming what it carries instead."""
    data = item_data(value)
    if isinstance(data, np.ndarray):
        return data
    if is_torch_tensor(data):
        raise ValueError(f"{op_name}: {key!r} is already a torch tensor — run {op_name} before ToTensor")
    raise ValueError(f"{op_name}: {key!r} holds a {type(value).__name__}, not an array")


def _replace(value: Any, payload: np.ndarray, new: np.ndarray) -> Any:
    """``new`` in ``value``'s place: a plain array stays plain, an item keeps its type and attrs."""
    return new if value is payload else with_data(value, new)


@configurable(category="op", group="numpy")
class Scale(Transform):
    """Map one value range onto another — ``[source_min, source_max]`` → ``[target_min, target_max]``.

    Changes the RANGE and nothing else: no clipping (a value outside the source range lands
    outside the target range), no channel change. The result is floating point — ``float32`` for
    an integer input, the input's own float type otherwise — because a range like ``0..1`` cannot
    be held in an integer. To change the element type without touching the values, use
    :class:`ToType`.

    A blank source bound is the integer type's full range, so a bare ``Scale()`` takes ``uint8``
    pixels to ``0..1``. A 12-bit sensor stored as ``uint16`` names its own range
    (``source_max: 4095``) — left blank it would scale by ``65535`` and read dark. A float has no
    full range, so a blank bound on a float payload is REFUSED rather than guessed: guessing
    ("anything above 1 must be 0..255") is exactly what squashed standardized images to ~0.

    Args:
        source_min: The input value that maps to ``target_min``; blank = the integer type's minimum.
        source_max: The input value that maps to ``target_max``; blank = the integer type's maximum.
        target_min: The output value ``source_min`` maps to.
        target_max: The output value ``source_max`` maps to.
        field: The one entry to scale (any array); blank = every Image, never a Mask.
    """

    handles = (NDArrayItem,)

    def __init__(
        self,
        source_min: Optional[float] = None,
        source_max: Optional[float] = None,
        target_min: float = 0.0,
        target_max: float = 1.0,
        field: str = "",
    ) -> None:
        super().__init__()
        self.source_min = source_min
        self.source_max = source_max
        self.target_min = target_min
        self.target_max = target_max
        self.field = field

    def __call__(self, record: Record) -> Record:
        out = dict(record)
        for key in entries_to_change("Scale", record, self.field):
            payload = _payload("Scale", key, record[key])
            out[key] = _replace(record[key], payload, self._scale(key, payload))
        return out

    def _scale(self, key: str, array: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(array):
            raise ValueError(
                f"Scale: {key!r} is complex — complex values have no order to take a range over; "
                "take the magnitude or the real part first"
            )
        low, high = self.source_min, self.source_max
        if low is None or high is None:
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(
                    f"Scale: {key!r} is {array.dtype} — only an integer type has a full range to default "
                    "to; give source_min and source_max"
                )
            info = np.iinfo(array.dtype)
            low = info.min if low is None else low
            high = info.max if high is None else high
        low, high = float(low), float(high)
        if low == high:
            raise ValueError(f"Scale: source_min and source_max are both {low} — an empty range cannot be mapped")
        out_type = array.dtype if np.issubdtype(array.dtype, np.floating) else np.dtype(np.float32)
        # Divide by the source span BEFORE multiplying by the target one: for the default 0..1
        # target that is exactly ``x / 255`` for uint8 — bit-identical to the rescale ToTensor used to do.
        span = float(self.target_max) - float(self.target_min)
        scaled = (array.astype(out_type) - low) / (high - low) * span + float(self.target_min)
        return np.asarray(scaled, dtype=out_type)


#: The element types :class:`ToType` casts to — numpy's own names, one spelling per type
#: (``float64``, never also ``double``). Closed so both GUIs render a dropdown from it.
ElementType = Literal["float16", "float32", "float64", "complex64", "complex128", "uint8", "int16", "int32", "int64"]
ELEMENT_TYPES: Tuple[str, ...] = get_args(ElementType)


@configurable(category="op", group="numpy")
class ToType(Transform):
    """Cast to a named element type — and nothing else: the values are not rescaled.

    ``uint8`` pixels cast to ``float64`` stay ``0..255``; use :class:`Scale` to change the range.
    Two casts numpy performs silently and wrongly are REFUSED instead: complex → real (numpy drops
    the imaginary part) and a value an integer type cannot hold (numpy wraps it round — ``300``
    becomes ``44`` as ``uint8``), NaN and infinity included. A fraction cast to an integer type is
    truncated toward zero, as numpy does (``2.7`` → ``2``).

    Args:
        dtype: The element type to cast to.
        field: The one entry to cast (any array); blank = every Image, never a Mask.
    """

    handles = (NDArrayItem,)

    def __init__(self, dtype: ElementType = "float32", field: str = "") -> None:
        super().__init__()
        self.dtype = dtype
        self.field = field

    def __call__(self, record: Record) -> Record:
        out = dict(record)
        for key in entries_to_change("ToType", record, self.field):
            payload = _payload("ToType", key, record[key])
            out[key] = _replace(record[key], payload, self._cast(key, payload))
        return out

    def _cast(self, key: str, array: np.ndarray) -> np.ndarray:
        target = np.dtype(self.dtype)
        if np.iscomplexobj(array) and target.kind != "c":
            raise ValueError(
                f"ToType: {key!r} is complex; {self.dtype} would drop the imaginary part — "
                "take the magnitude or the real part first"
            )
        if target.kind in "iu" and array.size:
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"ToType: {key!r} holds NaN or infinity, which {self.dtype} cannot represent")
            info = np.iinfo(target)
            low, high = array.min(), array.max()
            if low < info.min or high > info.max:
                raise ValueError(
                    f"ToType: {key!r} holds {low} .. {high}, which {self.dtype} ({info.min} .. {info.max}) "
                    "cannot hold — Scale it into that range first"
                )
        return array.astype(target)


__all__ = [
    "resolve_expression",
    "threshold_array",
    "connected_component_boxes",
    "entries_to_change",
    "ElementType",
    "ELEMENT_TYPES",
    "LowComparison",
    "HighComparison",
    "Threshold",
    "ConnectedComponents",
    "Scale",
    "ToType",
]
