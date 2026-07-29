"""Typed values — the vocabulary a record is made of, each value OWNING its metadata.

A record is a plain ``dict`` (the :data:`Record` alias) whose values are TYPED: an
:class:`Image` carries its ``layout``, a :class:`Label` its ``classes``, a
:class:`Regions` its ``canvas`` reference frame. Ops dispatch on these types (the
torchvision-v2 ``tv_tensors`` idea) — there is no wrapper container and no role tags;
key names ("image", "mask", "label") carry meaning, exactly like every torch batch dict.

The item model is HYBRID (the workspace decision):

* **Array-backed items subclass the payload** (:class:`NDArrayItem`, an ``np.ndarray``
  subclass) so a type-agnostic operation touches them AS an array while their extra
  attributes survive numpy operations (``__array_finalize__``). ``Image`` / ``Mask`` are
  these.
* **Structured items are dataclass wrappers** (:class:`Regions` / :class:`Label`) — a
  bounding-box set or a class label is not an array; a wrapper is also the right home for a
  payload a domain package does not want to subclass (e.g. complex-IQ signal data, where
  subclassing an ``np.complex64`` ndarray and preserving attributes through arithmetic is
  fragile).

This module is MODALITY-NEUTRAL — only generic items live here (images, masks, boxes,
labels). Domain items (a signal, a spectrogram) live in the domain package and register
into the SAME registry, per the workspace modality-neutral mandate. That IS the
extensibility story below.

Both shapes present a uniform payload accessor via :func:`item_data` / :func:`with_data`, so
a transform kernel never has to special-case "is this a subclass or a wrapper".

Extensibility: any type decorated with :func:`register_item` becomes a first-class item —
the dispatch registry (:mod:`recordstream.dispatch`) and a visual editor's socket-type map can
see it. A downstream package (a signal item, a user type) adds one class + one decorator,
no core edit.

NOTE (scope): array items are ``np.ndarray`` subclasses only; a torch-``Tensor``-subclass
item base (via ``__torch_function__``) is a documented follow-up — torch payloads ride in
wrapper items. Items are registered in the local :func:`register_item` registry rather than
carried on the confluid ``@configurable`` registry (an ``np.ndarray`` subclass builds
through ``__new__``, which fights confluid's ``__init__`` validation wrap).
"""

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, cast

import numpy as np

_ItemT = TypeVar("_ItemT")

#: A record record — a PLAIN dict of typed values. There is deliberately no container
#: class: ops receive and return ordinary dicts, so library transforms that already
#: understand dicts (torchvision v2) or named kwargs (albumentations) run as-is.
Record = Dict[str, Any]

__all__ = [
    "Record",
    "NDArrayItem",
    "Image",
    "Mask",
    "Regions",
    "Label",
    "MultiLabel",
    "is_class_id",
    "register_item",
    "item_types",
    "item_type_names",
    "get_item_type",
    "is_item",
    "item_data",
    "with_data",
]

# ---------------------------------------------------------------------------
# Item registry — the extensibility surface. A registered type is a first-class
# item the dispatch registry and a visual editor's socket-type map see.
# ---------------------------------------------------------------------------
_ITEM_TYPES: Dict[str, type] = {}


def register_item(cls: Type[Any]) -> Type[Any]:
    """Register ``cls`` as a first-class item type (usable as a class decorator).

    Re-registering the same name overwrites (consumers may deliberately replace a type).
    """
    _ITEM_TYPES[cls.__name__] = cls
    return cls


def item_types() -> Tuple[type, ...]:
    """Every registered item type (registration order)."""
    return tuple(_ITEM_TYPES.values())


def get_item_type(name: str) -> type:
    """The registered item type named ``name`` (a miss names the known types)."""
    try:
        return _ITEM_TYPES[name]
    except KeyError:
        known = ", ".join(sorted(_ITEM_TYPES)) or "<none>"
        raise KeyError(f"no item type registered as {name!r} (known: {known})") from None


def item_type_names() -> Tuple[str, ...]:
    """The registered item type NAMES (sorted) — the enumerable socket-type vocabulary."""
    return tuple(sorted(_ITEM_TYPES))


def is_item(obj: Any) -> bool:
    """True if ``obj`` is an instance of a registered item type."""
    types = tuple(_ITEM_TYPES.values())
    return bool(types) and isinstance(obj, types)


# ---------------------------------------------------------------------------
# Array-backed items — np.ndarray subclasses that preserve their extra attributes.
# ---------------------------------------------------------------------------
class NDArrayItem(np.ndarray):
    """Base for array-backed items: an ``np.ndarray`` subclass whose declared extra
    attributes (``_item_attrs``) survive numpy operations via ``__array_finalize__``.

    Subclasses declare their metadata attributes as ``_item_attrs`` plus a class-level
    default for each::

        class Image(NDArrayItem):
            _item_attrs = ("layout",)
            layout = "HWC"

        img = Image(rgb_hwc)            # img.layout == "HWC"
        img = Image(rgb_chw, layout="CHW")
        flipped = np.flip(img, axis=1)  # still an Image, flipped.layout == "CHW"
    """

    _item_attrs: Tuple[str, ...] = ()

    def __new__(cls, data: Any, **attrs: Any) -> "NDArrayItem":
        unknown = set(attrs) - set(cls._item_attrs)
        if unknown:
            raise TypeError(
                f"{cls.__name__}: unexpected attributes {sorted(unknown)} (allowed: {list(cls._item_attrs)})"
            )
        obj = np.asarray(data).view(cls)
        for name in cls._item_attrs:
            setattr(obj, name, attrs[name] if name in attrs else getattr(cls, name, None))
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        # Called on every construction path (view, slice, ufunc output). Carry the extra
        # attributes forward from the source array (class default when absent).
        if obj is None:
            return
        for name in getattr(type(self), "_item_attrs", ()):
            setattr(self, name, getattr(obj, name, getattr(type(self), name, None)))


@register_item
class Image(NDArrayItem):
    """An image array. ``layout`` is ``"HWC"`` (numpy convention, default) or ``"CHW"``."""

    _item_attrs = ("layout",)
    layout: str = "HWC"


@register_item
class Mask(NDArrayItem):
    """A segmentation / activity mask array (same spatial frame as its sibling image)."""


# ---------------------------------------------------------------------------
# Structured items — dataclass wrappers (not arrays).
# ---------------------------------------------------------------------------
@register_item
@dataclass
class Regions:
    """A set of rectangular regions / bounding boxes with optional labels and scores.

    Attributes:
        boxes: A list of boxes — pixel ``[x0, y0, x1, y1]`` or signal ``[f0, f1, t0, t1]``.
        labels: Optional per-box class labels.
        scores: Optional per-box confidence scores.
        canvas: Optional ``(H, W)`` reference frame — the coordinate system boxes live in,
            so a geometric transform (flip / resize) has a self-contained frame.
        extras: Auxiliary PER-BOX parallel arrays and region-set measurements keyed by name
            (e.g. per-box durations/bandwidths/power readings) — item-scoped metadata that
            travels WITH the boxes it describes.
    """

    boxes: List[Any] = field(default_factory=list)
    labels: Optional[List[Any]] = None
    scores: Optional[List[Any]] = None
    canvas: Optional[Tuple[int, int]] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def is_class_id(value: Any) -> bool:
    """True when ``value`` is an ENCODED class id (an integer), not a class name.

    The ONE rule for "is this label already encoded?" — so consumers dispatch on
    it instead of re-deriving a type check each time (a trainer used to sniff
    ``isinstance(target, str)`` itself).

    Recognises an integer in ANY framework: a Python ``int``, a numpy integer,
    and a **0-dimensional integer array or tensor** — a dataset that yields
    ``Label(torch.tensor(3))`` is as encoded as one yielding ``Label(3)``, and
    treating the tensor as a class NAME would send it through a LabelMap and
    key the mapping on ``"tensor(3)"``.

    ``bool`` is deliberately excluded: it is an ``int`` subclass, so a boolean
    flag mistakenly wired to the target key would silently become class id 1 and
    train without complaint.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    # 0-d array / tensor (numpy, torch, …) — unwrap via the array-scalar protocol
    # rather than importing a framework, so this stays modality- and engine-neutral.
    unwrap = getattr(value, "item", None)
    if callable(unwrap) and getattr(value, "ndim", None) == 0:
        try:
            return is_class_id(unwrap())
        except Exception:  # pragma: no cover - defensive: exotic 0-d payload
            return False
    return False


@register_item
@dataclass
class Label:
    """A single classification label plus its class vocabulary.

    The label is either a class NAME (needs a
    :class:`~recordstream.labels.LabelMap` to encode) or an already-encoded
    class ID — :attr:`is_encoded` is the one place that distinction is decided.

    Attributes:
        value: The label (a class id or name).
        classes: Optional ordered class vocabulary this label indexes into.
    """

    value: Any = None
    classes: Optional[List[Any]] = None

    @property
    def is_encoded(self) -> bool:
        """True when :attr:`value` is already a class id rather than a name."""
        return is_class_id(self.value)


@register_item
@dataclass
class MultiLabel:
    """Several classification labels for one record, plus their class vocabulary.

    The multi-label counterpart of :class:`Label` — a record belonging to more
    than one class. Giving it a TYPE is what lets consumers dispatch on the
    item instead of sniffing ``isinstance(value, (list, tuple, set))``, which
    cannot distinguish a genuine multi-label target from an ordinary sequence
    value that happens to sit under the target key.

    Like :class:`Label`, its values are always mappable to class ids through a
    :class:`~recordstream.labels.LabelMap`.

    Attributes:
        values: The labels (class ids or names). Order is not significant.
        classes: Optional ordered class vocabulary these labels index into.
    """

    values: List[Any] = field(default_factory=list)
    classes: Optional[List[Any]] = None

    @property
    def is_encoded(self) -> bool:
        """True when every value is already a class id (vacuously true when empty)."""
        return all(is_class_id(v) for v in self.values)


# ---------------------------------------------------------------------------
# Uniform payload accessors — so kernels never special-case subclass vs wrapper.
# ---------------------------------------------------------------------------
def item_data(item: Any) -> Any:
    """The underlying payload of an item: the plain array (array items) or ``.data`` (wrappers)."""
    if isinstance(item, NDArrayItem):
        return item.view(np.ndarray)
    if is_dataclass(item) and any(f.name == "data" for f in fields(item)):
        return getattr(item, "data")
    return item


def with_data(item: _ItemT, new_data: Any) -> _ItemT:
    """A copy of ``item`` carrying ``new_data`` as its payload, metadata preserved (same type).

    Works for both shapes: an array item is rebuilt with its declared attributes; a wrapper
    with a ``data`` field is ``dataclasses.replace``\\ d. An item with no payload slot raises.
    """
    if isinstance(item, NDArrayItem):
        attrs = {name: getattr(item, name, None) for name in type(item)._item_attrs}
        return cast(_ItemT, type(item)(new_data, **attrs))
    if is_dataclass(item) and not isinstance(item, type) and any(f.name == "data" for f in fields(item)):
        return cast(_ItemT, replace(cast(Any, item), data=new_data))
    raise TypeError(f"with_data: {type(item).__name__} has no payload slot to replace")
