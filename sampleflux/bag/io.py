"""The item codec registry — how a typed item serializes, for EVERY storage backend.

Storage backends never inspect item internals: they call :func:`encode_item` to get a flat
:class:`EncodedItem` (registered type name + array payload + scalar attrs) and
:func:`decode_item` to rebuild the item. The DEFAULT structural codec covers both item
shapes (an :class:`~sampleflux.bag.items.NDArrayItem` subclass → the array + its declared
attrs; a dataclass wrapper with a ``data`` field → the payload + the remaining fields), so
an externally-registered item type — a domain package's signal item, a user type — serializes
with ZERO storage-code changes. :func:`register_io` overrides the codec for types whose
structure the default cannot capture (e.g. a payload-less wrapper with non-scalar fields).

The registered TYPE NAME (via :func:`~sampleflux.bag.items.register_item` /
:func:`~sampleflux.bag.items.get_item_type`) is the on-disk type tag — decoding requires the
item type to be registered (imported) in the reading process, exactly like the confluid
``!class:`` contract.
"""

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Tuple, cast

from sampleflux.bag.items import NDArrayItem, get_item_type, item_data
from sampleflux.bag.sample import Role, TypedSample

__all__ = [
    "EncodedItem",
    "EncodedField",
    "register_io",
    "encode_item",
    "decode_item",
    "encode_sample",
    "decode_sample",
]


@dataclass(frozen=True)
class EncodedItem:
    """One item, flattened for storage: registered type name + payload + scalar attrs."""

    type_name: str
    payload: Any  # ndarray / tensor / scalar / None
    attrs: Dict[str, Any]


@dataclass(frozen=True)
class EncodedField:
    """One named field of a sample: the encoded item plus its key and role."""

    key: str
    role: Role
    item: EncodedItem


#: Encoder: item -> (payload, attrs). Decoder: (payload, attrs) -> item.
Encoder = Callable[[Any], Tuple[Any, Dict[str, Any]]]
Decoder = Callable[[Any, Dict[str, Any]], Any]

_CODECS: Dict[type, Tuple[Encoder, Decoder]] = {}


def register_io(item_cls: type, *, encode: Encoder, decode: Decoder) -> None:
    """Override the codec for ``item_cls`` (exact type — no MRO walk; a codec is a per-type contract).

    ``encode(item) -> (payload, attrs)`` and ``decode(payload, attrs) -> item``. Registering
    replaces any previous codec for the type.
    """
    _CODECS[item_cls] = (encode, decode)


def encode_item(item: Any) -> EncodedItem:
    """Flatten one item for storage (registered codec first, else the default structural codec)."""
    codec = _CODECS.get(type(item))
    if codec is not None:
        payload, attrs = codec[0](item)
        return EncodedItem(type_name=type(item).__name__, payload=payload, attrs=attrs)
    return EncodedItem(type_name=type(item).__name__, payload=_payload(item), attrs=_attrs(item))


def decode_item(encoded: EncodedItem) -> Any:
    """Rebuild an item from its encoded form (the type must be registered in this process)."""
    cls = cast(Any, get_item_type(encoded.type_name))
    codec = _CODECS.get(cls)
    if codec is not None:
        return codec[1](encoded.payload, dict(encoded.attrs))
    if issubclass(cls, NDArrayItem):
        return cls(encoded.payload, **encoded.attrs)
    if _has_data_field(cls):
        return cls(data=encoded.payload, **encoded.attrs)
    return cls(**encoded.attrs)


def encode_sample(sample: TypedSample) -> Tuple[EncodedField, ...]:
    """Encode every field of a sample, in insertion order."""
    return tuple(
        EncodedField(key=key, role=sample.role_of(key), item=encode_item(item)) for key, item in sample.items()
    )


def decode_sample(fields: Tuple[EncodedField, ...]) -> TypedSample:
    """Rebuild a :class:`TypedSample` from encoded fields (order preserved)."""
    items: Dict[str, Any] = {}
    roles: Dict[str, Role] = {}
    for field in fields:
        items[field.key] = decode_item(field.item)
        roles[field.key] = field.role
    return TypedSample(items, roles)


# --- the default structural codec -------------------------------------------
def _payload(item: Any) -> Any:
    """The array/data payload to store — the array for array items, ``.data`` for wrappers, else ``None``."""
    if isinstance(item, NDArrayItem):
        return item_data(item)
    if is_dataclass(item) and not isinstance(item, type) and _has_data_field(type(item)):
        return getattr(item, "data")
    return None


def _attrs(item: Any) -> Dict[str, Any]:
    """The reconstruction attributes (everything but the payload)."""
    if isinstance(item, NDArrayItem):
        return {name: getattr(item, name, None) for name in type(item)._item_attrs}
    if is_dataclass(item) and not isinstance(item, type):
        return {f.name: getattr(item, f.name) for f in dataclass_fields(item) if f.name != "data"}
    return {}


def _has_data_field(cls: type) -> bool:
    return is_dataclass(cls) and any(f.name == "data" for f in dataclass_fields(cls))


# --- optional helper: an item with a python-object payload (e.g. Regions boxes) ----
def default_encoded_attrs(item: Any) -> Dict[str, Any]:
    """The default codec's attrs view of ``item`` — reusable inside a custom encoder."""
    return _attrs(item)
