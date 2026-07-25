"""The item codec registry — how a typed value serializes, for EVERY storage backend.

Storage backends never inspect item internals: they call :func:`encode_item` to get a flat
:class:`EncodedItem` (registered type name + array payload + scalar attrs) and
:func:`decode_item` to rebuild the item. The DEFAULT structural codec covers both item
shapes (an :class:`~sampleflux.items.NDArrayItem` subclass → the array + its declared
attrs; a dataclass wrapper with a ``data`` field → the payload + the remaining fields), so
an externally-registered item type — a domain package's signal item, a user type — serializes
with ZERO storage-code changes. :func:`register_io` overrides the codec for types whose
structure the default cannot capture (e.g. a payload-less wrapper with non-scalar fields).

A PLAIN (non-item) record value — a float, a string, a bare array — encodes under the
pseudo type tag ``"plain"`` and decodes back verbatim, so scalar metadata keys ride the
same layout as typed values.

The registered TYPE NAME (via :func:`~sampleflux.items.register_item` /
:func:`~sampleflux.items.get_item_type`) is the on-disk type tag — decoding requires the
item type to be registered (imported) in the reading process, exactly like the confluid
``!class:`` contract.
"""

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Tuple, cast

from sampleflux.items import NDArrayItem, Record, get_item_type, is_item, item_data

__all__ = [
    "EncodedItem",
    "EncodedField",
    "register_io",
    "encode_item",
    "decode_item",
    "encode_record",
    "decode_record",
]

#: The on-disk type tag for a plain (non-item) record value — stored and restored verbatim.
PLAIN_TYPE = "plain"


@dataclass(frozen=True)
class EncodedItem:
    """One value, flattened for storage: registered type name (or ``"plain"``) + payload + scalar attrs."""

    type_name: str
    payload: Any  # ndarray / tensor / scalar / None
    attrs: Dict[str, Any]


@dataclass(frozen=True)
class EncodedField:
    """One named entry of a record: the encoded value plus its key."""

    key: str
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
    """Flatten one value for storage (registered codec first, else the default structural codec).

    A value that is not a registered item type encodes as ``"plain"`` — payload verbatim.
    """
    codec = _CODECS.get(type(item))
    if codec is not None:
        payload, attrs = codec[0](item)
        return EncodedItem(type_name=type(item).__name__, payload=payload, attrs=attrs)
    if not is_item(item):
        return EncodedItem(type_name=PLAIN_TYPE, payload=item, attrs={})
    return EncodedItem(type_name=type(item).__name__, payload=_payload(item), attrs=_attrs(item))


def decode_item(encoded: EncodedItem) -> Any:
    """Rebuild a value from its encoded form (an item type must be registered in this process)."""
    if encoded.type_name == PLAIN_TYPE:
        return encoded.payload
    cls = cast(Any, get_item_type(encoded.type_name))
    codec = _CODECS.get(cls)
    if codec is not None:
        return codec[1](encoded.payload, dict(encoded.attrs))
    if issubclass(cls, NDArrayItem):
        return cls(encoded.payload, **encoded.attrs)
    if _has_data_field(cls):
        return cls(data=encoded.payload, **encoded.attrs)
    return cls(**encoded.attrs)


def encode_record(record: Record) -> Tuple[EncodedField, ...]:
    """Encode every entry of a record, in insertion order."""
    return tuple(EncodedField(key=key, item=encode_item(value)) for key, value in record.items())


def decode_record(fields: Tuple[EncodedField, ...]) -> Record:
    """Rebuild a record dict from encoded fields (order preserved)."""
    return {field.key: decode_item(field.item) for field in fields}


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
