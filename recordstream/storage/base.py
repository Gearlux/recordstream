import json
from typing import Any, Dict, Iterator, Protocol, Self, Tuple, runtime_checkable

import numpy as np

from recordstream._compat import is_torch_tensor
from recordstream.items import Record

#: Root-attribute format tag stamped on stores written in the record key-group layout.
TYPED_FORMAT = "typedrecord-v1"

#: Attr/entry name a ``"plain"``-tagged SCALAR record value is stored under (the array
#: payload of a plain value rides the regular ``data`` slot instead).
PLAIN_VALUE = "value"

#: Prefix marking a JSON-encoded structured attr value (list/tuple/dict/None) in plain attrs.
_JSON_MARK = "__json__:"


def require_record_format(found: Any, where: str) -> None:
    """Raise unless ``found`` is the record-layout tag ``"typedrecord-v1"``.

    There is deliberately NO backward compatibility with pre-record layouts: a store whose
    ``recordstream_format`` tag is missing or different was written before the plain-dict
    record model and must be re-generated with a current sink.
    """
    if found == TYPED_FORMAT:
        return
    detail = "no recordstream_format tag" if found is None else f"recordstream_format={found!r}"
    raise ValueError(
        f"{where}: {detail} — expected {TYPED_FORMAT!r}. Pre-record-model datasets are not "
        "readable/appendable; re-generate them with a current sink."
    )


def to_numpy(data: Any) -> Any:
    """Convert a torch tensor to a numpy array for array-storage backends (HDF5 / Zarr).

    Detaches and moves to CPU first so tensors carrying grad or living on a GPU
    convert cleanly. Non-tensor values pass through unchanged.
    """
    if is_torch_tensor(data):
        return data.detach().cpu().numpy()
    return data


@runtime_checkable
class DataSource(Protocol):
    """Minimum contract for a RecordStream data source."""

    def __iter__(self) -> Iterator[Record]:
        """Iterate over records in the source."""
        ...

    def __len__(self) -> int:
        """Total number of records available."""
        ...


@runtime_checkable
class DataSink(Protocol):
    """Minimum contract for a RecordStream data sink."""

    def write(self, record: Record) -> None:
        """Write a single record to the sink."""
        ...

    def flush(self) -> None:
        """Ensure all pending writes are committed to storage."""
        ...


# --------------------------------------------------------------------------------------
# The shared attr wire-format for the record key-group layout (HDF5 attrs / Zarr .zattrs
# / directory JSON all speak it): scalars stay native (queryable), array values become
# separate datasets, and structured values (list/tuple/dict/None) ride a JSON string with
# TUPLE TAGGING so a round-trip preserves tuple-ness (Regions.canvas == (H, W), not [H, W]).
# --------------------------------------------------------------------------------------
def split_attrs(attrs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split an encoded value's attrs into ``(plain, arrays)`` for storage.

    ``plain`` holds natively-storable scalars/strings plus JSON-marked structured values;
    ``arrays`` holds ndarray/Tensor attr values (stored as their own datasets).
    """
    plain: Dict[str, Any] = {}
    arrays: Dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, np.ndarray) or is_torch_tensor(value):
            arrays[key] = to_numpy(value)
        elif isinstance(value, np.generic):
            plain[key] = value.item()
        elif isinstance(value, (bool, int, float, str)):
            plain[key] = value
        else:
            plain[key] = _JSON_MARK + json.dumps(_tag_json(value))
    return plain, arrays


def restore_attrs(plain: Dict[str, Any], arrays: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild an encoded value's attrs dict from :func:`split_attrs`' two halves."""
    attrs: Dict[str, Any] = {}
    for key, value in plain.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):  # h5py may hand string attrs back as bytes
            value = value.decode("utf-8")
        if isinstance(value, str) and value.startswith(_JSON_MARK):
            attrs[key] = _untag_json(json.loads(value[len(_JSON_MARK) :]))
        else:
            attrs[key] = value
    attrs.update(arrays)
    return attrs


def _tag_json(value: Any) -> Any:
    """JSON-safe view of ``value`` with tuples tagged (``{"__tuple__": [...]}``) so they survive."""
    if isinstance(value, tuple):
        return {"__tuple__": [_tag_json(v) for v in value]}
    if isinstance(value, list):
        return [_tag_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _tag_json(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _untag_json(value: Any) -> Any:
    """Reverse :func:`_tag_json` (restores tuples)."""
    if isinstance(value, dict):
        if set(value.keys()) == {"__tuple__"}:
            return tuple(_untag_json(v) for v in value["__tuple__"])
        return {k: _untag_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_untag_json(v) for v in value]
    return value


class Storage:
    """Base class for storage backends providing context manager support."""

    def open(self) -> Self:
        return self

    def close(self) -> None:
        pass  # pragma: no cover

    def __enter__(self) -> Self:
        # `Self`, not `Storage`: `with HDF5Sink(...) as sink` must keep the concrete
        # backend type so `sink.write(...)` type-checks at the call site.
        return self.open()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
