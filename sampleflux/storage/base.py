import json
from typing import Any, Dict, Iterator, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

from sampleflux.bag.sample import Sample

#: Root-attribute format tag stamped on stores written in the typed field-group layout.
TYPED_FORMAT = "typedsample-v1"

#: Prefix marking a JSON-encoded structured attr value (list/tuple/dict/None) in plain attrs.
_JSON_MARK = "__json__:"


def to_numpy(data: Any) -> Any:
    """Convert a torch tensor to a numpy array for array-storage backends (HDF5 / Zarr).

    Detaches and moves to CPU first so tensors carrying grad or living on a GPU
    convert cleanly. Non-tensor values pass through unchanged.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


@runtime_checkable
class DataSource(Protocol):
    """Minimum contract for a SampleFlux data source (LEGACY carrier — dies with the purge stage)."""

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples in the source."""
        ...

    def __len__(self) -> int:
        """Total number of samples available."""
        ...


@runtime_checkable
class DataSink(Protocol):
    """Minimum contract for a SampleFlux data sink (LEGACY carrier — dies with the purge stage)."""

    def write(self, sample: Sample) -> None:
        """Write a single sample to the sink."""
        ...

    def flush(self) -> None:
        """Ensure all pending writes are committed to storage."""
        ...


@runtime_checkable
class TypedDataSource(Protocol):
    """Minimum contract for a typed-bag data source."""

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over typed samples in the source."""
        ...

    def __len__(self) -> int:
        """Total number of samples available."""
        ...


@runtime_checkable
class TypedDataSink(Protocol):
    """Minimum contract for a typed-bag data sink."""

    def write(self, sample: Sample) -> None:
        """Write a single typed sample to the sink."""
        ...

    def flush(self) -> None:
        """Ensure all pending writes are committed to storage."""
        ...


# --------------------------------------------------------------------------------------
# The shared attr wire-format for the typed field-group layout (HDF5 attrs / Zarr .zattrs
# / directory JSON all speak it): scalars stay native (queryable), array values become
# separate datasets, and structured values (list/tuple/dict/None) ride a JSON string with
# TUPLE TAGGING so a round-trip preserves tuple-ness (Regions.canvas == (H, W), not [H, W]).
# --------------------------------------------------------------------------------------
def split_attrs(attrs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split an item's attrs into ``(plain, arrays)`` for storage.

    ``plain`` holds natively-storable scalars/strings plus JSON-marked structured values;
    ``arrays`` holds ndarray/Tensor attr values (stored as their own datasets).
    """
    plain: Dict[str, Any] = {}
    arrays: Dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            arrays[key] = to_numpy(value)
        elif isinstance(value, np.generic):
            plain[key] = value.item()
        elif isinstance(value, (bool, int, float, str)):
            plain[key] = value
        else:
            plain[key] = _JSON_MARK + json.dumps(_tag_json(value))
    return plain, arrays


def restore_attrs(plain: Dict[str, Any], arrays: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild an item's attrs dict from :func:`split_attrs`' two halves."""
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

    def open(self) -> "Storage":
        return self

    def close(self) -> None:
        pass  # pragma: no cover

    def __enter__(self) -> "Storage":
        return self.open()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
