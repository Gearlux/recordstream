import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, cast

import confluid
import numpy as np
import zarr

from recordstream.io import PLAIN_TYPE, EncodedItem, decode_item, encode_item
from recordstream.items import Record
from recordstream.storage.base import (
    PLAIN_VALUE,
    TYPED_FORMAT,
    DataSink,
    DataSource,
    Storage,
    require_record_format,
    restore_attrs,
    split_attrs,
    to_numpy,
)

#: Reserved key-group attr names in the record layout (never item attrs).
_TYPE_ATTR = "__item_type__"
_ORDER_ATTR = "__field_order__"


def _read_record(grp: "zarr.Group") -> Record:
    """Decode one ``record_NNNNNN`` group of the record key-group layout."""
    order = json.loads(str(grp.attrs[_ORDER_ATTR]))
    record: Record = {}
    payload: Any
    for name in order:
        fgrp = cast(zarr.Group, grp[name])
        fattrs = dict(fgrp.attrs)
        type_name = str(fattrs[_TYPE_ATTR])
        if type_name == PLAIN_TYPE:
            # A plain value: array payload as the ``data`` array, scalar payload as the
            # ``value`` attr (JSON-marked when structured) — see ZarrGroupSink._write_record.
            if "data" in fgrp.array_keys():
                payload = np.asarray(cast(zarr.Array, fgrp["data"])[:])
            else:
                payload = restore_attrs({PLAIN_VALUE: fattrs[PLAIN_VALUE]}, {})[PLAIN_VALUE]
            record[name] = decode_item(EncodedItem(type_name=type_name, payload=payload, attrs={}))
            continue
        plain = {k: v for k, v in fattrs.items() if k != _TYPE_ATTR}
        arrays: Dict[str, Any] = {}
        if "attrs" in fgrp:
            agrp = cast(zarr.Group, fgrp["attrs"])
            for key in agrp.array_keys():
                arrays[key] = np.asarray(cast(zarr.Array, agrp[key])[:])
        payload = np.asarray(cast(zarr.Array, fgrp["data"])[:]) if "data" in fgrp.array_keys() else None
        attrs = restore_attrs(plain, arrays)
        record[name] = decode_item(EncodedItem(type_name=type_name, payload=payload, attrs=attrs))
    return record


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@confluid.configurable(category="sink")
class ZarrGroupSink(Storage, DataSink):
    """
    Stores each record as a unique group within a Zarr group.
    Supports variable lengths while keeping data in a single bundle.
    """

    def __init__(self, path: Union[str, Path] = "", overwrite: bool = False) -> None:
        # Partial / zero-arg: store config only; the group is opened lazily in open().
        self.path = str(path)
        self.overwrite = overwrite
        self._root: Optional[zarr.Group] = None
        self._counter = 0

    def open(self) -> "ZarrGroupSink":
        if self._root is None:
            self._root = zarr.open_group(self.path, mode="a")
            if self.overwrite:
                # In a real app, we'd clear the group
                pass
        return self

    def write(self, record: Any) -> None:
        self.open()
        if self._root is None:
            raise RuntimeError("Zarr group not open")
        if not isinstance(record, dict):
            raise TypeError(f"ZarrGroupSink: expected a record dict, got {type(record).__name__}")
        self._write_record(record)

    def _write_record(self, record: Record) -> None:
        """One record in the key-group layout (the Zarr twin of HDF5Sink._write_record)."""
        assert self._root is not None
        existing = self._root.attrs.get("recordstream_format")
        if existing is None and not any(True for _ in self._root.group_keys()):
            self._root.attrs["recordstream_format"] = TYPED_FORMAT
        elif existing != TYPED_FORMAT:
            require_record_format(existing, "ZarrGroupSink")

        grp = self._root.require_group(f"record_{self._counter:06d}")
        grp.attrs[_ORDER_ATTR] = json.dumps(list(record.keys()))
        for key, value in record.items():
            encoded = encode_item(value)
            fgrp = grp.require_group(key)
            fgrp.attrs[_TYPE_ATTR] = encoded.type_name
            if encoded.type_name == PLAIN_TYPE:
                plain, arrays = split_attrs({PLAIN_VALUE: encoded.payload})
                if arrays:
                    fgrp.create_array("data", data=np.asarray(arrays[PLAIN_VALUE]), overwrite=True)
                else:
                    fgrp.attrs[PLAIN_VALUE] = plain[PLAIN_VALUE]
                continue
            plain, arrays = split_attrs(encoded.attrs)
            fgrp.attrs.update(plain)
            if encoded.payload is not None:
                fgrp.create_array("data", data=np.asarray(to_numpy(encoded.payload)), overwrite=True)
            for name, attr_value in arrays.items():
                fgrp.create_array(f"attrs/{name}", data=np.asarray(attr_value), overwrite=True)
        self._counter += 1

    def flush(self) -> None:
        pass  # pragma: no cover


@confluid.configurable
class ZarrGroupSource(Storage, DataSource):
    """Read records written by :class:`ZarrGroupSink` (one Zarr group per record).

    Mirrors the group sink's key-group layout; groups are iterated in sorted name
    order so the read order matches the write order.

    Args:
        path: Path to the Zarr group written by ZarrGroupSink.
    """

    def __init__(self, path: Union[str, Path] = "") -> None:
        # Partial / zero-arg: store config only; the group is opened lazily in open().
        self.path = str(path)
        self._root: Optional[zarr.Group] = None

    def open(self) -> "ZarrGroupSource":
        if self._root is None:
            root = zarr.open_group(self.path, mode="r")
            require_record_format(root.attrs.get("recordstream_format"), "ZarrGroupSource")
            self._root = root
        return self

    def close(self) -> None:
        self._root = None

    def __iter__(self) -> Iterator[Record]:
        self.open()
        if self._root is None:
            return
        for name in sorted(self._root.group_keys()):
            yield _read_record(cast(zarr.Group, self._root[name]))

    def __len__(self) -> int:
        self.open()
        if self._root is None:
            return 0
        return len(list(self._root.group_keys()))

    def iter_metadata(self) -> "Iterator[tuple[str, dict]]":
        """(group name, ``.zattrs`` metadata) per record WITHOUT loading arrays (SupportsMetadataScan)."""
        from recordstream.storage.query import scan_zarr_metadata

        yield from scan_zarr_metadata(self.path)


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@confluid.configurable(category="sink")
class ZarrBatchSink(Storage, DataSink):
    """
    Optimized for uniform data. Appends records into a single large Zarr array.
    """

    def __init__(
        self,
        path: Union[str, Path] = "",
        shape: Optional[List[int]] = None,
        dtype: str = "float32",
        chunks: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> None:
        # Partial / zero-arg: store config only; the array is created lazily in open() (an unset
        # path / shape surfaces there).
        self.path = str(path)
        self.shape = tuple(shape) if shape else ()
        self.dtype = dtype
        self.chunks = tuple(chunks) if chunks else None
        self.overwrite = overwrite
        self._data_arr: Optional[zarr.Array] = None
        self._target_arr: Optional[zarr.Array] = None
        self._counter = 0

    def open(self) -> "ZarrBatchSink":
        if self._data_arr is None:
            # We create a resizable array (unlimited along first dimension)
            self._data_arr = zarr.open_array(
                store=f"{self.path}/data",
                mode="a" if not self.overwrite else "w",
                shape=(0,) + self.shape,
                chunks=(1,) + self.shape if not self.chunks else self.chunks,
                dtype=self.dtype,
            )
        return self

    def write(self, record: Any) -> None:
        self.open()
        if self._data_arr is None:
            raise RuntimeError("Zarr array not open")
        if not isinstance(record, dict):
            raise TypeError(f"ZarrBatchSink: expected a record dict, got {type(record).__name__}")
        if not record:
            raise ValueError("ZarrBatchSink: cannot write an empty record")

        # The batch sink stores ONE uniform array: the FIRST record entry's payload per row
        # (insertion order), plus a one-time item template (type/key/attrs of the FIRST record)
        # so the source can rebuild typed rows. Uniform-batch by design — per-record attr
        # variation does not fit a single stacked array; use ZarrGroupSink for that.
        key, value = next(iter(record.items()))
        encoded = encode_item(value)
        existing = self._data_arr.attrs.get("recordstream_format")
        if existing is None:
            if self._data_arr.shape[0] > 0:
                require_record_format(None, "ZarrBatchSink")
            plain, arrays = split_attrs(encoded.attrs)
            if arrays:
                raise TypeError(
                    "ZarrBatchSink: array-valued item attrs do not fit the single-array batch "
                    "layout — use ZarrGroupSink."
                )
            self._data_arr.attrs.update(
                {"recordstream_format": TYPED_FORMAT, _TYPE_ATTR: encoded.type_name, "__field__": key, **plain}
            )
        elif existing != TYPED_FORMAT:
            require_record_format(existing, "ZarrBatchSink")
        self._data_arr.append([np.asarray(to_numpy(encoded.payload))], axis=0)
        self._counter += 1

    def flush(self) -> None:
        pass  # pragma: no cover


@confluid.configurable
class ZarrBatchSource(Storage, DataSource):
    """Read records written by :class:`ZarrBatchSink` (one stacked array).

    The batch sink appends every record's first entry's payload along axis 0 of a
    single ``data`` array plus a one-time uniform item template, so this source
    yields single-key records — one per row of the leading axis.

    Args:
        path: Path to the Zarr store written by ZarrBatchSink (the directory holding the ``data`` array).
    """

    def __init__(self, path: Union[str, Path] = "") -> None:
        # Partial / zero-arg: store config only; the array is opened lazily in open().
        self.path = str(path)
        self._data_arr: Optional[zarr.Array] = None

    def open(self) -> "ZarrBatchSource":
        if self._data_arr is None:
            arr = zarr.open_array(store=f"{self.path}/data", mode="r")
            require_record_format(arr.attrs.get("recordstream_format"), "ZarrBatchSource")
            self._data_arr = arr
        return self

    def close(self) -> None:
        self._data_arr = None

    def __iter__(self) -> Iterator[Record]:
        self.open()
        if self._data_arr is None:
            return
        attrs = dict(self._data_arr.attrs)
        # Record batch rows: rebuild each row as the stored item type under the stored key
        # (uniform template — see ZarrBatchSink.write).
        field = str(attrs["__field__"])
        type_name = str(attrs[_TYPE_ATTR])
        item_attrs = restore_attrs(
            {k: v for k, v in attrs.items() if k not in ("recordstream_format", _TYPE_ATTR, "__field__")}, {}
        )
        for i in range(self._data_arr.shape[0]):
            payload = np.asarray(self._data_arr[i])
            item = decode_item(EncodedItem(type_name=type_name, payload=payload, attrs=item_attrs))
            yield {field: item}

    def __len__(self) -> int:
        self.open()
        if self._data_arr is None:
            return 0
        return int(self._data_arr.shape[0])
