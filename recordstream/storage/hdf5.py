import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import h5py
import numpy as np
from confluid import configurable
from loggair import get_logger

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

logger = get_logger("recordstream.storage.hdf5")

#: Reserved key-group attr names in the record layout (never item attrs).
_TYPE_ATTR = "__item_type__"
_ORDER_ATTR = "__field_order__"


def _read_record(group: h5py.Group) -> Record:
    """Decode one ``sNNNNNN`` record group of the record key-group layout."""
    order = json.loads(group.attrs[_ORDER_ATTR])
    record: Record = {}
    payload: Any
    for name in order:
        fgrp = group[name]
        type_name = str(fgrp.attrs[_TYPE_ATTR])
        if type_name == PLAIN_TYPE:
            # A plain value: array payload as the ``data`` dataset, scalar payload as the
            # ``value`` attr (JSON-marked when structured) — see HDF5Sink._write_record.
            if "data" in fgrp:
                payload = fgrp["data"][()]
            else:
                payload = restore_attrs({PLAIN_VALUE: fgrp.attrs[PLAIN_VALUE]}, {})[PLAIN_VALUE]
            record[name] = decode_item(EncodedItem(type_name=type_name, payload=payload, attrs={}))
            continue
        plain = {k: v for k, v in fgrp.attrs.items() if k != _TYPE_ATTR}
        arrays: Dict[str, Any] = {}
        agrp = fgrp.get("attrs")
        if isinstance(agrp, h5py.Group):
            for key, dset in agrp.items():
                arrays[key] = dset[()]
        payload = fgrp["data"][()] if "data" in fgrp else None
        attrs = restore_attrs(dict(plain), arrays)
        record[name] = decode_item(EncodedItem(type_name=type_name, payload=payload, attrs=attrs))
    return record


@configurable
class HDF5Source(Storage, DataSource):
    """Read records written by :class:`HDF5Sink` (the record key-group layout).

    Args:
        path: Path to the HDF5 file written by HDF5Sink.
    """

    def __init__(self, path: Union[str, Path] = "") -> None:
        # Lazy / zero-arg: store config only; the file is opened lazily in open() (an unset path
        # surfaces there, not in __init__).
        self.path = Path(path)
        self._file: Optional[h5py.File] = None

    def open(self) -> "HDF5Source":
        if self._file is None:
            handle = h5py.File(self.path, "r")
            found = handle.attrs.get("recordstream_format")
            if found != TYPED_FORMAT:
                handle.close()
                require_record_format(found, "HDF5Source")
            self._file = handle
        return self

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __iter__(self) -> Iterator[Record]:
        self.open()
        if self._file is None:
            return
        for name in sorted(k for k in self._file.keys() if k.startswith("s")):
            yield _read_record(self._file[name])

    def __len__(self) -> int:
        self.open()
        if self._file is None:
            return 0
        return len([k for k in self._file.keys() if k.startswith("s")])

    def iter_metadata(self) -> "Iterator[tuple[str, dict]]":
        """(prefix, metadata) per record WITHOUT loading data arrays (SupportsMetadataScan).

        Array-valued metadata appears as shape/dtype stub strings — see
        :func:`recordstream.storage.query.scan_hdf5_metadata`.
        """
        from recordstream.storage.query import scan_hdf5_metadata

        yield from scan_hdf5_metadata(self.path)


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@configurable(category="sink")
class HDF5Sink(Storage, DataSink):
    """High-performance HDF5 data sink for plain record dicts."""

    def __init__(
        self,
        path: Union[str, Path] = "",
        compression: Optional[str] = "gzip",
        overwrite: bool = False,
    ) -> None:
        # Lazy / zero-arg: store config only; the file is opened lazily in open().
        self.path = Path(path)
        self.compression = compression
        self.overwrite = overwrite
        self._file: Optional[h5py.File] = None
        self._counter = 0

    def open(self) -> "HDF5Sink":
        if self._file is None:
            mode = "w" if self.overwrite and self._counter == 0 else "a"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Opening HDF5 file for writing: {self.path} (mode={mode})")
            self._file = h5py.File(self.path, mode)
        return self

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def write(self, record: Any) -> None:
        self.open()
        if self._file is None:
            return
        if not isinstance(record, dict):
            raise TypeError(f"HDF5Sink: expected a record dict, got {type(record).__name__}")
        self._write_record(record)

    def _write_record(self, record: Record) -> None:
        """One record in the key-group layout.

        Layout: root attr ``recordstream_format = "typedrecord-v1"``; per record a group
        ``sNNNNNN`` (attr ``__field_order__`` preserves insertion order) holding one subgroup
        per KEY with the ``__item_type__`` attr + the item's plain attrs, the payload as
        ``data``, and array-valued attrs as datasets under ``attrs/``. A ``"plain"`` value
        stores an array payload as ``data`` and any other payload as the ``value`` attr
        (JSON-marked when structured). Every value serializes through the
        :mod:`recordstream.io` codec, so externally-registered item types round-trip with no
        storage edits.
        """
        assert self._file is not None
        existing = self._file.attrs.get("recordstream_format")
        if existing is None and len(self._file) == 0:
            self._file.attrs["recordstream_format"] = TYPED_FORMAT
        elif existing != TYPED_FORMAT:
            require_record_format(existing, "HDF5Sink")

        group = self._file.create_group(f"s{self._counter:06d}")
        group.attrs[_ORDER_ATTR] = json.dumps(list(record.keys()))
        for key, value in record.items():
            encoded = encode_item(value)
            fgrp = group.create_group(key)
            fgrp.attrs[_TYPE_ATTR] = encoded.type_name
            if encoded.type_name == PLAIN_TYPE:
                plain, arrays = split_attrs({PLAIN_VALUE: encoded.payload})
                if arrays:
                    arr = np.asarray(arrays[PLAIN_VALUE])
                    kwargs = {"compression": self.compression} if self.compression and arr.ndim > 0 else {}
                    fgrp.create_dataset("data", data=arr, **kwargs)
                else:
                    fgrp.attrs[PLAIN_VALUE] = plain[PLAIN_VALUE]
                continue
            plain, arrays = split_attrs(encoded.attrs)
            for name, attr_value in plain.items():
                fgrp.attrs[name] = attr_value
            if encoded.payload is not None:
                payload = np.asarray(to_numpy(encoded.payload))
                kwargs = {"compression": self.compression} if self.compression and payload.ndim > 0 else {}
                fgrp.create_dataset("data", data=payload, **kwargs)
            for name, attr_value in arrays.items():
                arr = np.asarray(attr_value)
                kwargs = {"compression": self.compression} if self.compression and arr.ndim > 0 else {}
                fgrp.create_dataset(f"attrs/{name}", data=arr, **kwargs)
        self._counter += 1

    def flush(self) -> None:
        if self._file:
            self._file.flush()
