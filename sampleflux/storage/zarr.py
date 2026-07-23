import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, cast

import confluid
import numpy as np
import zarr

from sampleflux.bag.io import EncodedItem, decode_item, encode_item
from sampleflux.bag.sample import Sample, primary
from sampleflux.storage.base import TYPED_FORMAT, DataSink, DataSource, Storage, restore_attrs, split_attrs, to_numpy

#: Reserved field-group attr names in the typed layout (never item attrs).
_TYPE_ATTR = "__item_type__"
_ROLE_ATTR = "__role__"
_ORDER_ATTR = "__field_order__"


def _read_typed_group(grp: "zarr.Group") -> Sample:
    """Decode one ``sample_NNNNNN`` group of the typed field-group layout."""
    order = json.loads(str(grp.attrs[_ORDER_ATTR]))
    fields: Dict[str, Any] = {}
    roles: Dict[str, Any] = {}
    for name in order:
        fgrp = cast(zarr.Group, grp[name])
        fattrs = dict(fgrp.attrs)
        plain = {k: v for k, v in fattrs.items() if k not in (_TYPE_ATTR, _ROLE_ATTR)}
        arrays: Dict[str, Any] = {}
        if "attrs" in fgrp:
            agrp = cast(zarr.Group, fgrp["attrs"])
            for key in agrp.array_keys():
                arrays[key] = np.asarray(cast(zarr.Array, agrp[key])[:])
        payload = np.asarray(cast(zarr.Array, fgrp["data"])[:]) if "data" in fgrp.array_keys() else None
        attrs = restore_attrs(plain, arrays)
        fields[name] = decode_item(EncodedItem(type_name=str(fattrs[_TYPE_ATTR]), payload=payload, attrs=attrs))
        roles[name] = str(fattrs[_ROLE_ATTR])
    return Sample(fields, roles)


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@confluid.configurable(category="sink")
class ZarrGroupSink(Storage, DataSink):
    """
    Stores each sample as a unique array within a Zarr group.
    Supports variable lengths while keeping data in a single bundle.
    """

    def __init__(self, path: Union[str, Path] = "", overwrite: bool = False) -> None:
        # Lazy / zero-arg: store config only; the group is opened lazily in open().
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

    def write(self, sample: Any) -> None:
        self.open()
        if self._root is None:
            raise RuntimeError("Zarr group not open")
        if not isinstance(sample, Sample):
            raise TypeError(f"ZarrGroupSink: expected a Sample bag, got {type(sample).__name__}")
        self._write_typed(sample)

    def _write_typed(self, sample: Sample) -> None:
        """One sample in the typed field-group layout (the Zarr twin of HDF5Sink._write_typed)."""
        assert self._root is not None
        existing_format = self._root.attrs.get("sampleflux_format")
        if existing_format is None:
            if any(True for _ in self._root.group_keys()) and self._counter == 0:
                raise TypeError(
                    "ZarrGroupSink: this store carries the legacy Sample layout — cannot append a "
                    "Sample to it (one carrier per store)."
                )
            self._root.attrs["sampleflux_format"] = TYPED_FORMAT
        elif existing_format != TYPED_FORMAT:
            raise TypeError(f"ZarrGroupSink: unknown store format {existing_format!r}")

        grp = self._root.require_group(f"sample_{self._counter:06d}")
        grp.attrs[_ORDER_ATTR] = json.dumps(list(sample.keys()))
        for key, item in sample.items():
            encoded = encode_item(item)
            fgrp = grp.require_group(key)
            fgrp.attrs[_TYPE_ATTR] = encoded.type_name
            fgrp.attrs[_ROLE_ATTR] = sample.role_of(key)
            plain, arrays = split_attrs(encoded.attrs)
            fgrp.attrs.update(plain)
            if encoded.payload is not None:
                fgrp.create_array("data", data=np.asarray(to_numpy(encoded.payload)), overwrite=True)
            for name, value in arrays.items():
                fgrp.create_array(f"attrs/{name}", data=np.asarray(value), overwrite=True)
        self._counter += 1

    def flush(self) -> None:
        pass  # pragma: no cover


@confluid.configurable
class ZarrGroupSource(Storage, DataSource):
    """Read samples written by :class:`ZarrGroupSink` (one Zarr group per sample).

    Mirrors the group sink's layout: each ``sample_NNNNNN`` subgroup carries a
    ``data`` array, an optional ``target`` array, and the sample metadata as
    group attributes (``.zattrs``). Groups are iterated in sorted name order so
    the read order matches the write order.

    Args:
        path: Path to the Zarr group written by ZarrGroupSink.
        sample_key: Name of the per-sample array holding the primary input item's payload.
        target_key: Name of the per-sample array holding the target-role item's payload (absent when no target).
    """

    def __init__(
        self,
        path: Union[str, Path] = "",
        sample_key: str = "data",
        target_key: str = "target",
    ) -> None:
        # Lazy / zero-arg: store config only; the group is opened lazily in open().
        self.path = str(path)
        self.sample_key = sample_key
        self.target_key = target_key
        self._root: Optional[zarr.Group] = None

    def open(self) -> "ZarrGroupSource":
        if self._root is None:
            self._root = zarr.open_group(self.path, mode="r")
        return self

    def close(self) -> None:
        self._root = None

    @property
    def is_typed(self) -> bool:
        """True when the store carries the typed field-group layout (``sampleflux_format`` root attr)."""
        self.open()
        return self._root is not None and self._root.attrs.get("sampleflux_format") == TYPED_FORMAT

    def __iter__(self) -> Iterator[Any]:
        self.open()
        if self._root is None:
            return
        for name in sorted(self._root.group_keys()):
            yield _read_typed_group(cast(zarr.Group, self._root[name]))

    def __len__(self) -> int:
        self.open()
        if self._root is None:
            return 0
        return len(list(self._root.group_keys()))

    def iter_metadata(self) -> "Iterator[tuple[str, dict]]":
        """(group name, ``.zattrs`` metadata) per sample WITHOUT loading arrays (SupportsMetadataScan)."""
        from sampleflux.storage.query import scan_zarr_metadata

        yield from scan_zarr_metadata(self.path)


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@confluid.configurable(category="sink")
class ZarrBatchSink(Storage, DataSink):
    """
    Optimized for uniform data. Appends samples into a single large Zarr array.
    """

    def __init__(
        self,
        path: Union[str, Path] = "",
        shape: Optional[List[int]] = None,
        dtype: str = "float32",
        chunks: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> None:
        # Lazy / zero-arg: store config only; the array is created lazily in open() (an unset
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

    def write(self, sample: Any) -> None:
        self.open()
        if self._data_arr is None:
            raise RuntimeError("Zarr array not open")
        if not isinstance(sample, Sample):
            raise TypeError(f"ZarrBatchSink: expected a Sample bag, got {type(sample).__name__}")

        # The batch sink stores ONE uniform array: the PRIMARY input field's payload per row,
        # plus a one-time item template (type/field/attrs of the FIRST sample) so the source can
        # rebuild typed rows. Uniform-batch by design — per-sample attr variation does not fit a
        # single stacked array; use ZarrGroupSink for that.
        key, item = primary(sample)
        encoded = encode_item(item)
        if "sampleflux_format" not in self._data_arr.attrs:
            plain, arrays = split_attrs(encoded.attrs)
            if arrays:
                raise TypeError(
                    "ZarrBatchSink: array-valued item attrs do not fit the single-array batch "
                    "layout — use ZarrGroupSink."
                )
            self._data_arr.attrs.update(
                {"sampleflux_format": TYPED_FORMAT, _TYPE_ATTR: encoded.type_name, "__field__": key, **plain}
            )
        self._data_arr.append([np.asarray(to_numpy(encoded.payload))], axis=0)
        self._counter += 1

    def flush(self) -> None:
        pass  # pragma: no cover


@confluid.configurable
class ZarrBatchSource(Storage, DataSource):
    """Read samples written by :class:`ZarrBatchSink` (one stacked array).

    The batch sink appends every sample's input along axis 0 of a single
    ``data`` array and stores no per-sample target or metadata, so this source
    yields input-only :class:`~sampleflux.sample.Sample` objects — one per row of
    the leading axis.

    Args:
        path: Path to the Zarr store written by ZarrBatchSink (the directory holding the ``data`` array).
    """

    def __init__(self, path: Union[str, Path] = "") -> None:
        # Lazy / zero-arg: store config only; the array is opened lazily in open().
        self.path = str(path)
        self._data_arr: Optional[zarr.Array] = None

    def open(self) -> "ZarrBatchSource":
        if self._data_arr is None:
            self._data_arr = zarr.open_array(store=f"{self.path}/data", mode="r")
        return self

    def close(self) -> None:
        self._data_arr = None

    def __iter__(self) -> Iterator[Any]:
        self.open()
        if self._data_arr is None:
            return
        attrs = dict(self._data_arr.attrs)
        # Typed batch rows: rebuild each row as the stored item type under the stored field key
        # (uniform template — see ZarrBatchSink.write).
        field = str(attrs["__field__"])
        type_name = str(attrs[_TYPE_ATTR])
        item_attrs = restore_attrs(
            {k: v for k, v in attrs.items() if k not in ("sampleflux_format", _TYPE_ATTR, "__field__")}, {}
        )
        for i in range(self._data_arr.shape[0]):
            payload = np.asarray(self._data_arr[i])
            item = decode_item(EncodedItem(type_name=type_name, payload=payload, attrs=item_attrs))
            yield Sample({field: item})

    def __len__(self) -> int:
        self.open()
        if self._data_arr is None:
            return 0
        return int(self._data_arr.shape[0])
