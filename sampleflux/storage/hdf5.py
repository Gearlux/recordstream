import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import h5py
import numpy as np
import torch
from confluid import configurable
from loggair import get_logger

from sampleflux.bag.io import EncodedItem, decode_item, encode_item
from sampleflux.bag.sample import TypedSample
from sampleflux.sample import Sample
from sampleflux.storage.base import TYPED_FORMAT, DataSink, DataSource, Storage, restore_attrs, split_attrs, to_numpy

logger = get_logger("sampleflux.storage.hdf5")

#: Reserved field-group attr names in the typed layout (never item attrs).
_TYPE_ATTR = "__item_type__"
_ROLE_ATTR = "__role__"
_ORDER_ATTR = "__field_order__"


def _read_typed_sample(group: h5py.Group) -> TypedSample:
    """Decode one ``sNNNNNN`` sample group of the typed field-group layout."""
    order = json.loads(group.attrs[_ORDER_ATTR])
    fields: Dict[str, Any] = {}
    roles: Dict[str, Any] = {}
    for name in order:
        fgrp = group[name]
        plain = {k: v for k, v in fgrp.attrs.items() if k not in (_TYPE_ATTR, _ROLE_ATTR)}
        arrays: Dict[str, Any] = {}
        agrp = fgrp.get("attrs")
        if isinstance(agrp, h5py.Group):
            for key, dset in agrp.items():
                arrays[key] = dset[()]
        payload = fgrp["data"][()] if "data" in fgrp else None
        attrs = restore_attrs(dict(plain), arrays)
        fields[name] = decode_item(EncodedItem(type_name=str(fgrp.attrs[_TYPE_ATTR]), payload=payload, attrs=attrs))
        roles[name] = str(fgrp.attrs[_ROLE_ATTR])
    return TypedSample(fields, roles)


@configurable
class HDF5Source(Storage, DataSource):
    """Clean, high-performance HDF5 data source."""

    def __init__(
        self,
        path: Union[str, Path] = "",
        sample_key: str = "data",
        target_key: Optional[str] = "target",
    ) -> None:
        # Lazy / zero-arg: store config only; the file is opened lazily in open() (an unset path
        # surfaces there, not in __init__).
        self.path = Path(path)
        self.sample_key = sample_key
        self.target_key = target_key
        self._file: Optional[h5py.File] = None

    def open(self) -> "HDF5Source":
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    @property
    def is_typed(self) -> bool:
        """True when the file carries the typed field-group layout (``sampleflux_format`` root attr)."""
        self.open()
        return self._file is not None and self._file.attrs.get("sampleflux_format") == TYPED_FORMAT

    def __iter__(self) -> Iterator[Any]:
        self.open()
        if self._file is None:
            return

        if self.is_typed:
            for name in sorted(k for k in self._file.keys() if k.startswith("s")):
                yield _read_typed_sample(self._file[name])
            return

        prefixes = sorted([k.split("_data")[0] for k in self._file.keys() if k.endswith("_data")])

        for pref in prefixes:
            data = self._file[f"{pref}_data"][()]
            target = self._file[f"{pref}_target"][()] if f"{pref}_target" in self._file else None
            metadata = dict(self._file[f"{pref}_data"].attrs)
            # Merge array-valued metadata written as datasets under the per-sample meta group
            # (see HDF5Sink.write). Absent on files written before this layout — old files read unchanged.
            meta_grp = self._file.get(f"{pref}_meta")
            if isinstance(meta_grp, h5py.Group):
                for key, dset in meta_grp.items():
                    metadata[key] = dset[()]
            # Source returns Tensors to match schema
            yield Sample(input=torch.from_numpy(data), target=target, metadata=metadata)

    def __len__(self) -> int:
        self.open()
        if self._file is None:
            return 0
        if self.is_typed:
            return len([k for k in self._file.keys() if k.startswith("s")])
        return len([k for k in self._file.keys() if k.endswith("_data")])

    def iter_metadata(self) -> "Iterator[tuple[str, dict]]":
        """(prefix, metadata) per sample WITHOUT loading data arrays (SupportsMetadataScan).

        Array-valued metadata appears as shape/dtype stub strings — see
        :func:`sampleflux.storage.query.scan_hdf5_metadata`.
        """
        from sampleflux.storage.query import scan_hdf5_metadata

        yield from scan_hdf5_metadata(self.path)


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@configurable(category="sink")
class HDF5Sink(Storage, DataSink):
    """High-performance HDF5 data sink focused on Sample triplets."""

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

    def write(self, sample: Any) -> None:
        self.open()
        if self._file is None:
            return

        if isinstance(sample, TypedSample):
            self._write_typed(sample)
            return
        if self._file.attrs.get("sampleflux_format") == TYPED_FORMAT:
            raise TypeError(
                "HDF5Sink: this file carries the typed field-group layout — cannot append a legacy "
                "Sample to it (one carrier per file)."
            )

        prefix = f"{self._counter:05d}"

        # Convert tensors to numpy for h5py
        input_data = to_numpy(sample.input)
        target_data = to_numpy(sample.target)

        # 1. Write Data
        kwargs = {}
        if self.compression and hasattr(input_data, "shape") and len(input_data.shape) > 0:
            kwargs["compression"] = self.compression

        ds = self._file.create_dataset(f"{prefix}_data", data=input_data, **kwargs)

        # 2. Write Metadata. Scalars/strings go on the data dataset's HDF5 attributes (compact,
        # round-trips for the common case). Array-valued metadata (e.g. a segmentation mask) CANNOT
        # be stored as an attribute — HDF5 caps attribute size ("object header message is too large")
        # and the str() fallback would silently truncate the array — so it is written as its own
        # dataset under a per-sample group ``{prefix}_meta/<key>`` (the "/" makes h5py auto-create the
        # group; arbitrary metadata keys are safe as dataset names). HDF5Source merges both back.
        for k, v in sample.meta.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                arr = to_numpy(v)
                m_kwargs = {}
                if self.compression and getattr(arr, "ndim", 0) > 0:
                    m_kwargs["compression"] = self.compression
                self._file.create_dataset(f"{prefix}_meta/{k}", data=arr, **m_kwargs)
            else:
                try:
                    ds.attrs[k] = v
                except Exception:
                    ds.attrs[k] = str(v)

        # 3. Write Target
        if target_data is not None:
            t_kwargs = {}
            if self.compression and hasattr(target_data, "shape") and len(target_data.shape) > 0:
                t_kwargs["compression"] = self.compression

            self._file.create_dataset(f"{prefix}_target", data=target_data, **t_kwargs)

        self._counter += 1

    def _write_typed(self, sample: TypedSample) -> None:
        """One sample in the typed field-group layout — see ``docs/typed-model.md`` (storage).

        Layout: root attr ``sampleflux_format = "typedsample-v1"``; per sample a group
        ``sNNNNNN`` (attr ``__field_order__`` preserves insertion order) holding one subgroup
        per FIELD with attrs ``__item_type__``/``__role__`` + the item's plain attrs, the
        payload as ``data``, and array-valued attrs as datasets under ``attrs/``. Every item
        serializes through the :mod:`sampleflux.bag.io` codec, so externally-registered item
        types round-trip with no storage edits.
        """
        assert self._file is not None
        if self._counter == 0 and "sampleflux_format" not in self._file.attrs:
            if any(k.endswith("_data") for k in self._file.keys()):
                raise TypeError(
                    "HDF5Sink: this file carries the legacy Sample layout — cannot append a "
                    "TypedSample to it (one carrier per file)."
                )
            self._file.attrs["sampleflux_format"] = TYPED_FORMAT
        elif self._file.attrs.get("sampleflux_format") != TYPED_FORMAT:
            raise TypeError(
                "HDF5Sink: this file carries the legacy Sample layout — cannot append a "
                "TypedSample to it (one carrier per file)."
            )

        group = self._file.create_group(f"s{self._counter:06d}")
        group.attrs[_ORDER_ATTR] = json.dumps(list(sample.keys()))
        for key, item in sample.items():
            encoded = encode_item(item)
            fgrp = group.create_group(key)
            fgrp.attrs[_TYPE_ATTR] = encoded.type_name
            fgrp.attrs[_ROLE_ATTR] = sample.role_of(key)
            plain, arrays = split_attrs(encoded.attrs)
            for name, value in plain.items():
                fgrp.attrs[name] = value
            if encoded.payload is not None:
                payload = np.asarray(to_numpy(encoded.payload))
                kwargs = {"compression": self.compression} if self.compression and payload.ndim > 0 else {}
                fgrp.create_dataset("data", data=payload, **kwargs)
            for name, value in arrays.items():
                arr = np.asarray(value)
                kwargs = {"compression": self.compression} if self.compression and arr.ndim > 0 else {}
                fgrp.create_dataset(f"attrs/{name}", data=arr, **kwargs)
        self._counter += 1

    def flush(self) -> None:
        if self._file:
            self._file.flush()
