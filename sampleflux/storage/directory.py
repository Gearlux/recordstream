import json
from pathlib import Path
from typing import Any, Dict, Iterator, Union

import confluid
import numpy as np

from sampleflux.io import PLAIN_TYPE, EncodedItem, decode_item, encode_item
from sampleflux.items import Record
from sampleflux.storage.base import (
    PLAIN_VALUE,
    TYPED_FORMAT,
    DataSink,
    Storage,
    require_record_format,
    restore_attrs,
    split_attrs,
    to_numpy,
)

#: Record-layout filenames inside each per-sample directory.
_FIELDS_JSON = "fields.json"
_FIELDS_NPZ = "fields.npz"


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@confluid.configurable(category="sink")
class DirectorySink(Storage, DataSink):
    """
    High-concurrency sink that stores each record in its own directory.
    Perfect for irregular data lengths and massive parallel writing.
    """

    def __init__(self, path: Union[str, Path] = "", overwrite: bool = False, use_npz: bool = True) -> None:
        # Lazy / zero-arg: store config only; the directory is created lazily in open().
        self.path = Path(path)
        self.overwrite = overwrite
        self.use_npz = use_npz
        self._counter = 0

    def open(self) -> "DirectorySink":
        if self.overwrite and self.path.exists():
            # In a real app, we'd clear the directory
            pass
        self.path.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, record: Any) -> None:
        """Write a record to its own subdirectory."""
        if not isinstance(record, dict):
            raise TypeError(f"DirectorySink: expected a record dict, got {type(record).__name__}")
        self.open()
        self._write_record(record)

    def _write_record(self, record: Record) -> None:
        """One record in the key-group layout: ``fields.json`` + ``fields.npz``.

        ``fields.json`` describes every entry (order, item type, plain attrs — a ``"plain"``
        value's non-array payload rides its ``attrs`` under ``"value"``, JSON-marked when
        structured); ``fields.npz`` carries the array halves — payloads keyed by record key,
        array-valued attrs keyed ``<key>.<attr>``. Every value serializes through the
        :mod:`sampleflux.io` codec, so externally-registered item types round-trip with no
        storage edits.
        """
        sample_dir = self.path / f"{self._counter:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        spec: Dict[str, Any] = {"sampleflux_format": TYPED_FORMAT, "fields": []}
        payloads: Dict[str, Any] = {}
        for key, value in record.items():
            encoded = encode_item(value)
            if encoded.type_name == PLAIN_TYPE:
                plain, arrays = split_attrs({PLAIN_VALUE: encoded.payload})
                has_payload = bool(arrays)
                spec["fields"].append(
                    {
                        "key": key,
                        "type": encoded.type_name,
                        "attrs": plain,
                        "array_attrs": [],
                        "has_payload": has_payload,
                    }
                )
                if has_payload:
                    payloads[key] = np.asarray(arrays[PLAIN_VALUE])
                continue
            plain, arrays = split_attrs(encoded.attrs)
            spec["fields"].append(
                {
                    "key": key,
                    "type": encoded.type_name,
                    "attrs": plain,
                    "array_attrs": sorted(arrays),
                    "has_payload": encoded.payload is not None,
                }
            )
            if encoded.payload is not None:
                payloads[key] = np.asarray(to_numpy(encoded.payload))
            for name, attr_value in arrays.items():
                payloads[f"{key}.{name}"] = np.asarray(attr_value)

        (sample_dir / _FIELDS_JSON).write_text(json.dumps(spec, indent=2))
        if payloads:
            np.savez(sample_dir / _FIELDS_NPZ, **payloads)
        self._counter += 1

    def flush(self) -> None:
        pass  # Filesystem handles immediate writes


@confluid.configurable
class DirectorySource(Storage):
    """Read records written by :class:`DirectorySink` (one ``fields.json`` + ``fields.npz`` per record).

    The matching source of the sink's record layout (one directory per record, sorted by the
    zero-padded name, so read order matches write order).

    Args:
        path: Root directory written by DirectorySink.
    """

    def __init__(self, path: Union[str, Path] = "") -> None:
        # Lazy / zero-arg: store config only; the directory is scanned lazily on iteration.
        self.path = Path(path)

    def _sample_dirs(self) -> list:
        if not self.path.exists():
            raise FileNotFoundError(f"DirectorySource: {self.path} does not exist")
        return sorted(p for p in self.path.iterdir() if p.is_dir() and (p / _FIELDS_JSON).exists())

    def __iter__(self) -> Iterator[Record]:
        for sample_dir in self._sample_dirs():
            yield self._read(sample_dir)

    def __len__(self) -> int:
        return len(self._sample_dirs())

    @staticmethod
    def _read(sample_dir: Path) -> Record:
        spec = json.loads((sample_dir / _FIELDS_JSON).read_text())
        require_record_format(spec.get("sampleflux_format"), "DirectorySource")
        npz_path = sample_dir / _FIELDS_NPZ
        payloads = dict(np.load(npz_path, allow_pickle=False)) if npz_path.exists() else {}
        record: Record = {}
        payload: Any
        for entry in spec["fields"]:
            key = entry["key"]
            if entry["type"] == PLAIN_TYPE:
                # A plain value: array payload in the npz, non-array payload restored from the
                # ``value`` attr (see DirectorySink._write_record).
                if entry["has_payload"]:
                    payload = payloads[key]
                else:
                    payload = restore_attrs(dict(entry["attrs"]), {}).get(PLAIN_VALUE)
                record[key] = decode_item(EncodedItem(type_name=PLAIN_TYPE, payload=payload, attrs={}))
                continue
            arrays = {name: payloads[f"{key}.{name}"] for name in entry["array_attrs"]}
            attrs = restore_attrs(dict(entry["attrs"]), arrays)
            payload = payloads[key] if entry["has_payload"] else None
            record[key] = decode_item(EncodedItem(type_name=entry["type"], payload=payload, attrs=attrs))
        return record
