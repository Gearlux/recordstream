import json
from pathlib import Path
from typing import Any, Dict, Iterator, Union

import confluid
import numpy as np

from sampleflux.bag.io import EncodedItem, decode_item, encode_item
from sampleflux.bag.sample import Sample
from sampleflux.storage.base import DataSink, Storage, restore_attrs, split_attrs, to_numpy

#: Typed-layout filenames inside each per-sample directory.
_FIELDS_JSON = "fields.json"
_FIELDS_NPZ = "fields.npz"


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@confluid.configurable(category="sink")
class DirectorySink(Storage, DataSink):
    """
    High-concurrency sink that stores each Sample in its own directory.
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

    def write(self, sample: Any) -> None:
        """Write a sample to its own subdirectory."""
        if not isinstance(sample, Sample):
            raise TypeError(f"DirectorySink: expected a Sample bag, got {type(sample).__name__}")
        self.open()
        self._write_typed(sample)

    def _write_typed(self, sample: Sample) -> None:
        """One sample in the typed field-group layout: ``fields.json`` + ``fields.npz``.

        ``fields.json`` describes every field (order, item type, role, plain attrs);
        ``fields.npz`` carries the array halves — payloads keyed by field name, array-valued
        attrs keyed ``<field>.<attr>``. Every item serializes through the
        :mod:`sampleflux.bag.io` codec, so externally-registered item types round-trip with
        no storage edits.
        """
        sample_dir = self.path / f"{self._counter:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        spec: Dict[str, Any] = {"sampleflux_format": "typedsample-v1", "fields": []}
        payloads: Dict[str, Any] = {}
        for key, item in sample.items():
            encoded = encode_item(item)
            plain, arrays = split_attrs(encoded.attrs)
            spec["fields"].append(
                {
                    "key": key,
                    "type": encoded.type_name,
                    "role": sample.role_of(key),
                    "attrs": plain,
                    "array_attrs": sorted(arrays),
                    "has_payload": encoded.payload is not None,
                }
            )
            if encoded.payload is not None:
                payloads[key] = np.asarray(to_numpy(encoded.payload))
            for name, value in arrays.items():
                payloads[f"{key}.{name}"] = np.asarray(value)

        (sample_dir / _FIELDS_JSON).write_text(json.dumps(spec, indent=2))
        if payloads:
            np.savez(sample_dir / _FIELDS_NPZ, **payloads)
        self._counter += 1

    def flush(self) -> None:
        pass  # Filesystem handles immediate writes


@confluid.configurable
class DirectorySource(Storage):
    """Read typed samples written by :class:`DirectorySink` (one ``fields.json`` + ``fields.npz`` per sample).

    The matching source of the sink's TYPED layout (one directory per sample, sorted by the
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

    def __iter__(self) -> Iterator[Sample]:
        for sample_dir in self._sample_dirs():
            yield self._read(sample_dir)

    def __len__(self) -> int:
        return len(self._sample_dirs())

    @staticmethod
    def _read(sample_dir: Path) -> Sample:
        spec = json.loads((sample_dir / _FIELDS_JSON).read_text())
        npz_path = sample_dir / _FIELDS_NPZ
        payloads = dict(np.load(npz_path, allow_pickle=False)) if npz_path.exists() else {}
        fields: Dict[str, Any] = {}
        roles: Dict[str, Any] = {}
        for entry in spec["fields"]:
            key = entry["key"]
            arrays = {name: payloads[f"{key}.{name}"] for name in entry["array_attrs"]}
            attrs = restore_attrs(dict(entry["attrs"]), arrays)
            payload = payloads[key] if entry["has_payload"] else None
            fields[key] = decode_item(EncodedItem(type_name=entry["type"], payload=payload, attrs=attrs))
            roles[key] = entry["role"]
        return Sample(fields, roles)
