"""``DerivedStore`` — what was computed FROM records, kept by record id and by what computed it.

A prediction, an embedding, a score or a thumbnail is computed from a record by some producer — a chain of ops
with its settings and its weights. The store keeps each such value under three names:

* the **producer**, a hash of what computed it (:func:`producer_hash`): another model, another setting or
  another weights file is another producer, so a value is never stale and no file time is compared;
* the **output**'s name (``embedding``, ``predictions``);
* the **record id** (:mod:`recordstream.record_id`).

On disk a producer is a folder, an output a folder inside it (a *table*), and a table an ``index.json`` plus
numbered ``.npz`` shards, each written once and never changed::

    <root>/<producer>/producer.yaml          what the producer is, for a person
    <root>/<producer>/<output>/index.json    {record id: [shard, row]}
    <root>/<producer>/<output>/000000.npz    the values of one flush

Values go through :mod:`recordstream.io`, so a typed item comes back as its type. One process writes a table at
a time; any number read it.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from confluid import configurable
from loggair import get_logger

from recordstream._compat import is_torch_tensor
from recordstream.exceptions import RecordstreamError
from recordstream.io import EncodedItem, decode_item, encode_item
from recordstream.storage.base import Storage, _tag_json, _untag_json, restore_attrs, split_attrs, to_numpy

logger = get_logger(__name__)

#: The on-disk layout's name, written in every index; another layout is refused, never guessed at.
DERIVED_FORMAT = "derived-v1"

#: How many values a table holds in memory before it writes a shard on its own.
FLUSH_EVERY = 256

_INDEX = "index.json"
_PRODUCER_FILE = "producer.yaml"
_MISSING: Any = object()


class DerivedStoreError(RecordstreamError, ValueError):
    """A derived store was asked something it cannot answer: a bad name, another layout, an unstorable value."""


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def producer_text(producer: Any) -> str:
    """What ``producer`` is, as text: a string as it is, anything else as the YAML it builds from.

    Classes are written by their importable path and every setting is listed, so a setting left at its default
    and the same value written out read the same.
    """
    if isinstance(producer, str):
        return producer
    from confluid import dump
    from confluid.dumper import is_dumpable_object
    from confluid.fluid import Fluid

    members = producer if isinstance(producer, (list, tuple)) else [producer]
    for member in members:
        if not isinstance(member, Fluid) and not is_dumpable_object(member):
            raise DerivedStoreError(
                f"DerivedStore: a {type(member).__module__}.{type(member).__qualname__} cannot be written as a "
                "document, so two of them with different settings would be kept under one name. Make the class "
                "@configurable (or build it from a config), or name the producer with a text of your own."
            )
    return dump(producer, qualified=True)


def producer_hash(producer: Any, files: Sequence[Union[str, Path]] = ()) -> str:
    """The name a producer's values are kept under: a hash of what it is and of the files it reads.

    Args:
        producer: The ops (one, a list, any configured object) that compute the values, or a text naming them.
        files: Files whose CONTENT decides the values — a model's weights. Each is hashed whole.

    Example::

        producer_hash([Resize(size=224), Embed(model="resnet18")], files=["weights/resnet18.pt"])
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(DERIVED_FORMAT.encode())
    digest.update(b"\0")
    digest.update(producer_text(producer).encode())
    for path in files:
        file = Path(os.path.expanduser(os.path.expandvars(str(path))))
        digest.update(b"\0")
        digest.update(file.name.encode())
        digest.update(b"\0")
        digest.update(_file_digest(file).encode())
    return digest.hexdigest()


def _checked_name(name: str, what: str) -> str:
    if not name or name in (".", "..") or any(sep in name for sep in ("/", "\\", "\0")) or name.startswith("."):
        raise DerivedStoreError(f"DerivedStore: {what} {name!r} must be a plain name — no path, no leading dot.")
    return name


def _encode(value: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """One value as ``(meta, arrays)``: the JSON half and the arrays, the arrays named ``p`` and ``a.<attr>``."""
    encoded = encode_item(value)
    plain, attr_arrays = split_attrs(dict(encoded.attrs))
    meta: Dict[str, Any] = {"type": encoded.type_name, "attrs": plain, "arrays": sorted(attr_arrays)}
    arrays = {f"a.{name}": np.asarray(array) for name, array in attr_arrays.items()}
    payload = encoded.payload
    if payload is None:
        meta["payload"] = "none"
    elif isinstance(payload, (np.ndarray, np.generic)) or is_torch_tensor(payload):
        array = np.asarray(to_numpy(payload))
        if array.dtype == object:
            raise DerivedStoreError("DerivedStore: an array of Python objects cannot be kept; keep numbers.")
        meta["payload"] = "array"
        arrays["p"] = array
    else:
        try:
            meta["json"] = json.loads(json.dumps(_tag_json(payload)))
        except TypeError as exc:
            raise DerivedStoreError(
                f"DerivedStore: a {type(payload).__name__} holding {exc} cannot be kept as one value — "
                "keep each array as an output of its own."
            ) from exc
        meta["payload"] = "json"
    return meta, arrays


def _decode(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Any:
    attrs = restore_attrs(dict(meta["attrs"]), {name: arrays[f"a.{name}"] for name in meta["arrays"]})
    form = meta["payload"]
    payload: Any = None
    if form == "array":
        payload = arrays["p"]
    elif form == "json":
        payload = _untag_json(meta["json"])
    return decode_item(EncodedItem(type_name=meta["type"], payload=payload, attrs=attrs))


class DerivedTable:
    """One output of one producer: a value per record id.

    Made by :meth:`DerivedStore.table`. ``put`` holds a value in memory until :meth:`flush` (or
    :data:`FLUSH_EVERY` values) writes a shard; a value put is readable at once, flushed or not.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._rows: Optional[Dict[str, List[int]]] = None
        self._pending: Dict[str, Any] = {}
        self._shards: Dict[int, Tuple[List[Dict[str, Any]], Any]] = {}

    @property
    def rows(self) -> Dict[str, List[int]]:
        """``{record id: [shard, row]}`` of what is on disk, read once."""
        if self._rows is None:
            index = self.folder / _INDEX
            if index.is_file():
                document = json.loads(index.read_text())
                if document.get("format") != DERIVED_FORMAT:
                    raise DerivedStoreError(
                        f"DerivedStore: {index} is laid out as {document.get('format')!r}, not {DERIVED_FORMAT!r}. "
                        "Remove the producer's folder to have it computed again."
                    )
                self._rows = {key: list(place) for key, place in document["rows"].items()}
            else:
                self._rows = {}
        return self._rows

    def __len__(self) -> int:
        return len(self.rows) + sum(1 for key in self._pending if key not in self.rows)

    def __contains__(self, record_id: object) -> bool:
        return record_id in self._pending or record_id in self.rows

    def ids(self) -> List[str]:
        """Every record id that has a value, the kept ones first, each in the order it was put."""
        return list(self.rows) + [key for key in self._pending if key not in self.rows]

    def missing(self, record_ids: Iterable[str]) -> List[str]:
        """The ids of ``record_ids`` that have no value yet, in their order."""
        return [record_id for record_id in record_ids if record_id not in self]

    def put(self, record_id: str, value: Any) -> None:
        """Keep ``value`` for ``record_id``; a value already kept for it is replaced."""
        if not isinstance(record_id, str) or not record_id:
            raise DerivedStoreError(f"DerivedStore: a record id is a non-empty text, not {record_id!r}.")
        _encode(value)  # an unstorable value is refused where it is put, not at a later flush
        self._pending[record_id] = value
        if len(self._pending) >= FLUSH_EVERY:
            self.flush()

    def get(self, record_id: str, default: Any = _MISSING) -> Any:
        """The value kept for ``record_id``; ``default`` when there is none (a ``KeyError`` without one)."""
        if record_id in self._pending:
            return self._pending[record_id]
        place = self.rows.get(record_id)
        if place is None:
            if default is _MISSING:
                raise KeyError(record_id)
            return default
        metas, arrays = self._shard(place[0])
        prefix = f"{place[1]}."
        own = {name[len(prefix) :]: arrays[name] for name in arrays.files if name.startswith(prefix)}
        return _decode(metas[place[1]], own)

    def column(self, record_ids: Sequence[str]) -> List[Any]:
        """The values of ``record_ids``, in their order. Every id must have one (ask :meth:`missing` first)."""
        return [self.get(record_id) for record_id in record_ids]

    def array(self, record_ids: Sequence[str]) -> np.ndarray:
        """The values of ``record_ids`` stacked into one array, row ``i`` the value of ``record_ids[i]``."""
        return np.stack([np.asarray(value) for value in self.column(record_ids)])

    def _shard(self, number: int) -> Tuple[List[Dict[str, Any]], Any]:
        held = self._shards.get(number)
        if held is None:
            arrays = np.load(self.folder / f"{number:06d}.npz", allow_pickle=False)
            held = (json.loads(str(arrays["meta"])), arrays)
            self._shards[number] = held
        return held

    def flush(self) -> None:
        """Write the values held in memory as one new shard and the index naming them."""
        if not self._pending:
            return
        rows = self.rows
        self.folder.mkdir(parents=True, exist_ok=True)
        number = 1 + max((place[0] for place in rows.values()), default=-1)
        metas: List[Dict[str, Any]] = []
        arrays: Dict[str, np.ndarray] = {}
        for row, value in enumerate(self._pending.values()):
            meta, own = _encode(value)
            metas.append(meta)
            arrays.update({f"{row}.{name}": array for name, array in own.items()})
        shard = self.folder / f"{number:06d}.npz"
        partial = self.folder / f"{number:06d}.partial.npz"
        np.savez(partial, meta=np.array(json.dumps(metas)), **arrays)  # type: ignore[arg-type]
        partial.replace(shard)
        for row, record_id in enumerate(self._pending):
            rows[record_id] = [number, row]
        index = self.folder / _INDEX
        partial_index = self.folder / (_INDEX + ".partial")
        partial_index.write_text(json.dumps({"format": DERIVED_FORMAT, "rows": rows}))
        partial_index.replace(index)
        self._pending = {}

    def close(self) -> None:
        """Flush, and let go of the shards read."""
        self.flush()
        for _metas, arrays in self._shards.values():
            arrays.close()
        self._shards = {}


@configurable
class DerivedStore(Storage):
    """Values computed from records, kept under ``root`` by producer, output name and record id.

    One root serves any number of datasets and producers: a record id names its dataset, and a producer is a
    hash of what computed the value (:func:`producer_hash`).

    YAML::

        store: !class:recordstream.storage.derived.DerivedStore
          root: $RUN_DIR/derived

    Args:
        root: The folder everything is kept under. ``~`` and ``$VAR`` are expanded; created on first write.
    """

    def __init__(self, root: str = "") -> None:
        # Lazy / zero-arg: store config only; the folder is read and made on first use.
        self.root = str(root)
        self._tables: Dict[Tuple[str, str], DerivedTable] = {}

    @property
    def folder(self) -> Path:
        """``root`` as a path, ``~`` and ``$VAR`` expanded; refused when empty."""
        if not self.root:
            raise DerivedStoreError("DerivedStore: `root` is empty — name the folder the values are kept under.")
        return Path(os.path.expanduser(os.path.expandvars(self.root)))

    def table(self, producer: str, name: str) -> DerivedTable:
        """The table of output ``name`` of ``producer`` (the same object for the same two names)."""
        key = (_checked_name(producer, "producer"), _checked_name(name, "output"))
        if key not in self._tables:
            self._tables[key] = DerivedTable(self.folder / producer / name)
        return self._tables[key]

    def describe(self, producer: str, text: str) -> None:
        """Write what ``producer`` is beside its tables, for a person reading the folder. Written once."""
        file = self.folder / _checked_name(producer, "producer") / _PRODUCER_FILE
        if not file.is_file():
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)

    def producers(self) -> List[Dict[str, Any]]:
        """Every producer kept here: ``{producer, outputs: {name: records}, bytes, description}``, by name."""
        folder = self.folder
        if not folder.is_dir():
            return []
        found: List[Dict[str, Any]] = []
        for entry in sorted(path for path in folder.iterdir() if path.is_dir() and not path.name.startswith(".")):
            outputs = {
                table.name: len(self.table(entry.name, table.name))
                for table in sorted(entry.iterdir())
                if table.is_dir() and (table / _INDEX).is_file()
            }
            description = entry / _PRODUCER_FILE
            found.append(
                {
                    "producer": entry.name,
                    "outputs": outputs,
                    "bytes": sum(file.stat().st_size for file in entry.rglob("*") if file.is_file()),
                    "description": description.read_text() if description.is_file() else "",
                }
            )
        return found

    def remove(self, producer: str) -> None:
        """Delete everything kept for ``producer``. Its values are computed again when next asked for."""
        _checked_name(producer, "producer")
        for key in [key for key in self._tables if key[0] == producer]:
            self._tables.pop(key).close()
        target = self.folder / producer
        if target.is_dir():
            shutil.rmtree(target)
            logger.info(f"DerivedStore: removed producer {producer} from {self.folder}")

    def flush(self) -> None:
        """Write every table's values held in memory."""
        for table in self._tables.values():
            table.flush()

    def close(self) -> None:
        """Flush and close every table."""
        for table in self._tables.values():
            table.close()
        self._tables = {}


__all__ = [
    "DERIVED_FORMAT",
    "FLUSH_EVERY",
    "DerivedStore",
    "DerivedStoreError",
    "DerivedTable",
    "producer_hash",
    "producer_text",
]
