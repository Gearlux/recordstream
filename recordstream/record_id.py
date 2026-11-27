"""Record identity — an id a record keeps wherever it is read from.

A record is found by its place in a source, and a place says nothing once the source is split, sliced or
filtered: record 3 of a validation view is not record 3 of the dataset. Anything kept ABOUT a record — a
prediction, an embedding, a score — needs a handle that names the record itself. This module is that handle,
with the same shape as :mod:`recordstream.uri`: a ``Protocol`` (never a base class) and free functions that
walk a source's wrappers.

An id has two parts, ``<dataset>#<key>``:

* **the dataset** — the source's :func:`~recordstream.uri.dataset_uri`; else, for a source that reads a folder
  (``root``), that folder as a ``file://`` URI; else its ``name``; else nothing;
* **the key** — the source's own name for the record (:class:`SupportsRecordKeys`: a file's path under its
  root), or the record's place in the ROOT source when the source has no name for it.

A view (a split, a range, a difference, a filter) keeps the ids of the source it reads: the walk follows
``.source`` through the view's ``indices`` (or ``matches``), so the same record has the same id in every view.
A concatenation gives each member's ids, end to end.

A record made FROM another (one box of an image, one burst of a capture) is named by its parent's id and a
region, :func:`derived_id`, and carries its parent's id beside its own (:func:`derive`).

Design notes
------------
* **Asking loads no record.** ``record_keys()`` reads a listing or stored settings, never a payload — the
  one-listing answer, as ``FilesSource.paths()`` is.
* **A place is a key of last resort, and says so.** :func:`has_record_keys` tells a consumer whether the ids
  are the source's own names or places. An id that is a place holds while the source's records and their order
  do not change, and no longer.
* **Ops change what the walk can know.** A stream WITH ops may drop or add records, so its records are counted
  as its own; put :class:`~recordstream.sources.IdentifiedSource` under the ops and the id rides in the record.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from recordstream.items import Record

#: The entry a record's id is carried under.
RECORD_ID = "record_id"
#: The entry a derived record carries its parent's id under.
PARENT_ID = "parent_id"

#: How many wrappers the walk follows before giving up (a cyclic ``source`` must not hang it).
MAX_WRAPPER_DEPTH = 16


@runtime_checkable
class SupportsRecordKeys(Protocol):
    """A source that has its own name for each record.

    ``record_keys()`` answers one key per record, in record order, from ONE cheap listing: it reads no
    payload. Keys are unique within the source and do not change when records are added or removed around
    them (a file's path under the source's root does; its place in the listing does not).
    """

    def record_keys(self) -> Sequence[str]:
        """Every record's key, in record order."""
        ...


def _materialize(node: Any) -> Any:
    """Build ``node`` when it is a deferred config marker; hand a live object back untouched."""
    from confluid import Fluid, flow

    return flow(node) if isinstance(node, Fluid) else node


def _dataset_part(node: Any) -> str:
    """What names ``node``'s dataset in an id: its ``dataset_uri``, else the folder it reads (``root``), else its
    ``name``, else nothing. The folder comes before the name: two sources left at one default name read two
    datasets, and their records must not share ids."""
    uri = getattr(node, "dataset_uri", None)
    if isinstance(uri, str) and uri:
        return uri
    root = getattr(node, "root", None)
    if isinstance(root, str) and root:
        return Path(os.path.expanduser(os.path.expandvars(root))).resolve().as_uri()
    name = getattr(node, "name", None)
    return name if isinstance(name, str) else ""


def _resolve(node: Any, depth: int = 0) -> Tuple[List[str], bool]:
    """``node``'s record ids, and whether every one is the source's own name for its record."""
    node = _materialize(node)
    if node is None:
        raise TypeError("record_ids: there is no source to read ids from.")
    if depth > MAX_WRAPPER_DEPTH:
        raise TypeError(f"record_ids: more than {MAX_WRAPPER_DEPTH} wrapped sources — is one its own source?")

    own = getattr(node, "record_ids", None)
    if callable(own):
        ids = [str(value) for value in own()]
        return ids, bool(getattr(node, "record_ids_are_keys", True))

    wrapped = getattr(node, "source", None)
    if wrapped is not None and wrapped is not node:
        picked = getattr(node, "indices", None)
        if picked is None:
            picked = getattr(node, "matches", None)
        if picked is not None:
            parent, keyed = _resolve(wrapped, depth + 1)
            return [parent[index] for index in picked], keyed
        ops = getattr(node, "ops", None)
        if isinstance(ops, (list, tuple)) and not ops:
            return _resolve(wrapped, depth + 1)

    members = getattr(node, "sources", None)
    if isinstance(members, (list, tuple)) and members:
        ids = []
        keyed = True
        for member in members:
            member_ids, member_keyed = _resolve(member, depth + 1)
            ids.extend(member_ids)
            keyed = keyed and member_keyed
        return ids, keyed

    prefix = _dataset_part(node) + "#"
    if isinstance(node, SupportsRecordKeys):
        return [prefix + str(key) for key in node.record_keys()], True
    if not hasattr(node, "__len__"):
        raise TypeError(
            f"record_ids: {type(node).__name__} has no record keys and no length, so its records cannot be "
            "named. Give it `record_keys()`, or read it through an indexable source."
        )
    return [f"{prefix}{index}" for index in range(len(node))], False


def record_ids(source: Any) -> List[str]:
    """Every record's id, in ``source``'s record order.

    The same record has the same id through any view of its source. A deferred source (a ``!class:`` marker)
    is built first, as :func:`~recordstream.projection.project` does.

    Args:
        source: Any indexable source, view or stream.

    Example::

        files = FilesSource(root="/data/captures", pattern="**/*.png")
        record_ids(files)                        # ['file:///data/captures#a/1.png', 'file:///data/captures#a/2.png', …]
        record_ids(RangeSource(files, start=1))  # ['file:///data/captures#a/2.png', …] — the same ids
    """
    return _resolve(source)[0]


def has_record_keys(source: Any) -> bool:
    """Whether ``source``'s ids are its own names for its records, not their places.

    ``False`` means an id holds only while the source's records and their order stay as they are.
    """
    return _resolve(source)[1]


def derived_id(parent: str, region: Any) -> str:
    """The id of the record made from region ``region`` of the record ``parent``.

    Args:
        parent: The id of the record it is made from.
        region: What names the part within its parent — a box's place, a region's own id.
    """
    return f"{parent}@{region}"


def derive(record: Record, region: Any, entries: Optional[Record] = None) -> Record:
    """A new record made from region ``region`` of ``record``: ``entries`` plus its id and its parent's.

    The parent must carry its id (read it through :class:`~recordstream.sources.IdentifiedSource`).

    Args:
        record: The record the new one is made from.
        region: What names the part within its parent.
        entries: The new record's own entries.
    """
    parent = record.get(RECORD_ID)
    if not isinstance(parent, str) or not parent:
        raise KeyError(
            f"derive: the record carries no {RECORD_ID!r} entry, so what is made from it cannot be named. "
            "Read the dataset through IdentifiedSource."
        )
    child: Record = dict(entries or {})
    child[RECORD_ID] = derived_id(parent, region)
    child[PARENT_ID] = parent
    return child


__all__ = [
    "MAX_WRAPPER_DEPTH",
    "PARENT_ID",
    "RECORD_ID",
    "SupportsRecordKeys",
    "derive",
    "derived_id",
    "has_record_keys",
    "record_ids",
]
