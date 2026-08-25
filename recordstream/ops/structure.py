"""Structure ops — reshape a record dict's entries.

Plumbing over the plain-dict carrier: RENAME, COPY, DROP, or SELECT named entries. Each op
is a thin copy-on-write dict expression — no payload is touched.

All ops are lazy / zero-arg constructible (config validated in ``__call__``) and
``@configurable(category="op", group="structure")`` so they surface as canvas nodes.
"""

from typing import List, Optional

from confluid import configurable

from recordstream.items import Record

__all__ = ["RenameField", "DropField", "CopyField", "SelectFields"]


@configurable(category="op", group="structure")
class RenameField:
    """Rename a record entry. Renaming onto an existing key replaces it.

    The sanctioned way to avoid a deliberate fan-in collision: rename on the producing branch
    BEFORE the merge, instead of a merge-policy knob. Also the way to route a value into an
    albumentations op's vocabulary (``image`` / ``mask`` / ``bboxes``).

    Args:
        src: The entry to rename.
        dst: The new key.
    """

    def __init__(self, src: str = "", dst: str = "") -> None:
        self.src = src
        self.dst = dst

    def __call__(self, record: Record) -> Record:
        if not self.src or not self.dst:
            raise ValueError("RenameField: both 'src' and 'dst' are required")
        if self.src not in record:
            raise KeyError(f"RenameField: unknown key {self.src!r} (keys: {list(record)})")
        return {(self.dst if k == self.src else k): v for k, v in record.items()}


@configurable(category="op", group="structure")
class DropField:
    """Remove an entry from the record (e.g. free a heavy signal after its spectrogram is derived).

    Args:
        key: The entry to remove. Missing keys raise unless ``missing_ok``.
        missing_ok: Silently pass through when the entry is absent (default False).
    """

    def __init__(self, key: str = "", missing_ok: bool = False) -> None:
        self.key = key
        self.missing_ok = missing_ok

    def __call__(self, record: Record) -> Record:
        if not self.key:
            raise ValueError("DropField: 'key' (the entry to remove) is required")
        if self.key not in record:
            if self.missing_ok:
                return record
            raise KeyError(f"DropField: unknown key {self.key!r} (keys: {list(record)})")
        return {k: v for k, v in record.items() if k != self.key}


@configurable(category="op", group="structure")
class CopyField:
    """Duplicate an entry under a new key (same value object; values are treated as immutable).

    Args:
        src: The entry to copy.
        dst: The key of the copy. An existing ``dst`` is replaced.
    """

    def __init__(self, src: str = "", dst: str = "") -> None:
        self.src = src
        self.dst = dst

    def __call__(self, record: Record) -> Record:
        if not self.src or not self.dst:
            raise ValueError("CopyField: both 'src' and 'dst' are required")
        if self.src not in record:
            raise KeyError(f"CopyField: unknown key {self.src!r} (keys: {list(record)})")
        return {**record, self.dst: record[self.src]}


@configurable(category="op", group="structure")
class SelectFields:
    """Keep ONLY the named entries (order = the given order); everything else is dropped.

    Args:
        keys: The entries to keep. Unknown keys raise (a silent miss hides a typo).
    """

    def __init__(self, keys: Optional[List[str]] = None) -> None:
        self.keys = list(keys) if keys else []

    def __call__(self, record: Record) -> Record:
        if not self.keys:
            raise ValueError("SelectFields: 'keys' (the entries to keep) is required")
        missing = [k for k in self.keys if k not in record]
        if missing:
            raise KeyError(f"SelectFields: unknown keys {missing} (keys: {list(record)})")
        return {k: record[k] for k in self.keys}
