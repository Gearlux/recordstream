"""Structure ops — reshape a record dict's entries.

Plumbing over the plain-dict carrier: RENAME, COPY, DROP, or SELECT named entries. Each op
is a thin copy-on-write dict expression — no payload is touched.

All ops are lazy / zero-arg constructible (config validated in ``__call__``) and
``@configurable(category="op", group="structure")`` so they surface as canvas nodes.
"""

import random
from typing import Any, List, Optional

from confluid import configurable, output

from recordstream.items import Label, Record

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


@configurable(category="value", group="structure", random=True)
class RandomNumber:
    """A uniformly random number — a producer, with no record anywhere near it.

    The stand-in for any number a graph needs before the real source of it exists — the
    canonical use is a confidence score fed into :class:`PutField`. Wire the NODE itself into
    the consumer for a fresh draw per record; the ``value`` output is a single draw, and a
    visual editor folds it into the document as one FROZEN literal (documented there), which
    is almost never what a per-record score means.

    Args:
        low: Lower bound of the draw.
        high: Upper bound of the draw.
        seed: Set for a reproducible sequence — one generator per instance, so the same seed
            yields the same draws in order. ``None`` = fresh randomness.
    """

    def __init__(self, low: float = 0.0, high: float = 1.0, seed: Optional[int] = None) -> None:
        self.low = float(low)
        self.high = float(high)
        self.seed = seed
        self._rng: Optional[random.Random] = None  # per-instance, first-use (zero-arg rule)

    def __call__(self) -> float:
        """One fresh draw."""
        if self._rng is None:
            self._rng = random.Random(self.seed)
        return self._rng.uniform(self.low, self.high)

    @property
    @output
    def value(self) -> float:
        """A single draw, as a wireable output. Folds to a frozen literal at export."""
        return self()


@configurable(category="op", group="structure")
class PutField:
    """Put a value into the record under ``key`` — record + value + name, nothing else.

    The one generic way to stamp a value into a record, whatever produces it: a
    :class:`RandomNumber`, a model's confidence, a constant. A CALLABLE value is called PER
    RECORD (that is what makes a wired producer draw fresh values for every record instead of
    one frozen number for the whole dataset); anything else is stored as given.

    ``key`` uses dict semantics — an existing entry is replaced. Put means put: a model
    re-stamping its own entry on a second pass must not fail, and protecting the ground truth
    is the prediction convention's business, not this op's.

    Args:
        key: The record entry to write.
        value: What to store — a plain value, or a callable drawn per record.
    """

    def __init__(self, key: str = "", value: Optional[Any] = None) -> None:
        self.key = key
        # `Any` because the whole point is accepting whatever produces the value -- a number,
        # a producer node, a model output. Naming a type would exclude the next source.
        self.value = value

    def __call__(self, record: Record) -> Record:
        if not self.key:
            raise ValueError("PutField: 'key' (the record entry to write) is required")
        if self.value is None:
            raise ValueError("PutField: 'value' is required — wire a producer or set a number")
        drawn = self.value() if callable(self.value) else self.value
        return {**record, self.key: Label(drawn)}


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
