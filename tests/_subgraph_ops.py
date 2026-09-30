"""Test ops for the subgraph suites (``test_subgraph.py``, ``test_trace_subgraph.py``) and the demo image.

Every class name starts with ``Sg`` so it clashes with nothing in the confluid registry (a bare
``!class:Name`` resolves by registry name, and a namesake elsewhere would make it ambiguous). The
YAML in the suites spells the full path, ``!class:tests._subgraph_ops.SgRaiseBright``.

The ops declare their record entries BY NAME, so ``Tracer.check`` can reason about them — the image
ops of recordstream declare TYPES (or nothing), which turns a verdict into ``unverifiable``.
"""

from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
from confluid import configurable

from recordstream.algorithm import Algorithm, Input, Output
from recordstream.items import Image, Mask, Record

#: The two rectangles of the demo image, as ``(x0, y0, x1, y1)`` half-open pixel boxes.
DEMO_BOXES = [(15, 20, 61, 71), (90, 50, 146, 106)]
#: grey -> scale to 0..1 -> threshold at 0.5 keeps both rectangles: (51*46 + 56*56) / (120*160).
DEMO_MASK_MEAN = 0.2855


def demo_pixels() -> np.ndarray:
    """The 120x160 RGB demo image: a dark background and two light rectangles.

    Background (20, 20, 30) is grey level 21 — 0.08 after scaling, below a 0.5 threshold; the
    rectangles (230 and 200) scale to 0.90 and 0.78, above it.
    """
    pixels = np.empty((120, 160, 3), dtype=np.uint8)
    pixels[...] = (20, 20, 30)
    pixels[20:71, 15:61] = 230
    pixels[50:106, 90:146] = 200
    return pixels


def demo_record() -> Record:
    return {"image": Image(demo_pixels())}


@configurable(category="op", group="test")
class SgRaiseBright:
    """Raises the flag 'bright' when the image mean is above ``level``.

    Args:
        level: The mean (0..255) above which the image counts as bright.
    """

    flags: Tuple[str, ...] = ("bright",)
    consumes = {"image": "*"}
    produces = {"bright": "*"}

    def __init__(self, level: float = 50.0) -> None:
        self.level = level

    def __call__(self, record: Record) -> Record:
        return {**record, "bright": bool(float(np.asarray(record["image"]).mean()) > self.level)}


@configurable(category="op", group="test")
class SgGated:
    """Runs only when the flag named by ``requires`` is set in the record; writes ``gated_ran``.

    Args:
        requires: The flag that must be raised for this node to run. Blank = ungated.
    """

    consumes: Dict[str, str] = {}
    produces = {"gated_ran": "*"}

    def __init__(self, requires: str = "") -> None:
        self.requires = requires

    def __call__(self, record: Record) -> Record:
        ran = (not self.requires) or bool(record.get(self.requires))
        return {**record, "gated_ran": ran}


@configurable(category="op", group="test")
class SgStamp:
    """Writes ``value`` into the record under ``key`` — the target of a ``bind:``.

    Args:
        key: The record entry to write.
        value: The value written.
    """

    consumes: Dict[str, str] = {}
    produces = {"stamp": "*"}

    def __init__(self, key: str = "stamp", value: float = 0.0) -> None:
        self.key = key
        self.value = value

    def __call__(self, record: Record) -> Record:
        return {**record, self.key: self.value}


@configurable(category="op", group="test")
class SgSplit:
    """A 1->N EXPANDING op: yields ``parts`` records tagged ``part`` 0, 1, …

    Args:
        parts: How many records to yield.
    """

    EXPANDS = True

    def __init__(self, parts: int = 2) -> None:
        self.parts = parts

    def __call__(self, record: Record) -> List[Record]:
        return [{**record, "part": i} for i in range(self.parts)]


@configurable(category="op", group="test")
class SgHasResult:
    """Has a parameter named like the subgraph's own ``result`` — the broadcast probe.

    Args:
        result: Any string.
    """

    def __init__(self, result: str = "") -> None:
        self.result = result

    def __call__(self, record: Record) -> Record:
        return record


@configurable(category="op", group="test")
class SgCount:
    """A pass-through that counts its calls per ``label`` — proves which nodes ran again.

    Args:
        label: The counter this node increments.
    """

    counts: ClassVar[Dict[str, int]] = {}

    def __init__(self, label: str = "count") -> None:
        self.label = label

    def __call__(self, record: Record) -> Record:
        type(self).counts[self.label] = type(self).counts.get(self.label, 0) + 1
        return record


@configurable(category="op", group="test")
class SgExplode:
    """Raises at RUN time while ``fail`` is set; drops the record while ``drop`` is set.

    Args:
        fail: Raise when called.
        drop: Return ``None`` (filter the record out) when called.
    """

    def __init__(self, fail: bool = True, drop: bool = False) -> None:
        self.fail = fail
        self.drop = drop

    def __call__(self, record: Record) -> Any:
        if self.fail:
            raise RuntimeError("boom: SgExplode was told to fail")
        if self.drop:
            return None
        return {**record, "survived": True}


@configurable(category="op", group="test")
class SgMaskFraction(Algorithm):
    """The fraction of True pixels in a mask (declares its entries by name)."""

    mask: Mask = Input(doc="The boolean mask to read.")
    fraction: float = Output(doc="Fraction of True pixels.")

    def compute(self) -> Dict[str, Any]:
        return {"fraction": float(np.asarray(self.mask).mean())}


@configurable(category="op", group="test")
class SgDoubled(Algorithm):
    """Twice a number (a second named-declaration op, to chain inside and after a subgraph)."""

    fraction: float = Input(doc="The number to double.")
    doubled: float = Output(doc="Twice the number.")

    def compute(self) -> Dict[str, Any]:
        return {"doubled": 2.0 * float(self.fraction)}


@configurable(category="op", group="test")
class SgMeanLevel(Algorithm):
    """The image mean / 255 as a DECLARED ``@output`` — what ``bind: step.level`` reads."""

    image: Image = Input(doc="The image to measure.")
    level: float = Output(doc="Mean pixel value / 255.")

    def compute(self) -> Dict[str, Any]:
        return {"level": float(np.asarray(self.image).mean()) / 255.0}


@configurable(category="op", group="test")
class SgPut:
    """Writes ``value`` under ``key`` and appends ``key`` to the record's ``path`` — declares what it writes.

    Args:
        key: The record entry to write (also the name appended to ``path``).
        value: The value written.
    """

    consumes: Dict[str, str] = {}

    def __init__(self, key: str = "put", value: Any = 0.0) -> None:
        self.key = key
        self.value = value

    @property
    def produces(self) -> Dict[str, str]:
        return {self.key: "*", "path": "*"}

    def __call__(self, record: Record) -> Record:
        return {**record, self.key: self.value, "path": (*record.get("path", ()), self.key)}


@configurable(category="op", group="test")
class SgBump:
    """Edits the record it was given IN PLACE: ``record[key] += 1`` (a missing key counts as 0).

    Args:
        key: The record entry to bump.
    """

    def __init__(self, key: str = "k") -> None:
        self.key = key

    def __call__(self, record: Record) -> Record:
        record[self.key] = record.get(self.key, 0) + 1
        return record
