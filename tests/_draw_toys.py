"""A toy generator for the draw tests, with rules shaped like a real one's.

``ToyPlan`` is the settings object: closed sets (``width``, ``level``), an optional sub-object (``mode``) with a
setting that inherits from its parent when ``None`` (``depth`` — the shape of a TDD cell's control region in
subframes 1 and 6), and a list of elements (``slots``) that must fit together (distinct positions, cells inside the
width, positions the mode closes left free). ``ToyGenerator`` is the algorithm a source runs; it adds a rule of its
own that only its ``check()`` knows.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

from annotated_types import Interval
from confluid import configurable
from typing_extensions import Annotated

from recordstream import Algorithm, Input, Output, Param

Position = Annotated[int, Interval(ge=0, le=9)]
CellIndex = Annotated[int, Interval(ge=0, le=19)]
Share = Annotated[float, Interval(ge=0.0, le=1.0)]


@configurable
@dataclass(kw_only=True)
class ToyMode:
    """Closes every position p with p % 3 == pattern; ``depth`` None = the plan's level."""

    pattern: Literal[0, 1, 2] = 0
    depth: Optional[Literal[1, 2]] = None


@configurable
@dataclass(kw_only=True)
class ToySlot:
    position: Position = 0
    kind: Literal["a", "b"] = "a"
    cells: Optional[List[CellIndex]] = None
    weight: Share = 0.5

    def __post_init__(self) -> None:
        if self.cells is not None and (not self.cells or len(set(self.cells)) != len(self.cells)):
            raise ValueError(f"ToySlot: cells {self.cells} must be distinct and not empty")


@configurable
@dataclass(kw_only=True)
class ToyPlan:
    width: Literal[4, 8, 16] = 8
    level: Literal[1, 2, 3] = 2
    mode: Optional[ToyMode] = None
    slots: Optional[List[ToySlot]] = None

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        if self.level == 3 and self.width < 8:
            raise ValueError(f"ToyPlan: level 3 needs a width of 8 or more — width is {self.width}")
        if self.mode is not None:
            depth = self.mode.depth or self.level
            if depth > 2:
                raise ValueError(f"ToyPlan: a mode takes a depth of 1 or 2 — it inherits level {self.level}")
        seen = set()
        for slot in self.slots or ():
            if slot.position in seen:
                raise ValueError(f"ToyPlan: position {slot.position} is taken twice")
            seen.add(slot.position)
            if self.mode is not None and slot.position % 3 == self.mode.pattern:
                raise ValueError(f"ToyPlan: position {slot.position} is closed by pattern {self.mode.pattern}")
            if slot.kind == "b" and self.level < 2:
                raise ValueError("ToyPlan: a slot of kind b needs level 2 or more")
            for cell in slot.cells or ():
                if cell >= self.width:
                    raise ValueError(f"ToyPlan: cell {cell} is outside the width {self.width}")


@configurable
class ToyGenerator(Algorithm):
    """Turns a plan into one number; refuses a gain above 0.9 at level 3 — but only in ``check()``."""

    plan: ToyPlan = Param(default=ToyPlan(), doc="The plan.")
    gain: Share = Param(default=0.5, doc="A gain.")
    repeats: Annotated[int, Interval(ge=0, le=100)] = Param(default=1, doc="A count.")
    total: float = Output(doc="gain × width + slots.")

    def check(self) -> None:
        self.plan.check()
        if self.gain > 0.9 and self.plan.level == 3:
            raise ValueError(f"ToyGenerator: gain {self.gain:g} above 0.9 needs level 1 or 2")

    def compute(self) -> Dict[str, float]:
        self.check()
        return {"total": self.gain * self.plan.width + len(self.plan.slots or [])}


@configurable
class ToyNeedsInput(Algorithm):
    """A generator that cannot run on an empty record."""

    scale: float = Param(default=1.0, doc="A scale.")
    x: float = Input(doc="The input.")
    y: float = Output(doc="scale × x.")

    def compute(self) -> Dict[str, float]:
        return {"y": self.scale * self.x}


@configurable
@dataclass(kw_only=True)
class ToyStrip:
    """A list of bounded cells with two refusals of its own: ``closed`` takes no cells, ``limit`` caps their count."""

    cells: Optional[List[CellIndex]] = None
    closed: bool = False
    limit: Annotated[int, Interval(ge=1, le=20)] = 20
    weight: float = 0.5
    tags: Optional[List[int]] = None
    label: Union[int, str] = 0

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        if self.cells is None:
            return
        if self.closed:
            raise ValueError("ToyStrip: closed — it takes no cells")
        if len(self.cells) > self.limit:
            raise ValueError(f"ToyStrip: {len(self.cells)} cells, over the limit {self.limit}")


@configurable
@dataclass(kw_only=True)
class ToyNeedsSize:
    """An element that cannot be built from defaults alone."""

    size: int


@configurable
@dataclass(kw_only=True)
class ToyHolder:
    mode: Optional[ToyMode] = None


@configurable
@dataclass(kw_only=True)
class ToyShelf:
    holders: Optional[List[ToyHolder]] = None
    parts: Optional[List[ToyNeedsSize]] = None


@configurable
@dataclass(kw_only=True)
class ToyFloor:
    """A list whose elements hold lists of their own (plans, each with its slots), and a list whose elements may be
    None (a mode per room, or none)."""

    plans: Optional[List[ToyPlan]] = None
    modes: Optional[List[Optional[ToyMode]]] = None


@configurable
class ToyForgetful:
    """Stores its setting under another name, so nothing can read it back."""

    def __init__(self, size: int = 0) -> None:
        self._size = size


@configurable
class ToyWritesSettings(Algorithm):
    """A generator whose output takes the entry a DrawSource writes the settings file to."""

    scale: float = Param(default=1.0, doc="A scale.")
    settings: str = Output(doc="Its own settings entry.")

    def compute(self) -> Dict[str, str]:
        return {"settings": "mine"}
