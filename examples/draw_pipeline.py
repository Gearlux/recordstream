"""A training set of valid examples: a generator's settings drawn anew for every record.

``Traffic`` is a small generator whose settings interlock (lanes must fit the road and avoid its closed position).
``DrawSource`` draws them per record; every draw picks only among the values the generator accepts, so every record
is a valid example, and each carries its settings file. Run: ``python examples/draw_pipeline.py``.
"""

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import confluid
from confluid import configurable

from recordstream import Algorithm, Output, Param
from recordstream.draws import Choice, DrawRefused, Repeat, Uniform, draw_settings
from recordstream.sources import DrawSource


@configurable
@dataclass(kw_only=True)
class Lane:
    position: Literal[0, 1, 2, 3, 4, 5, 6, 7] = 0


@configurable
@dataclass(kw_only=True)
class Road:
    width: Literal[2, 4, 8] = 8
    closed: Optional[Literal[0, 1]] = None  # a closed lane, or None
    lanes: Optional[List[Lane]] = None

    def __post_init__(self) -> None:
        taken = [lane.position for lane in self.lanes or ()]
        if len(set(taken)) != len(taken):
            raise ValueError("Road: two lanes on one position")
        if any(p >= self.width or p == self.closed for p in taken):
            raise ValueError("Road: a lane off the road or on the closed position")


@configurable
class Traffic(Algorithm):
    road: Road = Param(default=Road(), doc="The road.")
    speed: float = Param(default=50.0, doc="The speed in km/h.")
    load: int = Output(doc="Lanes in use.")

    def compute(self) -> Dict[str, int]:
        return {"load": len(self.road.lanes or [])}


def main() -> None:
    source = DrawSource(
        generator=Traffic(),
        count=1000,
        seed=7,
        draws=[
            Choice(field="road.width"),
            Choice(field="road.closed", values=[None, 0], weights=[3, 1]),
            Repeat(field="road.lanes", count=(0, 8), each=[Choice(field="position")]),
            Uniform(field="speed", low=30, high=120),
        ],
    )
    first = source[0]
    print(f"record 0: load {first['load']}, drawn {source.draw(0).log}")
    print(first["settings"])

    drawn = [(confluid.load(record["settings"]).road.width, record["load"]) for record in source]
    widths = Counter(width for width, _ in drawn)
    most = {w: max(load for width, load in drawn if width == w) for w in (2, 4, 8)}
    print(f"widths over {len(source)} records: {dict(sorted(widths.items()))}; most lanes per width: {most}")

    try:
        draw_settings(
            Traffic(road=Road(lanes=[Lane(position=6)])), [Choice(field="road.width", values=[2, 4])], random.Random(0)
        )
    except DrawRefused as refusal:
        print(f"refused: {refusal}")


if __name__ == "__main__":
    main()
