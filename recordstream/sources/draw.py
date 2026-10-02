"""``DrawSource`` — one record per draw of a generator's settings: its outputs and its settings file."""

import random
from typing import Iterator, List, Optional

import confluid
from annotated_types import Interval
from confluid import configurable
from loggair import get_logger
from typing_extensions import Annotated

from recordstream.algorithm import Algorithm, algorithm_spec
from recordstream.draws import Draw, Drawn, draw_settings
from recordstream.items import Record

logger = get_logger(__name__)

#: The record entry holding the drawn generator's settings file (``confluid.dump`` of it).
SETTINGS_ENTRY = "settings"

RecordCount = Annotated[int, Interval(ge=0)]


@configurable(category="source")
class DrawSource:
    """Records from a generator whose settings are drawn anew for every record — a training set of valid examples.

    Record ``i`` runs :func:`~recordstream.draws.draw_settings` over ``generator`` with a random state seeded by
    ``(seed, i)`` alone, then runs the drawn generator on an empty record: the record holds the generator's outputs
    (under its own ``keys``) and, under ``settings``, the drawn generator's settings file — ``confluid.load`` of it
    gives the generator back, so any one example can be rebuilt and inspected on its own. Because a record depends
    on nothing but the seed, its index and the draws, any index range, worker split or machine produces the same
    records. Every draw picks only among the values the generator accepts given the draws above it, so every
    record is a valid example; the draws are described in :mod:`recordstream.draws`.

    Args:
        generator: The generator to draw: an Algorithm with no record inputs; its settings are where every draw starts.
        draws: The draws, run top to bottom (Choice, Uniform, Span, Repeat). None = every record is the generator.
        count: How many records the source draws.
        seed: The seed every record's random state is derived from, with the record's index.
    """

    def __init__(
        self,
        generator: Optional[Algorithm] = None,
        draws: Optional[List[Draw]] = None,
        count: RecordCount = 1,
        seed: int = 0,
    ) -> None:
        # Lazy / zero-arg: store config only; the generator is checked on first use.
        self.generator = generator
        self.draws = draws
        self.count = count
        self.seed = seed

    def _template(self) -> Algorithm:
        generator = self.generator
        if generator is None:
            raise ValueError("DrawSource: generator is not set — give the generator to draw (an Algorithm)")
        needs = [slot.name for slot in algorithm_spec(generator).inputs if slot.required]
        if needs:
            raise ValueError(
                f"DrawSource: {type(generator).__name__} needs the input(s) {', '.join(needs)} — a drawn generator "
                "runs on an empty record, so it must take none"
            )
        return generator

    def draw(self, index: int) -> Drawn:
        """The drawn generator of record ``index`` and what was drawn, without running it."""
        return draw_settings(self._template(), self.draws, random.Random(f"{self.seed}/{index}"))

    def __len__(self) -> int:
        return int(self.count)

    def __getitem__(self, index: int) -> Record:
        count = len(self)
        if not 0 <= index < count:
            raise IndexError(f"DrawSource: index {index} — the source draws {count} records (0 to {count - 1})")
        drawn = self.draw(index)
        logger.debug("DrawSource: record %d drew %s", index, drawn.log)
        record: Record = drawn.settings({})
        if SETTINGS_ENTRY in record:
            raise ValueError(
                f"DrawSource: {type(drawn.settings).__name__} writes an entry named {SETTINGS_ENTRY!r}, where the "
                "source puts the settings file — rename that output with the generator's keys"
            )
        record[SETTINGS_ENTRY] = confluid.dump(drawn.settings)
        return record

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]


__all__ = ["DrawSource", "SETTINGS_ENTRY"]
