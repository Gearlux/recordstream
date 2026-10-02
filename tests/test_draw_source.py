"""``DrawSource`` — one record per draw: the drawn generator's outputs and its settings file."""

from typing import List

import confluid
import pytest

from recordstream.core.stream import Stream
from recordstream.draws import Choice, Draw, Repeat, Uniform
from recordstream.sources import DrawSource
from recordstream.sources.draw import SETTINGS_ENTRY
from tests._draw_toys import ToyGenerator, ToyNeedsInput, ToyWritesSettings

SPEC: List[Draw] = [
    Choice(field="plan.width"),
    Choice(field="plan.level"),
    Repeat(field="plan.slots", count=(0, 4), each=[Choice(field="position")]),
    Uniform(field="gain"),
]


def _source(count: int = 5, seed: int = 3) -> DrawSource:
    return DrawSource(generator=ToyGenerator(), draws=SPEC, count=count, seed=seed)


def test_a_record_holds_the_generators_outputs_and_its_settings_file() -> None:
    record = _source()[0]
    assert set(record) == {"total", SETTINGS_ENTRY}
    rebuilt = confluid.load(record[SETTINGS_ENTRY])
    assert isinstance(rebuilt, ToyGenerator)
    assert rebuilt.run()["total"] == record["total"]
    assert confluid.dump(rebuilt) == record[SETTINGS_ENTRY]


def test_the_settings_file_is_the_drawn_generator() -> None:
    source = _source()
    drawn = source.draw(2).settings
    assert source[2][SETTINGS_ENTRY] == confluid.dump(drawn)


def test_it_has_count_records_and_iterates_them_in_order() -> None:
    source = _source(count=4)
    assert len(source) == 4
    assert [r[SETTINGS_ENTRY] for r in source] == [source[i][SETTINGS_ENTRY] for i in range(4)]


def test_a_record_depends_only_on_the_seed_and_its_index() -> None:
    """Record 3 alone, record 3 after the others, and record 3 of a longer source are the same record — so any
    worker split, and any single example, reproduces."""
    alone = _source(count=5)[3]
    in_order = list(_source(count=5))[3]
    longer = _source(count=50)[3]
    assert alone == in_order == longer
    assert _source(seed=4)[3] != alone


def test_records_differ_from_index_to_index() -> None:
    settings = {r[SETTINGS_ENTRY] for r in _source(count=20)}
    assert len(settings) > 15


def test_an_index_outside_the_count_is_an_index_error() -> None:
    with pytest.raises(IndexError, match=r"DrawSource: index 5 — the source draws 5 records \(0 to 4\)"):
        _source(count=5)[5]


def test_the_outputs_follow_the_generators_keys() -> None:
    source = DrawSource(generator=ToyGenerator(keys={"total": "value"}), draws=SPEC, count=1)
    assert set(source[0]) == {"value", SETTINGS_ENTRY}


def test_it_feeds_a_stream() -> None:
    stream = Stream(source=_source(count=3), ops=[])
    assert [r["total"] for r in stream] == [r["total"] for r in _source(count=3)]


def test_it_is_built_with_no_arguments_and_refuses_a_missing_generator_when_used() -> None:
    source = DrawSource()
    assert len(source) == 1
    with pytest.raises(ValueError, match=r"DrawSource: generator is not set — give the generator to draw"):
        source[0]


def test_a_generator_that_needs_an_input_is_refused() -> None:
    source = DrawSource(generator=ToyNeedsInput(), count=1)
    with pytest.raises(ValueError, match=r"DrawSource: ToyNeedsInput needs the input\(s\) x — a drawn generator runs"):
        source[0]


def test_no_draws_repeats_the_template() -> None:
    source = DrawSource(generator=ToyGenerator(), count=3)
    assert len({r[SETTINGS_ENTRY] for r in source}) == 1


def test_it_is_a_source_node() -> None:
    assert getattr(DrawSource, "__confluid_category__", None) == "source"


def test_a_generator_writing_its_own_settings_entry_is_refused() -> None:
    with pytest.raises(ValueError, match=r"DrawSource: ToyWritesSettings writes an entry named 'settings'"):
        DrawSource(generator=ToyWritesSettings())[0]
