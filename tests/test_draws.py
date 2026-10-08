"""The settings draws: each picks from its distribution, only among the values the generator accepts.

The toy generator (``tests/_draw_toys.py``) has the shapes a real one has — closed sets, an optional sub-object whose
setting inherits from its parent, a list of elements that must fit together, and a rule only ``check()`` knows.
"""

import random
from collections import Counter
from typing import Any, List

import confluid
import pytest
from confluid import to_pydantic

from recordstream.draws import (
    Choice,
    Draw,
    DrawRefused,
    DrawSpecError,
    Grid,
    Repeat,
    Span,
    Uniform,
    draw_settings,
    parse_quantity,
)
from recordstream.sources import DrawSource
from tests._draw_toys import (
    ToyBox,
    ToyCheckedTwice,
    ToyFloor,
    ToyForgetful,
    ToyGenerator,
    ToyMode,
    ToyPlan,
    ToySelfChecked,
    ToyShelf,
    ToySlot,
    ToyStrip,
)
from tests._draw_toys_postponed import ToyLater


def _draws(spec: List[Any], n: int = 300, template: Any = None) -> List[Any]:
    template = ToyGenerator() if template is None else template
    return [draw_settings(template, spec, random.Random(i)).settings for i in range(n)]


# -- Choice ------------------------------------------------------------------------------------------------------


def test_a_choice_with_no_values_draws_every_value_of_a_closed_setting() -> None:
    drawn = _draws([Choice(field="plan.width"), Choice(field="plan.level")])
    assert {g.plan.width for g in drawn} == {4, 8, 16}
    assert {g.plan.level for g in drawn} == {1, 2, 3}


def test_a_choice_never_draws_a_value_the_generator_refuses() -> None:
    """Level 3 needs a width of 8 or more: drawn after the width, level 3 never meets width 4."""
    drawn = _draws([Choice(field="plan.width"), Choice(field="plan.level")])
    assert not [g for g in drawn if g.plan.level == 3 and g.plan.width == 4]
    assert {g.plan.level for g in drawn if g.plan.width == 4} == {1, 2}


def test_a_choice_follows_its_weights() -> None:
    drawn = _draws([Choice(field="plan.width", values=[4, 16], weights=[1.0, 3.0])], n=2000)
    share = Counter(g.plan.width for g in drawn)[16] / 2000
    assert 0.72 < share < 0.78


def test_a_value_left_out_or_weighted_zero_is_never_drawn() -> None:
    drawn = _draws([Choice(field="plan.width", values=[4, 8, 16], weights=[1.0, 0.0, 1.0])])
    assert {g.plan.width for g in drawn} == {4, 16}


def test_a_choice_over_a_small_integer_range_draws_from_every_integer() -> None:
    drawn = _draws([Choice(field="repeats")], n=400)  # repeats: 0-100
    values = {g.repeats for g in drawn}
    assert values <= set(range(101)) and len(values) > 90


def test_an_optional_setting_draws_none_too() -> None:
    drawn = _draws([Choice(field="plan.mode", values=[None, ToyMode()]), Choice(field="plan.mode.depth")])
    depths = {g.plan.mode.depth for g in drawn if g.plan.mode is not None}
    assert depths == {1, 2, None}


def test_a_draw_inside_an_object_that_is_none_is_skipped() -> None:
    drawn = _draws([Choice(field="plan.mode", values=[None, ToyMode()]), Choice(field="plan.mode.pattern")])
    assert {g.plan.mode is None for g in drawn} == {True, False}
    assert {g.plan.mode.pattern for g in drawn if g.plan.mode is not None} == {0, 1, 2}


def test_the_order_of_the_draws_decides_what_a_later_default_allows() -> None:
    """A mode's depth None inherits the level, and a mode refuses depth 3: with the level drawn while the depth is
    still None, a plan with a mode never gets level 3; with the depth drawn first it does."""
    mode = Choice(field="plan.mode", values=[ToyMode()])
    depth = Choice(field="plan.mode.depth", values=[1, 2])
    level = Choice(field="plan.level")
    level_first = {g.plan.level for g in _draws([mode, level, depth])}
    depth_first = {g.plan.level for g in _draws([mode, depth, level])}
    assert level_first == {1, 2}
    assert depth_first == {1, 2, 3}


def test_a_choice_no_value_of_which_is_accepted_names_the_setting_the_earlier_draws_and_the_reason() -> None:
    spec: List[Draw] = [Choice(field="plan.width", values=[4]), Choice(field="plan.level", values=[3])]
    with pytest.raises(DrawRefused) as caught:
        draw_settings(ToyGenerator(), spec, random.Random(0))
    assert str(caught.value) == (
        "plan.level: none of [3] is accepted after plan.width=4 — "
        "ToyPlan: level 3 needs a width of 8 or more — width is 4"
    )


def test_a_value_outside_the_settings_type_is_refused_like_any_other() -> None:
    drawn = _draws([Choice(field="plan.width", values=[4, 5])], n=50)
    assert {g.plan.width for g in drawn} == {4}


def test_a_rule_only_the_generators_check_knows_is_respected() -> None:
    """ToyGenerator refuses a gain above 0.9 at level 3 in check() alone; its constructor would build it."""
    assert ToyGenerator(plan=ToyPlan(level=3), gain=0.95).gain == 0.95
    drawn = _draws([Choice(field="plan.level", values=[3]), Choice(field="gain", values=[0.5, 0.95])], n=50)
    assert {g.gain for g in drawn} == {0.5}


def test_a_class_whose_constructor_checks_it_is_checked_once_for_each_value_tried() -> None:
    """``checked_on_construction``: the draw builds the value and asks no second ``check()`` — the constructor's is
    the verdict (size 3 refused there). A class that does not say so is checked again after its constructor."""
    box = ToyBox(self_checked=ToySelfChecked(), checked_twice=ToyCheckedTwice())
    ToySelfChecked.checks = ToyCheckedTwice.checks = 0
    spec: List[Draw] = [Choice(field="self_checked.size", values=[3, 2], weights=[1.0, 1e-9])]
    drawn = draw_settings(box, spec, random.Random(0)).settings  # 3 picked first (all but surely), refused, then 2
    assert drawn.self_checked.size == 2 and ToySelfChecked.checks == 2  # 3 built, 2 built: one check each
    spec = [Choice(field="checked_twice.size", values=[3, 2], weights=[1.0, 1e-9])]
    drawn = draw_settings(box, spec, random.Random(0)).settings
    assert drawn.checked_twice.size == 2 and ToyCheckedTwice.checks == 2 + 1  # each built once, 2 checked again


def test_a_class_whose_constructor_checks_it_still_has_its_refusal_named() -> None:
    with pytest.raises(DrawRefused, match=r"self_checked.size: none of \[3\] is accepted .* size 3 is refused"):
        draw_settings(ToyBox(self_checked=ToySelfChecked()), [Choice(field="self_checked.size", values=[3])],
                      random.Random(0))  # fmt: skip


class TestAChoiceChecksOnlyTheValueItPicks:
    """A Choice picks a value by its weight and asks the generator about that one alone; a refused value is dropped and
    another picked among the rest — so each accepted value comes out with its weight's share of the accepted ones, as
    if every value had been tried first, at one check instead of one per value (measured on generated cellular records
    2026-10-08: 74 066 rebuilds → 23 498)."""

    def test_an_accepted_pick_is_the_only_value_checked(self) -> None:
        box = ToyBox(self_checked=ToySelfChecked(), checked_twice=ToyCheckedTwice())
        ToySelfChecked.checks = 0
        draw_settings(box, [Choice(field="self_checked.size", values=[1, 2])], random.Random(4))
        assert ToySelfChecked.checks == 1

    def test_a_refused_pick_is_dropped_and_another_picked(self) -> None:
        box = ToyBox(self_checked=ToySelfChecked(), checked_twice=ToyCheckedTwice())
        sizes, checks = [], []
        for seed in range(200):
            ToySelfChecked.checks = 0
            drawn = draw_settings(box, [Choice(field="self_checked.size", values=[1, 3, 2])], random.Random(seed))
            sizes.append(drawn.settings.self_checked.size)
            checks.append(ToySelfChecked.checks)
        assert set(sizes) == {1, 2}
        assert set(checks) == {1, 2}  # 3 is checked at most once a draw, and only when it was picked

    def test_each_accepted_value_comes_out_with_its_weights_share_of_the_accepted(self) -> None:
        """Weights 1, 2, 3 with the third refused: 1/3 and 2/3 — what picking among the accepted gives."""
        spec: List[Draw] = [Choice(field="self_checked.size", values=[1, 2, 3], weights=[1.0, 2.0, 3.0])]
        box = ToyBox(self_checked=ToySelfChecked(), checked_twice=ToyCheckedTwice())
        counts = Counter(draw_settings(box, spec, random.Random(n)).settings.self_checked.size for n in range(6000))
        assert counts[3] == 0
        assert abs(counts[1] / 6000 - 1 / 3) < 0.02 and abs(counts[2] / 6000 - 2 / 3) < 0.02

    def test_a_weight_of_zero_is_never_checked(self) -> None:
        box = ToyBox(self_checked=ToySelfChecked(), checked_twice=ToyCheckedTwice())
        ToySelfChecked.checks = 0
        spec: List[Draw] = [Choice(field="self_checked.size", values=[3, 1], weights=[0.0, 1.0])]
        assert draw_settings(box, spec, random.Random(0)).settings.self_checked.size == 1
        assert ToySelfChecked.checks == 1

    def test_when_every_value_is_refused_every_value_was_tried(self) -> None:
        box = ToyBox(self_checked=ToySelfChecked(), checked_twice=ToyCheckedTwice())
        ToySelfChecked.checks = 0
        with pytest.raises(
            DrawRefused, match=r"^self_checked.size: none of \[3, 3\] is accepted .* size 3 is refused$"
        ):
            draw_settings(box, [Choice(field="self_checked.size", values=[3, 3])], random.Random(0))
        assert ToySelfChecked.checks == 2


def test_a_choice_with_no_values_on_an_open_setting_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"gain: Choice needs values — the setting is .*float.*; draw it with"):
        draw_settings(ToyGenerator(), [Choice(field="gain")], random.Random(0))


def test_weights_that_do_not_match_the_values_are_a_spec_error() -> None:
    spec: List[Draw] = [Choice(field="plan.width", values=[4, 8], weights=[1.0])]
    with pytest.raises(DrawSpecError, match=r"plan.width: 2 values and 1 weights"):
        draw_settings(ToyGenerator(), spec, random.Random(0))


# -- Uniform -----------------------------------------------------------------------------------------------------


def test_a_uniform_with_no_bounds_takes_the_settings_own_range() -> None:
    drawn = _draws([Uniform(field="gain"), Uniform(field="repeats")], n=500)
    gains = [g.gain for g in drawn]
    assert 0.0 <= min(gains) < 0.05 and 0.95 < max(gains) <= 1.0
    assert all(isinstance(g.repeats, int) and 0 <= g.repeats <= 100 for g in drawn)


def test_a_uniform_keeps_inside_its_own_bounds_and_skips_refused_values() -> None:
    drawn = _draws([Choice(field="plan.level", values=[3]), Uniform(field="gain", low=0.8, high=1.0)], n=300)
    gains = [g.gain for g in drawn]
    assert 0.8 <= min(gains) and max(gains) <= 0.9


def test_a_uniform_none_of_whose_values_is_accepted_is_refused() -> None:
    spec: List[Draw] = [Choice(field="plan.level", values=[3]), Uniform(field="gain", low=0.95, high=1.0)]
    with pytest.raises(DrawRefused, match=r"gain: 100 values from 0.95 to 1 refused after plan.level=3 — ToyGenerator"):
        draw_settings(ToyGenerator(), spec, random.Random(0))


def test_a_uniform_on_a_setting_without_a_range_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"plan.width: Uniform needs a number setting"):
        draw_settings(ToyGenerator(), [Uniform(field="plan.width")], random.Random(0))


# -- Grid --------------------------------------------------------------------------------------------------------


def _grid_values(draw: Grid, n: int = 400, template: Any = None) -> List[Any]:
    field = draw.field
    return sorted(
        {getattr(g, field) for g in _draws([draw], n=n, template=ToyStrip() if template is None else template)}
    )


def test_a_linear_grid_draws_every_multiple_of_its_step_in_the_range() -> None:
    assert _grid_values(Grid(field="weight", low=0.2, high=0.6, step=0.1)) == [0.2, 0.3, 0.4, 0.5, 0.6]


def test_a_grid_reads_numbers_written_with_their_unit() -> None:
    """``low: 1MHz`` in YAML is the string '1MHz'; the grid reads it as 1e6."""
    drawn = _grid_values(Grid(field="weight", low="1MHz", high="2.5 MSa/s", step="100kHz"))
    assert drawn == [1.0e6 + k * 1.0e5 for k in range(16)]


def test_a_log_grid_spreads_count_points_evenly_on_a_log_scale_each_on_the_step() -> None:
    drawn = _grid_values(Grid(field="weight", low="1MHz", high="2.5MHz", step="100kHz", spacing="log", count=6))
    assert drawn == [1.0e6, 1.2e6, 1.4e6, 1.7e6, 2.1e6, 2.5e6]


@pytest.mark.parametrize(
    "ratio, high, expected",
    [
        (2.0, "8MHz", [1.0e6, 2.0e6, 4.0e6, 8.0e6]),
        (1.25, "2.5MHz", [1.0e6, 1.2e6, 1.6e6, 2.0e6, 2.4e6]),
    ],
)
def test_a_power_grid_steps_by_its_ratio_each_on_the_step(ratio: float, high: str, expected: List[float]) -> None:
    drawn = _grid_values(Grid(field="weight", low="1MHz", high=high, step="100kHz", spacing="power", ratio=ratio))
    assert drawn == expected


def test_a_grid_keeps_its_rounded_points_inside_the_range() -> None:
    """1.05 MHz rounds down to 1.0 MHz on a 100 kHz step, below low; the grid keeps 1.1 MHz instead."""
    drawn = _grid_values(Grid(field="weight", low="1.05MHz", high="1.35MHz", step="100kHz", spacing="log", count=3))
    assert drawn == [1.1e6, 1.2e6, 1.3e6]


def test_a_grid_never_draws_a_value_the_generator_refuses() -> None:
    drawn = _draws([Choice(field="plan.level", values=[3]), Grid(field="gain", low=0.5, high=1.0, step=0.1)], n=300)
    assert sorted({g.gain for g in drawn}) == [0.5, 0.6, 0.7, 0.8, 0.9]


def test_a_grid_takes_the_settings_own_range_when_it_has_no_bounds() -> None:
    assert _grid_values(Grid(field="gain", step=0.25), template=ToyGenerator()) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_a_grid_on_an_integer_setting_draws_integers() -> None:
    drawn = _grid_values(Grid(field="repeats", low=0, high=100, step=25), template=ToyGenerator())
    assert drawn == [0, 25, 50, 75, 100] and all(isinstance(v, int) for v in drawn)


def test_a_fine_grid_is_drawn_without_listing_its_points() -> None:
    """1 Hz steps from 1 to 2.5 MHz are 1.5 million points; a draw picks among them without building the list."""
    drawn = _draws([Grid(field="weight", low="1MHz", high="2.5MHz", step="1Hz")], n=50, template=ToyStrip())
    assert all(1.0e6 <= g.weight <= 2.5e6 and float(g.weight).is_integer() for g in drawn)
    assert len({g.weight for g in drawn}) == 50


def test_a_grid_none_of_whose_points_is_accepted_is_refused() -> None:
    spec: List[Draw] = [Choice(field="plan.level", values=[3]), Grid(field="gain", low=0.95, high=1.0, step=0.05)]
    with pytest.raises(
        DrawRefused,
        match=r"^gain: none of the 2 points from 0\.95 to 1 on a 0\.05 step is accepted after plan\.level=3 — "
        r"ToyGenerator: gain (0\.95|1) above 0\.9 needs level 1 or 2$",
    ):
        draw_settings(ToyGenerator(), spec, random.Random(0))


@pytest.mark.parametrize(
    "draw, message",
    [
        (
            Grid(field="weight", low="1.1MHz", high="1.9MHz", step="1MHz"),
            r"^weight: no multiple of 1e\+06 between 1\.1e\+06 and 1\.9e\+06$",
        ),
        (Grid(field="weight", low=1.0, high=2.0), r"^weight: Grid needs a step$"),
        (Grid(field="weight", low=1.0, high=2.0, step=0), r"^weight: Grid's step 0 is not above 0$"),
        (Grid(field="weight", low=2.0, high=1.0, step=0.1), r"^weight: Grid's low 2 is above its high 1$"),
        (Grid(field="weight", step=0.1), r"^weight: Grid needs low and high — the setting \(float\) has none$"),
        (Grid(field="weight", low=1, high=2, step=0.1, spacing="log"), r"^weight: a log Grid needs a count$"),
        (
            Grid(field="weight", low=0, high=2, step=0.1, spacing="log", count=3),
            r"^weight: a log Grid needs a low above 0; it is 0$",
        ),
        (Grid(field="weight", low=1, high=2, step=0.1, spacing="power"), r"^weight: a power Grid needs a ratio$"),
        (
            Grid(field="weight", low=1, high=2, step=0.1, spacing="power", ratio=1.0),
            r"^weight: a power Grid needs a ratio above 1; it is 1$",
        ),
        (
            Grid(field="weight", low=1, high=2, step=0.1, count=3),
            r"^weight: count is for spacing: log — this Grid's spacing is linear$",
        ),
        (
            Grid(field="weight", low=1, high=2, step=0.1, spacing="log", count=3, ratio=2.0),
            r"^weight: ratio is for spacing: power — this Grid's spacing is log$",
        ),
        (Grid(field="plan.width", step=4), r"^plan.width: Grid needs a number setting \(int or float\); it is "),
        (
            Grid(field="repeats", low=0, high=10, step=0.5),
            r"^repeats: Grid's step 0.5 on an integer setting — give a whole step$",
        ),
    ],
)
def test_a_grid_that_cannot_be_read_is_a_spec_error(draw: Grid, message: str) -> None:
    template = ToyGenerator() if draw.field in ("plan.width", "repeats") else ToyStrip()
    with pytest.raises(DrawSpecError, match=message):
        draw_settings(template, [draw], random.Random(0))


@pytest.mark.parametrize(
    "written, value",
    [
        (2.5e6, 2.5e6),
        (3, 3.0),
        ("1MHz", 1.0e6),
        ("2 MSa/s", 2.0e6),
        ("2MS/s", 2.0e6),
        ("100kHz", 1.0e5),
        ("100 k", 1.0e5),
        ("1e6", 1.0e6),
        ("2.5e+6 Hz", 2.5e6),
        ("5 ms", 0.005),
        ("20us", 2.0e-5),
        ("1.5GHz", 1.5e9),
        ("-3", -3.0),
    ],
)
def test_a_number_is_read_with_its_unit(written: Any, value: float) -> None:
    assert parse_quantity(written) == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("written", ["1 mhz", "MHz", "1 MHz Hz", "one", "1 kg", True])
def test_a_number_that_cannot_be_read_is_refused_with_the_spellings(written: Any) -> None:
    with pytest.raises(DrawSpecError, match=r"cannot be read as a number — write it as 1e6, 1MHz, 2 MSa/s"):
        parse_quantity(written)


def test_a_grid_written_in_yaml_reads_its_units() -> None:
    draws = confluid.load(
        """
        draws:
          - !class:recordstream.draws.Grid
            field: weight
            low: 1MHz
            high: 2MSa/s
            step: 500kHz
        """
    )["draws"]
    assert _grid_values(draws[0]) == [1.0e6, 1.5e6, 2.0e6]


# -- Span --------------------------------------------------------------------------------------------------------


def _slot_cells(share: Any, width: int = 16) -> List[List[int]]:
    spec: List[Draw] = [
        Choice(field="plan.width", values=[width]),
        Repeat(field="plan.slots", count=(1, 1), each=[Span(field="cells", share=share)]),
    ]
    return [g.plan.slots[0].cells for g in _draws(spec, n=300)]


def test_a_span_is_a_run_of_consecutive_values_the_generator_accepts() -> None:
    for cells in _slot_cells((0.0, 1.0), width=8):
        assert cells == list(range(cells[0], cells[0] + len(cells)))
        assert cells[-1] < 8


def test_a_spans_share_is_of_the_values_accepted_one_at_a_time() -> None:
    """Width 8 accepts cells 0-7 of the setting's 0-19: a share of 0.5 is 4 cells."""
    assert {len(c) for c in _slot_cells((0.5, 0.5), width=8)} == {4}
    lengths = [len(c) for c in _slot_cells((0.25, 1.0), width=16)]
    assert min(lengths) == 4 and max(lengths) == 16


def test_a_span_on_a_setting_that_is_not_a_list_of_bounded_integers_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"plan.width: Span needs a list of integers with a range"):
        draw_settings(ToyGenerator(), [Span(field="plan.width")], random.Random(0))


# -- Repeat ------------------------------------------------------------------------------------------------------


def test_a_repeat_grows_the_list_by_its_drawn_count() -> None:
    spec: List[Draw] = [Repeat(field="plan.slots", count=(2, 5), each=[Choice(field="position"), Choice(field="kind")])]
    counts = {len(g.plan.slots) for g in _draws(spec)}
    assert counts == {2, 3, 4, 5}


def test_a_repeat_ends_the_list_when_the_next_element_finds_no_room() -> None:
    """Pattern 0 closes positions 0, 3, 6, 9: six positions are left, so ten slots asked give six."""
    spec: List[Draw] = [
        Choice(field="plan.mode", values=[ToyMode(pattern=0, depth=2)]),
        Repeat(field="plan.slots", count=(10, 10), each=[Choice(field="position")]),
    ]
    for g in _draws(spec, n=20):
        assert sorted(s.position for s in g.plan.slots) == [1, 2, 4, 5, 7, 8]


def test_a_repeat_logs_how_many_elements_it_made_of_how_many_drawn() -> None:
    spec: List[Draw] = [
        Choice(field="plan.mode", values=[ToyMode(pattern=0, depth=2)]),
        Repeat(field="plan.slots", count=(10, 10), each=[Choice(field="position")]),
    ]
    log = dict(draw_settings(ToyGenerator(), spec, random.Random(0)).log)
    assert log["plan.slots"] == "6 of 10"


def test_a_refusal_after_an_elements_first_draw_is_the_specs_fault_and_is_raised() -> None:
    """Level 1 refuses kind b: the slot found room, so its kind having no accepted value is a contradiction."""
    spec: List[Draw] = [
        Choice(field="plan.level", values=[1]),
        Repeat(field="plan.slots", count=(1, 1), each=[Choice(field="position"), Choice(field="kind", values=["b"])]),
    ]
    with pytest.raises(DrawRefused, match=r"plan.slots\[0\].kind: none of \['b'\] is accepted after"):
        draw_settings(ToyGenerator(), spec, random.Random(0))


def test_a_repeat_with_nothing_to_draw_adds_default_elements_while_they_fit() -> None:
    """A default slot sits at position 0, so only one fits."""
    drawn = _draws([Repeat(field="plan.slots", count=(3, 3))], n=5)
    assert all(g.plan.slots == [ToySlot()] for g in drawn)


def test_a_repeat_inside_a_repeat_fills_each_element_as_far_as_that_element_allows() -> None:
    """A floor's plans each draw a mode or none, then grow their slots: ten asked, and each plan holds the ten
    positions less those its own mode closes (pattern 0 closes four, 1 and 2 close three)."""
    spec: List[Draw] = [
        Repeat(
            field="plans",
            count=(3, 3),
            each=[
                Choice(field="mode", values=[None, ToyMode(depth=2)]),
                Choice(field="mode.pattern"),
                Repeat(field="slots", count=(10, 10), each=[Choice(field="position")]),
            ],
        )
    ]
    closed = {None: 0, 0: 4, 1: 3, 2: 3}
    shapes = set()
    for seed in range(40):
        drawn = draw_settings(ToyFloor(), spec, random.Random(seed))
        plans = drawn.settings.plans
        assert len(plans) == 3
        for plan in plans:
            pattern = None if plan.mode is None else plan.mode.pattern
            assert len(plan.slots) == 10 - closed[pattern]
            shapes.add(pattern)
        log = dict(drawn.log)
        assert log["plans"] == "3 of 3" and log["plans[0].slots"].endswith(" of 10")
    assert shapes == {None, 0, 1, 2}


def test_a_repeat_as_an_elements_first_draw_places_the_element() -> None:
    spec: List[Draw] = [
        Repeat(field="plans", count=(3, 3), each=[Repeat(field="slots", count=(1, 2), each=[Choice(field="position")])])
    ]
    plans = draw_settings(ToyFloor(), spec, random.Random(5)).settings.plans
    assert len(plans) == 3 and all(1 <= len(plan.slots) <= 2 for plan in plans)


def test_an_inner_repeat_that_adds_nothing_as_the_first_draw_ends_the_outer_list() -> None:
    """The first draw decides whether there is room; an inner Repeat that placed nothing placed no element."""
    spec: List[Draw] = [Repeat(field="plans", count=(3, 3), each=[Repeat(field="slots", count=(0, 0))])]
    drawn = draw_settings(ToyFloor(), spec, random.Random(0))
    assert drawn.settings.plans is None and dict(drawn.log)["plans"] == "0 of 3"


def test_a_repeat_inside_a_repeat_loads_from_a_config_with_every_count_checked() -> None:
    document = """
draw: !class:recordstream.draws.Repeat
  field: plans
  count: [1, 2]
  each:
    - !class:recordstream.draws.Repeat
      field: slots
      count: COUNT
      each:
        - !class:recordstream.draws.Choice {field: position}
"""
    draw = confluid.load(document.replace("COUNT", "[0, 4]"))["draw"]
    assert isinstance(draw, Repeat) and draw.each is not None and isinstance(draw.each[0], Repeat)
    plans = draw_settings(ToyFloor(), [draw], random.Random(1)).settings.plans
    assert 1 <= len(plans) <= 2
    with pytest.raises(confluid.ConstructionError, match=r"Repeat at <unicode string>:6:7: .*\n?count"):
        confluid.load(document.replace("COUNT", "[-1, 4]"))


def test_a_repeat_on_a_setting_that_is_not_a_list_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"plan.width: Repeat needs a list setting"):
        draw_settings(ToyGenerator(), [Repeat(field="plan.width", count=(1, 2))], random.Random(0))


# -- an element of a list -----------------------------------------------------------------------------------------


def _two_slots() -> ToyGenerator:
    return ToyGenerator(plan=ToyPlan(slots=[ToySlot(position=1), ToySlot(position=5)]))


def test_a_draw_reaches_an_existing_element_of_a_list() -> None:
    """``slots[1]`` names the list's second element: its position is drawn among those the plan accepts — never the
    first slot's 1 —, the first slot is left as it was, and the log names the element."""
    drawn = _draws([Choice(field="plan.slots[1].position")], template=_two_slots())
    assert {g.plan.slots[1].position for g in drawn} == set(range(10)) - {1}
    assert {g.plan.slots[0].position for g in drawn} == {1}
    log = draw_settings(_two_slots(), [Choice(field="plan.slots[1].position")], random.Random(3)).log
    assert log[0][0] == "plan.slots[1].position"


def test_uniform_and_span_reach_an_element_too() -> None:
    spec = [Uniform(field="plan.slots[0].weight", low=0.1, high=0.2), Span(field="plan.slots[0].cells")]
    drawn = _draws(spec, n=50, template=_two_slots())
    assert all(0.1 <= g.plan.slots[0].weight <= 0.2 and g.plan.slots[1].weight == 0.5 for g in drawn)
    assert all(g.plan.slots[0].cells and max(g.plan.slots[0].cells) < 8 for g in drawn)


def test_an_element_inside_a_list_that_is_none_is_skipped() -> None:
    """As every path through a setting that is ``None``: there is no element to draw."""
    drawn = draw_settings(ToyGenerator(), [Choice(field="plan.slots[0].kind")], random.Random(1))
    assert drawn.settings.plan.slots is None and drawn.log == []


def test_a_whole_element_is_drawn_or_skipped() -> None:
    """``slots[1]`` as the last step draws the element itself — among those the plan accepts (position 1 is the first
    slot's) —, and is skipped when the list is ``None``."""
    choice = Choice(field="plan.slots[1]", values=[ToySlot(position=1), ToySlot(position=7)])
    assert {g.plan.slots[1].position for g in _draws([choice], n=40, template=_two_slots())} == {7}
    assert draw_settings(ToyGenerator(), [choice], random.Random(1)).settings.plan.slots is None


def test_an_element_that_is_none_is_drawn_like_any_value() -> None:
    """``modes[1]`` is None in the template: the step names that element, so the draw sets it — only an unset LIST
    skips — and a draw inside it follows."""
    spec: List[Draw] = [
        Choice(field="modes[1]", values=[ToyMode(pattern=2)]),
        Choice(field="modes[1].depth", values=[1]),
    ]
    drawn = draw_settings(ToyFloor(modes=[None, None]), spec, random.Random(0))
    assert drawn.settings.modes == [None, ToyMode(pattern=2, depth=1)]
    assert drawn.log == [("modes[1]", ToyMode(pattern=2)), ("modes[1].depth", 1)]


ELEMENT_SPEC_ERRORS = [
    (
        Choice(field="plan.slots[2].kind"),
        "plan.slots\\[2\\].kind: slots has 2 elements; \\[2\\] is past its end — only a Repeat adds elements",
    ),
    (
        Uniform(field="plan.width[0]"),
        "plan.width\\[0\\]: width is not a list; it is Literal\\[4, 8, 16\\] — an index \\[i\\] names an element of a "
        "list setting",
    ),
    (
        Choice(field="plan.slots[x].kind"),
        "Choice: field 'plan.slots\\[x\\].kind' has a step 'slots\\[x\\]' that is neither a name nor name\\[index\\] — "
        "write it as a.b\\[0\\].c",
    ),
    (Choice(field="plan.slots[0"), "has a step 'slots\\[0' that is neither a name nor name\\[index\\]"),
]


@pytest.mark.parametrize("draw,words", ELEMENT_SPEC_ERRORS)
def test_a_path_to_an_element_that_cannot_be_read_is_a_spec_error(draw: Draw, words: str) -> None:
    with pytest.raises(DrawSpecError, match=words):
        draw_settings(_two_slots(), [draw], random.Random(1))


# -- every draw --------------------------------------------------------------------------------------------------

FULL: List[Draw] = [
    Choice(field="plan.width"),
    Choice(field="plan.mode", values=[None, ToyMode()], weights=[1.0, 1.0]),
    Choice(field="plan.mode.pattern"),
    Choice(field="plan.mode.depth"),
    Choice(field="plan.level"),
    Repeat(
        field="plan.slots",
        count=(0, 10),
        each=[
            Choice(field="position"),
            Span(field="cells", share=(0.1, 1.0)),
            Choice(field="kind"),
            Uniform(field="weight"),
        ],
    ),
    Uniform(field="gain"),
    Uniform(field="repeats", low=1, high=5),
]


def test_every_draw_is_a_generator_the_generator_accepts() -> None:
    for generator in _draws(FULL, n=300):
        rebuilt = ToyGenerator(plan=generator.plan, gain=generator.gain, repeats=generator.repeats)
        rebuilt.check()
        assert rebuilt.run()["total"] >= 0


def test_the_template_is_never_changed() -> None:
    template = ToyGenerator()
    _draws(FULL, n=20, template=template)
    assert template.plan == ToyPlan() and template.gain == 0.5 and template.repeats == 1


def test_the_same_random_state_draws_the_same_settings() -> None:
    first = draw_settings(ToyGenerator(), FULL, random.Random(7))
    again = draw_settings(ToyGenerator(), FULL, random.Random(7))
    other = draw_settings(ToyGenerator(), FULL, random.Random(8))
    assert first.log == again.log and first.settings.plan == again.settings.plan
    assert first.log != other.log


def test_a_setting_that_does_not_exist_is_a_spec_error_naming_the_settings_there_are() -> None:
    with pytest.raises(DrawSpecError) as caught:
        draw_settings(ToyGenerator(), [Choice(field="plan.widht")], random.Random(0))
    assert str(caught.value) == "plan.widht: ToyPlan has no setting 'widht'; its settings are width, level, mode, slots"


def test_a_draw_without_a_field_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"Choice: field is empty — name the setting to draw by its path from the"):
        draw_settings(ToyGenerator(), [Choice()], random.Random(0))


def test_the_draws_are_cheap_to_build() -> None:
    """Zero-arg construction stores values only; nothing is checked until a draw."""
    assert Choice().field == "" and Uniform().low is None and Span().share == (0.0, 1.0) and Repeat().each is None
    assert Grid().step is None and Grid().spacing == "linear"


# -- the spec's own mistakes, and the shapes a setting can have -------------------------------------------------------


def test_a_bool_setting_draws_both_values() -> None:
    assert {g.closed for g in _draws([Choice(field="closed")], template=ToyStrip())} == {False, True}


def test_a_setting_written_with_postponed_annotations_is_read() -> None:
    assert {g.width for g in _draws([Choice(field="width")], template=ToyLater())} == {4, 8}


def test_a_setting_the_object_does_not_keep_cannot_be_rebuilt() -> None:
    with pytest.raises(DrawSpecError, match=r"ToyForgetful keeps no attribute for its setting 'size'"):
        draw_settings(ToyForgetful(), [Choice(field="size", values=[1])], random.Random(0))


def test_a_repeat_over_a_list_of_plain_values_is_a_spec_error() -> None:
    with pytest.raises(
        DrawSpecError, match=r"tags: Repeat needs a list setting of objects; .* draw a list of integers"
    ):
        draw_settings(ToyStrip(), [Repeat(field="tags", count=(1, 1))], random.Random(0))


def test_an_element_that_cannot_be_built_from_its_defaults_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"parts: Repeat builds each new element as ToyNeedsSize\(\) — "):
        draw_settings(ToyShelf(), [Repeat(field="parts", count=(1, 1))], random.Random(0))


def test_a_repeat_whose_first_draw_is_skipped_adds_nothing() -> None:
    """A holder's mode is None, so a first draw inside it is skipped and nothing places the element."""
    drawn = draw_settings(
        ToyShelf(), [Repeat(field="holders", count=(3, 3), each=[Choice(field="mode.pattern")])], random.Random(0)
    )
    assert drawn.settings.holders is None
    assert dict(drawn.log)["holders"] == "0 of 3"


def test_a_type_refusal_is_quoted_by_its_first_problem() -> None:
    with pytest.raises(DrawRefused) as caught:
        draw_settings(ToyGenerator(), [Choice(field="plan.width", values=[5])], random.Random(0))
    assert (
        str(caught.value)
        == "plan.width: none of [5] is accepted after nothing drawn — width: Input should be 4, 8 or 16"
    )


def test_a_field_with_an_empty_step_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"Choice: field 'plan..width' has an empty step"):
        draw_settings(ToyGenerator(), [Choice(field="plan..width")], random.Random(0))


def test_uniform_span_and_repeat_skip_a_setting_inside_an_object_that_is_none() -> None:
    spec: List[Draw] = [
        Uniform(field="plan.mode.depth"),
        Span(field="plan.mode.cells"),
        Repeat(field="plan.mode.slots", count=(1, 1)),
    ]
    drawn = draw_settings(ToyGenerator(), spec, random.Random(0))
    assert drawn.settings.plan == ToyPlan() and drawn.log == []


def test_a_choice_with_no_values_on_a_union_setting_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"label: Choice needs values — the setting is Union\[int, str\]"):
        draw_settings(ToyStrip(), [Choice(field="label")], random.Random(0))


def test_a_uniform_on_a_float_without_a_range_needs_low_and_high() -> None:
    with pytest.raises(DrawSpecError, match=r"weight: Uniform needs low and high — the setting \(float\) has none"):
        draw_settings(ToyStrip(), [Uniform(field="weight")], random.Random(0))


@pytest.mark.parametrize(
    "draw, message",
    [
        (Uniform(field="gain", low=0.9, high=0.1), r"gain: Uniform's low 0.9 is above its high 0.1"),
        (Span(field="cells", share=(0.8, 0.2)), r"cells: Span's share \(0.8, 0.2\) runs backwards"),
        (Repeat(field="plan.slots", count=(3, 1)), r"plan.slots: Repeat's count \(3, 1\) runs backwards"),
    ],
)
def test_a_range_written_backwards_is_a_spec_error(draw: Draw, message: str) -> None:
    template = ToyStrip() if isinstance(draw, Span) else ToyGenerator()
    with pytest.raises(DrawSpecError, match=message):
        draw_settings(template, [draw], random.Random(0))


def test_a_span_with_no_value_accepted_names_the_reason() -> None:
    with pytest.raises(DrawRefused) as caught:
        draw_settings(ToyStrip(closed=True), [Span(field="cells")], random.Random(0))
    assert str(caught.value) == (
        "cells: no value from 0 to 19 is accepted after nothing drawn — ToyStrip: closed — it takes no cells"
    )


def test_a_span_whose_run_is_refused_names_its_length() -> None:
    with pytest.raises(DrawRefused) as caught:
        draw_settings(ToyStrip(limit=3), [Span(field="cells", share=(1.0, 1.0))], random.Random(0))
    assert str(caught.value) == (
        "cells: no run of 20 of the 20 accepted values is accepted after nothing drawn — "
        "ToyStrip: 20 cells, over the limit 3"
    )


def test_no_template_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"draw_settings: there is no template to draw from"):
        draw_settings(None, [], random.Random(0))


def test_something_that_is_not_a_draw_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"draws\[0\] is a str — a draw is a Choice, Uniform, Grid, Span or Repeat"):
        draw_settings(ToyGenerator(), ["plan.width"], random.Random(0))  # type: ignore[list-item]


# -- the draws are ordinary configurable classes -------------------------------------------------------------------


@pytest.mark.parametrize("cls", [Choice, Uniform, Grid, Span, Repeat, DrawSource])
def test_confluid_builds_the_schema_every_form_and_tool_reads(cls: type) -> None:
    """A self-referring annotation left confluid with no schema: validation was switched OFF, with a warning only."""
    model = to_pydantic(cls)
    assert "field" in model.model_fields or "generator" in model.model_fields


def test_a_value_outside_a_draws_own_range_is_refused_when_it_is_built() -> None:
    with pytest.raises(ValueError, match=r"count"):
        Repeat(field="plan.slots", count=(-1, 2))
    with pytest.raises(ValueError, match=r"share"):
        Span(field="cells", share=(0.0, 1.5))
