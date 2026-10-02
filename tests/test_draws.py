"""The settings draws: each picks from its distribution, only among the values the generator accepts.

The toy generator (``tests/_draw_toys.py``) has the shapes a real one has — closed sets, an optional sub-object whose
setting inherits from its parent, a list of elements that must fit together, and a rule only ``check()`` knows.
"""

import random
from collections import Counter
from typing import Any, List

import pytest
from confluid import to_pydantic

from recordstream.draws import Choice, Draw, DrawRefused, DrawSpecError, Repeat, Span, Uniform, draw_settings
from recordstream.sources import DrawSource
from tests._draw_toys import ToyForgetful, ToyGenerator, ToyMode, ToyPlan, ToyShelf, ToySlot, ToyStrip
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


def test_a_choice_over_a_small_integer_range_tests_every_integer() -> None:
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


def test_a_repeat_on_a_setting_that_is_not_a_list_is_a_spec_error() -> None:
    with pytest.raises(DrawSpecError, match=r"plan.width: Repeat needs a list setting"):
        draw_settings(ToyGenerator(), [Repeat(field="plan.width", count=(1, 2))], random.Random(0))


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
    with pytest.raises(DrawSpecError, match=r"draws\[0\] is a str — a draw is a Choice, Uniform, Span or Repeat"):
        draw_settings(ToyGenerator(), ["plan.width"], random.Random(0))  # type: ignore[list-item]


# -- the draws are ordinary configurable classes -------------------------------------------------------------------


@pytest.mark.parametrize("cls", [Choice, Uniform, Span, Repeat, DrawSource])
def test_confluid_builds_the_schema_every_form_and_tool_reads(cls: type) -> None:
    """A self-referring annotation left confluid with no schema: validation was switched OFF, with a warning only."""
    model = to_pydantic(cls)
    assert "field" in model.model_fields or "generator" in model.model_fields


def test_a_value_outside_a_draws_own_range_is_refused_when_it_is_built() -> None:
    with pytest.raises(ValueError, match=r"count"):
        Repeat(field="plan.slots", count=(-1, 2))
    with pytest.raises(ValueError, match=r"share"):
        Span(field="cells", share=(0.0, 1.5))
