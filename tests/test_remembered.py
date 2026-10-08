"""``remembered``: a check (or any method of a settings object) answers a question it has answered before — the same
settings, field for field — from memory: the same return value, or the same refusal raised again. A draw asks a
generator's check thousands of times a record, a third of them about settings it has already judged.

The key is EVERY setting, exactly: a different value anywhere (deep in a list, the sign of a zero, ``True`` against
``1``), a setting changed after construction, or an argument of the method asks again; a value no fingerprint
describes is always asked; an error that is no refusal (``ValueError``/``TypeError``) is never remembered.
"""

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pytest

from recordstream import draws
from recordstream.draws import Choice, draw_settings, remembered, remembering

ASKED: List[str] = []


@dataclass
class Part:
    width: float = 1.0
    tags: Optional[List[str]] = None


@dataclass
class Settings:
    """A settings object whose check is remembered; it records every time it is really asked."""

    level: int = 0
    gain: float = 0.0
    parts: List[Part] = field(default_factory=list)
    note: Optional[str] = None
    table: Optional[np.ndarray] = None
    loose: object = None

    @remembered
    def check(self) -> None:
        ASKED.append(f"check level={self.level}")
        if self.level > 3:
            raise ValueError(f"Settings: level {self.level} is above 3")

    @remembered(ignore=("note",))
    def frame(self) -> int:
        ASKED.append("frame")
        return self.level * 10

    @remembered
    def scaled(self, factor: int) -> int:
        ASKED.append(f"scaled {factor}")
        return self.level * factor

    @remembered
    def broken(self) -> None:
        ASKED.append("broken")
        raise KeyError("not a refusal")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    ASKED.clear()
    for method in (Settings.check, Settings.frame, Settings.scaled, Settings.broken):
        method.forget()


class TestTheSameQuestionIsAnsweredFromMemory:
    def test_an_accepted_check_is_asked_once(self) -> None:
        Settings(level=1).check()
        Settings(level=1).check()  # another object, the same settings
        assert ASKED == ["check level=1"]

    def test_a_refusal_is_raised_again_with_its_message(self) -> None:
        for _ in range(3):
            with pytest.raises(ValueError, match="^Settings: level 5 is above 3$"):
                Settings(level=5).check()
        assert ASKED == ["check level=5"]

    def test_a_return_value_is_remembered(self) -> None:
        assert [Settings(level=2).frame() for _ in range(3)] == [20, 20, 20]
        assert ASKED == ["frame"]

    def test_a_raised_refusal_carries_no_earlier_traceback(self) -> None:
        tracebacks = []
        for _ in range(3):
            try:
                Settings(level=7).check()
            except ValueError as error:
                frames = 0
                tb = error.__traceback__
                while tb is not None:
                    frames, tb = frames + 1, tb.tb_next
                tracebacks.append(frames)
        assert tracebacks[1] == tracebacks[2]  # it does not grow with every answer


class TestAnyDifferenceAsksAgain:
    @pytest.mark.parametrize(
        "first, second",
        [
            (Settings(level=1), Settings(level=2)),
            (Settings(gain=0.0), Settings(gain=-0.0)),
            (Settings(level=True), Settings(level=1)),
            (Settings(parts=[Part(1.0)]), Settings(parts=[Part(1.5)])),
            (Settings(parts=[Part(tags=["a"])]), Settings(parts=[Part(tags=["b"])])),
            (Settings(table=np.array([1, 2, 3])), Settings(table=np.array([1, 2, 4]))),
            (Settings(table=np.array([1, 2, 3])), Settings(table=np.array([1.0, 2.0, 3.0]))),
            (Settings(note=None), Settings(note="x")),
        ],
    )
    def test_a_different_setting_is_another_question(self, first: Settings, second: Settings) -> None:
        first.check()
        second.check()
        assert len(ASKED) == 2

    def test_a_setting_changed_after_construction_is_another_question(self) -> None:
        settings = Settings(level=1)
        settings.check()
        settings.level = 5
        with pytest.raises(ValueError, match="level 5"):
            settings.check()
        settings.parts.append(Part())
        settings.level = 1
        settings.check()
        assert ASKED == ["check level=1", "check level=5", "check level=1"]

    def test_the_arguments_are_part_of_the_question(self) -> None:
        settings = Settings(level=3)
        assert (settings.scaled(2), settings.scaled(3), settings.scaled(2)) == (6, 9, 6)
        assert ASKED == ["scaled 2", "scaled 3"]

    def test_an_ignored_setting_is_not_part_of_the_question(self) -> None:
        assert Settings(level=2, note="a").frame() == Settings(level=2, note="b").frame() == 20
        assert ASKED == ["frame"]


class TestWhatIsNeverRemembered:
    def test_a_value_no_fingerprint_describes_is_always_asked(self) -> None:
        Settings(loose=object()).check()
        Settings(loose=object()).check()
        assert len(ASKED) == 2

    def test_an_error_that_is_no_refusal_is_raised_each_time(self) -> None:
        for _ in range(2):
            with pytest.raises(KeyError):
                Settings().broken()
        assert ASKED == ["broken", "broken"]

    def test_remembering_can_be_switched_off(self) -> None:
        with remembering(False):
            Settings(level=1).check()
            Settings(level=1).check()
        Settings(level=1).check()
        Settings(level=1).check()
        assert ASKED == ["check level=1"] * 3

    def test_the_oldest_answer_is_forgotten_beyond_the_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(draws, "REMEMBERED", 2)
        for level in (0, 1, 2, 0):  # 0 was pushed out by 1 and 2
            Settings(level=level).check()
        assert ASKED == [f"check level={n}" for n in (0, 1, 2, 0)]


def test_a_draw_asks_a_remembered_check_once_per_distinct_value() -> None:
    """The draws' own use: a Choice tries its values, several of which give settings already judged."""
    template = Settings(level=0)
    twice = [Choice(field="level", values=[0, 1, 2, 9]), Choice(field="level", values=[0, 1, 2, 9])]
    draw_settings(template, twice, random.Random(1))
    asked = sorted(set(ASKED))
    assert len(ASKED) == len(asked), f"a settings object was checked twice: {ASKED}"


def test_the_ignored_names_must_be_settings() -> None:
    message = "remembered: ignore names 'nonsense', which is not a setting of Settings2 — name a dataclass field"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):

        @dataclass
        class Settings2:
            level: int = 0

            @remembered(ignore=("nonsense",))
            def check(self) -> None:
                pass

        Settings2().check()
