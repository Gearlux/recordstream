"""``Quantity`` — a number written with its unit, as a node a graph wires into a number setting (user request
2026-10-09: "that graph should have a node that can convert the unit to the rate and connect that to the upsample
rate, only the unit node exposes its settings, make this node generic"; user pick 1A: a unit of another kind is
refused).

The node IS the number (a ``float``): referenced from a number setting it hands that setting ``2500000.0`` for
``2.5 MSa/s``, and a settings form shows its ``value`` field instead of the setting it feeds. ``kind`` names the kind
of number it stands for, so a rate written as a time is refused naming what it takes.
"""

import re
from typing import Any

import confluid
import pytest
from confluid import configurable, get_registry

from recordstream.draws import DrawSpecError, parse_quantity, quantity_unit
from recordstream.quantity import Quantity, QuantityKind


@configurable
class RateTaker:
    """A node with a number setting, for a quantity to be wired into.

    Args:
        rate: A rate in Hz.
        count: A whole number.
    """

    def __init__(self, rate: float = 0.0, count: int = 1) -> None:
        self.rate = rate
        self.count = count


class TestTheNumber:
    @pytest.mark.parametrize(
        "written,kind,value",
        [
            ("2.5 MSa/s", "rate", 2.5e6),
            ("250k", "rate", 2.5e5),
            ("250 kHz", "rate", 2.5e5),
            ("1 MS/s", "rate", 1.0e6),
            (250000, "rate", 2.5e5),
            ("1.5GHz", "frequency", 1.5e9),
            ("100 k", "frequency", 1.0e5),
            ("5 ms", "time", 0.005),
            ("20us", "time", 2.0e-5),
            ("5 ms", "any", 0.005),
            ("2 MSa/s", "any", 2.0e6),
            ("0", "rate", 0.0),
        ],
    )
    def test_it_is_the_number_it_reads(self, written: Any, kind: QuantityKind, value: float) -> None:
        number = Quantity(value=written, kind=kind)
        assert isinstance(number, float) and float(number) == pytest.approx(value, rel=1e-12)
        assert (number.value, number.kind) == (written, kind), "its settings stay readable"

    @pytest.mark.parametrize(
        "written,kind,message",
        [
            (
                "5 ms",
                "rate",
                "Quantity: '5 ms' is a time — a rate is written in Hz, Sa/s, S/s or with no unit (250k, 2.5 MSa/s)",
            ),
            (
                "2 MSa/s",
                "frequency",
                "Quantity: '2 MSa/s' is a rate — a frequency is written in Hz or with no unit (1 MHz, 100k)",
            ),
            ("1 kHz", "time", "Quantity: '1 kHz' is a frequency — a time is written in s or with no unit (5 ms, 20us)"),
        ],
    )
    def test_a_number_of_another_kind_is_refused_naming_what_it_takes(
        self, written: str, kind: QuantityKind, message: str
    ) -> None:
        with pytest.raises(DrawSpecError, match=f"^{re.escape(message)}$"):
            Quantity(value=written, kind=kind)

    def test_a_text_that_is_no_number_is_refused_with_the_spellings(self) -> None:
        with pytest.raises(DrawSpecError, match=r"cannot be read as a number — write it as 1e6, 1MHz, 2 MSa/s"):
            Quantity(value="fast", kind="rate")

    def test_zero_arg_it_is_zero(self) -> None:
        assert Quantity() == 0.0 and Quantity().kind == "any"


class TestTheReader:
    @pytest.mark.parametrize(
        "written,unit",
        [("2.5 MSa/s", "Sa/s"), ("5 ms", "s"), ("1 kHz", "Hz"), ("250k", None), (250000, None), ("x", None)],
    )
    def test_the_unit_a_number_was_written_in(self, written: Any, unit: Any) -> None:
        assert quantity_unit(written) == unit

    def test_the_units_a_reading_accepts_can_be_named(self) -> None:
        assert parse_quantity("250 kHz", units=("Hz", "Sa/s", "S/s")) == 2.5e5
        assert parse_quantity("250", units=("s",)) == 250.0, "a bare number has no unit to refuse"
        with pytest.raises(DrawSpecError, match="'5 ms' is written in s — only Hz, Sa/s, S/s here"):
            parse_quantity("5 ms", units=("Hz", "Sa/s", "S/s"))


class TestInAGraph:
    def test_a_reference_hands_the_setting_the_number(self) -> None:
        document = confluid.load(
            "upsample_rate_value:\n"
            "  _target_: recordstream.quantity.Quantity\n"
            "  value: 2.5 MSa/s\n"
            "  kind: rate\n"
            "taker:\n"
            f"  _target_: {__name__}.RateTaker\n"
            "  rate:\n"
            "    _ref_: upsample_rate_value\n"
        )
        assert document["taker"].rate == 2.5e6

    def test_a_refusal_in_a_graph_names_the_node(self) -> None:
        with pytest.raises(Exception, match="'5 ms' is a time — a rate is written in"):
            confluid.load(
                "rate_value:\n  _target_: recordstream.quantity.Quantity\n  value: 5 ms\n  kind: rate\n"
                f"taker:\n  _target_: {__name__}.RateTaker\n  rate:\n    _ref_: rate_value\n"
            )

    def test_it_is_registered_under_its_name(self) -> None:
        assert get_registry().get_class("Quantity") is Quantity
