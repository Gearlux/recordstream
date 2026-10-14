"""``Quantity`` — a number written with its unit, as a node a graph wires into a number setting.

A setting a person types is often a rate or a frequency, and a person writes it with its unit: ``250k``,
``2.5 MSa/s``, ``1 MHz``. A step that takes a plain number (Hz) cannot read that, and the workspace's rule is that a
number's unit is written in the value. So the reading is a NODE of its own: wire it into the number setting, and a
settings form shows the node's ``value`` where the step's number was. The node IS the number — a ``float`` — so a
reference to it hands the setting the number itself, validated as any number is; the reading is
:func:`recordstream.draws.parse_quantity`, the one rule a draw's ``low``/``high``/``step`` follow too.
"""

from typing import Dict, Literal, Tuple

from confluid import configurable

from recordstream.draws import DrawSpecError
from recordstream.draws import Quantity as Written
from recordstream.draws import Unit, parse_quantity, quantity_unit

#: The kind of number a quantity stands for — which decides the units it may be written in.
QuantityKind = Literal["any", "rate", "frequency", "time"]

#: The units each kind is written in (a number with no unit is every kind), and what a refusal shows as examples.
_KIND_UNITS: Dict[str, Tuple[Tuple[Unit, ...], str]] = {
    "rate": (("Hz", "Sa/s", "S/s"), "250k, 2.5 MSa/s"),
    "frequency": (("Hz",), "1 MHz, 100k"),
    "time": (("s",), "5 ms, 20us"),
}
#: The kind a unit writes, as a refusal names it.
_UNIT_KIND: Dict[str, str] = {"Hz": "frequency", "Sa/s": "rate", "S/s": "rate", "s": "time"}


@configurable(category="value", group="values")
class Quantity(float):
    """A number written with its unit — ``2.5 MSa/s`` is 2 500 000 — wired into a number setting.

    The node is the number itself: a reference to it from a setting hands that setting the number. ``kind`` is the kind
    of number it stands for — never the unit the number is in, which stays in the value. Written in a unit of another
    kind (a rate written as ``5 ms``), it is refused naming what that kind is written in. A number with no unit is
    taken as it is.

    Args:
        value: The number, with its unit or without: ``250k``, ``2.5 MSa/s``, ``1 MHz``, ``5 ms``, ``1e6``, 250000.
        kind: The kind of number — ``rate`` (Hz, Sa/s, S/s), ``frequency`` (Hz), ``time`` (s) or ``any``.
    """

    def __new__(cls, value: Written = "0", kind: QuantityKind = "any") -> "Quantity":
        # The reading happens here, not in __init__: a float's value is fixed when it is made. It is a pure function of
        # the two settings — no I/O, nothing read from a record — so the node stays cheap to build.
        if kind == "any":
            return super().__new__(cls, parse_quantity(value))
        units, examples = _KIND_UNITS[kind]
        written = quantity_unit(value)
        if written is not None and written not in units:
            raise DrawSpecError(
                f"Quantity: {value!r} is a {_UNIT_KIND[written]} — a {kind} is written in "
                f"{', '.join(units)} or with no unit ({examples})"
            )
        number = parse_quantity(value)
        return super().__new__(cls, number)

    def __init__(self, value: Written = "0", kind: QuantityKind = "any") -> None:
        self.value = value
        self.kind: QuantityKind = kind


__all__ = ["Quantity", "QuantityKind"]
