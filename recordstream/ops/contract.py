"""Record interface contracts — a pass-through op asserting what a record carries.

A pipeline's interface is a statement about the RECORDS crossing a point in the chain:
"whatever reaches here carries an ``Image`` under ``image`` and a ``Label`` under
``class``". :class:`RecordContract` makes that statement executable and VISIBLE — a
consuming workspace or visual editor places one at the boundary it cares about, and a
violating record fails loudly at that boundary instead of surfacing later as an empty
result.

One class serves both boundary roles, decided by POSITION (ops apply in list order):
the first op in a chain states what the host must feed (input contract), the last op
states what the pipeline guarantees to deliver (output contract). ``name`` labels the
boundary in errors ("classification input" vs "classification output"); there is
deliberately no ``role`` knob — position already IS the role.
"""

from typing import Dict, Optional

from confluid import configurable

from recordstream.items import Record, get_item_type

#: Declared-type sentinel meaning "the entry must be PRESENT, any type" — for a boundary
#: past a conversion that emits plain values (e.g. a live tensor, which no item wraps).
ANY_TYPE = "*"


class ContractError(ValueError):
    """A record violated a :class:`RecordContract` — the message names the boundary,
    the record ordinal, the offending entry, and what the record does carry."""


def _describe(record: Record) -> str:
    """The record's entries as ``key[TypeName]`` — what IS present, for the error message."""
    return ", ".join(f"{key}[{type(value).__name__}]" for key, value in record.items()) or "<empty record>"


@configurable(category="op", group="contract")
class RecordContract:
    """Pass-through op asserting each record carries the declared entries with the declared item types.

    Place it at a pipeline BOUNDARY: first in the ops list to state what the host must
    feed, last to state what the pipeline delivers. Every record is checked (a dict
    lookup plus an ``isinstance`` per declared entry); a violation raises
    :class:`ContractError` naming the boundary, the record ordinal, the offending entry,
    and the entries the record does carry. The record itself is returned UNCHANGED.

    Args:
        fields: Entry key -> registered item type name (e.g. ``{"image": "Image"}``); ``"*"`` = present, any type.
        name: Boundary label used in error messages (e.g. "classification input").
    """

    def __init__(self, fields: Optional[Dict[str, str]] = None, name: str = "") -> None:
        self.fields = fields or {}
        self.name = name
        self._count = 0  # runtime record counter for error messages — NOT config (per-instance)

    def __call__(self, record: Record) -> Record:
        label = self.name or "record contract"
        for key, type_name in self.fields.items():
            if key not in record:
                self._fail(f"{label}: record #{self._count} has no entry {key!r} (expected {type_name})", record)
            if type_name == ANY_TYPE:
                continue
            try:
                expected = get_item_type(type_name)
            except KeyError as error:
                self._fail(f"{label}: entry {key!r} declares {error.args[0]}", record)
            value = record[key]
            if not isinstance(value, expected):
                self._fail(
                    f"{label}: record #{self._count} entry {key!r} is a {type(value).__name__}, expected {type_name}",
                    record,
                )
        self._count += 1
        return record

    def _fail(self, message: str, record: Record) -> None:
        self._count += 1  # a failed record still advances the ordinal
        raise ContractError(f"{message}; present: {_describe(record)}") from None
