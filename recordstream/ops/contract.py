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

from typing import Any, Dict, List, Optional

from confluid import configurable, output

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


@configurable(category="value", constant=True, group="contract")
class ClassNamesOutput:
    """The class vocabulary a pipeline DELIVERS — a graph output a consuming workspace reads.

    :class:`RecordContract` states what each RECORD carries; this states something about the
    DATASET, so it cannot be a per-record op. A consuming workspace needs the ordered class list
    to show a label as a NAME rather than an integer.

    It READS, it never derives. There are three ways to give it an answer, and the graph author
    picks one explicitly — this class decides nothing on their behalf:

    1. **type them** — set :attr:`names` directly;
    2. **connect something that carries a vocabulary** — set :attr:`classes` to any object with a
       ``class_names`` attribute (a :class:`~recordstream.Stream`, a source that knows its own,
       a walker like :class:`ClassNamesScan`);
    3. **connect a walker** — :class:`ClassNamesScan` derives them by walking a source, for
       a source that declares none.

    Nothing here scans a dataset. A walk is seconds per thousand records and would run on every
    graph open, so it is a node the author PLACES (route 3), never a fallback this one takes.

    ``classes`` holds the whole object rather than a reference to its attribute because that is
    the only shape a config document can carry: attribute references (``!ref:src.class_names``)
    were removed from the config layer, and the documented replacement is exactly this — a
    selector parameter on the CONSUMER, referencing the whole object.

    ``constant=True`` so a visual editor hoists it to its own top-level key rather than inlining
    it, which is what makes it readable back out of the exported document.

    Args:
        names: The classes in class-id order — typed in, or wired from a ``class_names`` output.

    Example:
        >>> ClassNamesOutput(names=["cat", "dog"]).class_names
        ['cat', 'dog']
        >>> ClassNamesOutput().class_names          # nothing given: empty, never a guess
        []
    """

    def __init__(self, names: Optional[List[str]] = None) -> None:
        self.names = list(names or [])

    @property
    @output
    def class_names(self) -> List[str]:
        """The declared vocabulary, in class-id order — ``[]`` when nothing was given."""
        return [str(entry) for entry in self.names]

    @property
    def num_classes(self) -> int:
        """How many classes this output delivers (``0`` while nothing has been given)."""
        return len(self.class_names)

    def __call__(self) -> None:
        """No-arg call: a value producer, not a record op — it reads no record."""
        return None


@configurable(category="value", group="contract")
class ClassNamesScan:
    """Derive a class vocabulary by WALKING (scanning) a source's label column.

    Named ``…Scan`` rather than ``…FromSource`` deliberately: a class whose name ends in
    ``Source`` is read by the visual editor's node builder as a record SOURCE, and this one
    produces a vocabulary, not records.

    It is the node an author places when a source declares no vocabulary of its own — a folder reader,
    a CSV, anything the dataset format does not describe. It is deliberately a separate class
    from :class:`ClassNamesOutput`: a walk costs seconds per thousand records, so it happens
    because someone put this on the canvas, never as a silent fallback inside something else.

    The walk is key-restricted through :mod:`recordstream.projection`, so a projection-aware
    source is asked only for the label column.

    Values are reported sorted-unique and stringified — the same ordering
    :meth:`~recordstream.LabelMap.fit` uses, so a vocabulary derived here and one fitted there
    agree. A column of encoded ids therefore yields ``['0', '1', '2']``: a walk over encoded
    labels can only honestly report the ids it saw, and that is what a dataset's own metadata
    says too (MNIST's names really are ``'0'``…``'9'``).

    Args:
        source: The source to walk. ``None`` = nothing to walk, and an empty vocabulary.
        key: The record key holding the label.

    Example:
        >>> from recordstream import Label
        >>> ClassNamesScan(source=[{"class": Label(value="dog")}, {"class": Label(value="cat")}]).class_names
        ['cat', 'dog']
    """

    def __init__(self, source: Optional[Any] = None, key: str = "class") -> None:
        self.source = source
        self.key = key

    @property
    @output
    def class_names(self) -> List[str]:
        """The sorted-unique vocabulary found in the source's label column."""
        if self.source is None:
            return []
        from recordstream.projection import iter_key

        seen = {str(value) for value in iter_key(self.source, self.key) if value is not None}
        return sorted(seen)

    @property
    def num_classes(self) -> int:
        """How many distinct classes the walk found."""
        return len(self.class_names)

    def __call__(self) -> None:
        """No-arg call: a value producer, not a record op."""
        return None
