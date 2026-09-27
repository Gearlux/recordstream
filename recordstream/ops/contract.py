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

from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, get_args

from confluid import configurable, output

from recordstream.items import Record, get_item_type, item_type_names

#: Declared-type sentinel meaning "the entry must be PRESENT, any type" — for a boundary
#: past a conversion that emits plain values (e.g. a live tensor, which no item wraps).
ANY_TYPE = "*"


class ContractError(ValueError):
    """A record violated a :class:`RecordContract` — the message names the boundary,
    the record ordinal, the offending entry, and what the record does carry."""


class ChainContractError(ValueError):
    """An op chain cannot hold together — checked by :func:`check_chain` BEFORE it runs.

    The failure this exists to prevent is a SILENT one: an op that reads a record entry an
    earlier op was supposed to write just returns the record unchanged, so a mis-wired chain
    produces an empty result rather than an error, and the reader has nothing to go on.
    """


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

    It holds ONE thing, the list in :attr:`names`, and derives nothing. The graph author picks how
    the list gets there — this class decides nothing on their behalf:

    1. **type them** — ``names: [cat, dog]``;
    2. **take them from a source that knows its own** — wire the source's ``class_names`` output
       into ``names`` (a HuggingFace source reads its ``ClassLabel`` names from the metadata);
    3. **take them from a walker** — wire a :class:`ClassNamesScan`'s ``class_names`` output into
       ``names``, for a source that declares none.

    A wire becomes a LITERAL in the saved document: a document cannot reference one object's
    attribute from another (``!ref:scan.class_names`` is refused on load), so a visual editor
    evaluates the wire when it saves and writes the list into ``names``. There is deliberately
    no second input holding the producer: a second input is a second way to be wrong.

    Nothing here scans a dataset. A walk is seconds per thousand records and would run on every
    graph open, so it is a node the author PLACES (route 3), never a fallback this one takes.

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


# =======================================================================================
# The GRAPH-level contract: what one graph takes from its host and must deliver back
# =======================================================================================

#: The slot kinds a graph contract may declare, in the words a visual editor types its sockets
#: with. A registered item type name (``Image``, ``Label``, ``Boxes``, …) is a legal kind too:
#: the item registry is the open half, this set is the closed half.
GraphSlotKind = Literal["Stream", "Source", "Sink", "Op", "Record", "List[str]", "str", "int", "float", "bool"]
GRAPH_SLOT_KINDS: Tuple[str, ...] = get_args(GraphSlotKind)


def _slot_kind_known(kind: str) -> bool:
    return kind in GRAPH_SLOT_KINDS or kind in item_type_names()


def _is_stream(value: Any) -> bool:
    """A ``Stream`` by shape — it has ``ops`` and a ``source`` — so a consumer's own subclass qualifies."""
    return hasattr(value, "ops") and hasattr(value, "source") and not isinstance(value, (list, tuple, dict))


@configurable(category="contract", group="graph", constant=True)
class GraphContract:
    """What ONE graph takes from its host and must deliver back — declared as DATA.

    A consuming workspace opens several graphs — one that delivers its records, one that turns
    dropped files into records, one that consumes the records it labelled — and each must deliver
    something different. Stating that per graph with ONE class, as a map of named and typed slots,
    is what lets a visual editor draw every graph's boundary the same way (a root node whose input
    sockets are the ``outputs``; an input node whose output sockets are the ``inputs``), lets its
    export write what the author wired (``delivered``), and lets the host refuse a graph that
    delivers less than it promised — in the same words, because :meth:`check` is the ONE
    verification both of them call.

    Every entry of ``outputs`` is REQUIRED. An optional output would not be a contract: what a
    graph may or may not deliver is simply not declared.

    Args:
        name: The graph this contract is for, as a refusal names it (``"classification source"``).
        inputs: ``{slot: kind}`` the HOST provides when it runs the graph (dropped file paths, the
            records to annotate, a viewer's window). Kinds are :data:`GRAPH_SLOT_KINDS` or a
            registered item type name.
        outputs: ``{slot: kind}`` the graph must deliver. Same vocabulary.
        records: ``{entry: item type}`` every record of a delivered stream carries — the
            :class:`RecordContract` :meth:`stream` appends as the stream's last op, so a violating
            record is refused where the graph is applied, never when a record is opened later.
        delivered: ``{slot: value}`` — what the drawn graph wired into each output. Written by the
            editor's export, read by the host; never typed by hand.
    """

    def __init__(
        self,
        name: str = "",
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[Dict[str, str]] = None,
        records: Optional[Dict[str, str]] = None,
        delivered: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.inputs: Dict[str, str] = dict(inputs or {})
        self.outputs: Dict[str, str] = dict(outputs or {})
        self.records: Dict[str, str] = dict(records or {})
        self.delivered: Dict[str, Any] = dict(delivered or {})

    @property
    def label(self) -> str:
        return self.name or "graph contract"

    def missing(self) -> List[str]:
        """The outputs nothing was wired into, in declaration order — what an editor marks and a host refuses."""
        return [slot for slot in self.outputs if self.delivered.get(slot) is None]

    def check_declaration(self) -> None:
        """Refuse a DECLARATION outside the vocabulary — what a visual editor checks before it draws anything."""
        for side, slots in (("input", self.inputs), ("output", self.outputs)):
            for slot, kind in slots.items():
                if not _slot_kind_known(str(kind)):
                    raise ContractError(
                        f"{self.label}: {side} {slot!r} declares the type {kind!r}, which is neither a graph slot type "
                        f"({', '.join(GRAPH_SLOT_KINDS)}) nor a registered item type"
                    )

    def check(self) -> None:
        """Refuse a graph that does not deliver what it declares — :class:`ContractError`, in the reader's words.

        Checked in this order, so the first message a reader sees is the one to act on: a kind
        outside the vocabulary (the DECLARATION is wrong), an output nothing was wired into, then
        each delivered value against its kind — a ``Stream`` must be one and must have a source; a
        ``List[str]`` must be a non-empty list.
        """
        self.check_declaration()
        missing = self.missing()
        if missing:
            raise ContractError(
                f"{self.label}: {missing[0]!r} is not delivered — wire it in the graph "
                f"(this graph delivers: {', '.join(self.outputs)})"
            )
        for slot, kind in self.outputs.items():
            value = self.delivered[slot]
            if kind == "Stream":
                if not _is_stream(value):
                    raise ContractError(
                        f"{self.label}: {slot!r} is a {type(value).__name__}, not a Stream — put a Stream between them"
                    )
                if value.source is None:
                    raise ContractError(
                        f"{self.label}: the Stream delivered as {slot!r} has no source — connect a Source to it"
                    )
            elif kind == "List[str]":
                if not isinstance(value, (list, tuple)):
                    raise ContractError(f"{self.label}: {slot!r} is a {type(value).__name__}, not a list of names")
                if not value:
                    raise ContractError(
                        f"{self.label}: {slot!r} resolved to [] — wire a class_names output (a HuggingFace source "
                        "has one) or a Value list into it"
                    )

    def records_contract(self) -> Optional[RecordContract]:
        """The per-record contract as the op that enforces it, or ``None`` when none is declared."""
        return RecordContract(fields=dict(self.records), name=self.name) if self.records else None

    def stream(self) -> Any:
        """The delivered stream, with :attr:`records` enforced as its LAST op — a new Stream, the
        delivered one untouched (it is the document's object; the host must not rewrite the graph)."""
        slot = next((s for s, kind in self.outputs.items() if kind == "Stream"), None)
        if slot is None:
            raise ContractError(f"{self.label}: this graph delivers no stream (it delivers: {', '.join(self.outputs)})")
        self.check()
        from recordstream.core.stream import Stream  # the engine imports this module's ops; keep the edge one-way

        delivered = self.delivered[slot]
        contract = self.records_contract()
        ops = list(delivered.ops or [])
        if contract is not None:
            ops.append(contract)
        return Stream(source=delivered.source, ops=ops, class_names=getattr(delivered, "class_names", None))

    @property
    def class_names(self) -> List[str]:
        """The delivered vocabulary as strings, in its delivered order — ``[]`` when the graph delivers none."""
        return [str(name) for name in (self.delivered.get("class_names") or [])]


# =======================================================================================
# The PER-OP interface declaration, and the whole-chain check over it
# =======================================================================================


def _declared(op: Any, name: str, empty: Any) -> Any:
    """One declaration read off an op, tolerating an op that declares nothing."""
    return getattr(op, name, empty) or empty


def _node_name(op: Any) -> str:
    return type(op).__name__


def flag_producers(ops: Sequence[Any]) -> Dict[str, int]:
    """``{flag: index of the op that raises it}`` over a chain.

    The map is what makes a gate's reason ANSWERABLE — the report says which node decided a
    branch, and a visual editor rendering ``requires`` as a fork knows which node's output
    socket the wire leaves from. :func:`check_chain` refuses a chain where two ops declare one
    flag, so this map is total over every flag any op declares.
    """
    producers: Dict[str, int] = {}
    for index, op in enumerate(ops):
        for flag in _declared(op, "flags", ()):
            producers.setdefault(str(flag), index)
    return producers


def check_chain(ops: Sequence[Any], *, provided: Iterable[str] = (), where: str = "") -> None:
    """Check that a chain holds together, BEFORE it is run over a record.

    An op may declare its interface as class attributes — ``consumes`` / ``produces``
    (``{record key: registered item type}``, the same vocabulary :class:`RecordContract` uses,
    with :data:`ANY_TYPE` for "present, any type"), ``reports`` (its key in an analysis report;
    ``""`` marks a transform rather than an analysis) and ``flags`` (the boolean findings it
    raises) — plus the instance parameter ``requires``, naming the one flag that gates it.

    Four things are refused, each of which would otherwise surface as an empty result:

    * a ``consumes`` key no earlier op ``produces`` and ``provided`` does not carry;
    * a ``requires`` naming a flag no op declares, or one declared only LATER in the chain;
    * two ops declaring the same flag — ambiguous, so which node decided a branch would have
      no answer;
    * two ops reporting under the same name — the later finding would overwrite the earlier.

    Declaring is OPT-IN: an op with none of these attributes is checked for nothing, so a chain
    written before the mechanism existed passes unchanged. That also means a declaring op must
    not depend on an UNdeclared op's output — the fix is to declare on the producer, which the
    refusal says.

    Args:
        ops: The chain, in execution order.
        provided: Record keys already present when the chain starts (a graph's input contract).
        where: Location prefix for the message — a file, a graph name — so a refusal is located.
    """
    prefix = f"{where}: " if where else ""
    available = {str(key) for key in provided}
    reporters: Dict[str, str] = {}
    raised: Dict[str, str] = {}
    # Who writes what, over the WHOLE chain — so an unmet need can say whether the key is
    # simply absent or merely produced too late, which are different mistakes.
    written: Dict[str, str] = {}
    for op in ops:
        for key in _declared(op, "produces", {}):
            written.setdefault(str(key), _node_name(op))

    for op in ops:
        name = _node_name(op)
        for key in _declared(op, "consumes", {}):
            if str(key) not in available:
                later = written.get(str(key))
                where_from = (
                    f"{later} produces it, but LATER in the chain — move it before {name}"
                    if later
                    else f"the chain has {', '.join(sorted(available)) or 'nothing'} at that point "
                    "(an op that DOES write it must declare it in `produces`)"
                )
                raise ChainContractError(
                    f"{prefix}{name} needs the record entry {str(key)!r}, which nothing before it "
                    f"produces — {where_from}"
                )
        requires = str(getattr(op, "requires", "") or "")
        if requires and requires not in raised:
            known = ", ".join(sorted(raised)) or "none"
            raise ChainContractError(
                f"{prefix}{name} is gated on the flag {requires!r}, which no node before it "
                f"raises — the flags available at that point are: {known}"
            )
        for flag in _declared(op, "flags", ()):
            flag = str(flag)
            if flag in raised:
                raise ChainContractError(
                    f"{prefix}the flag {flag!r} is declared by both {raised[flag]} and {name} — "
                    "a gate naming it could not say which node decided it, so declare it once"
                )
            raised[flag] = name
        reports = str(_declared(op, "reports", ""))
        if reports:
            if reports in reporters:
                raise ChainContractError(
                    f"{prefix}both {reporters[reports]} and {name} report under {reports!r} — the "
                    "second finding would overwrite the first, so give each node its own name"
                )
            reporters[reports] = name
        for key in _declared(op, "produces", {}):
            available.add(str(key))
