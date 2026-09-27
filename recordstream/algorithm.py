"""Algorithms: declare the settings, inputs and outputs once — the recordstream op is derived.

An algorithm states three kinds of slot and computes::

    class BackgroundLevel(Algorithm):
        percentile: float = Param(default=25.0, doc="Percentile rank across the per-row medians.")
        image: Image = Input(doc="The image to read.")
        background: float = Output(doc="The background level.")

        def compute(self):
            per_row = np.median(np.asarray(self.image), axis=1)
            return {"background": float(np.percentile(per_row, self.percentile))}

* a **param** is a setting — a keyword constructor argument, so config, a visual editor and a
  generated tool schema see it exactly as they see any other ``@configurable`` parameter;
* an **input** is read from the record entry of the same name when the algorithm runs as an op;
* an **output** is written to the record entry of the same name (or, with ``replaces=``, back into
  the entry an input was read from).

The algorithm never sees a record. Reading the inputs out of one, checking their item type,
running ``compute`` and writing the outputs back is done ONCE, here, for every algorithm — the
part each hand-written "read named entries, write named entries" op used to repeat (a ``field`` /
``<input>_field`` / ``output`` parameter per slot, a ``resolve_item`` call, a hand-kept
``consumes`` / ``produces`` declaration, a private ``_last_*`` mirror behind an ``@output``
property). Everything a tool needs is DERIVED from the three declarations instead of written a
second time: the constructor (and so ``to_pydantic``), the ``Args:`` block, the confluid
``@output`` sockets, the chain checker's ``consumes`` / ``produces`` and :func:`algorithm_spec`.

Why the slots are spelled ``name: type = Param(...)`` and not ``name: Param[type] = default``:
the class is marked :func:`typing.dataclass_transform`, so a type checker synthesises the
constructor from the slots — and only the field-specifier spelling can tell it that an input or
an output is NOT a constructor argument (``init=False``). With annotation markers, mypy demanded
the inputs as constructor arguments and refused the correct call ``BackgroundLevel(percentile=30.0)``
(measured). Rationale and the rejected alternatives: ``docs/architecture.md`` (the Algorithm record);
usage: ``docs/algorithm.md``.
"""

import copy
import inspect
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from confluid import output as confluid_output

from recordstream.items import Record, item_types
from recordstream.ops.contract import ANY_TYPE

T = TypeVar("T")

#: The three kinds of slot — a closed set, so a tool can enumerate it.
SlotRole = Literal["param", "input", "output"]

# "No default was given" — distinct from ``None``, which is a legitimate default.
_NO_DEFAULT: Any = object()


@dataclass(frozen=True)
class AlgorithmSlot:
    """One declared slot of an :class:`Algorithm` — what :func:`algorithm_spec` reports.

    Args:
        name: The slot's name — the constructor argument (param) or the default record entry (input/output).
        role: Which kind of slot it is: ``param``, ``input`` or ``output``.
        annotation: The declared type, as written (an ``Annotated`` range mark on a param is kept).
        doc: The one-line description given with ``doc=``.
        required: A param or input that has no default must be given.
        default: The default of a param or an optional input (``None`` when there is none).
        replaces: For an output, the input whose record entry it is written back into; blank = its own name.
    """

    name: str
    role: SlotRole
    # ``Any``: the declared type is whatever the author wrote — an item class, a builtin, an
    # ``Annotated``/``Optional`` form — and is only read back, never type-checked against.
    annotation: Any = Any
    doc: str = ""
    required: bool = False
    default: Any = None
    replaces: str = ""


class AlgorithmSpec(NamedTuple):
    """An algorithm's interface, in declaration order (a base class's slots first)."""

    params: Tuple[AlgorithmSlot, ...]
    inputs: Tuple[AlgorithmSlot, ...]
    outputs: Tuple[AlgorithmSlot, ...]


# The three field specifiers. Each returns an AlgorithmSlot that ``Algorithm.__init_subclass__``
# collects and removes from the class. The ``init`` parameter does nothing at runtime: it is what
# ``dataclass_transform`` tells a type checker — a param IS a constructor argument, an input and an
# output are NOT — so it must stay keyword-only with a ``Literal`` default.


@overload
def Param(*, default: T, doc: str = "", init: Literal[True] = True) -> T: ...


@overload
def Param(*, doc: str = "", init: Literal[True] = True) -> Any: ...


def Param(*, default: Any = _NO_DEFAULT, doc: str = "", init: Literal[True] = True) -> Any:
    """Declare a setting: a keyword constructor argument, set in config, in an editor or by a tool.

    Args:
        default: The value when the argument is not given. Omit it to make the setting required.
        doc: One line describing the setting — it becomes the constructor's ``Args:`` entry.
        init: Always ``True``; tells a type checker this slot is a constructor argument.
    """
    required = default is _NO_DEFAULT
    return AlgorithmSlot(name="", role="param", doc=doc, required=required, default=None if required else default)


def Input(*, default: Any = _NO_DEFAULT, doc: str = "", init: Literal[False] = False) -> Any:
    """Declare an input: a value ``compute`` reads as ``self.<name>``, taken from the record entry of that name.

    Args:
        default: Makes the input optional — the value used when the record does not carry the entry.
        doc: One line describing the input.
        init: Always ``False``; tells a type checker an input is not a constructor argument.
    """
    required = default is _NO_DEFAULT
    return AlgorithmSlot(name="", role="input", doc=doc, required=required, default=None if required else default)


def Output(*, replaces: str = "", doc: str = "", init: Literal[False] = False) -> Any:
    """Declare an output: a value ``compute`` returns by name, written to the record entry of that name.

    Args:
        replaces: The name of an INPUT: the output is written back into the record entry that input was
            read from (following ``keys``), for an algorithm whose job is to replace what it read.
        doc: One line describing the output.
        init: Always ``False``; tells a type checker an output is not a constructor argument.
    """
    return AlgorithmSlot(name="", role="output", doc=doc, replaces=replaces)


@dataclass_transform(kw_only_default=True, field_specifiers=(Param, Input, Output))
class _DeclaresSlots:
    """The type checker's anchor: its SUBCLASSES get a constructor synthesised from their slots.

    A separate class because a type checker collects fields only from classes DERIVED from the
    decorated one — the ``keys`` argument declared on :class:`Algorithm` below would otherwise be
    invisible to it, and every correct ``SomeAlgorithm(keys={...})`` would be reported as an error.
    """


class Algorithm(_DeclaresSlots):
    """Base class: declare ``Param`` / ``Input`` / ``Output`` slots, then write ``compute``.

    ``compute(self)`` reads the settings and the inputs as ``self.<name>`` and returns a mapping
    holding every output by name. The rest is provided: the keyword constructor (one argument per
    param, plus ``keys``), ``run(**inputs)`` for a standalone call, ``__call__(record)`` for use as a
    recordstream op, one read-only ``@output`` property per output holding the last computed value,
    and the chain checker's ``consumes`` / ``produces``. Do not write ``__init__``; it is generated.
    """

    #: Every slot, inherited ones included, by name — set per subclass by ``__init_subclass__``.
    __algorithm_slots__: ClassVar[Dict[str, AlgorithmSlot]] = {}

    if TYPE_CHECKING:
        # What the generated constructor accepts on top of the params, told to the type checker only:
        # at runtime ``keys`` is the generated constructor's own argument, not a slot, so it never
        # appears in :func:`algorithm_spec`.
        keys: Optional[Dict[str, str]] = Param(default=None)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if "__init__" in vars(cls):
            raise TypeError(
                f"{name}: an Algorithm's constructor is generated from its Param slots — declare each "
                "setting as `name: type = Param(default=...)` instead of writing __init__"
            )
        own = {key: value for key, value in vars(cls).items() if isinstance(value, AlgorithmSlot)}
        taken = sorted(set(own) & _RESERVED)
        if taken:
            raise TypeError(f"{name}: the slot name(s) {taken} are taken by Algorithm itself; rename the slot")
        hints = _type_hints(cls)
        slots: Dict[str, AlgorithmSlot] = {}
        for base in reversed(cls.__mro__[1:]):
            slots.update(vars(base).get("__algorithm_slots__", {}))
        for key, slot in own.items():
            slots[key] = replace(slot, name=key, annotation=hints.get(key, Any))
        inputs = sorted(key for key, slot in slots.items() if slot.role == "input")
        for slot in slots.values():
            if slot.role == "output" and slot.replaces and slot.replaces not in inputs:
                raise TypeError(
                    f"{name}: the output {slot.name!r} replaces {slot.replaces!r}, which is not an input; "
                    f"the inputs are {inputs}"
                )
        cls.__algorithm_slots__ = slots
        for key in own:
            delattr(cls, key)  # the instance attribute (param) or per-run value (input) is the truth
        for slot in (slots[key] for key in own):
            if slot.role == "output":
                setattr(cls, slot.name, _output_property(cls, slot))
        # Generate the constructor for the first algorithm class, and again only when a NEW param
        # changes it: regenerating it otherwise would replace a parent's constructor that confluid's
        # ``@configurable`` has already wrapped with validation and capture (``functools.wraps`` copies
        # the marker onto that wrapper, so the check sees through it).
        inherited = getattr(cls.__init__, "__algorithm_generated__", False)
        if not inherited or any(slot.role == "param" for slot in own.values()):
            setattr(cls, "__init__", _generated_init(cls, [s for s in slots.values() if s.role == "param"]))

    # -- standalone ----------------------------------------------------------------------------

    def run(self, **inputs: Any) -> Dict[str, Any]:
        """Compute from the given inputs and return every output by name — no record involved.

        The inputs are set on a shallow COPY of this object, so the configured instance never holds
        a record's values between calls; the result is kept for the ``@output`` properties.
        """
        name = type(self).__name__
        spec = algorithm_spec(self)
        declared = [slot.name for slot in spec.inputs]
        unknown = sorted(set(inputs) - set(declared))
        if unknown:
            raise TypeError(f"{name}.run(): unknown input(s) {unknown}; inputs are {sorted(declared)}")
        work = copy.copy(self)
        for slot in spec.inputs:
            if slot.name in inputs:
                value = inputs[slot.name]
            elif slot.required:
                raise TypeError(f"{name}.run(): missing input {slot.name!r}")
            else:
                value = slot.default
            setattr(work, slot.name, value)
        result = work.compute()
        wanted = [slot.name for slot in spec.outputs]
        if not isinstance(result, Mapping) or set(result) != set(wanted):
            got = sorted(result) if isinstance(result, Mapping) else type(result).__name__
            raise TypeError(f"{name}.compute() returned {got}; it must return exactly {sorted(wanted)}")
        outputs = {key: result[key] for key in wanted}
        self._last_outputs = outputs
        return outputs

    def compute(self) -> Mapping[str, Any]:
        """Read the settings and inputs as ``self.<name>``; return every output by name."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute() — read the settings and inputs as "
            "self.<name> and return every output by name"
        )

    # -- as a recordstream op ------------------------------------------------------------------

    def __call__(self, record: Record) -> Record:
        """Read the inputs from ``record``, compute, and return the record with the outputs added."""
        name = type(self).__name__
        spec = algorithm_spec(self)
        entries = self._entries(spec)
        inputs: Dict[str, Any] = {}
        for slot in spec.inputs:
            entry = entries[slot.name]
            if entry not in record:
                if not slot.required:
                    continue
                hint = f" — if it is stored under another name, set keys: {{{slot.name}: <entry>}}"
                raise ValueError(
                    f"{name} needs the input {slot.name!r} ({_type_label(slot.annotation)}) from the record entry "
                    f"{entry!r}, which this record does not carry (it has: {', '.join(map(str, record)) or 'nothing'})"
                    + (hint if entry == slot.name else "")
                )
            value = record[entry]
            wanted = _item_class(slot.annotation)
            if wanted is not None and value is not None and not isinstance(value, wanted):
                raise ValueError(
                    f"{name}: the input {slot.name!r} must be of type {wanted.__name__}, but the record entry "
                    f"{entry!r} is of type {type(value).__name__}"
                )
            inputs[slot.name] = value
        outputs = self.run(**inputs)
        return {**record, **{entries[key]: value for key, value in outputs.items()}}

    def _entries(self, spec: AlgorithmSpec) -> Dict[str, str]:
        """``{slot name: record entry}`` for every input and output, after ``keys`` and ``replaces``."""
        name = type(self).__name__
        given = self.keys if self.keys is not None else {}
        if not isinstance(given, Mapping):
            raise ValueError(
                f"{name}: keys must map an input or output name to a record entry; got a value of type "
                f"{type(given).__name__}"
            )
        slots = sorted(slot.name for slot in spec.inputs + spec.outputs)
        unknown = sorted(set(given) - set(slots))
        if unknown:
            raise ValueError(f"{name}: keys names {unknown}, which are not inputs or outputs; those are {slots}")
        entries = {slot.name: str(given.get(slot.name, slot.name)) for slot in spec.inputs}
        for slot in spec.outputs:
            fallback = entries[slot.replaces] if slot.replaces else slot.name
            entries[slot.name] = str(given.get(slot.name, fallback))
        return entries

    @property
    def consumes(self) -> Dict[str, str]:
        """``{record entry: item type name}`` of the REQUIRED inputs — what the chain checker reads."""
        spec = algorithm_spec(self)
        entries = self._entries(spec)
        return {entries[slot.name]: _contract_type(slot.annotation) for slot in spec.inputs if slot.required}

    @property
    def produces(self) -> Dict[str, str]:
        """``{record entry: item type name}`` of the outputs — what a later op may rely on finding."""
        spec = algorithm_spec(self)
        entries = self._entries(spec)
        return {entries[slot.name]: _contract_type(slot.annotation) for slot in spec.outputs}


#: Names a slot may not take: everything public on the base class, plus the generated ``keys``.
_RESERVED = frozenset(key for key in dir(Algorithm) if not key.startswith("_")) | {"keys"}

#: The ``keys`` constructor argument every algorithm gets — its annotation and ``Args:`` line.
_KEYS_ANNOTATION = Optional[Dict[str, str]]
_KEYS_DOC = "Record entry per input/output name; a name left out reads or writes the entry of the same name."


def algorithm_spec(algorithm: Union[type, Algorithm]) -> AlgorithmSpec:
    """The declared params, inputs and outputs of an algorithm class (or instance), in declaration order."""
    cls = algorithm if isinstance(algorithm, type) else type(algorithm)
    slots = list(getattr(cls, "__algorithm_slots__", {}).values())
    return AlgorithmSpec(
        params=tuple(slot for slot in slots if slot.role == "param"),
        inputs=tuple(slot for slot in slots if slot.role == "input"),
        outputs=tuple(slot for slot in slots if slot.role == "output"),
    )


def _generated_init(cls: type, params: List[AlgorithmSlot]) -> Any:
    """The keyword-only constructor: one argument per param, plus ``keys``.

    It carries a real ``__signature__``, ``__annotations__`` and an ``Args:`` docstring, because that
    is what ``inspect.signature`` / ``to_pydantic`` / ``parse_param_docs`` read — so confluid, a
    visual editor and a generated tool schema see an ordinary constructor.
    """
    allowed = {slot.name for slot in params} | {"keys"}

    def __init__(self: Algorithm, **kwargs: Any) -> None:
        owner = type(self).__name__
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"{owner}() got unknown parameter(s) {unknown}")
        for slot in params:
            if slot.name in kwargs:
                value = kwargs[slot.name]
            elif slot.required:
                raise TypeError(f"{owner}() missing parameter {slot.name!r}")
            else:
                value = copy.deepcopy(slot.default)  # a list default is never shared between instances
            setattr(self, slot.name, value)
        self.keys = kwargs.get("keys")
        self._last_outputs = {}

    kw = inspect.Parameter.KEYWORD_ONLY
    signature = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    annotations: Dict[str, Any] = {}
    for slot in params:
        default = inspect.Parameter.empty if slot.required else slot.default
        signature.append(inspect.Parameter(slot.name, kw, default=default, annotation=slot.annotation))
        annotations[slot.name] = slot.annotation
    signature.append(inspect.Parameter("keys", kw, default=None, annotation=_KEYS_ANNOTATION))
    annotations["keys"] = _KEYS_ANNOTATION
    annotations["return"] = None
    documented = [slot for slot in params if slot.doc]  # an empty entry would swallow the next one
    setattr(__init__, "__signature__", inspect.Signature(signature))
    __init__.__annotations__ = annotations
    __init__.__doc__ = (
        "Args:\n" + "".join(f"    {slot.name}: {slot.doc}\n" for slot in documented) + (f"    keys: {_KEYS_DOC}\n")
    )
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__module__ = cls.__module__
    setattr(__init__, "__algorithm_generated__", True)
    return __init__


def _output_property(cls: type, slot: AlgorithmSlot) -> property:
    """A read-only confluid ``@output`` property: the value the last run computed (``None`` before).

    Read-only on purpose: confluid treats a SETTABLE property as a configuration knob, and an output
    is never configured — which is also why ``compute`` returns its outputs instead of assigning them.
    """
    key = slot.name

    def getter(self: Algorithm) -> Any:
        return vars(self).get("_last_outputs", {}).get(key)

    getter.__name__ = key
    getter.__qualname__ = f"{cls.__qualname__}.{key}"
    getter.__doc__ = slot.doc or f"The last computed {key!r}."
    getter.__annotations__ = {"return": slot.annotation}
    return property(confluid_output(getter))


def _type_hints(cls: type) -> Dict[str, Any]:
    """The class's annotations, resolved (``Annotated`` range marks kept)."""
    try:
        return get_type_hints(cls, include_extras=True)
    except NameError:
        # A forward reference that cannot be resolved (a class local to a function under postponed
        # evaluation): fall back to the annotations as written rather than refusing the class.
        merged: Dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("__annotations__", {}))
        return merged


def _unwrapped(annotation: Any) -> Any:
    """``Annotated[X, ...]`` -> ``X``; ``Optional[X]`` -> ``X``; anything else unchanged."""
    if hasattr(annotation, "__metadata__"):
        annotation = annotation.__origin__
    if get_origin(annotation) is Union:
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return _unwrapped(members[0])
    return annotation


def _item_class(annotation: Any) -> Optional[type]:
    """The registered item class an input is declared as, or ``None`` for any other type.

    Only item types are checked on the way in: a ``float`` input legitimately receives a numpy
    ``float32`` from a record, and refusing it would make every numeric input brittle.
    """
    candidate = _unwrapped(annotation)
    if isinstance(candidate, type) and issubclass(candidate, item_types()):
        return candidate
    return None


def _contract_type(annotation: Any) -> str:
    """The chain checker's vocabulary: the registered item type name, else ``ANY_TYPE``."""
    candidate = _item_class(annotation)
    if candidate is None:
        return ANY_TYPE
    registered = item_types()
    return next(klass.__name__ for klass in candidate.__mro__ if klass in registered)


def _type_label(annotation: Any) -> str:
    candidate = _unwrapped(annotation)
    return candidate.__name__ if isinstance(candidate, type) else str(candidate)
