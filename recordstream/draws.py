"""Draw a generator's settings at random — each value only among those the generator itself accepts.

A training set needs a different, valid example per record. Drawing every setting independently and discarding what
the generator refuses keeps the WRONG mix: on a radio-cell generator 145 of 400 random settings were valid, and of 993
drawn with one frame structure 4 survived, so the set silently became the other kind. Restating the generator's
rules in a sampler would give a second copy of them that drifts.

So the generator is the judge, asked one setting at a time. A list of draws runs top to bottom; each picks from its
own distribution, but only among the values the generator accepts given every draw above it — the settings not drawn
yet keep the template's values. "Accepts" means the setting's object, and every object above it up to the generator,
rebuild through their constructors (which check their annotations) and pass their ``check()`` when they have one.
Since every step keeps the whole object valid, the last one is valid too: there is no final rejection.

The four draws:

* :class:`Choice` — tests each value and picks among the accepted ones, by weight. Given no values it takes every
  value the setting's type allows: a closed ``Literal``, ``bool``, ``None`` for an optional setting, or an integer
  range of at most :data:`ENUMERATION_LIMIT` values.
* :class:`Uniform` — a number from ``low`` to ``high`` (default: the setting's own range), tried up to :data:`TRIES`
  times until one is accepted.
* :class:`Span` — for a list of bounded integers: a run of consecutive values covering a share of those the generator
  accepts one at a time (a share of the free ones, whatever the size of the space).
* :class:`Repeat` — grows a list one element at a time, each starting from its class's defaults and drawing its own
  settings with ``each``. When an element's FIRST draw finds no accepted value there is no room, and the list ends. A
  ``Repeat`` among ``each`` grows a list inside the element, as far as that element allows.

A ``field`` is a dotted path from the generator; a step ``name[i]`` is the i-th element (from 0) of an existing list
setting — ``exchange.transmissions[1].gap`` — which must be there: only a :class:`Repeat` adds elements.

Order matters, and it is the one thing to learn: a value can be refused because a setting drawn LATER still has its
default (a setting that inherits from its parent until set). Draw first what opens up the choices after it.

A draw no value of which is accepted raises :class:`DrawRefused`, naming the setting, the values, the draws before it
and the generator's own reason. A spec that cannot be read (a setting that does not exist, a ``Choice`` with no
values on an open setting) raises :class:`DrawSpecError`.
"""

import copy
import inspect
import math
import random
import re
import types
import typing
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from annotated_types import Ge, Gt, Interval, Le, Lt
from confluid import configurable
from typing_extensions import Annotated

#: How many values a :class:`Uniform` tries before it gives up.
TRIES = 100
#: The largest integer range a :class:`Choice` without values, or a :class:`Span`, tests value by value.
ENUMERATION_LIMIT = 1024
#: Element types a :class:`Repeat` cannot grow a list of: they have no settings for ``each`` to draw.
_PLAIN_VALUES = (int, float, complex, str, bytes, bool)

Weight = Annotated[float, Interval(ge=0.0)]
Share = Annotated[Tuple[float, float], Interval(ge=0.0, le=1.0)]
Count = Annotated[Tuple[int, int], Interval(ge=0)]

#: One step of a ``field``: a setting's name, or a list setting's name with an element's index (``lanes[1]``).
_STEP = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")


class _Element(NamedTuple):
    """A step a ``field`` writes as ``name[index]``: an element the list already has. A :class:`Repeat`'s step to the
    element it is adding is a plain ``(name, index)`` one past the end; this one never is."""

    name: str
    position: int


#: One step of a path to a setting: a setting's name, or (a list setting's name, an element's index).
Step = Union[str, Tuple[str, int]]
Path = Tuple[Step, ...]


class DrawSpecError(ValueError):
    """The list of draws cannot be read: a setting that does not exist, a draw that does not fit its setting."""


class DrawRefused(ValueError):
    """No value of a draw is accepted by the generator, given the draws before it."""


class Drawn(NamedTuple):
    """A drawn generator and what was drawn, in order: ``(setting path, value)``."""

    settings: Any
    log: List[Tuple[str, Any]]


# -- reading a setting's type ---------------------------------------------------------------------------------------


class _Shape(NamedTuple):
    """An annotation with ``Optional`` and ``Annotated`` taken off: the type, whether ``None`` is allowed, the range."""

    base: Any
    optional: bool
    low: Optional[float]
    high: Optional[float]
    low_open: bool
    high_open: bool


def _shape(annotation: Any) -> _Shape:
    optional, metadata = False, []
    while True:
        origin = typing.get_origin(annotation)
        if origin is Annotated:
            annotation, *extra = typing.get_args(annotation)
            metadata.extend(extra)
        elif origin in (Union, types.UnionType):
            members = [a for a in typing.get_args(annotation) if a is not type(None)]
            optional = optional or len(members) < len(typing.get_args(annotation))
            if len(members) != 1:
                break
            annotation = members[0]
        else:
            break
    low = high = None
    low_open = high_open = False
    for mark in metadata:
        for kind, attribute, opens in ((Ge, "ge", False), (Gt, "gt", True), (Le, "le", False), (Lt, "lt", True)):
            value = getattr(mark, attribute, None) if isinstance(mark, (Interval, kind)) else None
            if value is None:
                continue
            if attribute in ("ge", "gt"):
                low, low_open = value, opens
            else:
                high, high_open = value, opens
    return _Shape(annotation, optional, low, high, low_open, high_open)


def _integer_range(shape: _Shape) -> Optional[Tuple[int, int]]:
    """The inclusive integer range of an ``int`` setting with both bounds, else None."""
    if shape.base is not int or shape.low is None or shape.high is None:
        return None
    low = int(shape.low) + 1 if shape.low_open else math.ceil(shape.low)
    high = int(shape.high) - 1 if shape.high_open else math.floor(shape.high)
    return low, high


def _describe(annotation: Any) -> str:
    """A setting's type in words: ``float from 0 to 1``, ``Literal[4, 8, 16]``, ``optional ToyMode``."""
    shape = _shape(annotation)
    name = shape.base.__name__ if isinstance(shape.base, type) else repr(shape.base).replace("typing.", "")
    if shape.low is not None and shape.high is not None:
        name += f" from {shape.low:g} to {shape.high:g}"
    return ("optional " if shape.optional else "") + name


def _closed_values(annotation: Any) -> Optional[List[Any]]:
    """Every value a setting's type allows, when that is a short closed list; else None."""
    shape = _shape(annotation)
    values: Optional[List[Any]] = None
    if typing.get_origin(shape.base) is typing.Literal:
        values = list(typing.get_args(shape.base))
    elif shape.base is bool:
        values = [False, True]
    else:
        bounds = _integer_range(shape)
        if bounds is not None and bounds[1] - bounds[0] < ENUMERATION_LIMIT:
            values = list(range(bounds[0], bounds[1] + 1))
    if values is None:
        return None
    return values + ([None] if shape.optional else [])


def _list_item(annotation: Any) -> Optional[Any]:
    """The element annotation of a (possibly optional) list setting, else None."""
    base = _shape(annotation).base
    if typing.get_origin(base) not in (list, List):
        return None
    arguments = typing.get_args(base)
    return arguments[0] if arguments else None


# -- reading and rebuilding an object -------------------------------------------------------------------------------


#: Each class's settings, read once: a draw rebuilds objects many times and a signature is not free to compute.
_PARAMETERS: Dict[type, Dict[str, inspect.Parameter]] = {}


def _parameters(cls: type) -> Dict[str, inspect.Parameter]:
    """A class's settings: its constructor's named parameters (they stay in the signature — the workspace rule)."""
    if cls not in _PARAMETERS:
        variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _PARAMETERS[cls] = {n: p for n, p in inspect.signature(cls).parameters.items() if p.kind not in variadic}
    return _PARAMETERS[cls]


def _annotation(cls: type, name: str) -> Any:
    annotation = _parameters(cls)[name].annotation
    if isinstance(annotation, str):
        # A module written with postponed annotations hands the signature strings; resolve them where they live.
        annotation = typing.get_type_hints(getattr(cls, "__init__"), include_extras=True).get(name, annotation)
    return annotation


def _settings(obj: Any) -> Dict[str, Any]:
    """An object's current settings, read back from the attributes its constructor stored."""
    values = {}
    for name in _parameters(type(obj)):
        if not hasattr(obj, name):
            raise DrawSpecError(
                f"{type(obj).__name__} keeps no attribute for its setting {name!r}, so a draw cannot rebuild it"
            )
        values[name] = getattr(obj, name)
    return values


def _rebuild(obj: Any, name: str, value: Any) -> Any:
    """``obj`` with one setting changed, built through its constructor and checked — a refusal raises."""
    built = type(obj)(**{**_settings(obj), name: value})
    check = getattr(built, "check", None)
    if callable(check):
        check()
    return built


def _name(step: Step) -> str:
    return step if isinstance(step, str) else step[0]


def _element(owner: type, name: str, where: str) -> Any:
    """A new element of the list setting ``owner.name``, from its class's defaults."""
    item = _list_item(_annotation(owner, name))
    cls = _shape(item).base if item is not None else None
    if not isinstance(cls, type) or cls in _PLAIN_VALUES:
        raise DrawSpecError(
            f"{where}: Repeat needs a list setting of objects; it is {_describe(_annotation(owner, name))} — draw "
            "a list of integers with Span"
        )
    try:
        return cls()
    except (ValueError, TypeError) as error:
        raise DrawSpecError(f"{where}: Repeat builds each new element as {cls.__name__}() — {_reason(error)}") from None


def _existing(obj: Any, step: _Element, where: str) -> Any:
    """The element a ``name[index]`` step names, None when the list is None; refused when the setting is no list or
    the index is past its end."""
    name, index = step
    if _list_item(_annotation(type(obj), name)) is None:
        raise DrawSpecError(
            f"{where}: {name} is not a list; it is {_describe(_annotation(type(obj), name))} — an index [i] names an "
            "element of a list setting"
        )
    items = getattr(obj, name)
    if items is None:
        return None
    if index >= len(items):
        count = f"{len(items)} element{'' if len(items) == 1 else 's'}"
        raise DrawSpecError(f"{where}: {name} has {count}; [{index}] is past its end — only a Repeat adds elements")
    return items[index]


def _child(obj: Any, step: Step, where: str) -> Any:
    """The object one step down; one index past a list's end is a new element (what a Repeat is adding)."""
    if isinstance(step, str):
        return getattr(obj, step)
    if isinstance(step, _Element):
        return _existing(obj, step, where)
    name, index = step
    items = list(getattr(obj, name) or [])
    return items[index] if index < len(items) else _element(type(obj), name, where)


def _with(obj: Any, path: Path, value: Any, where: str) -> Any:
    """``obj`` with the setting at ``path`` set to ``value``, every object on the way rebuilt and so re-checked."""
    step, rest = path[0], path[1:]
    if isinstance(step, str):
        inner = value if not rest else _with(getattr(obj, step), rest, value, where)
        return _rebuild(obj, step, inner)
    name, index = step
    items = list(getattr(obj, name) or [])
    if index == len(items):
        items.append(_element(type(obj), name, where))
    items[index] = value if not rest else _with(items[index], rest, value, where)
    return _rebuild(obj, name, items)


def _attempt(root: Any, path: Path, value: Any, where: str) -> Tuple[Any, Optional[BaseException]]:
    """``(the rebuilt root, None)`` when the generator accepts ``value`` at ``path``, ``(None, its refusal)`` when it
    does not. A refusal of the spec itself is no answer about the value, so it is raised."""
    try:
        return _with(root, path, value, where), None
    except DrawSpecError:
        raise
    except (ValueError, TypeError) as error:
        return None, error


def _target(root: Any, path: Path, where: str) -> Optional[Tuple[type, str]]:
    """The class holding the path's last setting and that setting's name; None when an object on the way is None."""
    obj = root
    for depth, step in enumerate(path):
        name = _name(step)
        settings = _parameters(type(obj))
        if name not in settings:
            listed = ", ".join(n for n in settings if n != "keys")
            raise DrawSpecError(f"{where}: {type(obj).__name__} has no setting {name!r}; its settings are {listed}")
        if depth == len(path) - 1:
            if isinstance(step, _Element):
                _existing(obj, step, where)  # refuses a setting that is no list, or an index past its end
                if getattr(obj, name) is None:
                    return None  # the list is unset; an element that is None is drawn like any value
            return type(obj), name
        obj = _child(obj, step, where)
        if obj is None:
            return None
    raise DrawSpecError(f"{where}: the path is empty")  # pragma: no cover — a field always has a step


def _where(path: Path) -> str:
    return ".".join(step if isinstance(step, str) else f"{step[0]}[{step[1]}]" for step in path)


def _reason(error: Optional[BaseException]) -> str:
    """The generator's refusal in one line (a pydantic error's first problem rather than its whole report)."""
    if error is None:  # pragma: no cover — every caller tested at least one value before it asks
        return "nothing was tested"
    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            first = errors()[0]
        except (TypeError, IndexError):  # pragma: no cover — not a pydantic error after all
            return str(error)
        location = ".".join(str(part) for part in first.get("loc", ()))
        return f"{location}: {first.get('msg', error)}" if location else str(first.get("msg", error))
    return str(error)


def _short(value: Any) -> str:
    if isinstance(value, list) and len(value) > 6:
        return f"[{value[0]!r} … {value[-1]!r}] ({len(value)})"
    return repr(value)


def _earlier(log: Sequence[Tuple[str, Any]]) -> str:
    return ", ".join(f"{where}={_short(value)}" for where, value in log) or "nothing drawn"


class _Draw:
    """What every draw shares: the setting it draws, as a dotted path from the generator (or a Repeat's element)."""

    field: str

    def _path(self, base: Path) -> Tuple[Path, str]:
        kind = type(self).__name__
        if not self.field:
            raise DrawSpecError(
                f"{kind}: field is empty — name the setting to draw by its path from the generator, a.b.c"
            )
        names = tuple(self.field.split("."))
        if not all(names):
            raise DrawSpecError(f"{kind}: field {self.field!r} has an empty step — write it as a.b.c")
        steps: List[Step] = []
        for name in names:
            found = _STEP.match(name)
            if found is None:
                raise DrawSpecError(
                    f"{kind}: field {self.field!r} has a step {name!r} that is neither a name nor name[index] — write "
                    "it as a.b[0].c"
                )
            setting, index = found.groups()
            steps.append(setting if index is None else _Element(setting, int(index)))
        path = base + tuple(steps)
        return path, _where(path)

    def _draw(self, root: Any, base: Path, rng: random.Random, log: List[Tuple[str, Any]]) -> Any:
        raise NotImplementedError  # pragma: no cover


# -- the draws -------------------------------------------------------------------------------------------------------


@configurable
class Choice(_Draw):
    """Draw a setting from a list of values, by weight, among those the generator accepts.

    Args:
        field: The setting to draw: its dotted path from the generator (``config.mode``), or from a Repeat's element.
        values: The values to choose from. None = all the setting's type allows (a Literal, bool, None, ≤ 1024 ints).
        weights: One weight per value (0 = never). None = all equal.
    """

    def __init__(
        self, field: str = "", values: Optional[List[Any]] = None, weights: Optional[List[Weight]] = None
    ) -> None:
        # ``Any``: a value is whatever the drawn setting holds — a number, a word, None, an object.
        self.field = field
        self.values = values
        self.weights = weights

    def _draw(self, root: Any, base: Path, rng: random.Random, log: List[Tuple[str, Any]]) -> Any:
        path, where = self._path(base)
        located = _target(root, path, where)
        if located is None:
            return root
        owner, name = located
        values = self.values
        if values is None:
            values = _closed_values(_annotation(owner, name))
            if values is None:
                raise DrawSpecError(
                    f"{where}: Choice needs values — the setting is {_describe(_annotation(owner, name))}; draw it "
                    "with Uniform, or give values"
                )
        weights = [1.0] * len(values) if self.weights is None else list(self.weights)
        if len(weights) != len(values):
            raise DrawSpecError(f"{where}: {len(values)} values and {len(weights)} weights — give one weight per value")
        accepted: List[Tuple[Any, Any, float]] = []
        reason: Optional[BaseException] = None
        for value, weight in zip(values, weights):
            if weight <= 0:
                continue
            built, refusal = _attempt(root, path, copy.deepcopy(value), where)
            if refusal is None:
                accepted.append((built, value, weight))
            else:
                reason = refusal
        if not accepted:
            because = "every weight is 0" if reason is None else _reason(reason)
            raise DrawRefused(f"{where}: none of {list(values)!r} is accepted after {_earlier(log)} — {because}")
        built, value, _ = rng.choices(accepted, weights=[weight for _, _, weight in accepted])[0]
        log.append((where, value))
        return built


@configurable
class Uniform(_Draw):
    """Draw a number setting uniformly from a range, trying values until the generator accepts one.

    Args:
        field: The setting to draw: its dotted path from the generator (``config.gain``), or from a Repeat's element.
        low: The smallest value. None = the setting's own lower bound.
        high: The largest value. None = the setting's own upper bound.
    """

    def __init__(self, field: str = "", low: Optional[float] = None, high: Optional[float] = None) -> None:
        self.field = field
        self.low = low
        self.high = high

    def _draw(self, root: Any, base: Path, rng: random.Random, log: List[Tuple[str, Any]]) -> Any:
        path, where = self._path(base)
        located = _target(root, path, where)
        if located is None:
            return root
        annotation = _annotation(*located)
        shape = _shape(annotation)
        if shape.base not in (int, float):
            raise DrawSpecError(
                f"{where}: Uniform needs a number setting (int or float); it is {_describe(annotation)} — draw it "
                "with Choice"
            )
        own = _integer_range(shape) if shape.base is int else (shape.low, shape.high)
        low = self.low if self.low is not None else (own[0] if own else None)
        high = self.high if self.high is not None else (own[1] if own else None)
        if low is None or high is None:
            raise DrawSpecError(f"{where}: Uniform needs low and high — the setting ({_describe(annotation)}) has none")
        if low > high:
            raise DrawSpecError(f"{where}: Uniform's low {low:g} is above its high {high:g}")
        reason: Optional[BaseException] = None
        for _ in range(TRIES):
            value: float = (
                rng.randint(math.ceil(low), math.floor(high)) if shape.base is int else rng.uniform(low, high)
            )
            built, refusal = _attempt(root, path, value, where)
            if refusal is None:
                log.append((where, value))
                return built
            reason = refusal
        raise DrawRefused(
            f"{where}: {TRIES} values from {low:g} to {high:g} refused after {_earlier(log)} — {_reason(reason)}"
        )


@configurable
class Span(_Draw):
    """Draw a run of consecutive integers for a list setting, covering a share of the values the generator accepts.

    Args:
        field: The list setting to draw: its dotted path from the generator, or from a Repeat's element.
        share: The run's length, drawn from this range, as a share of the values accepted one at a time (at least 1).
    """

    def __init__(self, field: str = "", share: Share = (0.0, 1.0)) -> None:
        self.field = field
        self.share = share

    def _draw(self, root: Any, base: Path, rng: random.Random, log: List[Tuple[str, Any]]) -> Any:
        path, where = self._path(base)
        located = _target(root, path, where)
        if located is None:
            return root
        annotation = _annotation(*located)
        item = _list_item(annotation)
        bounds = _integer_range(_shape(item)) if item is not None else None
        if bounds is None or bounds[1] - bounds[0] >= ENUMERATION_LIMIT:
            raise DrawSpecError(
                f"{where}: Span needs a list of integers with a range of at most {ENUMERATION_LIMIT}; it is "
                f"{_describe(annotation)}"
            )
        smallest, largest = self.share
        if smallest > largest:
            raise DrawSpecError(f"{where}: Span's share {self.share} runs backwards — write it smallest first")
        singles: List[int] = []
        reason: Optional[BaseException] = None
        for value in range(bounds[0], bounds[1] + 1):
            _, refusal = _attempt(root, path, [value], where)
            if refusal is None:
                singles.append(value)
            else:
                reason = refusal
        if not singles:
            raise DrawRefused(
                f"{where}: no value from {bounds[0]} to {bounds[1]} is accepted after {_earlier(log)} — "
                f"{_reason(reason)}"
            )
        size = max(1, round(rng.uniform(smallest, largest) * len(singles)))
        free = set(singles)
        starts = [v for v in singles if all(v + k in free for k in range(size))]
        rng.shuffle(starts)
        for start in starts:
            run = list(range(start, start + size))
            built, refusal = _attempt(root, path, run, where)
            if refusal is None:
                log.append((where, run))
                return built
            reason = refusal
        because = _reason(reason) if reason is not None else f"no {size} consecutive ones among them"
        raise DrawRefused(
            f"{where}: no run of {size} of the {len(singles)} accepted values is accepted after {_earlier(log)} — "
            f"{because}"
        )


@configurable
class Repeat(_Draw):
    """Grow a list setting one element at a time, each element drawing its own settings, while there is room.

    Args:
        field: The list setting to grow: its dotted path from the generator (``config.items``).
        count: How many elements to add, drawn from this range (ends included); fewer when the next one has no room.
        each: The draws of each new element, paths from it — a Repeat among them grows a list inside the element.
            None = defaults while they fit.
    """

    def __init__(self, field: str = "", count: Count = (0, 1), each: Optional[List["Draw"]] = None) -> None:
        self.field = field
        self.count = count
        self.each = each

    def _draw(self, root: Any, base: Path, rng: random.Random, log: List[Tuple[str, Any]]) -> Any:
        path, where = self._path(base)
        located = _target(root, path, where)
        if located is None:
            return root
        owner, name = located
        if _list_item(_annotation(owner, name)) is None:
            raise DrawSpecError(f"{where}: Repeat needs a list setting; it is {_describe(_annotation(owner, name))}")
        smallest, largest = self.count
        if smallest > largest:
            raise DrawSpecError(f"{where}: Repeat's count {self.count} runs backwards — write it smallest first")
        wanted = rng.randint(smallest, largest)
        parent: Path = path[:-1]
        made = 0
        for _ in range(wanted):
            holder = root
            for step in parent:
                holder = _child(holder, step, where)
            index = len(getattr(holder, name) or [])
            element: Path = parent + ((name, index),)
            if not self.each:
                grown, refusal = _attempt(root, element, _element(owner, name, where), where)
                if refusal is not None:
                    break  # the default element does not fit
                root = grown
            else:
                first, *others = self.each
                try:
                    grown = first._draw(root, element, rng, log)
                except DrawRefused:
                    break  # no room for another element
                if grown is root:
                    break  # the first draw was skipped: nothing placed the element
                for later in others:
                    grown = later._draw(grown, element, rng, log)
                root = grown
            made += 1
        log.append((where, f"{made} of {wanted}"))
        return root


#: Any one draw — at the top of a list of draws or among a Repeat's ``each``.
Draw = Union[Choice, Uniform, Span, Repeat]


def draw_settings(template: Any, draws: Optional[Sequence[Draw]], rng: random.Random) -> Drawn:
    """Run ``draws`` top to bottom over ``template`` (never changed) and return the drawn object with its log."""
    if template is None:
        raise DrawSpecError("draw_settings: there is no template to draw from — give the generator")
    log: List[Tuple[str, Any]] = []
    settings = template
    for index, draw in enumerate(draws or ()):
        if not isinstance(draw, _Draw):
            raise DrawSpecError(
                f"draws[{index}] is a {type(draw).__name__} — a draw is a Choice, Uniform, Span or Repeat"
            )
        settings = draw._draw(settings, (), rng, log)
    return Drawn(settings, log)


__all__ = [
    "Choice",
    "Draw",
    "DrawRefused",
    "DrawSpecError",
    "Drawn",
    "ENUMERATION_LIMIT",
    "Repeat",
    "Span",
    "TRIES",
    "Uniform",
    "draw_settings",
]
