"""``Sample`` — the named bag of typed items that replaces ``Sample(input, target, metadata)``.

A sample is an ordered mapping ``name -> item`` (see :mod:`sampleflux.bag.items`), plus a
per-key ROLE tag. This gives every field BOTH a name (the key — the albumentations dispatch
axis) and a type (the item — the torchvision dispatch axis), and it makes ``input`` / ``target``
ordinary tags read only at the train / collate / sink boundary rather than fixed tuple
positions. A field can change role without moving keys; auxiliary items (masks, derived
params) are simply tagged ``aux`` and excluded from both ``inputs()`` and ``targets()``.

``Sample`` is immutable — every mutator returns a NEW sample (copy-on-write), mirroring
the ``Sample._replace`` idiom the legacy engine already relies on, so a transform never
aliases its input.
"""

from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
from typing_extensions import Literal, get_args

__all__ = ["Sample", "Role", "ROLES", "primary"]

#: The closed set of field roles. ``input`` / ``target`` drive the train boundary; ``aux`` is
#: a helper field (mask, derived param) in neither; ``pred`` is a model prediction. Closed
#: ``Literal`` so a typo fails at the call site and UIs enumerate the choices via ``get_args``.
Role = Literal["input", "target", "aux", "pred"]
ROLES: Tuple[str, ...] = get_args(Role)

_DEFAULT_ROLE: Role = "input"


class Sample:
    """An ordered, immutable bag of typed items with per-field role tags.

    Construct from a mapping of items (roles default to ``input``); pass ``roles`` to tag
    specific keys::

        s = Sample(
            {"image": Image(rgb), "regions": Regions(boxes), "class": Label("drone")},
            roles={"regions": "target", "class": "target"},
        )
        s.inputs()   # {"image": Image(...)}
        s.targets()  # {"regions": Regions(...), "class": Label(...)}
        s2 = s.set_role("regions", "aux")      # copy-on-write
    """

    __slots__ = ("_fields", "_roles")

    def __init__(
        self,
        fields: Optional[Mapping[str, Any]] = None,
        roles: Optional[Mapping[str, Role]] = None,
    ) -> None:
        self._fields: Dict[str, Any] = dict(fields or {})
        roles = roles or {}
        for key, role in roles.items():
            if key not in self._fields:
                raise KeyError(f"Sample: role given for unknown field {key!r}")
            if role not in ROLES:
                raise ValueError(f"Sample: invalid role {role!r} for {key!r} (allowed: {list(ROLES)})")
        self._roles: Dict[str, Role] = {key: roles.get(key, _DEFAULT_ROLE) for key in self._fields}

    # --- read views -------------------------------------------------------
    @property
    def fields(self) -> Dict[str, Any]:
        """A shallow copy of the ``name -> item`` mapping (mutating it does not touch the sample)."""
        return dict(self._fields)

    @property
    def roles(self) -> Dict[str, Role]:
        """A shallow copy of the ``name -> role`` mapping."""
        return dict(self._roles)

    def __getitem__(self, key: str) -> Any:
        return self._fields[key]

    def __contains__(self, key: object) -> bool:
        return key in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def keys(self) -> Iterator[str]:
        return iter(self._fields)

    def items(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._fields.items())

    def role_of(self, key: str) -> Role:
        """The role tag of ``key``."""
        return self._roles[key]

    def of_role(self, role: Role) -> Dict[str, Any]:
        """The ``name -> item`` fields tagged ``role`` (insertion order preserved)."""
        return {key: item for key, item in self._fields.items() if self._roles[key] == role}

    def inputs(self) -> Dict[str, Any]:
        """The fields tagged ``input`` — what the model consumes."""
        return self.of_role("input")

    def targets(self) -> Dict[str, Any]:
        """The fields tagged ``target`` — what the loss consumes."""
        return self.of_role("target")

    def aux(self) -> Dict[str, Any]:
        """The fields tagged ``aux`` — helpers in neither inputs nor targets."""
        return self.of_role("aux")

    def items_of_type(self, *types: type) -> Iterator[Tuple[str, Any]]:
        """Yield ``(key, item)`` for every field whose item is an instance of one of ``types``."""
        for key, item in self._fields.items():
            if isinstance(item, types):
                yield key, item

    # --- copy-on-write mutators ------------------------------------------
    def replace_field(self, key: str, item: Any) -> "Sample":
        """A copy with ``key`` set to ``item`` (added if new; role preserved, else ``input``)."""
        fields = dict(self._fields)
        fields[key] = item
        return Sample(fields, {**self._roles, key: self._roles.get(key, _DEFAULT_ROLE)})

    def set_role(self, key: str, role: Role) -> "Sample":
        """A copy with ``key``'s role set to ``role``."""
        if key not in self._fields:
            raise KeyError(f"Sample.set_role: unknown field {key!r}")
        if role not in ROLES:
            raise ValueError(f"Sample.set_role: invalid role {role!r} (allowed: {list(ROLES)})")
        return Sample(dict(self._fields), {**self._roles, key: role})

    def drop(self, key: str) -> "Sample":
        """A copy without ``key``."""
        fields = dict(self._fields)
        roles = dict(self._roles)
        fields.pop(key, None)
        roles.pop(key, None)
        return Sample(fields, roles)

    def rename(self, src: str, dst: str) -> "Sample":
        """A copy with field ``src`` renamed to ``dst`` (role travels; position moves to the end).

        Renaming onto an existing ``dst`` replaces it (last-write-wins, consistent with
        :meth:`merge`). Unknown ``src`` raises.
        """
        if src not in self._fields:
            raise KeyError(f"Sample.rename: unknown field {src!r}")
        fields = dict(self._fields)
        roles = dict(self._roles)
        item = fields.pop(src)
        role = roles.pop(src)
        fields.pop(dst, None)
        roles.pop(dst, None)
        fields[dst] = item
        roles[dst] = role
        return Sample(fields, roles)

    # --- fan-in ------------------------------------------------------------
    @classmethod
    def merge(cls, *samples: "Sample") -> "Sample":
        """The ordered UNION of several samples' fields — the typed fan-in primitive.

        Fields AND their roles are united in listed order; on a key collision the
        LAST-listed sample wins (value and role) — the deterministic slot-order rule that
        replaces the classic metadata dict-merge. Avoid a deliberate collision by renaming
        on the producing branch (:meth:`rename` / the ``RenameField`` op), not with merge
        policy knobs.
        """
        fields: Dict[str, Any] = {}
        roles: Dict[str, Role] = {}
        for sample in samples:
            if not isinstance(sample, Sample):
                raise TypeError(f"Sample.merge: expected Sample, got {type(sample).__name__}")
            fields.update(sample._fields)
            roles.update(sample._roles)
        return cls(fields, roles)

    # --- equality / repr --------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sample):
            return NotImplemented
        if self._roles != other._roles or list(self._fields) != list(other._fields):
            return False
        return all(_field_equal(self._fields[k], other._fields[k]) for k in self._fields)

    def __repr__(self) -> str:
        parts = ", ".join(f"{key}={type(item).__name__}[{self._roles[key]}]" for key, item in self._fields.items())
        return f"Sample({parts})"


def primary(sample: Sample, role: Role = "input") -> Tuple[str, Any]:
    """The FIRST field of ``role`` in insertion order, as ``(key, item)``.

    The sanctioned answer to "the input" / "the target" of a bag: engines, ``bind``, and
    ``Apply``-style parameter injection use it when no explicit field key is given. Raises
    ``KeyError`` (naming the sample's fields) when no field carries the role.
    """
    for key, item in sample.items():
        if sample.role_of(key) == role:
            return key, item
    raise KeyError(f"primary: no field with role {role!r} (fields: {list(sample.keys()) or '<empty>'})")


def _field_equal(a: Any, b: Any) -> bool:
    """Value equality that is robust to array-valued items (elementwise ``==`` is not a bool)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return type(a) is type(b) and np.array_equal(np.asarray(a), np.asarray(b))
    try:
        return bool(a == b)
    except Exception:  # pragma: no cover - exotic payloads fall back to identity
        return a is b
