"""Structure ops for the typed bag — reshape a :class:`~sampleflux.bag.sample.Sample`'s fields.

The typed analogue of the classic triple-slot plumbing (``MetadataToTargetOp``, the stash/swap
family): where the old model moved values between the fixed ``input``/``target`` slots and the
shared metadata dict, the bag model just RENAMES, RETAGS, COPIES, or DROPS named fields. Each op
is a thin copy-on-write wrapper over a ``Sample`` mutator — no payload is touched.

All ops are lazy / zero-arg constructible (config validated in ``__call__``) and
``@configurable(category="op", group="structure")`` so they surface as canvas nodes.
"""

from typing import List, Optional

from confluid import configurable
from typing_extensions import get_args

from sampleflux.bag.sample import ROLES, Role, Sample

__all__ = ["SetRole", "RenameField", "DropField", "CopyField", "SelectFields"]


@configurable(category="op", group="structure")
class SetRole:
    """Retag a field's role (``input`` / ``target`` / ``aux`` / ``pred``) without moving it.

    The typed replacement for the classic "metadata value becomes the target" op — in the bag
    model a field's role is a tag, so promotion is a retag, not a move.

    Args:
        key: The field to retag.
        role: The new role — one of ``input`` / ``target`` / ``aux`` / ``pred``.
    """

    def __init__(self, key: str = "", role: Role = "input") -> None:
        self.key = key
        self.role = role

    def __call__(self, sample: Sample) -> Sample:
        if not self.key:
            raise ValueError("SetRole: 'key' (the field to retag) is required")
        if self.role not in get_args(Role):
            raise ValueError(f"SetRole: invalid role {self.role!r} (allowed: {list(ROLES)})")
        return sample.set_role(self.key, self.role)


@configurable(category="op", group="structure")
class RenameField:
    """Rename a field (role travels with it). Renaming onto an existing key replaces it.

    The sanctioned way to avoid a deliberate fan-in collision: rename on the producing branch
    BEFORE the merge, instead of a merge-policy knob.

    Args:
        src: The field to rename.
        dst: The new field name.
    """

    def __init__(self, src: str = "", dst: str = "") -> None:
        self.src = src
        self.dst = dst

    def __call__(self, sample: Sample) -> Sample:
        if not self.src or not self.dst:
            raise ValueError("RenameField: both 'src' and 'dst' are required")
        return sample.rename(self.src, self.dst)


@configurable(category="op", group="structure")
class DropField:
    """Remove a field from the bag (e.g. free a heavy Signal after its Spectrogram is derived).

    Args:
        key: The field to remove. Missing keys raise unless ``missing_ok``.
        missing_ok: Silently pass through when the field is absent (default False).
    """

    def __init__(self, key: str = "", missing_ok: bool = False) -> None:
        self.key = key
        self.missing_ok = missing_ok

    def __call__(self, sample: Sample) -> Sample:
        if not self.key:
            raise ValueError("DropField: 'key' (the field to remove) is required")
        if self.key not in sample:
            if self.missing_ok:
                return sample
            raise KeyError(f"DropField: unknown field {self.key!r} (fields: {list(sample.keys())})")
        return sample.drop(self.key)


@configurable(category="op", group="structure")
class CopyField:
    """Duplicate a field under a new name (same item object; items are treated as immutable).

    Args:
        src: The field to copy.
        dst: The name of the copy. An existing ``dst`` is replaced.
        role: Optional role for the copy; ``None`` keeps the source field's role.
    """

    def __init__(self, src: str = "", dst: str = "", role: Optional[Role] = None) -> None:
        self.src = src
        self.dst = dst
        self.role = role

    def __call__(self, sample: Sample) -> Sample:
        if not self.src or not self.dst:
            raise ValueError("CopyField: both 'src' and 'dst' are required")
        if self.src not in sample:
            raise KeyError(f"CopyField: unknown field {self.src!r} (fields: {list(sample.keys())})")
        out = sample.replace_field(self.dst, sample[self.src])
        return out.set_role(self.dst, self.role if self.role is not None else sample.role_of(self.src))


@configurable(category="op", group="structure")
class SelectFields:
    """Keep ONLY the named fields (order = the given order); everything else is dropped.

    Args:
        keys: The fields to keep. Unknown keys raise (a silent miss hides a typo).
    """

    def __init__(self, keys: Optional[List[str]] = None) -> None:
        self.keys = list(keys) if keys else []

    def __call__(self, sample: Sample) -> Sample:
        if not self.keys:
            raise ValueError("SelectFields: 'keys' (the fields to keep) is required")
        missing = [k for k in self.keys if k not in sample]
        if missing:
            raise KeyError(f"SelectFields: unknown fields {missing} (fields: {list(sample.keys())})")
        return Sample({k: sample[k] for k in self.keys}, {k: sample.role_of(k) for k in self.keys})
