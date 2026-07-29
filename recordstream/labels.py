"""``LabelMap`` — a bidirectional class-name ↔ integer-id map.

The *fittable* companion to the config-pinned :class:`~recordstream.ops.target.EncodeTarget` /
:class:`~recordstream.ops.target.DecodeTarget`. Those ops carry an explicit ``mapping`` that is
**pinned in config, NOT fitted** at run time, so train / eval / predict share one identical
label→id ordering. :class:`LabelMap` is the piece that *produces* such a pinned mapping:

* :meth:`LabelMap.fit` derives a deterministic name→id mapping from a stream of raw targets
  (backed by scikit-learn's ``LabelEncoder``) — the one-time fit that happens at **train** time.
* :meth:`LabelMap.save` / :meth:`LabelMap.load` persist it (in marainer's ``class_names.json``
  format) so **eval / predict** reload the *same* mapping rather than refitting on a subset.
* :meth:`LabelMap.encode_op` / :meth:`LabelMap.decode_op` hand back the recordstream ops that apply it.

So fitting happens once, then the mapping is pinned/persisted — it does NOT contradict the
"mapping pinned in config, not fitted" discipline of the ops; it is how the pin gets created.

Zero-arg constructible (``LabelMap()`` succeeds with an empty mapping) and side-effect-free in
``__init__`` per the workspace "Lazy Initialization & Zero-Arg Construction" convention; the
non-empty requirement is validated lazily in the properties, not in the constructor.

A label is ALWAYS mappable to ids: :meth:`LabelMap.to_ids` accepts a ``Label`` / ``MultiLabel``
item, a bare name or id, or a sequence of those, and passes already-encoded values through — so a
consumer never branches on "are these names or ids?".
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

from confluid import configurable

from recordstream.items import Label, MultiLabel, is_class_id
from recordstream.ops.target import DecodeTarget, EncodeTarget


def _iter_label_values(target: Any) -> Iterator[Any]:
    """Yield the individual label values of ``target``, whatever shape it takes.

    A :class:`~recordstream.MultiLabel` yields each of its values, a
    :class:`~recordstream.Label` its single value, a bare sequence its elements, and anything
    else itself. One walker so :meth:`LabelMap.fit` and :meth:`LabelMap.to_ids` agree on what
    "the labels of this target" means.
    """
    if isinstance(target, MultiLabel):
        yield from target.values
    elif isinstance(target, Label):
        yield target.value
    elif isinstance(target, (list, tuple, set)):
        yield from target
    elif target is not None:
        yield target


@configurable
class LabelMap:
    """Bidirectional class-name ↔ integer-id map (the fittable companion to ``EncodeTarget``).

    Holds an explicit name→id ``mapping`` (pinned in config), or one fitted from a target stream
    via :meth:`fit`. Exposes :attr:`num_classes` / :attr:`label_names`, builds the
    :class:`~recordstream.ops.target.EncodeTarget` / :class:`~recordstream.ops.target.DecodeTarget`
    that apply it, and round-trips to disk in marainer's ``class_names.json`` format.

    Args:
        mapping: Explicit name→id lookup, e.g. ``{"cat": 0, "dog": 1}``. ``None`` (default) builds an
            empty map — valid to construct (zero-arg convention), but the properties raise until it
            is populated (by passing a mapping, or via :meth:`fit` / :meth:`from_label_names`).
    """

    def __init__(self, mapping: Optional[Dict[str, int]] = None) -> None:
        # Lazy / zero-arg: store config only. An empty map is a valid object; the non-empty
        # requirement is enforced lazily in the properties, never here.
        self.mapping: Dict[str, int] = {str(k): int(v) for k, v in mapping.items()} if mapping else {}

    def _require(self) -> Dict[str, int]:
        if not self.mapping:
            raise ValueError(
                "LabelMap is empty — pass a `mapping`, or build one via LabelMap.fit(targets) / "
                "LabelMap.from_label_names(names) / LabelMap.load(path) before use."
            )
        return self.mapping

    @property
    def num_classes(self) -> int:
        """Class count = ``max(id) + 1`` (covers the largest id even if some don't appear)."""
        return max(self._require().values()) + 1

    @property
    def label_names(self) -> List[str]:
        """``id → name`` list (index == class id). Ids without a name fall back to ``str(id)``."""
        inverse = self.inverse
        return [inverse.get(i, str(i)) for i in range(self.num_classes)]

    @property
    def inverse(self) -> Dict[int, str]:
        """``id → name`` lookup (the inverse of :attr:`mapping`)."""
        return {v: k for k, v in self._require().items()}

    def encode_op(self, ignore_unknown: bool = False, default: Any = 0) -> EncodeTarget:
        """Return an :class:`~recordstream.ops.target.EncodeTarget` transform that maps name → id via this map."""
        return EncodeTarget(mapping=dict(self._require()), ignore_unknown=ignore_unknown, default=default)

    def decode_op(self, ignore_unknown: bool = False, default: Any = None) -> DecodeTarget:
        """Return a :class:`~recordstream.ops.target.DecodeTarget` transform that maps id → name via this map."""
        return DecodeTarget(mapping=dict(self.inverse), ignore_unknown=ignore_unknown, default=default)

    @classmethod
    def fit(cls, targets: Iterable[Any]) -> "LabelMap":
        """Fit a deterministic name→id map from a stream of raw targets.

        Ordering is sorted-unique, so the same set of labels always yields the same mapping —
        train and (a refit on the same labels at) eval agree. In practice eval should :meth:`load`
        the pinned training map rather than refit on a subset.

        A :class:`~recordstream.MultiLabel` (or any sequence) target contributes EVERY one of its
        labels, so a multi-label dataset fits from the same call as a single-label one.

        (This used to delegate to scikit-learn's ``LabelEncoder``, whose ``classes_`` is exactly
        ``sorted(set(...))`` — the dependency bought nothing but made a data package require an ML
        library, so it was dropped. Ordering is unchanged.)

        Args:
            targets: Iterable of raw labels — names, ids, ``Label``/``MultiLabel`` items, or
                sequences of any of those. Must yield at least one label.
        """
        labels: List[str] = []
        for target in targets:
            labels.extend(str(v) for v in _iter_label_values(target))
        if not labels:
            raise ValueError("LabelMap.fit: no targets to fit on (empty stream).")
        return cls(mapping={name: idx for idx, name in enumerate(sorted(set(labels)))})

    @classmethod
    def from_label_names(cls, names: Sequence[str]) -> "LabelMap":
        """Build a map from an ordered ``id → name`` list (the inverse of :attr:`label_names`).

        Args:
            names: Ordered class names; the list index becomes the class id. Must be non-empty.
        """
        if not names:
            raise ValueError("LabelMap.from_label_names: `names` is empty.")
        return cls(mapping={str(name): int(i) for i, name in enumerate(names)})

    def to_ids(self, target: Any) -> List[int]:
        """Class ids for ``target`` — the "a label is ALWAYS mappable to ints" contract.

        Accepts every shape a target takes: a :class:`~recordstream.Label` or
        :class:`~recordstream.MultiLabel` item, a bare name/id, or a sequence of those. Values
        that are ALREADY encoded (:func:`~recordstream.items.is_class_id`) pass through, so this
        works on an integer-target dataset even when the map is EMPTY — which is what lets a
        consumer stop branching on "are these names or ids?" entirely.

        Raises:
            ValueError: A class NAME arrived but this map is empty (nothing to encode with).
            KeyError: A name is not in the mapping.

        Example::

            LabelMap({"cat": 0, "dog": 1}).to_ids(Label("dog"))     # [1]
            LabelMap().to_ids(Label(2))                             # [2] — no map needed
            LabelMap({"a": 0, "b": 1}).to_ids(MultiLabel(["a", "b"]))  # [0, 1]
        """
        ids: List[int] = []
        for value in _iter_label_values(target):
            if is_class_id(value):
                ids.append(int(value))
                continue
            if not self.mapping:
                raise ValueError(
                    f"LabelMap.to_ids: {value!r} is a class NAME but this LabelMap is empty — "
                    "fit or load a mapping first (LabelMap.fit(targets) / LabelMap.load(path))."
                )
            name = str(value)
            if name not in self.mapping:
                raise KeyError(f"LabelMap.to_ids: {name!r} is not in the mapping (classes: {list(self.mapping)[:8]})")
            ids.append(self.mapping[name])
        return ids

    def save(self, path: Union[str, Path]) -> None:
        """Persist as ``{"class_names": [...], "num_classes": N}`` — marainer's ``class_names.json`` format.

        Args:
            path: Destination file. Parent directories are created as needed.
        """
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"class_names": self.label_names, "num_classes": self.num_classes}
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LabelMap":
        """Restore from a ``class_names.json``-shaped file written by :meth:`save` or marainer.

        Args:
            path: Source file shaped ``{"class_names": [...]}`` (the ``num_classes`` key is optional;
                the ordering of ``class_names`` is authoritative).
        """
        data = json.loads(Path(path).expanduser().read_text())
        names = data.get("class_names")
        if not names:
            raise ValueError(f"LabelMap.load: {path} has no non-empty 'class_names' list.")
        return cls.from_label_names([str(n) for n in names])


__all__ = ["LabelMap"]
