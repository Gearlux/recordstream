"""``LabelMap`` — a bidirectional class-name ↔ integer-id map.

The *fittable* companion to the config-pinned :class:`~sampleflux.ops.target.EncodeTargetOp` /
:class:`~sampleflux.ops.target.DecodeTargetOp`. Those ops carry an explicit ``mapping`` that is
**pinned in config, NOT fitted** at run time, so train / eval / predict share one identical
label→id ordering. :class:`LabelMap` is the piece that *produces* such a pinned mapping:

* :meth:`LabelMap.fit` derives a deterministic name→id mapping from a stream of raw targets
  (backed by scikit-learn's ``LabelEncoder``) — the one-time fit that happens at **train** time.
* :meth:`LabelMap.save` / :meth:`LabelMap.load` persist it (in marainer's ``class_names.json``
  format) so **eval / predict** reload the *same* mapping rather than refitting on a subset.
* :meth:`LabelMap.encode_op` / :meth:`LabelMap.decode_op` hand back the sampleflux ops that apply it.

So fitting happens once, then the mapping is pinned/persisted — it does NOT contradict the
"mapping pinned in config, not fitted" discipline of the ops; it is how the pin gets created.

Zero-arg constructible (``LabelMap()`` succeeds with an empty mapping) and side-effect-free in
``__init__`` per the workspace "Lazy Initialization & Zero-Arg Construction" convention; the
non-empty requirement is validated lazily in the properties, not in the constructor. scikit-learn
is imported lazily inside :meth:`fit` so importing sampleflux never pulls it in.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from confluid import configurable

from sampleflux.ops.target import DecodeTargetOp, EncodeTargetOp


@configurable
class LabelMap:
    """Bidirectional class-name ↔ integer-id map (the fittable companion to ``EncodeTargetOp``).

    Holds an explicit name→id ``mapping`` (pinned in config), or one fitted from a target stream
    via :meth:`fit`. Exposes :attr:`num_classes` / :attr:`label_names`, builds the
    :class:`~sampleflux.ops.target.EncodeTargetOp` / :class:`~sampleflux.ops.target.DecodeTargetOp`
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

    def encode_op(self, ignore_unknown: bool = False, default: Any = 0) -> EncodeTargetOp:
        """Return an :class:`~sampleflux.ops.target.EncodeTargetOp` that maps name → id via this map."""
        return EncodeTargetOp(mapping=dict(self._require()), ignore_unknown=ignore_unknown, default=default)

    def decode_op(self, ignore_unknown: bool = False, default: Any = None) -> DecodeTargetOp:
        """Return a :class:`~sampleflux.ops.target.DecodeTargetOp` that maps id → name via this map."""
        return DecodeTargetOp(mapping=dict(self.inverse), ignore_unknown=ignore_unknown, default=default)

    @classmethod
    def fit(cls, targets: Iterable[Any]) -> "LabelMap":
        """Fit a deterministic name→id map from a stream of raw targets via sklearn ``LabelEncoder``.

        Ordering is scikit-learn's sorted-unique ordering, so the same set of labels always yields
        the same mapping — train and (a refit on the same labels at) eval agree. In practice eval
        should :meth:`load` the pinned training map rather than refit on a subset.

        Args:
            targets: Iterable of raw labels (strings, or anything ``str``-coercible). Must be non-empty.
        """
        from sklearn.preprocessing import LabelEncoder

        labels = [str(t) for t in targets]
        if not labels:
            raise ValueError("LabelMap.fit: no targets to fit on (empty stream).")
        encoder = LabelEncoder()
        encoder.fit(labels)
        return cls(mapping={str(name): int(idx) for idx, name in enumerate(encoder.classes_)})

    @classmethod
    def from_label_names(cls, names: Sequence[str]) -> "LabelMap":
        """Build a map from an ordered ``id → name`` list (the inverse of :attr:`label_names`).

        Args:
            names: Ordered class names; the list index becomes the class id. Must be non-empty.
        """
        if not names:
            raise ValueError("LabelMap.from_label_names: `names` is empty.")
        return cls(mapping={str(name): int(i) for i, name in enumerate(names)})

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
