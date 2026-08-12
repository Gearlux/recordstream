"""``LabelMap`` — a bidirectional class-name ↔ integer-id map.

The *fittable* companion to the config-pinned :class:`~recordstream.ops.target.EncodeTarget` /
:class:`~recordstream.ops.target.DecodeTarget`. Those ops carry an explicit ``mapping`` that is
**pinned in config, NOT fitted** at run time, so train / eval / predict share one identical
label→id ordering. :class:`LabelMap` is the piece that *produces* such a pinned mapping:

* :meth:`LabelMap.fit` derives a deterministic name→id mapping from a stream of raw targets
  (backed by scikit-learn's ``LabelEncoder``) — the one-time fit that happens at **train** time.
* :meth:`LabelMap.save` / :meth:`LabelMap.load` persist it (in matrainer's ``class_names.json``
  format) so **eval / predict** reload the *same* mapping rather than refitting on a subset.
* :meth:`LabelMap.encode_op` / :meth:`LabelMap.decode_op` hand back the recordstream ops that apply it.

So fitting happens once, then the mapping is pinned/persisted — it does NOT contradict the
"mapping pinned in config, not fitted" discipline of the ops; it is how the pin gets created.

Zero-arg constructible (``LabelMap()`` succeeds with an empty mapping) and side-effect-free in
``__init__`` per the workspace "Partial Initialization & Zero-Arg Construction" convention; the
non-empty requirement is validated lazily in the properties, not in the constructor.

A label is ALWAYS mappable to ids: :meth:`LabelMap.to_ids` accepts a ``Label`` / ``MultiLabel``
item, a bare name or id, or a sequence of those, and passes already-encoded values through — so a
consumer never branches on "are these names or ids?".
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
from confluid import configurable

from recordstream.items import Label, MultiLabel, is_class_id
from recordstream.ops.target import DecodeTarget, EncodeTarget

if TYPE_CHECKING:  # Stream imports labels indirectly — keep this annotation-only
    from recordstream.core import Stream


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
    via :meth:`fit`. Exposes :attr:`num_classes` / :attr:`class_names`, builds the
    :class:`~recordstream.ops.target.EncodeTarget` / :class:`~recordstream.ops.target.DecodeTarget`
    that apply it, and round-trips to disk in matrainer's ``class_names.json`` format.

    Args:
        mapping: Explicit name→id lookup, e.g. ``{"cat": 0, "dog": 1}``. ``None`` (default) builds an
            empty map — valid to construct (zero-arg convention), but the properties raise until it
            is populated (by passing a mapping, or via :meth:`fit` / :meth:`from_class_names`).
    """

    def __init__(self, mapping: Optional[Dict[str, int]] = None) -> None:
        # Partial / zero-arg: store config only. An empty map is a valid object; the non-empty
        # requirement is enforced lazily in the properties, never here.
        self.mapping: Dict[str, int] = {str(k): int(v) for k, v in mapping.items()} if mapping else {}

    def _require(self) -> Dict[str, int]:
        if not self.mapping:
            raise ValueError(
                "LabelMap is empty — pass a `mapping`, or build one via LabelMap.fit(targets) / "
                "LabelMap.from_class_names(names) / LabelMap.load(path) before use."
            )
        return self.mapping

    @property
    def num_classes(self) -> int:
        """Class count = ``max(id) + 1`` (covers the largest id even if some don't appear)."""
        return max(self._require().values()) + 1

    @property
    def class_names(self) -> List[str]:
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

    def encode(self, source: Any) -> "Stream":
        """Wrap ``source`` in a :class:`~recordstream.Stream` that applies this map's encode op.

        The one-call form of the two-step idiom every consumer of a name-labelled dataset
        writes — ``Stream(source=source, ops=[label_map.encode_op()])``. It lives here because
        it is a :class:`LabelMap` operation over a source, not knowledge about any particular
        task: a classifier, a detector and a tagger all need the identical wrap.

        Args:
            source: Any source/stream the engine accepts. A deferred ``!class:`` marker is
                flowed first, so a config-wired source works without the caller flowing it.

        Returns:
            A :class:`~recordstream.Stream` yielding the same records with their labels mapped
            to integer ids, and carrying this map's ``class_names`` so the vocabulary travels
            with the encoded data (read it back with :func:`~recordstream.class_names`).
            Which key is encoded follows :class:`EncodeTarget`'s own rule (its blank
            ``field`` picks the first :class:`~recordstream.Label`); pass a configured
            ``encode_op()`` into a ``Stream`` yourself when you need to pin a different key or
            tolerate unknowns.

        Raises:
            KeyError: lazily, while iterating, when a label is not in the mapping. That
            includes an ALREADY-ENCODED id — unlike :meth:`to_ids`, which passes ids through,
            the op is a straight lookup, so double-encoding fails loudly instead of silently
            remapping. Wrap a source only when its labels are names (ask
            :func:`~recordstream.is_class_id`), or build the op with ``ignore_unknown=True``.

        Example::

            label_map = LabelMap.fit(iter_key(train_source, "class"))
            train_set = label_map.encode(train_source)
        """
        from confluid import flow

        from recordstream.core import Stream

        return Stream(source=flow(source), ops=[self.encode_op()], class_names=self.class_names)

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
    def from_class_names(cls, names: Sequence[str]) -> "LabelMap":
        """Build a map from an ordered ``id → name`` list (the inverse of :attr:`class_names`).

        Args:
            names: Ordered class names; the list index becomes the class id. Must be non-empty.
        """
        if not names:
            raise ValueError("LabelMap.from_class_names: `names` is empty.")
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
        """Persist as ``{"class_names": [...], "num_classes": N}`` — matrainer's ``class_names.json`` format.

        Args:
            path: Destination file. Parent directories are created as needed.
        """
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"class_names": self.class_names, "num_classes": self.num_classes}
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LabelMap":
        """Restore from a ``class_names.json``-shaped file written by :meth:`save` or matrainer.

        Args:
            path: Source file shaped ``{"class_names": [...]}`` (the ``num_classes`` key is optional;
                the ordering of ``class_names`` is authoritative).
        """
        data = json.loads(Path(path).expanduser().read_text())
        names = data.get("class_names")
        if not names:
            raise ValueError(f"LabelMap.load: {path} has no non-empty 'class_names' list.")
        return cls.from_class_names([str(n) for n in names])


def class_counts(targets: Iterable[Any], num_classes: int, label_map: Optional[LabelMap] = None) -> np.ndarray:
    """How often each class id occurs in ``targets`` — the label statistic behind class balancing.

    Every target shape is accepted, because :meth:`LabelMap.to_ids` normalizes them: a
    :class:`~recordstream.Label` or :class:`~recordstream.MultiLabel` item, a bare name/id, or a
    sequence. A multi-label target counts for EVERY class it names. ``None`` targets are skipped.

    Args:
        targets: Already-walked target values (see the note on walking below).
        num_classes: Width of the returned vector. Ids outside ``[0, num_classes)`` are IGNORED
            rather than raising — a stray label must not abort a training run.
        label_map: Map for class-NAME targets. Omit for integer targets: an empty
            :class:`LabelMap` passes already-encoded ids through, which is the ``to_ids`` contract.

    Returns:
        A ``float64`` vector of length ``num_classes``.

    Note:
        This takes ALREADY-WALKED targets, not a source, on purpose. A caller typically walks the
        target stream once (``iter_key(source, key)``) and reuses that single pass for several
        answers — fitting a :class:`LabelMap`, deriving the class count, and weighting — and a
        convenience that walked internally would silently double the passes over the dataset.

    Example::

        class_counts([Label("cat"), Label("dog"), Label("cat")], 2, LabelMap({"cat": 0, "dog": 1}))
        # array([2., 1.])
    """
    mapper = label_map if label_map is not None else LabelMap()
    counts = np.zeros(int(num_classes), dtype=np.float64)
    for target in targets:
        if target is None:
            continue
        for class_id in mapper.to_ids(target):
            if 0 <= class_id < num_classes:
                counts[class_id] += 1.0
    return counts


def inverse_frequency_weights(
    targets: Iterable[Any], num_classes: int, label_map: Optional[LabelMap] = None
) -> Optional[np.ndarray]:
    """Per-class weights inversely proportional to observed frequency.

    ``w[c] = total / (num_classes * count[c])`` — a class at exactly the mean frequency gets
    ``1.0``, rarer classes more, commoner classes less. Training on a skewed label distribution
    biases a model toward the majority class; feeding these weights to a loss (or a framework's
    ``class_weight`` knob) is the standard remedy.

    This is a statistic OVER THE DATA, which is why it lives here rather than beside a loss: what
    a consuming framework then does with the vector — ``torch.nn``'s ``weight=`` constructor
    argument, Keras's ``class_weight`` on ``fit()`` — is that framework's convention, and the
    numbers are the same either way. Hence the **numpy** return (the same rule as
    :mod:`recordstream.batch`: only ``batch_tensor`` is torch); a torch caller writes
    ``torch.as_tensor(weights)``.

    Args:
        targets: Already-walked target values (see :func:`class_counts` on why not a source).
        num_classes: Width of the returned vector.
        label_map: Map for class-NAME targets; omit for integer targets.

    Returns:
        A ``float32`` vector of length ``num_classes``, or ``None`` when nothing was counted (an
        empty or fully out-of-range target set) — so a caller can tell "no weights" from
        "all-zero weights". A class observed **zero** times gets weight ``0.0``, not infinity.

    Example::

        inverse_frequency_weights([Label(0), Label(0), Label(0), Label(1)], num_classes=2)
        # array([0.6667, 2.0], dtype=float32)
    """
    counts = class_counts(targets, num_classes, label_map)
    total = float(counts.sum())
    if total <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(counts > 0, total / (num_classes * counts), 0.0)
    return weights.astype(np.float32)


__all__ = ["LabelMap", "class_counts", "inverse_frequency_weights"]
