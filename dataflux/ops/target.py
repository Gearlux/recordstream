"""Move and encode the supervised ``target`` field.

Companions to the input↔metadata movers (:class:`~dataflux.ops.stash.StashInputOp` /
:class:`~dataflux.ops.stash.UnstashInputOp`) and
:class:`~dataflux.ops.swap.SwapInputTargetOp`:

* :class:`MetadataToTargetOp` moves a value from ``metadata`` onto ``sample.target``.
* :class:`EncodeTargetOp` / :class:`DecodeTargetOp` map ``sample.target`` through an
  explicit lookup and back — the declarative analogue of scikit-learn's
  ``LabelEncoder``. The label→id mapping is pinned in config, NOT fitted from
  whatever labels happen to appear, so train / eval / predict share one identical
  ordering.

These are deliberately small, value-agnostic plumbing ops (no ``ACCEPTS`` /
``PRODUCES`` contract, like ``copy`` / ``swap`` / ``stash``). The encoded value is
written verbatim (e.g. a plain ``int``); wrap it into a framework tensor downstream
(e.g. a collate function) when a loss needs one.
"""

from typing import Any, Dict, Optional

from confluid import configurable

from dataflux.sample import Sample


def _lookup(value: Any, mapping: Dict[Any, Any], ignore_unknown: bool, default: Any, op_name: str) -> Any:
    """Return ``mapping[value]``, or ``default`` when missing and ``ignore_unknown``.

    Shared by :class:`EncodeTargetOp` / :class:`DecodeTargetOp`. A plain
    module-level function (NOT a base class) so the ops stay independent
    callables — DataFlux Functional Purity.
    """
    if value in mapping:
        return mapping[value]
    if ignore_unknown:
        return default
    sample_keys = list(mapping)[:8]
    suffix = "..." if len(mapping) > 8 else ""
    raise KeyError(
        f"{op_name}: value {value!r} not in mapping (keys: {sample_keys}{suffix}). "
        "Pass ignore_unknown=True to substitute `default` instead."
    )


@configurable(category="op", group="structure")
class MetadataToTargetOp:
    """Set ``sample.target := metadata[key]``; optionally copy it to ``metadata[target_key]``.

    The metadata→target counterpart of :class:`~dataflux.ops.stash.StashInputOp` /
    :class:`~dataflux.ops.stash.UnstashInputOp` (which move input↔metadata). Typical
    use: a raw label rides in ``metadata`` and must become the supervised ``target``
    before :class:`EncodeTargetOp` overwrites it with a class id.

    Args:
        key: Metadata key to read the value from into ``sample.target``.
        target_key: When set, the value is also written to ``metadata[target_key]``
            (so the raw label survives a later ``EncodeTargetOp`` and can be decoded
            back). ``None`` (default) leaves ``metadata`` untouched.
    """

    def __init__(self, key: str, target_key: Optional[str] = None) -> None:
        self.key = str(key)
        self.target_key = str(target_key) if target_key is not None else None

    def __call__(self, sample: Sample) -> Sample:
        if self.key not in sample.metadata:
            raise KeyError(
                f"MetadataToTargetOp: sample.metadata has no key {self.key!r}. "
                f"Available keys: {sorted(sample.metadata)}"
            )
        value = sample.metadata[self.key]
        if self.target_key is not None:
            sample.metadata[self.target_key] = value
        return sample._replace(target=value)


@configurable(category="op", group="structure")
class EncodeTargetOp:
    """Encode ``sample.target`` through an explicit lookup ``mapping``.

    The declarative analogue of scikit-learn's ``LabelEncoder``: maps a raw target
    (typically a string label) to its class id via a config-pinned ``mapping``.
    Pinning the mapping — rather than fitting it from whatever labels appear — keeps
    train / eval / predict on one identical label→id ordering. The plain mapping value
    is written (framework-agnostic); tensorize the target downstream when a loss needs it.

    Args:
        mapping: Lookup from raw target → encoded value, e.g. ``{"DJI AVATA2": 2, ...}``.
            Must be non-empty.
        ignore_unknown: When ``False`` (default), raise on a target missing from
            ``mapping``; when ``True``, substitute ``default``.
        default: Value written for an unknown target when ``ignore_unknown=True``.
            Defaults to ``0``.
    """

    def __init__(self, mapping: Dict[Any, Any], ignore_unknown: bool = False, default: Any = 0) -> None:
        if not mapping:
            raise ValueError("EncodeTargetOp: mapping must contain at least one entry.")
        self.mapping = dict(mapping)
        self.ignore_unknown = bool(ignore_unknown)
        self.default = default

    def __call__(self, sample: Sample) -> Sample:
        encoded = _lookup(sample.target, self.mapping, self.ignore_unknown, self.default, "EncodeTargetOp")
        return sample._replace(target=encoded)


@configurable(category="op", group="structure")
class DecodeTargetOp:
    """Decode ``sample.target`` through a lookup ``mapping`` (inverse of :class:`EncodeTargetOp`).

    Maps an encoded target (e.g. an integer class id) back to its label (e.g. a class
    name) — the readback half used in prediction / reporting.

    Args:
        mapping: Lookup from encoded value → decoded value, e.g. ``{2: "DJI AVATA2", ...}``.
            Must be non-empty.
        ignore_unknown: When ``False`` (default), raise on a target missing from
            ``mapping``; when ``True``, substitute ``default``.
        default: Value written for an unknown target when ``ignore_unknown=True``.
            Defaults to ``None``.
    """

    def __init__(self, mapping: Dict[Any, Any], ignore_unknown: bool = False, default: Any = None) -> None:
        if not mapping:
            raise ValueError("DecodeTargetOp: mapping must contain at least one entry.")
        self.mapping = dict(mapping)
        self.ignore_unknown = bool(ignore_unknown)
        self.default = default

    def __call__(self, sample: Sample) -> Sample:
        decoded = _lookup(sample.target, self.mapping, self.ignore_unknown, self.default, "DecodeTargetOp")
        return sample._replace(target=decoded)


__all__ = ["MetadataToTargetOp", "EncodeTargetOp", "DecodeTargetOp"]
