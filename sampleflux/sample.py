import json
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Tuple, Union, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sampleflux.typespec import SampleType

# Reserved metadata keys carrying a sample's stored type description (JSON strings so they survive
# every storage backend's metadata round-trip — HDF5 attrs / Zarr attrs / Directory YAML / HF / Confluid).
# ``__features__`` holds a ``datasets.Features`` dict (the standard, concrete structural description);
# ``__spec__`` holds the sidecar refinements Features can't express (framework / ranges / Any / Union).
FEATURES_KEY = "__features__"
SPEC_KEY = "__spec__"
TYPE_KEYS = (FEATURES_KEY, SPEC_KEY)

# A Sample's metadata is EITHER a single ``dict`` (one item — the normal pipeline form every op
# produces/consumes) OR a ``list`` of per-item dicts (a BATCH — produced by the collate functions when
# stacking N samples into one). The two forms are how a Sample distinguishes a single item from a batch:
# per-sample ops always see (and require) the dict form; the list form appears only AFTER collate, in the
# batched Sample fed to the model / loss / predictions sinks, and never flows back through a per-sample op.
Metadata = Union[Dict[str, Any], List[Dict[str, Any]]]


# The named field VIEWS of a Sample — the closed vocabulary of what a transform can
# process (the taxonomy `sampleflux.kinds` introspects and a visual editor can surface as
# socket types). A view is a real runtime NamedTuple, so an op annotated with one
# receives an object with named fields; the engine binds the view from the flowing
# carrier and merges the result back (untouched fields preserved).
class Pair(NamedTuple):
    """The classic metadata-free AI pair ``(input, target)`` — a named 2-tuple view."""

    input: Any
    target: Any = None


class InputMeta(NamedTuple):
    """The ``(input, metadata)`` view — a transform that reads/writes the input WITH its metadata.

    ``metadata`` follows the same single-vs-batch duality as ``Sample.metadata``: one dict
    per item, a list of dicts after collation.
    """

    input: Any
    metadata: Metadata = {}


class TargetMeta(NamedTuple):
    """The ``(target, metadata)`` view — a transform that reads/writes the target WITH its metadata.

    ``metadata`` follows the same single-vs-batch duality as ``Sample.metadata``.
    """

    target: Any
    metadata: Metadata = {}


# Standardized Sample: (input, target, metadata)
# This allows SampleFlux to handle complex pipelines while remaining
# compatible with simple PyTorch/HF (input, target) pairs.
class Sample(NamedTuple):
    input: Any
    target: Any = None
    metadata: Metadata = {}

    def to_tuple(self) -> Tuple[Any, Any, Metadata]:
        return (self.input, self.target, self.metadata)

    def to_pair(self) -> Tuple[Any, Any]:
        """The metadata-free ``(input, target)`` view (the native-engine pair carrier)."""
        return (self.input, self.target)

    def input_meta(self) -> "InputMeta":
        """The ``(input, metadata)`` view — the SAME metadata dict (mutation propagates)."""
        return InputMeta(self.input, self.meta)

    def target_meta(self) -> "TargetMeta":
        """The ``(target, metadata)`` view — the SAME metadata dict (mutation propagates)."""
        return TargetMeta(self.target, self.meta)

    @property
    def is_batched(self) -> bool:
        """True if this Sample holds a BATCH — ``metadata`` is a ``list`` of per-item dicts (one per
        stacked item, as the collate functions produce); False for a single item (``metadata`` is a
        ``dict``). The single source of truth for telling batch from single."""
        return isinstance(self.metadata, list)

    @property
    def meta(self) -> Dict[str, Any]:
        """The single-item metadata **dict** — the narrowing accessor per-sample ops/sources use to
        read or mutate ``metadata`` (``sample.meta[key]`` / ``sample.meta[key] = v``). Returns the same
        underlying dict (mutation propagates). Raises ``TypeError`` on a batched Sample, where there is
        no single dict — use :attr:`batch_meta` instead."""
        if isinstance(self.metadata, list):
            raise TypeError(
                "Sample.meta is the single-item metadata dict, but this Sample is batched (metadata is a "
                "list of per-item dicts) — use Sample.batch_meta."
            )
        return self.metadata

    @property
    def batch_meta(self) -> List[Dict[str, Any]]:
        """The per-item metadata **list** of a batched Sample (one dict per stacked item) — the
        narrowing accessor batch consumers (collate-fed losses / predictions sinks) use. Raises
        ``TypeError`` on a single Sample, whose metadata is one dict — use :attr:`meta` instead."""
        if not isinstance(self.metadata, list):
            raise TypeError(
                "Sample.batch_meta is the per-item metadata list of a batch, but this Sample is single "
                "(metadata is one dict) — use Sample.meta."
            )
        return self.metadata

    def describe(self) -> "SampleType":
        """Return this sample's :class:`~sampleflux.typespec.SampleType`.

        Prefers the stored type (the reserved metadata keys, set explicitly via :meth:`with_type` or
        carried by a serialized dataset); otherwise infers it from the live ``input`` / ``target``. A
        batched sample carries no per-reserved-key type, so it always infers from the live data.
        """
        from sampleflux.typespec import SampleType, infer_sample_type

        meta = self.metadata
        if isinstance(meta, dict):
            raw_features = meta.get(FEATURES_KEY)
            raw_extras = meta.get(SPEC_KEY)
            if raw_features is not None or raw_extras is not None:
                features = json.loads(raw_features) if isinstance(raw_features, str) else (raw_features or {})
                extras = json.loads(raw_extras) if isinstance(raw_extras, str) else raw_extras
                return SampleType.from_hf_features(features, extras)
        return infer_sample_type(self)

    def with_type(self, sample_type: "SampleType") -> "Sample":
        """Return a copy carrying ``sample_type`` in the reserved metadata keys (copy-on-write, so the
        original sample's metadata is not mutated). Only defined for a single (non-batched) sample —
        a batch carries no single stored type."""
        if self.is_batched:
            raise TypeError(
                "Sample.with_type is only defined for a single (non-batched) sample; this Sample carries "
                "list (batched) metadata."
            )
        base = cast(Dict[str, Any], self.metadata)
        features, extras = sample_type.to_hf_features()
        metadata = {**base, FEATURES_KEY: json.dumps(features.to_dict()), SPEC_KEY: json.dumps(extras)}
        return self._replace(metadata=metadata)

    @classmethod
    def from_any(cls, obj: Any) -> "Sample":
        """Coerce raw data from various sources into a Sample.

        The named field VIEWS are recognised BEFORE the generic tuple rule — an
        ``InputMeta``/``TargetMeta``/``Pair`` IS a tuple, and positional coercion would
        silently misread ``(input, metadata)`` as ``(input, target)``.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, InputMeta):
            return cls(obj.input, None, obj.metadata)
        if isinstance(obj, TargetMeta):
            return cls(None, obj.target, obj.metadata)
        if isinstance(obj, Pair):
            return cls(obj.input, obj.target, {})
        if isinstance(obj, tuple):
            if len(obj) >= 3:
                return cls(obj[0], obj[1], obj[2] or {})
            if len(obj) == 2:
                return cls(obj[0], obj[1], {})
            if len(obj) == 1:
                return cls(obj[0], None, {})
            # Empty tuple
            return cls(None, None, {})
        if isinstance(obj, dict):
            return cls(
                input=obj.get("input"),
                target=obj.get("target"),
                metadata=obj.get("metadata", {}),
            )
        return cls(obj, None, {})
