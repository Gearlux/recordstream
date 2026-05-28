import json
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataflux.typespec import SampleType

# Reserved metadata keys carrying a sample's stored type description (JSON strings so they survive
# every storage backend's metadata round-trip — HDF5 attrs / Zarr attrs / Directory YAML / HF / Confluid).
# ``__features__`` holds a ``datasets.Features`` dict (the standard, concrete structural description);
# ``__spec__`` holds the sidecar refinements Features can't express (framework / ranges / Any / Union).
FEATURES_KEY = "__features__"
SPEC_KEY = "__spec__"
TYPE_KEYS = (FEATURES_KEY, SPEC_KEY)


# Standardized Sample: (input, target, metadata)
# This allows DataFlux to handle complex pipelines while remaining
# compatible with simple PyTorch/HF (input, target) pairs.
class Sample(NamedTuple):
    input: Any
    target: Any = None
    metadata: Dict[str, Any] = {}

    def to_tuple(self) -> Tuple[Any, Any, Dict[str, Any]]:
        return (self.input, self.target, self.metadata)

    def describe(self) -> "SampleType":
        """Return this sample's :class:`~dataflux.typespec.SampleType`.

        Prefers the stored type (the reserved metadata keys, set explicitly via :meth:`with_type` or
        carried by a serialized dataset); otherwise infers it from the live ``input`` / ``target``.
        """
        from dataflux.typespec import SampleType, infer_sample_type

        raw_features = self.metadata.get(FEATURES_KEY)
        raw_extras = self.metadata.get(SPEC_KEY)
        if raw_features is not None or raw_extras is not None:
            features = json.loads(raw_features) if isinstance(raw_features, str) else (raw_features or {})
            extras = json.loads(raw_extras) if isinstance(raw_extras, str) else raw_extras
            return SampleType.from_hf_features(features, extras)
        return infer_sample_type(self)

    def with_type(self, sample_type: "SampleType") -> "Sample":
        """Return a copy carrying ``sample_type`` in the reserved metadata keys (copy-on-write, so the
        original sample's metadata is not mutated)."""
        features, extras = sample_type.to_hf_features()
        metadata = {**self.metadata, FEATURES_KEY: json.dumps(features.to_dict()), SPEC_KEY: json.dumps(extras)}
        return self._replace(metadata=metadata)

    @classmethod
    def from_any(cls, obj: Any) -> "Sample":
        """Coerce raw data from various sources into a Sample."""
        if isinstance(obj, cls):
            return obj
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
