from typing import Any, Dict, NamedTuple, Tuple


# Standardized Sample: (input, target, metadata)
# This allows DataFlux to handle complex pipelines while remaining
# compatible with simple PyTorch/HF (input, target) pairs.
class Sample(NamedTuple):
    input: Any
    target: Any = None
    metadata: Dict[str, Any] = {}

    def to_tuple(self) -> Tuple[Any, Any, Dict[str, Any]]:
        return (self.input, self.target, self.metadata)

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
