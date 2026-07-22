"""TEMPORARY bridge between the legacy ``Sample`` triple and :class:`TypedSample`.

MIGRATION NOTE: this module dies in the purge stage (when legacy ``Sample`` is deleted).
The structural item codec it used to own moved to :mod:`sampleflux.bag.io` (the storage
serializer registry); this file keeps only the legacy-carrier bridge so typed pipelines can
run against not-yet-migrated Sample sources/sinks during the transition.

The lowering is LOSSLESS — :func:`to_legacy` embeds the encoded typed bag in the legacy
metadata (under :data:`ENCODE_KEY`) while ALSO exposing the primary input / target payloads
on ``Sample.input`` / ``Sample.target`` so a legacy consumer still sees them;
:func:`to_typed` reconstructs the exact bag (``to_typed(to_legacy(x)) == x``).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from sampleflux.bag.io import EncodedField, EncodedItem, decode_sample, encode_sample
from sampleflux.bag.items import item_data
from sampleflux.bag.sample import TypedSample
from sampleflux.sample import Sample

__all__ = ["to_legacy", "to_typed", "ENCODE_KEY"]

#: Metadata key under which :func:`to_legacy` stores the lossless typed-bag encoding.
ENCODE_KEY = "__typed__"


def to_legacy(sample: TypedSample) -> Sample:
    """Lower a :class:`TypedSample` to a legacy ``Sample`` (lossless; see the module docstring)."""
    inputs = sample.inputs()
    targets = sample.targets()
    legacy_input = item_data(next(iter(inputs.values()))) if inputs else None
    legacy_target = item_data(next(iter(targets.values()))) if targets else None
    encoded: List[Dict[str, Any]] = [
        {"key": f.key, "role": f.role, "type": f.item.type_name, "payload": f.item.payload, "attrs": f.item.attrs}
        for f in encode_sample(sample)
    ]
    return Sample(input=legacy_input, target=legacy_target, metadata={ENCODE_KEY: {"fields": encoded}})


def to_typed(sample: Sample, builder: Optional[Callable[[Sample], TypedSample]] = None) -> TypedSample:
    """Lift a legacy ``Sample`` to a :class:`TypedSample`.

    A sample carrying an embedded encoding (produced by :func:`to_legacy`) is reconstructed
    exactly. Otherwise ``builder`` is called to map the sample's fields to typed items; without
    one, a clear error is raised (there is no universal legacy→typed mapping).
    """
    meta = sample.metadata
    if isinstance(meta, dict) and ENCODE_KEY in meta:
        fields: Tuple[EncodedField, ...] = tuple(
            EncodedField(
                key=spec["key"],
                role=spec["role"],
                item=EncodedItem(type_name=spec["type"], payload=spec["payload"], attrs=spec["attrs"]),
            )
            for spec in meta[ENCODE_KEY]["fields"]
        )
        return decode_sample(fields)
    if builder is not None:
        return builder(sample)
    raise ValueError(
        "to_typed: legacy Sample has no embedded typed encoding — pass builder=... to map its "
        "input/target/metadata onto typed items (per-dataset schema)."
    )
