"""
RecordStream sources — the classes that yield (or derive a view of) record dicts.

One class per module, re-exported here so ``from recordstream.sources import X`` keeps
working; the canonical dotted path a config / form-spec / MCP schema spells out is the
SUBMODULE one (``!class:recordstream.sources.huggingface.HuggingFaceSource``), exactly as
for :mod:`recordstream.ops`. Both spellings resolve — ``confluid.resolve_class`` falls back
to a module-path import, and this package re-exports every name — but generated configs and
the enrichment table key on the submodule path, because that is what ``cls.__module__`` says.

Submodules:
    - recordstream.sources.huggingface: HuggingFaceSource (+ the METADATA_ALL_FEATURES sentinel)
    - recordstream.sources.split: DatasetSplit (+ the SplitName Literal)
    - recordstream.sources.range: RangeSource (a contiguous ``[start:stop)`` slice)
    - recordstream.sources.concat: ConcatSource (several indexable sources end to end)

``__all__`` below is load-bearing, not decoration: a visual editor's node bridge scans this
module in two passes, and the first (``recordstream.discovery.scan_module``) filters on
``member.__module__``, so it sees NOTHING here now that the classes live in submodules. The
second pass — the one that surfaces these nodes — walks exactly this ``__all__``.
"""

from recordstream.sources.concat import ConcatSource
from recordstream.sources.huggingface import METADATA_ALL_FEATURES, HuggingFaceSource
from recordstream.sources.range import RangeSource
from recordstream.sources.split import DatasetSplit, SplitName

__all__ = [
    "ConcatSource",
    "DatasetSplit",
    "HuggingFaceSource",
    "METADATA_ALL_FEATURES",
    "RangeSource",
    "SplitName",
]
