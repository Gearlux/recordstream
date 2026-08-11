"""A still-deferred marker in a view source's ``source:`` slot explains itself.

A ``_partial_: true`` source leaves a Confluid marker in the slot — a CONFIG error, since a
source slot needs a live object and nothing here will flow it — and the view sources answer
it with the same actionable guidance ``Stream`` gives (naming the slot, the deferred target,
and the fix) instead of the cryptic ``got Partial`` their bare ``hasattr`` checks used to
produce. Deliberately message-only: the slot is never flowed (the raise-with-guidance
convention for ``source:`` slots — distinct from the free functions ``project`` /
``dataset_uri``).

**The TRIGGER changed 2026-08-11.** It used to be the parens-less ``!class:X`` spelling,
which left a deferred stub; confluid merged its eager and deferred markers, so both
spellings build and that footgun is gone. Deferral is now asked for explicitly, which is
the only way left to reach this guard — and the message changed with it (it used to say
"add parens", advice that would now change nothing).
"""

import pytest
from confluid import load

from recordstream import Stream
from recordstream.sources import ConcatSource, DatasetSplit, RangeSource

_LEAF = "recordstream.sources.concat.ConcatSource"


def _expect_guidance(excinfo: "pytest.ExceptionInfo[TypeError]", slot: str) -> None:
    """The error names the slot, says WHAT was deferred, and states the parens fix."""
    message = str(excinfo.value)
    assert slot in message
    assert "deferred Confluid marker" in message
    assert "ConcatSource" in message  # the deferred target, not just "Class"
    assert "_partial_: true" in message  # the actionable fix


def test_a_partial_marker_under_range_source_raises_the_stream_guidance() -> None:
    cfg = load(
        f"""
range_src:
  _target_: recordstream.sources.range.RangeSource
  source: {{_target_: {_LEAF}, _partial_: true}}
  stop: 3
""",
        flow=True,
    )
    with pytest.raises(TypeError) as excinfo:
        cfg["range_src"].indices
    _expect_guidance(excinfo, "RangeSource.source")


def test_a_partial_marker_under_concat_source_names_the_offending_index() -> None:
    cfg = load(
        f"""
concat_src:
  _target_: recordstream.sources.concat.ConcatSource
  sources:
    - {{_target_: {_LEAF}}}
    - {{_target_: {_LEAF}, _partial_: true}}
""",
        flow=True,
    )
    with pytest.raises(TypeError) as excinfo:
        cfg["concat_src"].offsets
    _expect_guidance(excinfo, "ConcatSource.sources[1]")


def test_a_partial_marker_under_dataset_split_raises_the_stream_guidance() -> None:
    cfg = load(
        f"""
split_src:
  _target_: recordstream.sources.split.DatasetSplit
  source: {{_target_: {_LEAF}, _partial_: true}}
""",
        flow=True,
    )
    with pytest.raises(TypeError) as excinfo:
        len(cfg["split_src"].train)
    _expect_guidance(excinfo, "DatasetSplit.source")


def test_the_parens_spelling_materializes_and_the_view_sources_work() -> None:
    """The positive counterpart: ``!class:X()`` becomes a live instance at load time."""
    cfg = load(
        f"""
range_src: !class:recordstream.sources.range.RangeSource()
  source: !class:{_LEAF}()
  stop: 3
concat_src: !class:recordstream.sources.concat.ConcatSource()
  sources:
    - !class:{_LEAF}()
split_src: !class:recordstream.sources.split.DatasetSplit()
  source: !class:{_LEAF}()
""",
        flow=True,
    )
    range_src: RangeSource = cfg["range_src"]
    concat_src: ConcatSource = cfg["concat_src"]
    split_src: DatasetSplit = cfg["split_src"]
    assert isinstance(range_src.source, ConcatSource)
    assert range_src.indices == []  # an empty leaf clamps [0:3) to nothing — but it computed
    assert concat_src.offsets == [0]
    assert len(split_src.train) == 0


def test_streams_own_guidance_still_names_its_slot() -> None:
    """The shared message helper's default slot stays ``Stream.source``."""
    cfg = load(f"deferred: {{_target_: {_LEAF}, _partial_: true}}", flow=True)
    with pytest.raises(TypeError, match="Stream.source is still a deferred Confluid marker"):
        len(Stream(source=cfg["deferred"]))
