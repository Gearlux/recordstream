"""Tests for :class:`dataflux.ops.enable.Enable` and :class:`dataflux.ops.sink.SampleSinkOp`.

These modality-neutral compose helpers moved here from ``waivefront.processing`` —
they thread any ``Sample`` through any ops and have no signal dependency.
"""

from typing import Any, Dict, List

import confluid
import pytest

from dataflux.ops.enable import Enable
from dataflux.ops.sink import SampleSinkOp
from dataflux.sample import Sample


class _CountingOp:
    """Plain callable that records every invocation on a shared counter dict."""

    def __init__(self, counter: Dict[str, int]) -> None:
        self.counter = counter

    def __call__(self, sample: Sample) -> Sample:
        self.counter["calls"] += 1
        new_meta = dict(sample.metadata)
        new_meta["counted"] = True
        return sample._replace(metadata=new_meta)


def _sample() -> Sample:
    return Sample(input=None, target=None, metadata={"x": 1})


def _wrap(ops_: Any, **toggle: bool) -> Enable:
    """Build an Enable + set the toggle attribute the way Confluid would.

    ``ops_`` accepts either a single callable (wrapped into a 1-list) or a
    list, so existing single-op test cases stay concise.
    """
    if not isinstance(ops_, list):
        ops_ = [ops_]
    e = Enable(ops=ops_)
    for k, v in toggle.items():
        setattr(e, k, v)
    return e


def test_enable_disabled_passes_sample_through_untouched() -> None:
    counter: Dict[str, int] = {"calls": 0}
    wrapped = _wrap(_CountingOp(counter), visualize=False)
    out = wrapped(_sample())
    assert counter["calls"] == 0
    assert "counted" not in out.metadata


def test_enable_enabled_invokes_inner_op() -> None:
    counter: Dict[str, int] = {"calls": 0}
    wrapped = _wrap(_CountingOp(counter), visualize=True)
    out = wrapped(_sample())
    assert counter["calls"] == 1
    assert out.metadata["counted"] is True
    assert wrapped.flag_name == "visualize"


def test_enable_requires_exactly_one_boolean_attribute() -> None:
    no_toggle = Enable(ops=[lambda s: s])
    with pytest.raises(RuntimeError, match="exactly one boolean toggle"):
        no_toggle(_sample())

    two_toggles = _wrap(lambda s: s, visualize=True, debug=False)
    with pytest.raises(RuntimeError, match="exactly one boolean toggle"):
        two_toggles(_sample())


def test_enable_flag_name_is_arbitrary() -> None:
    """Any kwarg name works — the chosen name is the CLI flag the user types."""
    counter: Dict[str, int] = {"calls": 0}
    wrapped = _wrap(_CountingOp(counter), debug_overlay=True)
    wrapped(_sample())
    assert counter["calls"] == 1
    assert wrapped.flag_name == "debug_overlay"


def test_enable_multi_op_threads_sample_through_each() -> None:
    counter_a: Dict[str, int] = {"calls": 0}
    counter_b: Dict[str, int] = {"calls": 0}

    class _TagOp:
        def __init__(self, tag: str, counter: Dict[str, int]) -> None:
            self.tag = tag
            self.counter = counter

        def __call__(self, sample: Sample) -> Sample:
            self.counter["calls"] += 1
            new_meta = dict(sample.metadata)
            tags = list(new_meta.get("tags", []))
            tags.append(self.tag)
            new_meta["tags"] = tags
            return sample._replace(metadata=new_meta)

    wrapped = _wrap(
        [_TagOp("first", counter_a), _TagOp("second", counter_b)],
        visualize=True,
    )
    out = wrapped(_sample())
    assert counter_a["calls"] == 1
    assert counter_b["calls"] == 1
    assert out.metadata["tags"] == ["first", "second"]


def test_enable_multi_op_disabled_skips_entire_chain() -> None:
    counter: Dict[str, int] = {"calls": 0}
    wrapped = _wrap(
        [_CountingOp(counter), _CountingOp(counter)],
        visualize=False,
    )
    wrapped(_sample())
    assert counter["calls"] == 0


def test_enable_zero_arg_construction_then_rejects_empty_ops_on_call() -> None:
    """Zero-arg / empty-ops construction succeeds (lazy convention); the non-empty
    requirement is enforced on first call, not in ``__init__``."""
    empty = Enable()  # zero-arg construction must work
    assert empty.ops == []
    empty.visualize = True  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="non-empty 'ops' list"):
        empty(_sample())


def test_enable_preserves_name_attr_and_toggle_independence() -> None:
    """Two Enable instances with distinct names carry their names through.

    Confirms that (a) ``name`` is a plain string attr that survives
    construction + post-construction setattr, (b) a string-valued name
    doesn't collide with ``_toggle()``'s bool-attr filter, and (c) the
    two wrappers can be toggled independently when each has its own
    boolean attribute.
    """
    counter_a: Dict[str, int] = {"calls": 0}
    counter_b: Dict[str, int] = {"calls": 0}

    overlay = Enable(ops=[_CountingOp(counter_a)])
    overlay.name = "overlay"  # type: ignore[attr-defined]  # set by Confluid at flow time
    overlay.visualize = True  # type: ignore[attr-defined]

    ls = Enable(ops=[_CountingOp(counter_b)])
    ls.name = "labelstudio"  # type: ignore[attr-defined]
    ls.visualize = False  # type: ignore[attr-defined]

    overlay(_sample())
    ls(_sample())

    assert counter_a["calls"] == 1
    assert counter_b["calls"] == 0
    # Names are preserved verbatim; _toggle's bool filter ignores them.
    assert overlay.name == "overlay"  # type: ignore[attr-defined]
    assert ls.name == "labelstudio"  # type: ignore[attr-defined]
    assert overlay.flag_name == "visualize"
    assert ls.flag_name == "visualize"


def test_enable_yaml_load_with_cli_style_override(tmp_path: Any) -> None:
    """Mimic what Liquify's --visualize true override does to Fluid kwargs."""
    yaml_text = """\
wrapper:
  !class:dataflux.ops.enable.Enable
  visualize: false
  ops:
    - !class:dataflux.ops.copy.CopySampleOp {}
"""
    cfg = tmp_path / "enable.yaml"
    cfg.write_text(yaml_text)
    loaded = confluid.load(cfg)
    enable_fluid = loaded["wrapper"]
    assert "visualize" in enable_fluid.kwargs
    # Liquify CLI override path mutates Fluid.kwargs in-place before flow().
    enable_fluid.kwargs["visualize"] = True
    materialized = confluid.flow(enable_fluid)
    assert isinstance(materialized, Enable)
    assert materialized.enabled is True
    assert materialized.flag_name == "visualize"


# --- SampleSinkOp & Enable.close propagation --------------------------------


class _RecordingSink:
    """Captures the open/write/flush/close lifecycle for assertions."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.writes: List[Sample] = []

    def open(self) -> "_RecordingSink":
        self.calls.append("open")
        return self

    def write(self, sample: Sample) -> None:
        self.calls.append("write")
        self.writes.append(sample)

    def flush(self) -> None:
        self.calls.append("flush")

    def close(self) -> None:
        self.calls.append("close")


def test_sample_sink_op_lazy_open_then_writes() -> None:
    """``open()`` fires once on first call, write() per call, returning the sample."""
    sink = _RecordingSink()
    op = SampleSinkOp(sink=sink)

    s1 = Sample(input=None, target=None, metadata={"i": 0})
    s2 = Sample(input=None, target=None, metadata={"i": 1})
    out1 = op(s1)
    out2 = op(s2)

    assert out1 is s1 and out2 is s2  # pass-through
    assert sink.calls == ["open", "write", "write"]
    assert sink.writes == [s1, s2]


def test_sample_sink_op_close_flushes_and_closes() -> None:
    """``close()`` calls flush() then close() so buffered sinks finalize cleanly."""
    sink = _RecordingSink()
    op = SampleSinkOp(sink=sink)
    op(Sample(input=None, target=None, metadata={}))
    op.close()
    assert sink.calls == ["open", "write", "flush", "close"]


def test_sample_sink_op_zero_arg_construction_then_rejects_none_on_call() -> None:
    """Zero-arg construction works (lazy convention); a missing sink raises on first call."""
    op = SampleSinkOp()  # zero-arg construction must work
    assert op.sink is None
    with pytest.raises(ValueError, match="non-None 'sink'"):
        op(Sample(input=None, target=None, metadata={}))


def test_enable_close_propagates_into_inner_ops() -> None:
    """Closing an Enable wrapper drives close() on every inner op that owns one."""
    sink = _RecordingSink()
    inner = SampleSinkOp(sink=sink)
    wrapped = _wrap(inner, visualize=True)
    wrapped(Sample(input=None, target=None, metadata={}))
    wrapped.close()
    assert sink.calls == ["open", "write", "flush", "close"]


def test_enable_close_on_disabled_wrapper_still_propagates() -> None:
    """Even when toggled off (and thus never invoked), close() must still reach
    inner ops in case they were opened independently — it must NEVER raise."""
    sink = _RecordingSink()
    inner = SampleSinkOp(sink=sink)
    wrapped = _wrap(inner, visualize=False)
    # Never called; close still safe.
    wrapped.close()
    # The SampleSinkOp was never opened, so there's no write — but flush+close
    # are forwarded unconditionally by SampleSinkOp.close.
    assert sink.calls == ["flush", "close"]
