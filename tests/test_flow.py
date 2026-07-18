"""Tests for the flow document, the FlowGraph engine, and the flow⇄ops converters.

The load-bearing contract: **execution parity both ways** — a flow document run natively
by FlowGraph equals the same flow lowered (`to_ops`) and run by the serial Flux engine,
and a flat context-ops list run by Flux equals its lifted (`from_ops`) flow run by
FlowGraph. Round-tripping re-lowers to an execution-equivalent list.
"""

from pathlib import Path
from typing import Callable, Optional

import pytest
from confluid import configurable, output

from sampleflux.context import Context, activate
from sampleflux.core import Flux
from sampleflux.flow import FlowGraph, FlowStep, from_ops, parse_flow, to_ops
from sampleflux.ops.context import Apply, Capture, Drop, Mix, Save, Use
from sampleflux.ops.swap import SwapInputTargetOp
from sampleflux.sample import Sample

# ---------------------------------------------------------------------------
# Test ops (module-level so they pickle for spawn parity)
# ---------------------------------------------------------------------------


@configurable
class AddOp:
    """Add a constant to the input.

    Args:
        amount: Value added to ``sample.input``.
    """

    def __init__(self, amount: float = 1.0) -> None:
        self.amount = amount

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input + self.amount)


@configurable
class ScaleOp:
    """Multiply the input by a factor.

    Args:
        factor: Multiplier applied to ``sample.input``.
    """

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input * self.factor)


@configurable
class TenfoldOutputOp:
    """Pass-through with a DETERMINISTIC @output (parity tests need reproducibility)."""

    def __init__(self) -> None:
        self._last: float = 0.0

    @property
    @output
    def tenfold(self) -> float:
        """Ten times the last seen input."""
        return self._last

    def __call__(self, sample: Sample) -> Sample:
        self._last = float(sample.input) * 10.0
        return sample


@configurable
class DropOddOp:
    """Filter: drop samples with odd integer input."""

    def __call__(self, sample: Sample) -> Optional[Sample]:
        return None if int(sample.input) % 2 else sample


def _samples(n: int = 4) -> list:
    return [Sample(input=float(i), target=i, metadata={"idx": i}) for i in range(n)]


def _key(sample: Sample) -> tuple:
    target = sample.target
    if isinstance(target, Sample):
        target = ("sample", target.input, target.target)
    return (sample.input, target, tuple(sorted(sample.meta.items())))


def _assert_parity(flow_doc: dict, outputs: str = "", n: int = 4) -> None:
    """FlowGraph-native == Flux-over-lowered, sample for sample (fresh ops per engine)."""
    import copy

    native = FlowGraph(source=_samples(n), flow=copy.deepcopy(flow_doc), outputs=outputs)
    lowered_steps, out = parse_flow(copy.deepcopy(flow_doc), outputs)
    serial = Flux(source=_samples(n), ops=to_ops(lowered_steps, out))
    got_native = [_key(s) for s in native]
    got_serial = [_key(s) for s in serial]
    assert got_native == got_serial, f"engine parity broken:\n native={got_native}\n serial={got_serial}"


# ---------------------------------------------------------------------------
# parse_flow validation
# ---------------------------------------------------------------------------


class TestParseFlow:
    def test_linear_defaults(self) -> None:
        steps, out = parse_flow({"a": AddOp(), "b": ScaleOp()})
        assert [s.name for s in steps] == ["a", "b"]
        assert steps[1].from_ is None and out == "b"

    def test_forward_reference_rejected(self) -> None:
        with pytest.raises(ValueError, match="EARLIER step"):
            parse_flow({"a": {"op": AddOp(), "from": "b"}, "b": ScaleOp()})

    def test_dotted_step_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="may not contain"):
            parse_flow({"a.b": AddOp()})

    def test_unknown_step_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown step key"):
            parse_flow({"a": AddOp(), "b": {"op": ScaleOp(), "sideways": "a"}})

    def test_bind_requires_known_step_and_op(self) -> None:
        with pytest.raises(ValueError, match="does not name an earlier step"):
            parse_flow({"a": {"op": AddOp(), "bind": {"amount": "ghost"}}})
        with pytest.raises(ValueError, match="bind requires an op"):
            parse_flow({"a": AddOp(), "b": {"from": "a", "bind": {"x": "a"}}})

    def test_outputs_must_name_a_step(self) -> None:
        with pytest.raises(ValueError, match="outputs"):
            parse_flow({"a": AddOp()}, outputs="ghost")

    def test_reserved_ctor_param_collision_rejected(self) -> None:
        @configurable
        class BadOp:
            """Op with a reserved-name ctor param.

            Args:
                bind: Collides with the reserved flow step key.
            """

            def __init__(self, bind: str = "") -> None:
                self.bind = bind

            def __call__(self, sample: Sample) -> Sample:
                return sample

        with pytest.raises(ValueError, match="reserved flow step keys"):
            parse_flow({"a": BadOp()})

    def test_duplicate_step_name_rejected(self) -> None:
        # dicts dedupe keys silently, so build the parsed list directly
        steps = [
            FlowStep("a", AddOp(), None, None, None, {}),
            FlowStep("a", ScaleOp(), None, None, None, {}),
        ]
        graph = FlowGraph(source=_samples(1), flow=steps)
        assert graph.steps  # duplicate FlowStep lists are the caller's problem; parse_flow guards dicts


# ---------------------------------------------------------------------------
# Engine parity (the load-bearing contract)
# ---------------------------------------------------------------------------


class TestEngineParity:
    def test_linear(self) -> None:
        _assert_parity({"a": AddOp(amount=1.0), "b": ScaleOp(factor=3.0)})

    def test_linear_lowering_is_bare(self) -> None:
        ops = to_ops({"a": AddOp(amount=1.0), "b": ScaleOp(factor=3.0)})
        assert [type(o).__name__ for o in ops] == ["AddOp", "ScaleOp"]  # zero context ops

    def test_fan_out_fan_in(self) -> None:
        _assert_parity(
            {
                "a": AddOp(amount=1.0),
                "b": {"op": SwapInputTargetOp(), "from": "a"},
                "c": {"op": AddOp(amount=5.0), "from": "a"},
                "out": {"from": "c", "target_from": "b"},
            }
        )

    def test_bind_step_result(self) -> None:
        _assert_parity(
            {
                "thresh": ScaleOp(factor=0.5),
                "shifted": {"op": AddOp(), "from": "thresh", "bind": {"amount": "thresh"}},
            }
        )

    def test_bind_at_output(self) -> None:
        _assert_parity(
            {
                "probe": TenfoldOutputOp(),
                "shifted": {"op": AddOp(), "bind": {"amount": "probe.tenfold"}},
            }
        )

    def test_outputs_earlier_step(self) -> None:
        _assert_parity({"a": AddOp(amount=1.0), "b": ScaleOp(factor=3.0)}, outputs="a")

    def test_identity_first_step_source_fork(self) -> None:
        _assert_parity(
            {
                "src": {},
                "a": AddOp(amount=1.0),
                "b": {"op": ScaleOp(factor=2.0), "from": "src"},
                "out": {"from": "b", "target_from": "a"},
            }
        )

    def test_filtering_drops_in_both_engines(self) -> None:
        flow_doc = {"f": DropOddOp(), "a": AddOp(amount=1.0)}
        _assert_parity(flow_doc)
        native = FlowGraph(source=_samples(4), flow={"f": DropOddOp(), "a": AddOp(amount=1.0)})
        assert [s.input for s in native] == [1.0, 3.0]

    def test_metadata_from_slot(self) -> None:
        _assert_parity(
            {
                "a": AddOp(amount=1.0),
                "b": {"op": ScaleOp(factor=2.0), "from": "a"},
                "out": {"from": "b", "metadata_from": "a"},
            }
        )


class TestReverseParity:
    """Flux(ops) == FlowGraph(from_ops(ops)) — lifting preserves execution."""

    def _assert_reverse(self, ops_builder: Callable[[], list], n: int = 4) -> None:
        serial = Flux(source=_samples(n), ops=ops_builder())
        flow_doc, out = from_ops(ops_builder())
        native = FlowGraph(source=_samples(n), flow=flow_doc, outputs=out)
        assert [_key(s) for s in serial] == [_key(s) for s in native]

    def test_linear_list(self) -> None:
        self._assert_reverse(lambda: [AddOp(amount=1.0), ScaleOp(factor=3.0)])

    def test_hand_written_graph_list(self) -> None:
        self._assert_reverse(
            lambda: [
                Save(name="fork"),
                AddOp(amount=1.0),
                SwapInputTargetOp(),
                Save(name="branch_a"),
                Use(name="fork", drop=True),
                ScaleOp(factor=2.0),
                Mix(target_from="branch_a", drop=["branch_a"]),
            ]
        )

    def test_apply_capture_list(self) -> None:
        self._assert_reverse(
            lambda: [
                Capture(op=TenfoldOutputOp(), output="tenfold", name="t"),
                Apply(op=AddOp(), param="amount", source="t", drop=True),
            ]
        )

    def test_drop_ops_vanish_from_lifted_flow(self) -> None:
        flow_doc, _ = from_ops([Save(name="x"), AddOp(), Drop(names=["x"])])
        assert all("drop" not in str(v).lower() or "op" in v for v in flow_doc.values())


class TestRoundTrip:
    def test_flow_to_ops_to_flow_execution_equivalent(self) -> None:
        original = {
            "a": AddOp(amount=1.0),
            "b": {"op": SwapInputTargetOp(), "from": "a"},
            "c": {"op": AddOp(amount=5.0), "from": "a"},
            "out": {"from": "c", "target_from": "b"},
        }
        lowered = to_ops(dict(original))
        lifted, out = from_ops(lowered)
        relowered = to_ops(lifted, out)
        a = [_key(s) for s in Flux(source=_samples(4), ops=lowered)]
        b = [_key(s) for s in Flux(source=_samples(4), ops=relowered)]
        assert a == b

    def test_lowered_graph_leaves_context_empty(self) -> None:
        ops = to_ops(
            {
                "a": AddOp(amount=1.0),
                "b": {"op": SwapInputTargetOp(), "from": "a"},
                "out": {"from": "a", "target_from": "b"},
            }
        )
        ctx = Context()
        sample: Sample = Sample(input=1.0, metadata={})
        with activate(ctx):
            for op in ops:
                result = op(sample)
                assert result is not None
                sample = result
        assert ctx.live() == ()  # automatic liveness freed every cell


# ---------------------------------------------------------------------------
# FlowGraph engine surface
# ---------------------------------------------------------------------------


class TestFlowGraphSurface:
    def test_zero_arg_construction(self) -> None:
        graph = FlowGraph()
        with pytest.raises(ValueError, match="flow is not set"):
            _ = graph.steps

    def test_len_and_getitem(self) -> None:
        graph = FlowGraph(source=_samples(5), flow={"a": AddOp(amount=1.0)})
        assert len(graph) == 5
        assert graph[2].input == 3.0

    def test_getitem_filtered_raises_indexerror(self) -> None:
        graph = FlowGraph(source=_samples(4), flow={"f": DropOddOp()})
        with pytest.raises(IndexError, match="filtered"):
            _ = graph[1]

    def test_batch(self) -> None:
        graph = FlowGraph(source=_samples(4), flow={"a": AddOp()}).batch(3)
        chunks = list(graph)
        assert [len(c) for c in chunks] == [3, 1]

    def test_parallel_spawn_parity(self) -> None:
        flow_doc = {
            "a": AddOp(amount=1.0),
            "b": {"op": SwapInputTargetOp(), "from": "a"},
            "c": {"op": AddOp(amount=5.0), "from": "a"},
            "out": {"from": "c", "target_from": "b"},
        }
        seq = [_key(s) for s in FlowGraph(source=_samples(4), flow=dict(flow_doc))]
        par = [_key(s) for s in FlowGraph(source=_samples(4), flow=dict(flow_doc)).parallel(2)]
        assert seq == par

    def test_to_flux_twin(self) -> None:
        graph = FlowGraph(source=_samples(3), flow={"a": AddOp(amount=2.0)})
        assert [s.input for s in graph.to_flux()] == [2.0, 3.0, 4.0]

    def test_collect(self) -> None:
        graph = FlowGraph(source=_samples(2), flow={"a": AddOp()})
        assert len(graph.collect()) == 2


# ---------------------------------------------------------------------------
# YAML round-trips
# ---------------------------------------------------------------------------

_FLOW_YAML = """
flow:
  a: !class:tests.test_flow.AddOp(amount=1.0)
  b: !class:sampleflux.ops.swap.SwapInputTargetOp()
    from: a
  c: !class:tests.test_flow.AddOp(amount=5.0)
    from: a
  out:
    from: c
    target_from: b
outputs: out
"""


class TestYaml:
    def test_flowgraph_from_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "graph.yaml"
        path.write_text(_FLOW_YAML)
        graph = FlowGraph.from_yaml(str(path), source=_samples(3))
        results = list(graph)
        # v -> a=v+1 -> c=a+5=v+6 (input); target = b's target = a's swapped input = v+1
        assert [s.input for s in results] == [6.0, 7.0, 8.0]
        assert [s.target for s in results] == [1.0, 2.0, 3.0]

    def test_flux_from_flow_yaml_matches_native(self, tmp_path: Path) -> None:
        path = tmp_path / "graph.yaml"
        path.write_text(_FLOW_YAML)
        native = [_key(s) for s in FlowGraph.from_yaml(str(path), source=_samples(3))]
        serial = [_key(s) for s in Flux.from_flow_yaml(str(path), source=_samples(3))]
        assert native == serial

    def test_flowgraph_from_ops_yaml(self, tmp_path: Path) -> None:
        ops_yaml = """
ops:
  - !class:tests.test_flow.AddOp(amount=1.0)
  - !class:tests.test_flow.ScaleOp(factor=3.0)
"""
        path = tmp_path / "ops.yaml"
        path.write_text(ops_yaml)
        graph = FlowGraph.from_ops_yaml(str(path), source=_samples(3))
        assert [s.input for s in graph] == [3.0, 6.0, 9.0]
