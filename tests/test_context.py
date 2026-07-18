"""Tests for the per-sample Context (`sampleflux.context`) and the context ops
(`sampleflux.ops.context`: Save / Use / Drop / Apply / Capture / Mix).

Covers the Phase-2 contract of the graph execution model:
- a flat op list containing context ops executes a fan-out/fan-in graph on the plain
  Flux engine (sequential, spawn-parallel, streamed, and random-access routes);
- Context never touches ``sample.metadata`` (the metadata-untouched invariant);
- copy-vs-move semantics mirror the stash family (`Use` deep-copies unless it drops);
- every context op round-trips through confluid dump/load (Pipeline Parity).
"""

from pathlib import Path

import numpy as np
import pytest
from confluid import configurable, dump, load, materialize, output

from sampleflux.context import Context, activate, current, require
from sampleflux.core import Flux
from sampleflux.ops.context import Apply, Capture, Drop, Mix, Save, Use
from sampleflux.ops.swap import SwapInputTargetOp
from sampleflux.sample import Sample

# ---------------------------------------------------------------------------
# Test ops (module-level so they pickle for the spawn route)
# ---------------------------------------------------------------------------


@configurable
class AddOp:
    """Add a constant to the (numeric or array) input.

    Args:
        amount: Value added to ``sample.input`` on every call.
    """

    def __init__(self, amount: float = 1.0) -> None:
        self.amount = amount

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input + self.amount)


@configurable
class ScaleOp:
    """Multiply the input by a factor.

    Args:
        factor: Multiplier applied to ``sample.input`` on every call.
    """

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input * self.factor)


@configurable
class StampOp:
    """Write one metadata key (tests branch-metadata survival through Mix).

    Args:
        key: Metadata key to write.
        value: Value written under ``key``.
    """

    def __init__(self, key: str = "stamp", value: str = "x") -> None:
        self.key = key
        self.value = value

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(metadata={**sample.meta, self.key: self.value})


@configurable
class DrawOp:
    """Pass-through op with a stochastic-style @output (captures must read the real run)."""

    def __init__(self) -> None:
        self._last: float = 0.0
        self._calls: int = 0

    @property
    @output
    def drawn(self) -> float:
        """The value produced by the last application."""
        return self._last

    def __call__(self, sample: Sample) -> Sample:
        self._calls += 1
        self._last = float(sample.input) * 10.0 + self._calls
        return sample


@configurable
class MutateInPlaceOp:
    """Deliberately mutate the input array IN PLACE (isolation tests)."""

    def __call__(self, sample: Sample) -> Sample:
        sample.input[0] = -999.0
        return sample


def _samples(n: int = 3) -> list:
    return [Sample(input=float(i), target=i, metadata={"idx": i}) for i in range(n)]


# ---------------------------------------------------------------------------
# Context core
# ---------------------------------------------------------------------------


class TestContext:
    def test_put_get_delete_live(self) -> None:
        ctx = Context()
        ctx.put("a", 1)
        ctx.put("b", 2)
        assert ctx.get("a") == 1
        assert ctx.live() == ("a", "b")
        assert "a" in ctx and len(ctx) == 2
        ctx.delete("a")
        assert ctx.live() == ("b",)

    def test_get_missing_is_actionable(self) -> None:
        with pytest.raises(KeyError, match="live cells"):
            Context().get("nope")

    def test_delete_missing_is_actionable(self) -> None:
        with pytest.raises(KeyError, match="missing cell"):
            Context().delete("nope")

    def test_copy_is_shallow_with_independent_cell_set(self) -> None:
        ctx = Context()
        payload = [1, 2]
        ctx.put("a", payload)
        clone = ctx.copy()
        clone.delete("a")
        assert "a" in ctx  # independent cell set
        assert ctx.get("a") is payload  # shared values (shallow)

    def test_activate_sets_and_resets(self) -> None:
        assert current() is None
        ctx = Context()
        with activate(ctx):
            assert current() is ctx
        assert current() is None

    def test_require_outside_engine_is_actionable(self) -> None:
        with pytest.raises(RuntimeError, match="no active Context"):
            require("Save")


# ---------------------------------------------------------------------------
# Context ops — unit behavior
# ---------------------------------------------------------------------------


class TestContextOps:
    def test_save_requires_name(self) -> None:
        with activate(Context()):
            with pytest.raises(ValueError, match="'name'"):
                Save()(Sample(1))

    def test_use_requires_name(self) -> None:
        with activate(Context()):
            with pytest.raises(ValueError, match="'name'"):
                Use()(Sample(1))

    def test_save_then_use_copies_by_default(self) -> None:
        s = Sample(input=np.array([1.0, 2.0]), metadata={"m": 1})
        with activate(Context()) as ctx:
            Save(name="cell")(s)
            restored = Use(name="cell")(Sample(input=None))
            assert restored.input is not s.input  # deep copy
            np.testing.assert_array_equal(restored.input, s.input)
            assert "cell" in ctx  # kept

    def test_use_with_drop_moves_without_copy(self) -> None:
        s = Sample(input=np.array([1.0, 2.0]))
        with activate(Context()) as ctx:
            Save(name="cell")(s)
            restored = Use(name="cell", drop=True)(Sample(input=None))
            assert restored.input is s.input  # move: no copy
            assert "cell" not in ctx  # freed

    def test_two_readers_are_isolated_against_inplace_mutation(self) -> None:
        s = Sample(input=np.array([1.0, 2.0]))
        with activate(Context()):
            Save(name="fork")(s)
            branch_a = Use(name="fork")(Sample(input=None))
            MutateInPlaceOp()(branch_a)  # mutates branch A's copy in place
            branch_b = Use(name="fork", drop=True)(Sample(input=None))
            assert branch_b.input[0] == 1.0  # untouched by branch A

    def test_use_coerces_raw_cell_value(self) -> None:
        with activate(Context()) as ctx:
            ctx.put("raw", 42.0)
            restored = Use(name="raw", drop=True)(Sample(input=None))
            assert restored.input == 42.0

    def test_drop_frees_cells_and_flags_liveness_bugs(self) -> None:
        with activate(Context()) as ctx:
            ctx.put("a", 1)
            ctx.put("b", 2)
            Drop(names=["a", "b"])(Sample(1))
            assert ctx.live() == ()
            with pytest.raises(KeyError, match="missing cell"):
                Drop(names=["a"])(Sample(1))

    def test_drop_empty_is_noop_without_context(self) -> None:
        # No names -> never needs the Context (works outside an engine too).
        assert Drop()(Sample(1)).input == 1

    def test_apply_sets_param_from_sample_cell_input(self) -> None:
        with activate(Context()) as ctx:
            ctx.put("thresh", Sample(input=5.0))
            op = Apply(op=AddOp(amount=0.0), param="amount", source="thresh", drop=True)
            result = op(Sample(input=1.0))
            assert result is not None and result.input == 6.0
            assert "thresh" not in ctx

    def test_apply_sets_param_from_raw_cell_value(self) -> None:
        with activate(Context()) as ctx:
            ctx.put("factor", 3.0)
            result = Apply(op=ScaleOp(), param="factor", source="factor")(Sample(input=2.0))
            assert result is not None and result.input == 6.0

    def test_apply_validations(self) -> None:
        with activate(Context()):
            with pytest.raises(ValueError, match="'op'"):
                Apply(param="p", source="s")(Sample(1))
            with pytest.raises(ValueError, match="'param'"):
                Apply(op=AddOp(), source="s")(Sample(1))
            with pytest.raises(ValueError, match="'source'"):
                Apply(op=AddOp(), param="p")(Sample(1))

    def test_capture_records_live_output_into_cell(self) -> None:
        with activate(Context()) as ctx:
            draw = DrawOp()
            result = Capture(op=draw, output="drawn", name="snr")(Sample(input=2.0))
            assert result is not None
            assert ctx.get("snr") == 21.0  # 2*10 + 1st call — the REAL run's value
            # A second application overwrites with the fresh draw (stochastic-correct).
            Capture(op=draw, output="drawn", name="snr")(Sample(input=2.0))
            assert ctx.get("snr") == 22.0

    def test_capture_reads_through_apply_wrapper(self) -> None:
        with activate(Context()) as ctx:
            ctx.put("noop", 0.0)
            inner = DrawOp()
            wrapped = Apply(op=inner, param="_unused", source="noop")
            Capture(op=wrapped, output="drawn", name="d")(Sample(input=1.0))
            assert ctx.get("d") == 11.0

    def test_capture_missing_output_is_actionable(self) -> None:
        with activate(Context()):
            with pytest.raises(AttributeError, match="has no @output attribute"):
                Capture(op=AddOp(), output="nope")(Sample(1.0))

    def test_capture_then_apply_wires_output_to_param(self) -> None:
        with activate(Context()):
            Capture(op=DrawOp(), output="drawn", name="d")(Sample(input=1.0))
            result = Apply(op=AddOp(amount=0.0), param="amount", source="d", drop=True)(Sample(input=0.5))
            assert result is not None and result.input == 0.5 + 11.0

    def test_mix_slots_and_metadata_merge_order(self) -> None:
        with activate(Context()) as ctx:
            ctx.put("a", Sample(input="A", target="tA", metadata={"who": "a", "a_only": 1}))
            ctx.put("b", Sample(input="B", target="tB", metadata={"who": "b", "b_only": 2}))
            incoming = Sample(input="in", target="t_in", metadata={"who": "incoming", "in_only": 0})
            mixed = Mix(input_from="a", target_from="b", drop=["a", "b"])(incoming)
            assert mixed is not None
            assert mixed.input == "A" and mixed.target == "tB"
            # incoming first, then input_from, then target_from (last write wins)
            assert mixed.meta["who"] == "b"
            assert mixed.meta["in_only"] == 0 and mixed.meta["a_only"] == 1 and mixed.meta["b_only"] == 2
            assert ctx.live() == ()

    def test_mix_metadata_from_wins_last(self) -> None:
        with activate(Context()) as ctx:
            ctx.put("a", Sample(input="A", metadata={"who": "a"}))
            ctx.put("m", Sample(input=None, metadata={"who": "meta"}))
            mixed = Mix(input_from="a", metadata_from="m", drop=["a", "m"])(Sample(input="in", metadata={"who": "i"}))
            assert mixed is not None and mixed.meta["who"] == "meta"

    def test_mix_metadata_from_accepts_raw_dict_and_rejects_nondict(self) -> None:
        with activate(Context()) as ctx:
            ctx.put("m", {"k": "v"})
            mixed = Mix(metadata_from="m")(Sample(input="in"))
            assert mixed is not None and mixed.meta["k"] == "v"
            ctx.put("bad", 3.0)
            with pytest.raises(TypeError, match="metadata_from"):
                Mix(metadata_from="bad")(Sample(input="in"))

    def test_mix_empty_slots_keep_incoming(self) -> None:
        with activate(Context()):
            incoming = Sample(input="in", target="t", metadata={"m": 1})
            mixed = Mix()(incoming)
            assert mixed is not None
            assert mixed.input == "in" and mixed.target == "t" and mixed.meta == {"m": 1}

    def test_ops_outside_engine_raise_actionable(self) -> None:
        with pytest.raises(RuntimeError, match="no active Context"):
            Save(name="x")(Sample(1))


# ---------------------------------------------------------------------------
# Engine integration — the four execution routes
# ---------------------------------------------------------------------------

# A fan-out/fan-in graph as a flat op list:
#   fork the incoming value; branch A computes value+1 and swaps it into its TARGET slot
#   (Mix's target_from reads the cell-sample's target field); branch B computes value*2 on
#   the stream; Mix yields input = B's (stream), target = A's (cell).
_GRAPH_OPS = [
    Save(name="fork"),
    AddOp(amount=1.0),  # branch A rides the stream
    SwapInputTargetOp(),  # park A's result in the target field for Mix
    Save(name="branch_a"),
    Use(name="fork", drop=True),  # branch B restarts from the fork
    ScaleOp(factor=2.0),
    Mix(target_from="branch_a", drop=["branch_a"]),  # input = B (stream), target = A
]


def _expected_graph(values: list) -> list:
    return [(v * 2.0, v + 1.0) for v in values]


class TestEngineRoutes:
    def test_sequential_graph_execution(self) -> None:
        flux = Flux(source=_samples(4), ops=list(_GRAPH_OPS))
        got = [(s.input, s.target) for s in flux]
        assert _expected_graph([0.0, 1.0, 2.0, 3.0]) == got

    def test_metadata_untouched_invariant(self) -> None:
        # Context wiring must not leak anything into sample.metadata.
        flux = Flux(source=_samples(3), ops=list(_GRAPH_OPS))
        for i, s in enumerate(flux):
            assert s.meta == {"idx": i}

    def test_random_access_getitem(self) -> None:
        flux = Flux(source=_samples(5), ops=list(_GRAPH_OPS))
        s = flux[3]
        assert s.input == 6.0

    def test_spawn_parallel_parity(self) -> None:
        seq = [s.input for s in Flux(source=_samples(4), ops=list(_GRAPH_OPS))]
        par = [s.input for s in Flux(source=_samples(4), ops=list(_GRAPH_OPS)).parallel(2)]
        assert seq == par

    def test_streamed_route_with_parallel_op(self) -> None:
        from sampleflux.ops.parallel import Parallel

        # Whole graph INSIDE Parallel: each worker's _worker_task provides the Context.
        flux = Flux(source=_samples(4), ops=[Parallel(ops=list(_GRAPH_OPS), workers=2)])
        got = [(s.input, s.target) for s in flux]
        assert got == _expected_graph([0.0, 1.0, 2.0, 3.0])

    def test_streamed_route_cells_may_not_cross_stream_boundary(self) -> None:
        from sampleflux.ops.parallel import Parallel

        flux = Flux(source=_samples(2), ops=[Save(name="fork"), Parallel(ops=[AddOp()], workers=1)])
        with pytest.raises(RuntimeError, match="stream-level op"):
            list(flux)

    def test_streamed_route_context_ops_before_and_after_boundary(self) -> None:
        from sampleflux.ops.parallel import Parallel

        # Cells used and FREED before the boundary, new cells after — both legal.
        ops = [
            Save(name="pre"),
            Use(name="pre", drop=True),
            Parallel(ops=[AddOp(amount=1.0)], workers=1),
            Save(name="post"),
            Mix(target_from="post", drop=["post"]),
        ]
        results = list(Flux(source=_samples(3), ops=ops))
        assert [s.input for s in results] == [1.0, 2.0, 3.0]

    def test_context_is_fresh_per_sample(self) -> None:
        # A cell saved for sample N must never be visible to sample N+1: use a
        # drop-less Save; if contexts leaked across samples, Use would see the
        # PREVIOUS sample's fork (values would shift) or cells would pile up.
        ops = [Save(name="fork"), AddOp(amount=100.0), Use(name="fork")]  # no drop
        results = list(Flux(source=_samples(3), ops=ops))
        assert [s.input for s in results] == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Confluid round-trip (Pipeline Parity) + YAML
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_every_context_op_dump_load_round_trips(self) -> None:
        ops = [
            Save(name="fork"),
            Use(name="fork", drop=True),
            Drop(names=["a", "b"]),
            Apply(op=AddOp(amount=2.0), param="amount", source="cell", drop=True),
            Capture(op=DrawOp(), output="drawn", name="snr"),
            Mix(input_from="a", target_from="b", metadata_from="m", drop=["a"]),
        ]
        for op in ops:
            text = dump(op)
            rebuilt = materialize(load(text))
            assert type(rebuilt) is type(op)
            for attr, value in vars(op).items():
                if attr.startswith("_") or attr == "op":
                    continue  # nested op compared structurally below
                assert getattr(rebuilt, attr) == value, f"{type(op).__name__}.{attr}"

        rebuilt_apply = materialize(load(dump(ops[3])))
        assert type(rebuilt_apply.op).__name__ == "AddOp" and rebuilt_apply.op.amount == 2.0

    def test_graph_ops_yaml_executes_via_from_ops_yaml(self, tmp_path: Path) -> None:
        yaml_text = """
ops:
  - !class:sampleflux.ops.context.Save(name=fork)
  - !class:tests.test_context.AddOp(amount=1.0)
  - !class:sampleflux.ops.swap.SwapInputTargetOp()
  - !class:sampleflux.ops.context.Save(name=branch_a)
  - !class:sampleflux.ops.context.Use(name=fork,drop=true)
  - !class:tests.test_context.ScaleOp(factor=2.0)
  - !class:sampleflux.ops.context.Mix(target_from=branch_a)
    drop: [branch_a]
"""
        path = tmp_path / "graph_ops.yaml"
        path.write_text(yaml_text)
        flux = Flux.from_ops_yaml(str(path), source=_samples(3))
        results = list(flux)
        assert [s.input for s in results] == [0.0, 2.0, 4.0]
        assert [s.target for s in results] == [1.0, 2.0, 3.0]

    def test_linear_pipeline_metadata_byte_identical(self) -> None:
        # A straight sequence (no context ops) — Context threading must be invisible.
        flux = Flux(source=_samples(3), ops=[AddOp(amount=1.0)])
        for i, s in enumerate(flux):
            assert s.meta == {"idx": i}
            assert s.input == float(i) + 1.0
