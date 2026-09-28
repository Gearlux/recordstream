"""The contract of the tracing engine, ``recordstream.flow.trace`` — written BEFORE the engine.

A tracer runs ONE record through a ``Stream`` or a ``FlowGraph`` on recordstream's own kernel, keeps a
snapshot of what every node received and produced, stops before a named node, steps, resumes, and
reruns one node with new parameters so that only the nodes after it recompute. This file pins that
behaviour with synthetic ops over small numpy records, so the engine is built against it.

The module does not exist yet: ``pytest.importorskip`` below SKIPS the whole file (it never fails)
until ``recordstream/flow/trace.py`` lands, and says so in the skip reason. Only a MISSING module
skips — a module that exists but is broken fails, as it should.
"""

import json
import time
from typing import Any, Callable, ClassVar, Dict, List, Literal, Tuple

import confluid
import numpy as np
import pytest
from confluid import configurable

from recordstream import FlowGraph, Image, Label, Mask, Record, Stream
from recordstream.ops.contract import ANY_TYPE, ChainContractError, check_chain

trace = pytest.importorskip(
    "recordstream.flow.trace",
    reason="recordstream.flow.trace (the phase-1 engine) has not landed yet — this file is its contract",
    exc_type=ModuleNotFoundError,
)
Tracer = trace.Tracer

# --------------------------------------------------------------------------------------------
# synthetic ops — dict records, small numpy arrays, no domain package
# --------------------------------------------------------------------------------------------


def _pixels() -> np.ndarray:
    """A 16x16 float32 ramp from 0 to 1 — deterministic, so every threshold below has a known answer."""
    return np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)


def _record() -> Record:
    return {"image": Image(_pixels()), "class": Label(value="cat")}


def _background(percentile: float) -> float:
    """What ``TraceBackgroundLevel`` computes for the ramp — recomputed here so a test states its expectation."""
    per_row = np.median(np.asarray(_pixels(), dtype=np.float64), axis=1)
    return float(np.percentile(per_row, percentile))


@configurable(category="op", group="test")
class TraceBackgroundLevel:
    """Writes ``background``: a percentile of the image's per-row medians. Declares consumes/produces BY NAME."""

    consumes = {"image": "Image"}
    produces = {"background": ANY_TYPE}
    # Counted on the CLASS so a node the tracer REBUILT (a new instance) still counts its runs.
    calls: ClassVar[int] = 0

    def __init__(self, percentile: float = 25.0) -> None:
        self.percentile = percentile

    def __call__(self, record: Record) -> Record:
        type(self).calls += 1
        per_row = np.median(np.asarray(record["image"], dtype=np.float64), axis=1)
        return {**record, "background": float(np.percentile(per_row, self.percentile))}


@configurable(category="op", group="test")
class TraceThreshold:
    """Writes ``mask``: the pixels above (or below) ``level``.

    ``mode`` is a closed ``Literal`` so a value outside it is REFUSED by the constructor — confluid's
    validation — which is what a rerun with a bad parameter must run into.
    """

    consumes = {"image": "Image"}
    produces = {"mask": "Mask"}
    calls: ClassVar[int] = 0

    def __init__(self, level: float = 0.5, mode: Literal["above", "below"] = "above") -> None:
        self.level = level
        self.mode = mode

    def __call__(self, record: Record) -> Record:
        type(self).calls += 1
        pixels = np.asarray(record["image"])
        on = pixels > self.level if self.mode == "above" else pixels < self.level
        return {**record, "mask": Mask(on)}


@configurable(category="op", group="test")
class CountOn:
    """Writes ``on``: how many mask pixels are set. Has no parameters at all."""

    consumes = {"mask": "Mask"}
    produces = {"on": ANY_TYPE}
    calls: ClassVar[int] = 0

    def __call__(self, record: Record) -> Record:
        type(self).calls += 1
        return {**record, "on": int(np.asarray(record["mask"]).sum())}


@configurable(category="op", group="test")
class Blank:
    """MUTATES the record it is given: overwrites ``image`` with zeros in place and returns the same dict."""

    consumes = {"image": "Image"}
    produces = {"image": "Image"}

    def __call__(self, record: Record) -> Record:
        record["image"] = Image(np.zeros_like(np.asarray(record["image"])))
        return record


@configurable(category="op", group="test")
class Explode:
    """Raises at RUN time while ``fail`` is set — a value the constructor accepts and the op rejects."""

    def __init__(self, fail: bool = True) -> None:
        self.fail = fail

    def __call__(self, record: Record) -> Record:
        if self.fail:
            raise RuntimeError("boom: Explode was told to fail")
        return {**record, "survived": True}


@configurable(category="op", group="test")
class Classify:
    """Raises the flag ``bright`` when the background is above ``cutoff`` — the node a gate points back at."""

    consumes = {"background": ANY_TYPE}
    flags: Tuple[str, ...] = ("bright",)

    def __init__(self, cutoff: float = 0.1) -> None:
        self.cutoff = cutoff

    def __call__(self, record: Record) -> Record:
        return {**record, "flags": {"bright": record["background"] > self.cutoff}}


@configurable(category="op", group="test")
class Decode:
    """Gated on the flag named by ``requires``: writes ``decoded`` only when that flag is raised."""

    consumes = {"image": "Image"}
    produces = {"decoded": ANY_TYPE}

    def __init__(self, requires: str = "") -> None:
        self.requires = requires

    def __call__(self, record: Record) -> Record:
        if self.requires and not record.get("flags", {}).get(self.requires, False):
            return dict(record)
        return {**record, "decoded": "ramp"}


@configurable(category="op", group="test")
class Counting:
    """Counts its CONSTRUCTIONS — built from a ``!class:`` marker to show WHEN the tracer builds a node."""

    built: ClassVar[int] = 0

    def __init__(self) -> None:
        type(self).built += 1

    def __call__(self, record: Record) -> Record:
        return record


def _chain() -> List[Any]:
    """The three-node chain most tests run: background -> mask -> count (nodes ops[0], ops[1], ops[2])."""
    return [TraceBackgroundLevel(), TraceThreshold(level=0.5), CountOn()]


def _forked_flow() -> Dict[str, Any]:
    """A graph that forks at ``start`` and re-binds ``gated.level`` from what ``floor`` computed."""
    return {
        "start": {},
        "floor": {"op": TraceBackgroundLevel(percentile=25.0), "from": "start"},
        "gated": {"op": TraceThreshold(), "from": "start", "bind": {"level": "floor[background]"}},
        "count": {"op": CountOn(), "from": "gated"},
    }


def _longest_list(value: object) -> int:
    """The longest list anywhere inside a JSON-like value — a dumped 16x16 array betrays itself as 16."""
    if isinstance(value, list):
        return max([len(value)] + [_longest_list(item) for item in value])
    if isinstance(value, dict):
        return max([0] + [_longest_list(item) for item in value.values()])
    return 0


def _best_of(repetitions: int, measure: Callable[[], None]) -> float:
    """The fastest of several timings — what survives a GC pause or a busy neighbour on the machine."""
    best = float("inf")
    for _ in range(repetitions):
        started = time.perf_counter()
        measure()
        best = min(best, time.perf_counter() - started)
    return best


# --------------------------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------------------------


class TestConstructionIsLazy:
    """``Tracer(graph)`` does no work: nothing is parsed or built until the first use."""

    def test_a_graph_that_cannot_be_parsed_is_accepted_and_refused_on_first_use(self) -> None:
        # `late` reads a step that comes AFTER it — parse_flow refuses that; the tracer must not parse yet.
        graph = FlowGraph(flow={"late": {"op": TraceThreshold(), "from": "early"}, "early": TraceBackgroundLevel()})
        tracer = Tracer(graph)
        with pytest.raises(ValueError, match="EARLIER step"):
            _ = tracer.names

    def test_no_node_is_built_until_first_use(self) -> None:
        # A settled marker is unbuilt; Counting counts its constructions, so the moment of building is visible.
        document = confluid.load("flow:\n  count: !class:tests.test_trace.Counting {}\n", until="settled")
        built = Counting.built
        tracer = Tracer(FlowGraph(flow=document["flow"]))
        assert Counting.built == built, "the constructor built the node"
        assert tracer.names == ["count"]
        assert Counting.built == built + 1, "the first use builds it"

    def test_a_flow_graph_without_a_flow_is_accepted_and_refused_on_first_use(self) -> None:
        tracer = Tracer(FlowGraph())
        with pytest.raises(ValueError, match="flow is not set"):
            _ = tracer.names

    def test_before_any_run_every_node_is_not_reached(self) -> None:
        tracer = Tracer(Stream(ops=_chain()))
        assert tracer.statuses == {"ops[0]": "not reached", "ops[1]": "not reached", "ops[2]": "not reached"}
        assert tracer.paused_at is None
        assert tracer.result is None


# --------------------------------------------------------------------------------------------
# the static check
# --------------------------------------------------------------------------------------------


class TestCheckBeforeRun:
    """``check(seed)`` refuses an unmet need BEFORE anything runs, and passes a chain whose gate is met."""

    def test_a_need_nothing_produces_is_refused_by_name_and_nothing_runs(self) -> None:
        seed = {"picture": Image(_pixels()), "class": Label(value="cat")}  # `picture`, not `image`
        tracer = Tracer(Stream(ops=_chain()))
        calls = TraceBackgroundLevel.calls
        with pytest.raises(ChainContractError) as refusal:
            tracer.check(seed)
        message = str(refusal.value)
        assert "TraceBackgroundLevel needs the record entry 'image'" in message
        assert "ops[0]" in message, "the refusal is located at the node, so a canvas can mark it"
        with pytest.raises(ChainContractError):
            tracer.run(seed)
        assert tracer.statuses == {"ops[0]": "not reached", "ops[1]": "not reached", "ops[2]": "not reached"}
        assert tracer.result is None
        assert TraceBackgroundLevel.calls == calls, "no op ran"

    def test_a_gate_on_a_flag_nobody_raises_is_refused_naming_the_flag(self) -> None:
        tracer = Tracer(Stream(ops=[TraceBackgroundLevel(), Decode(requires="bright")]))
        with pytest.raises(ChainContractError) as refusal:
            tracer.check(_record())
        assert "bright" in str(refusal.value) and "Decode" in str(refusal.value)

    def test_a_gated_chain_passes_the_check_and_runs_whole(self) -> None:
        """The flag is carried ACROSS nodes: checked one node at a time, the gate would be refused."""
        available = {"image", "class", "background", "flags"}
        with pytest.raises(ChainContractError, match="gated on the flag 'bright'"):
            check_chain([Decode(requires="bright")], provided=available)  # the per-node check: the CON case
        tracer = Tracer(Stream(ops=[TraceBackgroundLevel(), Classify(), Decode(requires="bright")]))
        calls = TraceBackgroundLevel.calls
        report = tracer.check(_record())
        assert TraceBackgroundLevel.calls == calls, "checking runs nothing"
        assert [row["node"] for row in report] == ["ops[0]", "ops[1]", "ops[2]"]
        tracer.run(_record())
        assert tracer.statuses == {"ops[0]": "ok", "ops[1]": "ok", "ops[2]": "ok"}
        assert tracer.result[0]["decoded"] == "ramp"

    def test_the_report_has_one_json_row_per_node(self) -> None:
        tracer = Tracer(Stream(ops=_chain()))
        report = tracer.check(_record())
        assert [row["node"] for row in report] == tracer.names
        json.dumps(report)


# --------------------------------------------------------------------------------------------
# run, stop before a node, step, resume
# --------------------------------------------------------------------------------------------


class TestRunStepResume:
    """``run`` yields the kernel's own answer; ``until`` pauses BEFORE a node; ``step``/``resume`` continue."""

    def test_a_full_run_yields_what_a_plain_stream_yields(self) -> None:
        tracer = Tracer(Stream(ops=_chain()))
        assert tracer.run(_record()) is tracer
        (plain,) = list(Stream(source=[_record()], ops=_chain()))
        assert tracer.result[0]["on"] == plain["on"] == 128
        assert np.array_equal(np.asarray(tracer.result[0]["mask"]), np.asarray(plain["mask"]))
        assert tracer.statuses == {"ops[0]": "ok", "ops[1]": "ok", "ops[2]": "ok"}
        assert tracer.paused_at is None

    def test_until_stops_before_the_named_node_with_everything_so_far_recorded(self) -> None:
        tracer = Tracer(Stream(ops=_chain()))
        calls = TraceThreshold.calls
        tracer.run(_record(), until="ops[1]")
        assert tracer.paused_at == "ops[1]"
        assert tracer.statuses == {"ops[0]": "ok", "ops[1]": "paused", "ops[2]": "not reached"}
        assert tracer.result is None
        assert TraceThreshold.calls == calls, "the paused node did not run"
        assert tracer.value("ops[1]", "background", side="input") == pytest.approx(_background(25.0))

    def test_pausing_at_the_first_node_runs_nothing(self) -> None:
        tracer = Tracer(Stream(ops=_chain()))
        calls = TraceBackgroundLevel.calls
        tracer.run(_record(), until="ops[0]")
        assert tracer.statuses == {"ops[0]": "paused", "ops[1]": "not reached", "ops[2]": "not reached"}
        assert TraceBackgroundLevel.calls == calls

    def test_step_runs_the_paused_node_and_pauses_at_the_next(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record(), until="ops[1]")
        counted = CountOn.calls
        assert tracer.step() is tracer
        assert tracer.paused_at == "ops[2]"
        assert tracer.statuses == {"ops[0]": "ok", "ops[1]": "ok", "ops[2]": "paused"}
        assert tracer.result is None and CountOn.calls == counted
        finished = tracer.step()  # the last node: the run completes
        assert finished.paused_at is None
        assert finished.statuses == {"ops[0]": "ok", "ops[1]": "ok", "ops[2]": "ok"}
        assert finished.result[0]["on"] == 128

    def test_resume_runs_to_the_end(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record(), until="ops[1]")
        assert tracer.resume() is tracer
        assert tracer.paused_at is None
        assert tracer.statuses == {"ops[0]": "ok", "ops[1]": "ok", "ops[2]": "ok"}
        assert tracer.result[0]["on"] == 128

    def test_step_and_resume_refuse_when_nothing_is_paused(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record())
        with pytest.raises(RuntimeError):
            tracer.step()
        with pytest.raises(RuntimeError):
            tracer.resume()

    def test_until_naming_no_node_is_refused_by_name(self) -> None:
        with pytest.raises(KeyError, match="nosuch"):
            Tracer(Stream(ops=_chain())).run(_record(), until="nosuch")


# --------------------------------------------------------------------------------------------
# rerun one node
# --------------------------------------------------------------------------------------------


class TestRerun:
    """``rerun_from`` rebuilds ONE node through its constructor and recomputes from there on."""

    def test_only_that_node_and_the_ones_after_it_recompute(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record())
        before = (TraceBackgroundLevel.calls, TraceThreshold.calls, CountOn.calls)
        assert tracer.rerun_from("ops[1]", level=0.75) is tracer
        assert TraceBackgroundLevel.calls == before[0], "the node before did not run again"
        assert (TraceThreshold.calls, CountOn.calls) == (before[1] + 1, before[2] + 1)
        assert tracer.generations == {"ops[0]": 0, "ops[1]": 1, "ops[2]": 1}
        assert tracer.statuses == {"ops[0]": "ok", "ops[1]": "ok", "ops[2]": "ok"}
        assert tracer.result[0]["on"] == int((_pixels() > 0.75).sum())

    def test_a_rerun_without_parameters_recomputes_from_that_node_unchanged(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record())
        tracer.rerun_from("ops[2]")
        assert tracer.generations == {"ops[0]": 0, "ops[1]": 0, "ops[2]": 1}
        assert tracer.result[0]["on"] == 128

    def test_a_value_the_host_set_afterwards_survives_the_rebuild(self) -> None:
        """The rebuild starts from the node's CURRENT values, not from what its constructor was given."""
        gate = TraceThreshold(level=0.5)
        gate.level = 0.7  # the host sets it after construction (a viewer's window, a selection)
        tracer = Tracer(Stream(ops=[TraceBackgroundLevel(), gate, CountOn()])).run(_record())
        tracer.rerun_from("ops[1]", mode="below")
        mask = np.asarray(tracer.value("ops[1]", "mask"))
        assert np.array_equal(mask, _pixels() < 0.7), "level 0.7 survived, mode changed"
        assert not np.array_equal(mask, _pixels() < 0.5), "a rebuild from the constructor's kwargs would give this"
        assert tracer.to_dict()["nodes"][1]["params"]["level"] == 0.7

    def test_a_value_the_constructor_refuses_leaves_the_trace_untouched(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record())
        before = tracer.to_dict()
        calls = (TraceThreshold.calls, CountOn.calls)
        with pytest.raises(ValueError, match="sideways"):
            tracer.rerun_from("ops[1]", mode="sideways")
        assert tracer.to_dict() == before
        assert tracer.generations == {"ops[0]": 0, "ops[1]": 0, "ops[2]": 0}
        assert (TraceThreshold.calls, CountOn.calls) == calls, "nothing reran"
        tracer.rerun_from("ops[1]")  # the live node is not poisoned: it still runs with mode 'above'
        assert tracer.result[0]["on"] == 128

    def test_a_node_that_raises_leaves_the_later_nodes_not_reached_and_a_rerun_recovers(self) -> None:
        tracer = Tracer(Stream(ops=[TraceBackgroundLevel(), Explode(fail=True), TraceThreshold()]))
        with pytest.raises(RuntimeError, match="boom"):
            tracer.run(_record())
        assert tracer.statuses == {"ops[0]": "ok", "ops[1]": "error", "ops[2]": "not reached"}
        assert tracer.result is None
        assert "boom" in tracer.to_dict()["nodes"][1]["error"]
        recovered = tracer.rerun_from("ops[1]", fail=False)
        assert recovered.statuses == {"ops[0]": "ok", "ops[1]": "ok", "ops[2]": "ok"}
        assert recovered.generations == {"ops[0]": 0, "ops[1]": 1, "ops[2]": 1}
        assert recovered.result[0]["survived"] is True

    def test_a_rerun_from_a_node_never_reached_is_refused(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record(), until="ops[1]")
        with pytest.raises(RuntimeError, match=r"ops\[2\]"):
            tracer.rerun_from("ops[2]")

    def test_a_rerun_from_an_unknown_node_is_refused_by_name(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record())
        with pytest.raises(KeyError, match="nosuch"):
            tracer.rerun_from("nosuch")

    def test_on_a_forked_graph_a_rerun_rebinds_what_later_nodes_read(self) -> None:
        tracer = Tracer(FlowGraph(flow=_forked_flow(), outputs="count")).run(_record())
        low = _background(25.0)
        assert tracer.to_dict()["nodes"][2]["bound"]["level"] == pytest.approx(low)
        assert tracer.result[0]["on"] == int((_pixels() > low).sum())
        tracer.rerun_from("floor", percentile=90.0)
        high = _background(90.0)
        assert tracer.generations == {"start": 0, "floor": 1, "gated": 1, "count": 1}
        assert tracer.to_dict()["nodes"][2]["bound"]["level"] == pytest.approx(high)
        assert tracer.result[0]["on"] == int((_pixels() > high).sum())


# --------------------------------------------------------------------------------------------
# snapshots
# --------------------------------------------------------------------------------------------


class TestSnapshots:
    """Snapshots are references unless ``copy_snapshots=True``; an in-place edit is flagged."""

    def test_snapshots_are_references_by_default(self) -> None:
        seed = _record()
        tracer = Tracer(Stream(ops=_chain())).run(seed)
        assert tracer.value("ops[0]", "image", side="input") is seed["image"]
        assert tracer.value("ops[1]", "image", side="input") is seed["image"]

    def test_copy_snapshots_makes_input_and_output_independent_copies(self) -> None:
        seed = _record()
        tracer = Tracer(Stream(ops=_chain()), copy_snapshots=True).run(seed)
        received = tracer.value("ops[0]", "image", side="input")
        returned = tracer.value("ops[0]", "image", side="output")
        assert received is not seed["image"] and returned is not seed["image"]
        assert received is not returned
        assert np.array_equal(np.asarray(received), _pixels())
        assert tracer.value("ops[1]", "mask") is not tracer.value("ops[2]", "mask", side="input")

    def test_an_op_that_mutates_its_input_in_place_is_flagged(self) -> None:
        nodes = Tracer(Stream(ops=[TraceBackgroundLevel(), Blank()])).run(_record()).to_dict()["nodes"]
        assert nodes[0].get("in_place", False) is False
        assert nodes[1]["in_place"] is True

    def test_with_copies_the_input_snapshot_keeps_what_the_op_overwrote(self) -> None:
        tracer = Tracer(Stream(ops=[Blank()]), copy_snapshots=True).run(_record())
        assert float(np.asarray(tracer.value("ops[0]", "image", side="input")).max()) == 1.0
        assert float(np.asarray(tracer.value("ops[0]", "image", side="output")).max()) == 0.0


# --------------------------------------------------------------------------------------------
# the trace a page or a tool consumes
# --------------------------------------------------------------------------------------------


class TestTrace:
    """``to_dict`` is plain JSON with arrays summarised; ``value`` hands out the real object on request."""

    def test_to_dict_is_json_serialisable_without_a_fallback_encoder(self) -> None:
        json.dumps(Tracer(Stream(ops=_chain())).run(_record()).to_dict())

    def test_arrays_are_summarised_never_dumped(self) -> None:
        report = Tracer(Stream(ops=_chain())).run(_record()).to_dict()
        image = report["nodes"][0]["input"]["image"]
        assert image["type"] == "Image" and image["shape"] == [16, 16] and image["dtype"] == "float32"
        assert image["min"] == pytest.approx(0.0) and image["max"] == pytest.approx(1.0)
        mask = report["nodes"][1]["output"]["mask"]
        assert mask["type"] == "Mask" and mask["shape"] == [16, 16]
        assert mask["true_fraction"] == pytest.approx(0.5)
        assert report["nodes"][0]["input"]["class"] == {"type": "Label", "value": "cat"}
        assert _longest_list(report) <= 8, "a 16x16 array dumped as nested lists would show as 16"

    def test_every_node_that_ran_reports_its_wall_time_in_ms(self) -> None:
        for node in Tracer(Stream(ops=_chain())).run(_record()).to_dict()["nodes"]:
            assert node["status"] == "ok"
            assert isinstance(node["ms"], (int, float)) and node["ms"] >= 0

    def test_the_report_mirrors_names_statuses_generations_params_and_the_result(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record())
        report = tracer.to_dict()
        assert [node["node"] for node in report["nodes"]] == tracer.names
        assert {node["node"]: node["status"] for node in report["nodes"]} == tracer.statuses
        assert {node["node"]: node["generation"] for node in report["nodes"]} == tracer.generations
        assert report["nodes"][1]["params"]["level"] == 0.5
        assert report["paused_at"] is None
        assert report["result"][0]["mask"]["type"] == "Mask"

    def test_a_paused_run_reports_the_pause(self) -> None:
        report = Tracer(Stream(ops=_chain())).run(_record(), until="ops[1]").to_dict()
        assert report["paused_at"] == "ops[1]" and report["result"] is None
        assert [node["status"] for node in report["nodes"]] == ["ok", "paused", "not reached"]

    def test_value_returns_the_real_object_not_a_summary(self) -> None:
        mask = Tracer(Stream(ops=_chain())).run(_record()).value("ops[1]", "mask")
        assert isinstance(mask, Mask) and np.asarray(mask).shape == (16, 16)
        assert int(np.asarray(mask).sum()) == 128

    def test_value_of_an_entry_or_node_that_does_not_exist_is_refused_by_name(self) -> None:
        tracer = Tracer(Stream(ops=_chain())).run(_record())
        with pytest.raises(KeyError, match="nosuch"):
            tracer.value("ops[1]", "nosuch")
        with pytest.raises(KeyError, match="nosuch"):
            tracer.value("nosuch", "mask")


# --------------------------------------------------------------------------------------------
# node names
# --------------------------------------------------------------------------------------------


class TestNaming:
    """One naming scheme: a Stream's ops are ``ops[i]``, a flow's steps are their keys."""

    def test_a_stream_names_its_nodes_by_position_in_the_ops_list(self) -> None:
        assert Tracer(Stream(ops=_chain())).names == ["ops[0]", "ops[1]", "ops[2]"]

    def test_the_same_op_class_twice_is_two_nodes(self) -> None:
        assert Tracer(Stream(ops=[TraceThreshold(), TraceThreshold(level=0.9)])).names == ["ops[0]", "ops[1]"]

    def test_a_flow_graph_names_its_nodes_by_step_key(self) -> None:
        assert Tracer(FlowGraph(flow=_forked_flow(), outputs="count")).names == ["start", "floor", "gated", "count"]

    def test_a_fan_in_step_without_an_op_is_a_node_too(self) -> None:
        flow = {"start": {}, "level": TraceBackgroundLevel(), "out": {"from": "level", "merge_from": ["start"]}}
        tracer = Tracer(FlowGraph(flow=flow)).run(_record())
        assert tracer.names == ["start", "level", "out"]
        assert tracer.statuses == {"start": "ok", "level": "ok", "out": "ok"}
        assert tracer.result[0]["background"] == pytest.approx(_background(25.0))


# --------------------------------------------------------------------------------------------
# overhead
# --------------------------------------------------------------------------------------------


class TestOverhead:
    """Tracing by reference stays within a loose bound of a plain run (measured 1.3x on the prototype)."""

    def test_tracing_costs_at_most_three_times_a_plain_run(self) -> None:
        records = [_record() for _ in range(200)]
        plain = Stream(source=records, ops=_chain())
        tracer = Tracer(Stream(ops=_chain()))

        def run_plain() -> None:
            for _ in plain:
                pass

        def run_traced() -> None:
            for record in records:
                tracer.run(record)

        run_plain()  # warm both paths once
        run_traced()
        plain_s, traced_s = _best_of(5, run_plain), _best_of(5, run_traced)
        assert (
            traced_s <= 3 * plain_s
        ), f"traced {traced_s * 1e3 / 200:.4f} ms/record vs plain {plain_s * 1e3 / 200:.4f} ms/record"
