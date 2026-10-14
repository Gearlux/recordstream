# mypy: disable-error-code="index"
# The tests read FlowStep.op and a subgraph's result (Optional by type: a fan-in step has no op, an op may
# drop its record) and Tracer.result (None until a run completes) right after a run that produced them.
"""The ``Tracer`` looks INSIDE a subgraph step — written before the descent (DESIGN §1.3).

A step whose op is a ``Subgraph`` — in a ``FlowGraph`` or a ``Stream``'s ops list — is traced by a
CHILD tracer over its inner steps, on the same kernel, and its inner nodes appear as
``<outer>/<inner>`` (``prep/grey``; a Stream member ``ops[1]/grey``; nested ``prep/inner/x``). Every
reading call takes those names; ``run(until="prep/scaled")`` pauses INSIDE the subgraph and ``step()`` /
``resume()`` carry on from there — through the rest of the inside, then the outer steps.

The gate and late-entry cases are the critic's measured breaks (§2.3, §2.4): each graph below runs
correctly, and before the descent either the check refused it or it was refused only while running.
"""

import json
from typing import Any, Dict

import numpy as np
import pytest
from pydantic import ValidationError

from recordstream import FlowGraph, Image, Mask, Stream
from recordstream.flow import Subgraph, Tracer
from recordstream.ops.contract import ChainContractError
from recordstream.ops.image import ConvertMode
from recordstream.ops.numpy import Scale, Threshold
from tests._subgraph_ops import (
    DEMO_MASK_MEAN,
    SgBump,
    SgCount,
    SgDoubled,
    SgExplode,
    SgGated,
    SgMaskFraction,
    SgPut,
    SgRaiseBright,
    demo_record,
)

NAMES = ["head", "prep", "prep/first", "prep/grey", "prep/scaled", "mask"]


@pytest.fixture(autouse=True)
def _reset_counts() -> None:
    SgCount.counts.clear()


def _prep() -> Subgraph:
    return Subgraph(
        steps={
            "first": SgCount(label="first"),
            "grey": ConvertMode(mode="L"),
            "scaled": Scale(source_min=0.0, source_max=255.0, target_min=0.0, target_max=1.0),
        }
    )


def _graph() -> FlowGraph:
    return FlowGraph(flow={"head": SgCount(label="head"), "prep": _prep(), "mask": Threshold(low_level=0.5)})


def _mask_mean(record: Dict[str, Any]) -> float:
    return round(float(np.asarray(record["mask"]).mean()), 4)


def _rows(tracer: Tracer) -> Dict[str, Dict[str, Any]]:
    return {row["node"]: row for row in tracer.to_dict()["nodes"]}


class TestNamesAndReading:
    def test_the_names_are_pre_order_and_qualified(self) -> None:
        assert Tracer(_graph()).names == NAMES

    def test_a_run_records_every_inner_node(self) -> None:
        tracer = Tracer(_graph()).run(demo_record())
        assert tracer.statuses == {name: "ok" for name in NAMES}
        assert tracer.generations == {name: 0 for name in NAMES}
        assert _mask_mean(tracer.result[0]) == DEMO_MASK_MEAN
        assert float(np.asarray(tracer.value("prep/scaled", "image")).max()) == pytest.approx(230 / 255)
        assert np.asarray(tracer.value("prep/scaled", "image", side="input")).dtype == np.uint8

    def test_the_traced_run_is_the_plain_run(self) -> None:
        traced = Tracer(_graph()).run(demo_record()).result[0]
        plain = _graph()._run(demo_record())
        assert np.array_equal(np.asarray(traced["mask"]), np.asarray(plain["mask"]))

    def test_to_dict_nests_by_parent_and_inner(self) -> None:
        tracer = Tracer(_graph()).run(demo_record())
        document = tracer.to_dict()
        json.dumps(document)  # plain JSON, no encoder
        assert [row["node"] for row in document["nodes"]] == NAMES
        rows = _rows(tracer)
        assert rows["prep"]["op"] == "Subgraph"
        assert rows["prep"]["inner"] == ["prep/first", "prep/grey", "prep/scaled"]
        assert "parent" not in rows["prep"] and "inner" not in rows["mask"]
        assert {rows[name]["parent"] for name in ("prep/first", "prep/grey", "prep/scaled")} == {"prep"}
        assert rows["prep"]["output"]["image"]["dtype"] == "float32"
        assert rows["prep/grey"]["op"] == "ConvertMode"

    def test_the_check_rows_nest_the_same_way(self) -> None:
        report = Tracer(_graph()).check(demo_record())
        assert [row["node"] for row in report] == NAMES
        by_node = {row["node"]: row for row in report}
        assert by_node["prep"]["inner"] == ["prep/first", "prep/grey", "prep/scaled"]
        assert by_node["prep/grey"]["parent"] == "prep"
        assert by_node["prep/grey"]["available"] == ["image"]

    def test_an_unknown_inner_name_is_refused_with_every_name(self) -> None:
        tracer = Tracer(_graph(), where="demo.yaml")
        for name in ("prep/nosuch", "mask/x", "nosuch/grey"):
            with pytest.raises(KeyError) as refused:
                tracer.run(demo_record(), until=name)
            assert refused.value.args[0] == f"demo.yaml: no node named {name!r} — the nodes are {NAMES}"

    def test_value_of_an_inner_node_never_reached(self) -> None:
        with pytest.raises(RuntimeError, match=r"^demo.yaml:prep/grey: 'grey' was never reached"):
            Tracer(_graph(), where="demo.yaml").value("prep/grey", "image")

    def test_a_stream_member_is_descended_too(self) -> None:
        stream = Stream(ops=[SgCount(label="head"), _prep(), Threshold(low_level=0.5)])
        tracer = Tracer(stream).run(demo_record())
        assert tracer.names == ["ops[0]", "ops[1]", "ops[1]/first", "ops[1]/grey", "ops[1]/scaled", "ops[2]"]
        assert set(tracer.statuses.values()) == {"ok"}
        assert _mask_mean(tracer.result[0]) == DEMO_MASK_MEAN

    def test_nested_subgraphs(self) -> None:
        graph = FlowGraph(
            flow={"prep": Subgraph(steps={"inner": Subgraph(steps={"x": SgCount(label="x")}), "y": SgCount(label="y")})}
        )
        tracer = Tracer(graph).run(demo_record())
        assert tracer.names == ["prep", "prep/inner", "prep/inner/x", "prep/y"]
        rows = _rows(tracer)
        assert rows["prep/inner"]["parent"] == "prep" and rows["prep/inner"]["inner"] == ["prep/inner/x"]
        assert rows["prep/inner/x"]["parent"] == "prep/inner"


class TestRerunInside:
    def test_rerun_an_inner_node_recomputes_from_there_and_after_the_subgraph(self) -> None:
        tracer = Tracer(_graph()).run(demo_record())
        tracer.rerun_from("prep/scaled", target_max=0.5)  # the rectangles now scale below the 0.5 threshold
        assert _mask_mean(tracer.result[0]) == 0.0
        assert tracer.generations == {
            "head": 0,
            "prep": 1,
            "prep/first": 0,
            "prep/grey": 0,
            "prep/scaled": 1,
            "mask": 1,
        }
        assert SgCount.counts == {"head": 1, "first": 1}  # nothing before 'scaled' ran again
        assert _rows(tracer)["prep/scaled"]["params"]["target_max"] == 0.5

    def test_a_refused_inner_value_leaves_the_trace_untouched(self) -> None:
        tracer = Tracer(_graph()).run(demo_record())
        before = tracer.to_dict()
        with pytest.raises(ValidationError):
            tracer.rerun_from("prep/grey", mode="XX")
        assert tracer.to_dict() == before

    def test_rerun_the_subgraph_node_reruns_its_whole_inside(self) -> None:
        tracer = Tracer(_graph()).run(demo_record())
        tracer.rerun_from("prep")
        assert SgCount.counts == {"head": 1, "first": 2}
        assert tracer.generations == {"head": 0, **{name: 1 for name in NAMES[1:]}}

    def test_rerun_the_subgraph_node_with_a_new_result(self) -> None:
        tracer = Tracer(_graph()).run(demo_record())
        tracer.rerun_from("prep", result="grey")  # the threshold now sees grey levels 21..230, all above 0.5
        assert _mask_mean(tracer.result[0]) == 1.0
        assert tracer.names == NAMES

    def test_rerun_an_inner_node_never_reached(self) -> None:
        with pytest.raises(RuntimeError, match=r"'prep/grey' was never reached"):
            Tracer(_graph()).rerun_from("prep/grey", mode="L")

    def test_a_nested_rerun(self) -> None:
        graph = FlowGraph(
            flow={"prep": Subgraph(steps={"inner": Subgraph(steps={"x": SgCount(label="x")}), "y": SgCount(label="y")})}
        )
        tracer = Tracer(graph).run(demo_record())
        tracer.rerun_from("prep/inner/x", label="x")
        assert SgCount.counts == {"x": 2, "y": 2}
        assert tracer.generations == {"prep": 1, "prep/inner": 1, "prep/inner/x": 1, "prep/y": 1}


class TestPausingInside:
    def test_run_until_an_inner_node_pauses_inside(self) -> None:
        tracer = Tracer(_graph()).run(demo_record(), until="prep/scaled")
        assert tracer.paused_at == "prep/scaled"
        assert tracer.result is None
        assert tracer.statuses == {
            "head": "ok",
            "prep": "paused",
            "prep/first": "ok",
            "prep/grey": "ok",
            "prep/scaled": "paused",
            "mask": "not reached",
        }
        assert np.asarray(tracer.value("prep/scaled", "image", side="input")).dtype == np.uint8
        assert tracer.to_dict()["paused_at"] == "prep/scaled"

    def test_step_runs_the_rest_of_the_inside_then_pauses_at_the_next_outer_node(self) -> None:
        tracer = Tracer(_graph()).run(demo_record(), until="prep/scaled").step()
        assert tracer.paused_at == "mask"
        assert tracer.statuses["prep"] == "ok" and tracer.statuses["prep/scaled"] == "ok"
        assert tracer.statuses["mask"] == "paused"
        tracer.resume()
        assert tracer.paused_at is None
        assert _mask_mean(tracer.result[0]) == DEMO_MASK_MEAN
        assert SgCount.counts == {"head": 1, "first": 1}

    def test_step_visits_every_name_once_in_order(self) -> None:
        tracer = Tracer(_graph()).run(demo_record(), until="prep")
        seen = [tracer.paused_at]
        while tracer.paused_at is not None:
            tracer.step()
            seen.append(tracer.paused_at)
        assert seen == ["prep", "prep/first", "prep/grey", "prep/scaled", "mask", None]
        assert _mask_mean(tracer.result[0]) == DEMO_MASK_MEAN
        assert SgCount.counts == {"head": 1, "first": 1}  # no node ran twice

    def test_resume_from_inside(self) -> None:
        tracer = Tracer(_graph()).run(demo_record(), until="prep/grey").resume()
        assert tracer.statuses == {name: "ok" for name in NAMES}
        assert _mask_mean(tracer.result[0]) == DEMO_MASK_MEAN

    def test_pausing_in_a_nested_subgraph(self) -> None:
        graph = FlowGraph(
            flow={"prep": Subgraph(steps={"inner": Subgraph(steps={"x": SgCount(label="x")}), "y": SgCount(label="y")})}
        )
        tracer = Tracer(graph).run(demo_record(), until="prep/inner/x")
        assert tracer.paused_at == "prep/inner/x"
        assert tracer.statuses == {
            "prep": "paused",
            "prep/inner": "paused",
            "prep/inner/x": "paused",
            "prep/y": "not reached",
        }
        assert tracer.step().paused_at == "prep/y"
        tracer.resume()
        assert set(tracer.statuses.values()) == {"ok"}
        assert SgCount.counts == {"x": 1, "y": 1}

    def test_a_rerun_of_an_earlier_outer_node_forgets_the_inside(self) -> None:
        tracer = Tracer(_graph()).run(demo_record(), until="prep/scaled")
        tracer.rerun_from("head")
        assert tracer.paused_at is None
        assert set(tracer.statuses.values()) == {"ok"}


class TestErrorsAndDropsInside:
    def test_an_inner_node_that_raises(self) -> None:
        graph = FlowGraph(flow={"prep": Subgraph(steps={"boom": SgExplode()}), "after": SgCount(label="after")})
        tracer = Tracer(graph)
        with pytest.raises(RuntimeError, match="boom: SgExplode was told to fail"):
            tracer.run(demo_record())
        assert tracer.statuses == {"prep": "error", "prep/boom": "error", "after": "not reached"}
        assert _rows(tracer)["prep"]["error"] == "RuntimeError: boom: SgExplode was told to fail"
        tracer.rerun_from("prep/boom", fail=False)
        assert set(tracer.statuses.values()) == {"ok"}
        assert tracer.result[0]["survived"] is True

    def test_an_inner_node_that_drops_the_record(self) -> None:
        graph = FlowGraph(
            flow={"prep": Subgraph(steps={"drop": SgExplode(fail=False, drop=True)}), "after": SgCount(label="after")}
        )
        tracer = Tracer(graph).run(demo_record())
        assert tracer.statuses == {"prep": "dropped", "prep/drop": "dropped", "after": "not reached"}
        assert tracer.result == []


class TestTheCheckLooksInside:
    def test_a_flag_raised_inside_reaches_an_outer_gate(self) -> None:
        # critic §2.4 gate/out_inline.yaml: runs right, and was refused with "Gated is gated on the flag
        # 'bright', which no node before it raises" — the subgraph now declares the flags its inside raises
        graph = FlowGraph(
            flow={"detect": Subgraph(steps={"bright": SgRaiseBright()}), "gated": SgGated(requires="bright")}
        )
        tracer = Tracer(graph).run(demo_record())
        assert tracer.statuses == {"detect": "ok", "detect/bright": "ok", "gated": "ok"}
        assert tracer.value("gated", "gated_ran") is True

    def test_a_flag_raised_outside_reaches_an_inner_gate(self) -> None:
        # critic §2.4 gate/in_inline.yaml: the descent prototype refused it at `finish/gated`
        graph = FlowGraph(
            flow={"bright": SgRaiseBright(), "finish": Subgraph(steps={"gated": SgGated(requires="bright")})}
        )
        tracer = Tracer(graph).run(demo_record())
        assert tracer.statuses == {"bright": "ok", "finish": "ok", "finish/gated": "ok"}
        assert tracer.value("finish/gated", "gated_ran") is True

    def test_an_inner_gate_nothing_raises_is_refused_before_anything_runs(self) -> None:
        graph = FlowGraph(
            flow={"head": SgCount(label="head"), "finish": Subgraph(steps={"gated": SgGated(requires="bright")})}
        )
        tracer = Tracer(graph, where="gate.yaml")
        with pytest.raises(ChainContractError) as refused:
            tracer.run(demo_record())
        assert str(refused.value) == (
            "gate.yaml:finish/gated: SgGated is gated on the flag 'bright', which no node before it raises — "
            "the flags available at that point are: none"
        )
        assert set(tracer.statuses.values()) == {"not reached"}
        assert SgCount.counts == {}
        assert tracer.report[-1]["node"] == "finish/gated" and tracer.report[-1]["verdict"] == "refused"

    def test_one_flag_raised_outside_and_inside_is_ambiguous(self) -> None:
        graph = FlowGraph(flow={"bright": SgRaiseBright(), "detect": Subgraph(steps={"again": SgRaiseBright()})})
        with pytest.raises(ChainContractError) as refused:
            Tracer(graph, where="gate.yaml").check(demo_record())
        assert str(refused.value).startswith(
            "gate.yaml:detect/again: the flag 'bright' is declared by both a node before this chain and SgRaiseBright"
        )

    def test_a_refusal_at_a_subgraph_is_re_asked_of_its_inside(self) -> None:
        # critic §2.3 late/inline.yaml: the inner step reads 'fraction', which the outer step writes only AFTER
        graph = FlowGraph(flow={"stats": Subgraph(steps={"twice": SgDoubled()}), "frac": SgMaskFraction()})
        seed = {**demo_record(), "mask": Mask(np.ones((4, 4), dtype=bool))}
        tracer = Tracer(graph, where="late.yaml")
        with pytest.raises(ChainContractError) as refused:
            tracer.run(seed)
        assert str(refused.value) == (
            "late.yaml:stats/twice: SgDoubled needs the record entry 'fraction', which nothing before it "
            "produces — the chain has image, mask at that point (an op that DOES write it must declare it in "
            "`produces`)"
        )
        verdicts = {row["node"]: row["verdict"] for row in tracer.report}
        assert verdicts == {"stats": "refused", "stats/twice": "refused"}
        assert set(tracer.statuses.values()) == {"not reached"}

    def test_the_same_steps_in_the_right_order_run(self) -> None:
        graph = FlowGraph(flow={"frac": SgMaskFraction(), "stats": Subgraph(steps={"twice": SgDoubled()})})
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2] = True
        tracer = Tracer(graph).run({"image": Image(np.zeros((4, 4), dtype=np.uint8)), "mask": Mask(mask)})
        assert tracer.value("stats/twice", "doubled") == pytest.approx(1.0)

    def test_an_inner_check_on_incomplete_knowledge_reports_instead_of_raising(self) -> None:
        # a types-only node ahead of the subgraph: what reaches the inside is not known by name
        graph = FlowGraph(flow={"mask": Threshold(low_level=0.5), "stats": Subgraph(steps={"twice": SgDoubled()})})
        report = Tracer(graph).check(demo_record())
        by_node = {row["node"]: row for row in report}
        assert by_node["stats/twice"]["complete"] is False
        assert by_node["stats/twice"]["verdict"] == "unverifiable"


class TestRefusalsAndSnapshots:
    def test_a_mistake_inside_is_refused_on_first_use_before_anything_runs(self) -> None:
        graph = FlowGraph(
            flow={"head": SgCount(label="head"), "prep": Subgraph(steps={"x": SgCount(label="x")}, result="nosuch")}
        )
        with pytest.raises(ValueError, match=r"^flow step 'prep' \(a subgraph\): Subgraph: result 'nosuch'"):
            Tracer(graph).check(demo_record())
        assert SgCount.counts == {}

    def test_a_rerun_the_inner_check_refuses_leaves_the_trace_untouched(self) -> None:
        graph = FlowGraph(
            flow={"bright": SgRaiseBright(), "finish": Subgraph(steps={"gated": SgGated(requires="bright")})}
        )
        tracer = Tracer(graph, where="gate.yaml").run(demo_record())
        before = tracer.to_dict()
        with pytest.raises(ChainContractError, match=r"^gate.yaml:finish/gated: SgGated is gated on the flag 'nosuch'"):
            tracer.rerun_from("finish/gated", requires="nosuch")
        assert tracer.to_dict() == before
        tracer.rerun_from("finish/gated")  # the restored node still runs
        assert tracer.value("finish/gated", "gated_ran") is True

    def test_a_rerun_of_the_subgraph_node_with_a_refused_inside_leaves_the_trace_untouched(self) -> None:
        tracer = Tracer(_graph()).run(demo_record())
        before = tracer.to_dict()
        with pytest.raises(
            ValueError, match=r"^prep \(a subgraph\): Subgraph: result 'nosuch' does not name an inner step"
        ):
            tracer.rerun_from("prep", result="nosuch")
        assert tracer.to_dict() == before

    def test_copied_snapshots_inside(self) -> None:
        tracer = Tracer(_graph(), copy_snapshots=True).run(demo_record())
        assert set(tracer.statuses.values()) == {"ok"}
        assert tracer.value("prep/scaled", "image") is not tracer.result[0]["image"]

    def test_an_inside_that_edits_its_record_in_place_flags_the_subgraph_node(self) -> None:
        def blank(record: Dict[str, Any]) -> Dict[str, Any]:  # overwrites 'image' in the record it was given
            record["image"] = Image(np.zeros_like(np.asarray(record["image"])))
            return record

        tracer = Tracer(FlowGraph(flow={"prep": Subgraph(steps={"blank": blank})})).run(demo_record())
        rows = _rows(tracer)
        assert rows["prep"]["in_place"] is True and rows["prep/blank"]["in_place"] is True


class TestARerunInsideThatNowDrops:
    def test_nothing_after_the_subgraph_runs(self) -> None:
        graph = FlowGraph(
            flow={"prep": Subgraph(steps={"gate": SgExplode(fail=False)}), "after": SgCount(label="after")}
        )
        tracer = Tracer(graph).run(demo_record())
        assert tracer.result[0]["survived"] is True
        tracer.rerun_from("prep/gate", drop=True)
        assert tracer.statuses == {"prep": "dropped", "prep/gate": "dropped", "after": "not reached"}
        assert tracer.result == []
        assert SgCount.counts == {"after": 1}


# --------------------------------------------------------------------------------------------
# review fixes (2026-09-29)
# --------------------------------------------------------------------------------------------


class TestARerunOfTheSubgraphKeepsItsInnerReruns:
    def test_the_inner_edit_survives_a_rebuild_of_the_subgraph(self) -> None:
        # measured before: rerun_from('prep', result='scaled') reverted prep/scaled to target_max 1.0, mask mean 0.175
        tracer = Tracer(_graph()).run(demo_record())
        tracer.rerun_from("prep/scaled", target_max=0.5)
        assert _mask_mean(tracer.result[0]) == 0.0
        tracer.rerun_from("prep", result="scaled")
        assert _rows(tracer)["prep/scaled"]["params"]["target_max"] == 0.5
        assert _mask_mean(tracer.result[0]) == 0.0

    def test_a_nested_inner_edit_survives_a_rebuild_of_the_outer_subgraph(self) -> None:
        inner = Subgraph(steps={"x": SgPut(key="x", value=1.0)})
        graph = FlowGraph(flow={"prep": Subgraph(steps={"inner": inner, "y": SgPut(key="y")})})
        tracer = Tracer(graph).run({})
        tracer.rerun_from("prep/inner/x", value=7.0)
        assert tracer.result[0]["x"] == 7.0
        tracer.rerun_from("prep", result="y")
        assert tracer.result[0]["x"] == 7.0
        assert _rows(tracer)["prep/inner/x"]["params"]["value"] == 7.0

    def test_new_steps_given_to_the_rerun_replace_the_inside(self) -> None:
        tracer = Tracer(FlowGraph(flow={"prep": Subgraph(steps={"x": SgPut(key="x", value=1.0)})})).run({})
        tracer.rerun_from("prep/x", value=2.0)
        tracer.rerun_from("prep", steps={"z": SgPut(key="z", value=3.0)})
        assert tracer.names == ["prep", "prep/z"] and tracer.result[0]["z"] == 3.0


class TestCopiedSnapshotsSeedARerunFromACopy:
    """``copy_snapshots=True``: a rerun of a FIRST node starts from the record as it arrived, every time."""

    def _node(self, tracer: Tracer, name: str) -> Dict[str, Any]:
        return _rows(tracer)[name]

    def test_the_first_node_inside_a_subgraph(self) -> None:
        # measured before: run k=1, rerun k=2 (input showed k: 1, prep in_place flipped to False), again k=3
        graph = FlowGraph(
            flow={"a": SgPut(key="a"), "sg": Subgraph(steps={"ip": SgBump(key="k"), "t": SgPut(key="t")})}
        )
        tracer = Tracer(graph, copy_snapshots=True).run({})
        assert tracer.result[0]["k"] == 1 and self._node(tracer, "sg")["in_place"] is True
        for _ in range(2):
            tracer.rerun_from("sg/ip")
            assert tracer.result[0]["k"] == 1
            assert "k" not in self._node(tracer, "sg/ip")["input"]
            assert self._node(tracer, "sg")["in_place"] is True
        tracer.rerun_from("sg")
        assert tracer.result[0]["k"] == 1

    def test_the_first_node_of_the_graph(self) -> None:
        # measured before: run k=1, rerun k=2, again k=3 (the same line fixes both)
        tracer = Tracer(FlowGraph(flow={"ip": SgBump(key="k"), "t": SgPut(key="t")}), copy_snapshots=True)
        seed: Dict[str, Any] = {}
        tracer.run(seed)
        assert tracer.result[0]["k"] == 1
        for _ in range(2):
            tracer.rerun_from("ip")
            assert tracer.result[0]["k"] == 1 and "k" not in self._node(tracer, "ip")["input"]

    def test_reference_snapshots_keep_their_documented_limit(self) -> None:
        # without copies the recorded input IS the edited record — copy_snapshots is the remedy
        tracer = Tracer(FlowGraph(flow={"ip": SgBump(key="k"), "t": SgPut(key="t")})).run({})
        tracer.rerun_from("ip")
        assert tracer.result[0]["k"] == 2


class TestAStructuralRefusalCarriesTheTracersWhere:
    def test_a_stream_member_refused_while_the_plan_is_built(self) -> None:
        # measured before: ValueError: Stream.ops[1] (a subgraph): … — no file at all
        stream = Stream(
            source=[{"x": 1}], ops=[SgCount(label="a"), Subgraph(steps={"grey": SgCount()}, result="nosuch")]
        )
        with pytest.raises(ValueError) as refused:
            Tracer(stream, where="ops_bad2.yaml").check({})
        assert str(refused.value) == (
            "ops_bad2.yaml: Stream.ops[1] (a subgraph): Subgraph: result 'nosuch' does not name an inner step "
            "(the steps are: ['grey'])"
        )

    def test_without_a_where_the_message_is_the_engines(self) -> None:
        stream = Stream(source=[{"x": 1}], ops=[Subgraph(steps={"grey": SgCount()}, result="nosuch")])
        with pytest.raises(ValueError, match=r"^Stream.ops\[0\] \(a subgraph\): Subgraph: result 'nosuch'"):
            Tracer(stream).names

    def test_an_expanding_inner_op_keeps_its_kind(self) -> None:
        from tests._subgraph_ops import SgSplit

        graph = FlowGraph(flow={"prep": Subgraph(steps={"split": SgSplit()})})
        with pytest.raises(TypeError, match=r"^fan.yaml: flow step 'prep' \(a subgraph\): Subgraph: step 'split'"):
            Tracer(graph, where="fan.yaml").check({})


class TestABoundResultIsTracedAsItRuns:
    def _graph(self) -> FlowGraph:
        return FlowGraph(
            flow={
                "pick": SgPut(key="which", value="a"),
                "sg": {
                    "op": Subgraph(steps={"a": SgPut(key="a"), "b": SgPut(key="b")}),
                    "bind": {"result": "pick[which]"},
                },
            }
        )

    def test_the_traced_run_is_the_plain_run(self) -> None:
        # measured before: plain ('which', 'a'), traced ('which', 'a', 'b')
        plain = self._graph()._run({})
        tracer = Tracer(self._graph()).run({})
        assert tracer.result[0]["path"] == plain["path"] == ("which", "a")
        assert _rows(tracer)["sg"]["bound"] == {"result": "a"}
        assert tracer.statuses["sg/a"] == "ok"

    def test_a_rerun_of_an_inner_node_under_the_bound_result(self) -> None:
        tracer = Tracer(self._graph()).run({})
        tracer.rerun_from("sg/a", value=5.0)
        assert tracer.result[0]["a"] == 5.0 and tracer.result[0]["path"] == ("which", "a")
