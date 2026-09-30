# mypy: disable-error-code="attr-defined,union-attr,index,arg-type,no-any-return"
# The tests read FlowStep.op and a subgraph's result (Optional by type: a fan-in step has no op, an op may
# drop its record) and Tracer.result (None until a run completes) right after a run that produced them.
"""The ``Subgraph`` op and the flow grammar around it — written BEFORE ``recordstream/flow/subgraph.py``.

A subgraph is a ``flow:`` mapping used as ONE op, written inline in the step that uses it:

.. code-block:: yaml

    flow:
      prep:
        op: !class:recordstream.flow.subgraph.Subgraph
          steps:
            grey:   {op: !class:recordstream.ops.image.ConvertMode {mode: L}}
            scaled: {op: !class:recordstream.ops.numpy.Scale {...}}
          result: scaled
      mask:
        op: !class:recordstream.ops.numpy.Threshold {low_level: 0.5}
    outputs: mask

Its inside runs on the SAME kernel a FlowGraph runs (no second executor). Every structural mistake —
a ``result:`` naming no inner step, an expanding inner op, an inner reference to a step outside, a
``from:`` swallowed by a bare inner marker, an outer ``bind:`` reaching inside — is refused in words
BEFORE the first record runs. The measured breaks these tests pin come from the critic's report of the
subgraph design (§2.1 names, §2.3 late entry, §2.4 gates, §2.5 bind, §2.6 broadcast, §2.8 expansion);
the tracer half lives in ``test_trace_subgraph.py``.
"""

from pathlib import Path
from typing import Any, Dict

import confluid
import numpy as np
import pytest
from confluid.pydantic_export import _qualname
from PIL import Image as PILImage

import recordstream.flow as flow_pkg
from recordstream import FlowGraph, Stream
from recordstream.flow import StepReferenceError, Subgraph, Tracer, parse_flow
from recordstream.flow.subgraph import Subgraph as CanonicalSubgraph
from recordstream.ops.contract import ChainContractError, check_chain
from recordstream.ops.image import ConvertMode
from recordstream.ops.numpy import Scale, Threshold
from tests._subgraph_ops import (
    DEMO_BOXES,
    DEMO_MASK_MEAN,
    SgCount,
    SgDoubled,
    SgGated,
    SgMaskFraction,
    SgPut,
    SgRaiseBright,
    demo_pixels,
    demo_record,
)

OPS = "tests._subgraph_ops"
SUBGRAPH = "!class:recordstream.flow.subgraph.Subgraph"
SCALE = "!class:recordstream.ops.numpy.Scale {source_min: 0.0, source_max: 255.0, target_min: 0.0, target_max: 1.0}"

#: The document the design shows (DESIGN §0.2), verbatim apart from the scale line being one constant.
DEMO_YAML = f"""flow:
  prep:
    op: {SUBGRAPH}
      steps:
        grey:
          op: !class:recordstream.ops.image.ConvertMode {{mode: L}}
        scaled:
          op: {SCALE}
      result: scaled
  mask:
    op: !class:recordstream.ops.numpy.Threshold {{low_level: 0.5}}
outputs: mask
"""


def _write(tmp_path: Path, text: str, name: str = "graph.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _mask_mean(record: Dict[str, Any]) -> float:
    return round(float(np.asarray(record["mask"]).mean()), 4)


def _prep_steps() -> Dict[str, Any]:
    return {
        "grey": ConvertMode(mode="L"),
        "scaled": Scale(source_min=0.0, source_max=255.0, target_min=0.0, target_max=1.0),
    }


@pytest.fixture(autouse=True)
def _reset_counts() -> None:
    SgCount.counts.clear()


# --------------------------------------------------------------------------------------------
# the op itself
# --------------------------------------------------------------------------------------------


class TestTheOp:
    def test_it_is_an_op_in_the_compose_group_at_its_submodule_path(self) -> None:
        assert Subgraph is CanonicalSubgraph
        assert "Subgraph" in flow_pkg.__all__
        assert _qualname(Subgraph) == "recordstream.flow.subgraph.Subgraph"
        assert Subgraph.__confluid_category__ == "op"
        assert Subgraph.__confluid_group__ == "compose"

    def test_the_constructor_stores_only(self) -> None:
        # A reference to a step that does not exist: nothing is parsed, so nothing is refused yet.
        sub = Subgraph(steps={"late": {"from": "nosuch"}}, result="late")
        assert sub.steps == {"late": {"from": "nosuch"}}
        assert sub.result == "late"
        with pytest.raises(ValueError, match="'nosuch' .* which is not a step inside this subgraph"):
            sub.flow_steps

    def test_zero_arg_construction_declares_nothing_and_refuses_to_run(self) -> None:
        sub = Subgraph()
        assert (sub.steps, sub.result) == (None, "")
        assert (sub.consumes, sub.produces, sub.flags) == ({}, {}, ())
        with pytest.raises(ValueError, match=r"^Subgraph: steps is empty — give it at least one step$"):
            sub(demo_record())

    def test_it_runs_its_steps_on_the_record_it_receives(self) -> None:
        sub = Subgraph(steps=_prep_steps(), result="scaled")
        scaled = sub(demo_record())
        assert float(np.asarray(scaled["image"]).max()) == pytest.approx(230 / 255)
        assert _mask_mean(Threshold(low_level=0.5)(scaled)) == DEMO_MASK_MEAN

    def test_a_blank_result_is_the_last_inner_step(self) -> None:
        sub = Subgraph(steps=_prep_steps())
        assert sub.output_step == "scaled"
        assert float(np.asarray(sub(demo_record())["image"]).max()) <= 1.0

    def test_result_may_name_an_earlier_inner_step(self) -> None:
        grey = Subgraph(steps=_prep_steps(), result="grey")(demo_record())
        assert np.asarray(grey["image"]).dtype == np.uint8  # the scale never reached the result

    def test_the_parse_is_cached_and_follows_a_reassigned_steps(self) -> None:
        sub = Subgraph(steps=_prep_steps())
        assert sub.flow_steps is sub.flow_steps
        sub.steps = {"grey": ConvertMode(mode="L")}
        assert [step.name for step in sub.flow_steps] == ["grey"]

    def test_a_dropped_record_is_dropped(self) -> None:
        from tests._subgraph_ops import SgExplode

        assert Subgraph(steps={"drop": SgExplode(fail=False, drop=True)})(demo_record()) is None


class TestTheDerivedDeclarations:
    """``consumes`` / ``produces`` / ``flags`` are READ off the inner ops, never written twice."""

    def test_names_when_every_inner_op_declares_names(self) -> None:
        sub = Subgraph(steps={"frac": SgMaskFraction(), "twice": SgDoubled()})
        # 'fraction' is produced inside, so only 'mask' is needed from outside
        assert sub.consumes == {"mask": "Mask"}
        assert sub.produces == {"fraction": "*", "doubled": "*"}

    def test_types_when_an_inner_op_declares_types(self) -> None:
        sub = Subgraph(steps={"grey": ConvertMode(mode="L"), "mask": Threshold(low_level=0.5)})
        assert isinstance(sub.consumes, tuple) and isinstance(sub.produces, tuple)
        assert any(getattr(kind, "__name__", "") == "Mask" for kind in sub.produces)

    def test_ops_declaring_nothing_make_an_empty_named_boundary(self) -> None:
        assert (Subgraph(steps=_prep_steps()).consumes, Subgraph(steps=_prep_steps()).produces) == ({}, {})

    def test_flags_are_the_union_of_the_inner_flags(self) -> None:
        assert Subgraph(steps={"bright": SgRaiseBright()}).flags == ("bright",)
        assert Subgraph(steps=_prep_steps()).flags == ()


# --------------------------------------------------------------------------------------------
# a real run: the design's document, the demo image
# --------------------------------------------------------------------------------------------


class TestTheDesignDocumentRuns:
    def test_from_yaml_on_the_demo_image(self, tmp_path: Path) -> None:
        graph = FlowGraph.from_yaml(str(_write(tmp_path, DEMO_YAML)))
        assert [(step.name, type(step.op).__name__) for step in graph.steps] == [
            ("prep", "Subgraph"),
            ("mask", "Threshold"),
        ]
        assert graph.steps[0].op.output_step == "scaled"
        assert _mask_mean(graph._run(demo_record())) == DEMO_MASK_MEAN

    def test_through_confluid_load_too(self, tmp_path: Path) -> None:
        document = confluid.load(str(_write(tmp_path, DEMO_YAML)))
        graph = FlowGraph(flow=document["flow"], outputs=document["outputs"])
        assert _mask_mean(graph._run(demo_record())) == DEMO_MASK_MEAN

    def test_read_from_a_png_file_then_boxes(self, tmp_path: Path) -> None:
        png = tmp_path / "demo.png"
        PILImage.fromarray(demo_pixels()).save(png)
        text = f"""flow:
  read: {{op: !class:recordstream.ops.image.ReadImage {{}}}}
  prep:
    op: {SUBGRAPH}
      steps:
        grey: {{op: !class:recordstream.ops.image.ConvertMode {{mode: L}}}}
        scaled: {{op: {SCALE}}}
  mask: {{op: !class:recordstream.ops.numpy.Threshold {{low_level: 0.5}}}}
  boxes: {{op: !class:recordstream.ops.numpy.ConnectedComponents {{}}}}
outputs: boxes
"""
        graph = FlowGraph.from_yaml(str(_write(tmp_path, text)), source=[{"file": str(png)}])
        (record,) = graph.collect()
        assert _mask_mean(record) == DEMO_MASK_MEAN
        assert [tuple(box) for box in record["boxes"].boxes] == DEMO_BOXES

    def test_a_subgraph_is_a_member_of_an_ops_list_too(self) -> None:
        stream = Stream(source=[demo_record()], ops=[Subgraph(steps=_prep_steps()), Threshold(low_level=0.5)])
        (record,) = list(stream)
        assert _mask_mean(record) == DEMO_MASK_MEAN


# --------------------------------------------------------------------------------------------
# refusals — every one BEFORE the first record runs, each saying what to do
# --------------------------------------------------------------------------------------------


class TestRefusedBeforeTheFirstRecord:
    def test_a_result_naming_no_inner_step(self) -> None:
        graph = FlowGraph(
            source=[demo_record()],
            flow={"head": SgCount(label="head"), "prep": {"op": Subgraph(steps=_prep_steps(), result="nosuch")}},
        )
        with pytest.raises(ValueError) as refused:
            graph.collect()
        assert str(refused.value) == (
            "flow step 'prep' (a subgraph): Subgraph: result 'nosuch' does not name an inner step "
            "(the steps are: ['grey', 'scaled'])"
        )
        assert SgCount.counts == {}  # nothing ran — not even the step before the subgraph

    def test_a_refusal_from_a_document_is_located_at_the_subgraph(self, tmp_path: Path) -> None:
        path = _write(tmp_path, DEMO_YAML.replace("result: scaled", "result: nosuch"))
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(path)).steps
        message = str(refused.value)
        assert message.startswith(
            "flow step 'prep' (a subgraph): Subgraph: result 'nosuch' does not name an inner step"
        )
        assert message.endswith(f"(at {path}:3:9)")

    def test_an_expanding_inner_op(self, tmp_path: Path) -> None:
        # critic §2.8: `expand/inline.yaml` died at the first record with the kernel's TypeError
        text = f"""flow:
  fan:
    op: {SUBGRAPH}
      steps:
        split: {{op: !class:{OPS}.SgSplit {{parts: 2}}}}
        stamp: {{op: !class:{OPS}.SgStamp {{key: stamp, value: 1.0}}}}
  grey: {{op: !class:{OPS}.SgCount {{label: after}}}}
outputs: grey
"""
        graph = FlowGraph.from_yaml(str(_write(tmp_path, text)), source=[demo_record()])
        with pytest.raises(TypeError) as refused:
            graph.collect()
        assert str(refused.value).startswith(
            "flow step 'fan' (a subgraph): Subgraph: step 'split' (SgSplit) is a 1→N expanding op, and a "
            "subgraph returns one record per record it receives — move the step out of the subgraph"
        )
        assert SgCount.counts == {}

    def test_an_inner_from_naming_an_outer_step(self, tmp_path: Path) -> None:
        text = f"""flow:
  grey: {{op: !class:recordstream.ops.image.ConvertMode {{mode: L}}}}
  prep:
    op: {SUBGRAPH}
      steps:
        mask: {{op: !class:recordstream.ops.numpy.Threshold {{low_level: 100}}, from: grey}}
outputs: prep
"""
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(_write(tmp_path, text))).steps
        assert str(refused.value).startswith(
            "flow step 'prep' (a subgraph): Subgraph: step 'mask' reads 'grey' (from: 'grey'), which is not a "
            "step inside this subgraph — a step inside a subgraph reads only the record the subgraph receives "
            "and the steps before it inside (the steps are: ['mask'])"
        )

    def test_an_inner_merge_from_naming_an_outer_step(self) -> None:
        sub = Subgraph(steps={"a": SgCount(label="a"), "b": {"merge_from": ["grey"]}})
        with pytest.raises(ValueError, match=r"^Subgraph: step 'b' reads 'grey' \(merge_from: 'grey'\), which is not"):
            sub.flow_steps

    def test_an_inner_bind_naming_an_outer_step(self, tmp_path: Path) -> None:
        # critic §2.5, out2in_inline.yaml: refused only at the first record, unlocated, before
        text = f"""flow:
  level: {{op: !class:{OPS}.SgMeanLevel {{}}}}
  finish:
    op: {SUBGRAPH}
      steps:
        stamp: {{op: !class:{OPS}.SgStamp {{key: stamp}}, bind: {{value: level.level}}}}
outputs: finish
"""
        path = _write(tmp_path, text)
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(path)).steps
        assert str(refused.value) == (
            "flow step 'finish' (a subgraph): Subgraph: step 'stamp' reads 'level' (bind: value='level.level'), "
            "which is not a step inside this subgraph — a step inside a subgraph reads only the record the "
            f"subgraph receives and the steps before it inside (the steps are: ['stamp']) (at {path}:4:9)"
        )

    def test_an_inner_reference_to_a_later_inner_step_keeps_the_flow_message(self) -> None:
        sub = Subgraph(steps={"a": {"from": "b"}, "b": SgCount(label="b")})
        with pytest.raises(ValueError, match=r"^flow step 'a': from: 'b' does not name an EARLIER step"):
            sub.flow_steps

    def test_from_written_inside_a_bare_inner_marker(self, tmp_path: Path) -> None:
        # Measured before this refusal: confluid set `from` as a plain attribute on the Threshold (with a
        # warning), the step read the previous step instead of `grey`, and the mask came out all False —
        # mean 0.0 where the mapping spelling below gives 0.2855. Silently wrong, so refused.
        text = f"""flow:
  prep:
    op: {SUBGRAPH}
      steps:
        grey: !class:recordstream.ops.image.ConvertMode {{mode: L}}
        unit: {SCALE}
        mask: !class:recordstream.ops.numpy.Threshold {{low_level: 100, from: grey}}
      result: mask
outputs: prep
"""
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(_write(tmp_path, text))).steps
        assert str(refused.value).startswith(
            "flow step 'prep' (a subgraph): Subgraph: step 'mask' carries 'from' inside its op's !class: marker, "
            "where Threshold takes it as a plain attribute and the step never reads it — write the step as a "
            "mapping: mask: {op: !class:recordstream.ops.numpy.Threshold {...}, from: grey}"
        )

    def test_the_mapping_spelling_of_that_step_runs_right(self, tmp_path: Path) -> None:
        text = f"""flow:
  prep:
    op: {SUBGRAPH}
      steps:
        grey: !class:recordstream.ops.image.ConvertMode {{mode: L}}
        unit: {SCALE}
        mask: {{op: !class:recordstream.ops.numpy.Threshold {{low_level: 100}}, from: grey}}
      result: mask
outputs: prep
"""
        graph = FlowGraph.from_yaml(str(_write(tmp_path, text)))
        assert _mask_mean(graph._run(demo_record())) == DEMO_MASK_MEAN

    def test_merge_from_written_inside_an_inner_marker(self, tmp_path: Path) -> None:
        text = f"""flow:
  prep:
    op: {SUBGRAPH}
      steps:
        a: !class:{OPS}.SgStamp {{key: a, value: 1.0}}
        b: {{op: !class:{OPS}.SgStamp {{key: b, value: 2.0, merge_from: [a]}}}}
outputs: prep
"""
        path = _write(tmp_path, text)
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(path)).steps
        assert str(refused.value) == (
            "flow step 'prep' (a subgraph): Subgraph: step 'b' carries 'merge_from' inside its op's !class: "
            "marker, where SgStamp takes it as a plain attribute and the step never reads it — write the step as "
            f"a mapping: b: {{op: !class:{OPS}.SgStamp {{...}}, merge_from: [a]}} (at {path}:3:9)"
        )

    def test_empty_steps(self) -> None:
        with pytest.raises(ValueError) as refused:
            parse_flow({"prep": Subgraph(steps={})})
        assert (
            str(refused.value) == "flow step 'prep' (a subgraph): Subgraph: steps is empty — give it at least one step"
        )

    def test_nested_subgraphs_name_the_whole_path(self) -> None:
        inner = Subgraph(steps={"x": SgCount(label="x")}, result="nosuch")
        with pytest.raises(ValueError) as refused:
            parse_flow({"prep": Subgraph(steps={"inner": inner})})
        assert str(refused.value) == (
            "flow step 'prep' (a subgraph): flow step 'inner' (a subgraph): Subgraph: result 'nosuch' does not "
            "name an inner step (the steps are: ['x'])"
        )

    def test_a_subgraph_in_an_ops_list_is_refused_when_the_stream_compiles(self) -> None:
        stream = Stream(
            source=[demo_record()], ops=[SgCount(label="head"), Subgraph(steps=_prep_steps(), result="nosuch")]
        )
        with pytest.raises(ValueError) as refused:
            list(stream)
        assert str(refused.value) == (
            "Stream.ops[1] (a subgraph): Subgraph: result 'nosuch' does not name an inner step "
            "(the steps are: ['grey', 'scaled'])"
        )
        assert SgCount.counts == {}

    def test_an_unbuilt_parse_does_not_open_the_subgraph(self, tmp_path: Path) -> None:
        # build=False is the structural read: a marker stays a marker, and nothing inside it is judged
        document = confluid.load(
            str(_write(tmp_path, DEMO_YAML.replace("result: scaled", "result: nosuch"))), until="settled"
        )
        steps, _ = parse_flow(document["flow"], build=False)
        assert [step.name for step in steps] == ["prep", "mask"]


class TestStepNames:
    """DESIGN §0.2 decision 4: inner nodes are named ``prep/grey``, so a step name may not contain ``/``."""

    def test_a_slash_in_a_step_name_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            parse_flow({"prep/grey": ConvertMode(mode="L")})
        assert str(refused.value) == (
            "flow: step name 'prep/grey' may not contain '/' (reserved for the nodes inside a subgraph: "
            "'prep/grey' is the step 'grey' inside the subgraph 'prep')"
        )

    def test_a_slash_inside_a_subgraph_is_refused_too(self) -> None:
        with pytest.raises(ValueError, match=r"^flow step 'prep' \(a subgraph\): flow: step name 'a/b' may not"):
            parse_flow({"prep": Subgraph(steps={"a/b": SgCount()})})

    def test_the_dot_rule_still_holds(self) -> None:
        with pytest.raises(ValueError, match="may not contain '.'"):
            parse_flow({"a.b": SgCount()})

    def test_an_outer_and_an_inner_step_may_share_a_name(self, tmp_path: Path) -> None:
        # critic §2.1 (names/inline.yaml): inner names are scoped to their subgraph
        text = f"""flow:
  grey: {{op: !class:recordstream.ops.image.ConvertMode {{mode: L}}}}
  prep:
    op: {SUBGRAPH}
      steps:
        grey: {{op: {SCALE}}}
        mask: {{op: !class:recordstream.ops.numpy.Threshold {{low_level: 0.5}}, from: grey}}
      result: mask
  boxes: {{op: !class:recordstream.ops.numpy.ConnectedComponents {{}}}}
outputs: boxes
"""
        record = FlowGraph.from_yaml(str(_write(tmp_path, text)))._run(demo_record())
        assert [tuple(box) for box in record["boxes"].boxes] == DEMO_BOXES


# --------------------------------------------------------------------------------------------
# DESIGN §0.2 decision 2: an OUTER bind into a subgraph is refused
# --------------------------------------------------------------------------------------------


MEASURE = f"""flow:
  measure:
    op: {SUBGRAPH}
      steps:
        level: {{op: !class:{OPS}.SgMeanLevel {{}}}}
  stamp:
    op: !class:{OPS}.SgStamp {{key: stamp}}
    bind: {{value: REF}}
outputs: stamp
"""


class TestAnOuterBindIntoASubgraph:
    def test_step_attr_of_an_inner_step_is_refused(self, tmp_path: Path) -> None:
        # critic §2.5, in2out_inline.yaml: the run died with an AttributeError at the first record while
        # Tracer.check passed
        path = _write(tmp_path, MEASURE.replace("REF", "measure.level"))
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(path)).steps
        assert str(refused.value) == (
            "flow step 'stamp': bind value='measure.level' reads 'level' of 'measure', which is a subgraph — a "
            "step outside a subgraph cannot read a value of a step inside it; move that step out of the subgraph "
            f"(at {path}:7:9)"
        )

    def test_the_qualified_spelling_is_refused_the_same_way(self, tmp_path: Path) -> None:
        path = _write(tmp_path, MEASURE.replace("REF", "measure/level.level"))
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(path)).steps
        assert str(refused.value).startswith(
            "flow step 'stamp': bind value='measure/level.level' reads the inner step 'level' of 'measure', which "
            "is a subgraph — a step outside a subgraph cannot read a value of a step inside it; move that step "
            "out of the subgraph"
        )

    def test_an_attribute_of_the_subgraph_itself_may_be_bound(self, tmp_path: Path) -> None:
        # `result` is the subgraph's own parameter, not a value of a step inside it
        text = MEASURE.replace("REF", "measure.result").replace("      steps:", "      result: level\n      steps:")
        record = FlowGraph.from_yaml(str(_write(tmp_path, text)))._run(demo_record())
        assert record["stamp"] == "level"

    def test_moved_out_of_the_subgraph_it_runs(self, tmp_path: Path) -> None:
        text = f"""flow:
  level: {{op: !class:{OPS}.SgMeanLevel {{}}}}
  stamp:
    op: !class:{OPS}.SgStamp {{key: stamp}}
    bind: {{value: level.level}}
outputs: stamp
"""
        record = FlowGraph.from_yaml(str(_write(tmp_path, text)))._run(demo_record())
        assert record["stamp"] == pytest.approx(float(demo_pixels().mean()) / 255.0)


# --------------------------------------------------------------------------------------------
# critic §2.6: the subgraph's own `result:` neither receives nor spreads a broadcast
# --------------------------------------------------------------------------------------------


class TestBroadcast:
    TEXT = f"""flow:
  prep:
    op: {SUBGRAPH}
      steps:
        grey: {{op: !class:recordstream.ops.image.ConvertMode {{mode: L}}}}
        probe: {{op: !class:{OPS}.SgHasResult {{}}}}
      result: grey
  probe2: {{op: !class:{OPS}.SgHasResult {{}}}}
outputs: probe2
"""

    @pytest.mark.parametrize("how", ["confluid.load", "FlowGraph.from_yaml"])
    def test_result_stays_where_it_is_written(self, tmp_path: Path, how: str) -> None:
        path = str(_write(tmp_path, self.TEXT))
        if how == "confluid.load":
            steps = FlowGraph(flow=confluid.load(path)["flow"], outputs="probe2").steps
        else:
            steps = FlowGraph.from_yaml(path).steps
        prep = steps[0].op
        assert prep.result == "grey"
        assert [s.op.result for s in prep.flow_steps if s.name == "probe"] == [""]  # not pushed down
        assert steps[1].op.result == ""  # not pushed sideways

    def test_the_files_outputs_does_not_reach_a_blank_result(self, tmp_path: Path) -> None:
        document = confluid.load(str(_write(tmp_path, DEMO_YAML.replace("      result: scaled\n", ""))))
        prep = FlowGraph(flow=document["flow"], outputs=document["outputs"]).steps[0].op
        assert prep.result == ""
        assert prep.output_step == "scaled"


# --------------------------------------------------------------------------------------------
# DESIGN §0.3: using one subgraph twice — the `include:` construction
# --------------------------------------------------------------------------------------------


class TestReuse:
    def test_one_steps_file_included_twice_gives_two_independent_insides(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "# the shared preparation\n"
            "grey:\n  op: !class:recordstream.ops.image.ConvertMode {mode: L}\n"
            f"scaled:\n  op: {SCALE}\n",
            "prep.steps.yaml",
        )
        text = f"""flow:
  read: {{}}
  prep:
    op: {SUBGRAPH}
      steps:
        include: prep.steps.yaml
      result: scaled
  mask: {{op: !class:recordstream.ops.numpy.Threshold {{low_level: 0.5}}}}
  prep2:
    from: read
    op: {SUBGRAPH}
      steps:
        include: prep.steps.yaml
outputs: mask
"""
        graph = FlowGraph.from_yaml(str(_write(tmp_path, text)))
        first, second = graph.steps[1].op, graph.steps[3].op
        assert list(first.steps) == list(second.steps) == ["grey", "scaled"]
        assert first.steps["grey"]["op"] is not second.steps["grey"]["op"]  # separate objects per use
        assert _mask_mean(graph._run(demo_record())) == DEMO_MASK_MEAN

    def test_two_written_copies_are_two_independent_insides(self) -> None:
        first, second = Subgraph(steps=_prep_steps()), Subgraph(steps=_prep_steps())
        first.flow_steps[1].op.target_max = 0.5
        assert second.flow_steps[1].op.target_max == 1.0


class TestSerialization:
    def test_a_dumped_subgraph_loads_back_and_runs_the_same(self) -> None:
        sub = Subgraph(
            steps={"grey": ConvertMode(mode="L"), "scaled": {"op": _prep_steps()["scaled"]}}, result="scaled"
        )
        back = confluid.load(confluid.dump(sub))
        assert type(back) is Subgraph and back.result == "scaled"
        assert [step.name for step in back.flow_steps] == ["grey", "scaled"]
        mask = Threshold(low_level=0.5)
        assert _mask_mean(mask(back(demo_record()))) == _mask_mean(mask(sub(demo_record()))) == DEMO_MASK_MEAN


class TestUnbuiltInnerSteps:
    """A subgraph built in code may hold still-unbuilt ``!class:`` markers as its steps."""

    TEXT = f"""steps:
  a: !class:{OPS}.SgStamp {{key: a, value: 1.0}}
  b: STEP_B
"""

    def _steps(self, step_b: str) -> Dict[str, Any]:
        return confluid.load(self.TEXT.replace("STEP_B", step_b), until="settled")["steps"]

    def test_a_bare_marker_keeps_its_step_keys(self) -> None:
        # parse_flow pops the reserved keys off an unbuilt bare marker, so this spelling is right
        sub = Subgraph(steps=self._steps(f"!class:{OPS}.SgStamp {{key: b, value: 2.0, from: a}}"))
        assert [(step.name, step.from_) for step in sub.flow_steps] == [("a", None), ("b", "a")]

    def test_a_marker_under_op_carrying_a_step_key_is_refused(self) -> None:
        sub = Subgraph(steps=self._steps(f"{{op: !class:{OPS}.SgStamp {{key: b, from: a}}}}"))
        with pytest.raises(ValueError) as refused:
            sub.flow_steps
        assert str(refused.value) == (
            "Subgraph: step 'b' carries 'from' inside its op's !class: marker, where SgStamp takes it as a plain "
            f"attribute and the step never reads it — write the step as a mapping: b: {{op: !class:{OPS}.SgStamp "
            "{...}, from: a}"
        )

    def test_steps_that_are_not_a_mapping(self) -> None:
        # the constructor's own validation refuses a list; a host setting it afterwards reaches the parse
        sub = Subgraph()
        sub.steps = ["a"]  # type: ignore[assignment]
        with pytest.raises(ValueError, match=r"^Subgraph: steps must be a mapping of step name -> op, got list$"):
            sub.flow_steps


class TestTheTypesBoundary:
    def test_a_type_produced_inside_is_not_needed_from_outside(self) -> None:
        from recordstream.items import Mask
        from recordstream.ops.numpy import ConnectedComponents

        sub = Subgraph(steps={"mask": Threshold(low_level=0.5), "boxes": ConnectedComponents()})
        assert Mask not in sub.consumes  # Threshold makes the Mask ConnectedComponents reads
        assert sub.consumes == tuple(Threshold(low_level=0.5).consumes)

    def test_named_declarations_beside_types_are_listed_by_type_name(self) -> None:
        sub = Subgraph(steps={"mask": Threshold(low_level=0.5), "frac": SgMaskFraction()})
        assert "Mask" in sub.consumes  # SgMaskFraction's {"mask": "Mask"}, as its type name
        assert "*" in sub.produces  # SgMaskFraction's {"fraction": "*"}


class TestAcrossProcesses:
    """Parallel routes pickle every step op (the spawn start method)."""

    def test_a_subgraph_pickles_for_a_spawn_worker(self) -> None:
        import pickle

        sub = Subgraph(steps=_prep_steps())
        sub.flow_steps  # parsed and cached before it crosses the process boundary
        back = pickle.loads(pickle.dumps(sub))
        assert back.flow_steps is back.flow_steps  # the cache survived and still matches its steps
        mask = Threshold(low_level=0.5)
        assert _mask_mean(mask(back(demo_record()))) == DEMO_MASK_MEAN

    def test_a_flow_holding_a_subgraph_runs_on_spawn_workers(self) -> None:
        graph = FlowGraph(
            source=[demo_record(), demo_record()],
            flow={"prep": Subgraph(steps=_prep_steps()), "mask": Threshold(low_level=0.5)},
        ).parallel(2)
        assert [_mask_mean(record) for record in graph] == [DEMO_MASK_MEAN, DEMO_MASK_MEAN]


# --------------------------------------------------------------------------------------------
# review fixes (2026-09-29): what the boundary declares, parsing twice, where a refusal points
# --------------------------------------------------------------------------------------------


class TestTheBoundaryIsTheRecordTheResultReturns:
    """``produces`` / ``flags`` follow the lineage of ``result`` — a sibling branch's entries never come back."""

    def test_an_earlier_result_does_not_produce_what_a_later_step_writes(self) -> None:
        sub = Subgraph(steps={"a": SgPut(key="x"), "b": SgPut(key="y")}, result="a")
        assert sub.produces == {"x": "*", "path": "*"}
        with pytest.raises(ChainContractError, match="SgDoubled needs the record entry 'fraction'"):
            check_chain([Subgraph(steps={"a": SgPut(key="x"), "b": SgPut(key="fraction")}, result="a"), SgDoubled()])

    def test_a_sibling_branch_is_not_in_what_comes_back(self) -> None:
        sub = Subgraph(
            steps={
                "s0": SgPut(key="x"),
                "yb": {"op": SgPut(key="fraction"), "from": "s0"},
                "zb": {"op": SgPut(key="z"), "from": "s0"},
            },
            result="zb",
        )
        assert sub.produces == {"x": "*", "path": "*", "z": "*"}
        graph = FlowGraph(flow={"sub": sub, "need": SgDoubled()})
        with pytest.raises(ChainContractError, match=r"^b.yaml:need: SgDoubled needs the record entry 'fraction'"):
            Tracer(graph, where="b.yaml").check({})

    def test_the_result_lineage_still_produces_what_it_writes(self) -> None:
        sub = Subgraph(
            steps={"s0": SgPut(key="x"), "yb": {"op": SgPut(key="fraction"), "from": "s0"}, "zb": SgPut(key="z")},
            result="zb",
        )
        # zb reads yb (the previous step), so yb's entry does come back
        assert sub.produces == {"x": "*", "fraction": "*", "path": "*", "z": "*"}
        check_chain([sub, SgDoubled()])  # passes

    def test_merge_from_brings_its_branch_along(self) -> None:
        sub = Subgraph(
            steps={
                "s0": SgPut(key="x"),
                "yb": {"op": SgPut(key="fraction"), "from": "s0"},
                "zb": {"op": SgPut(key="z"), "from": "s0", "merge_from": ["yb"]},
            },
            result="zb",
        )
        assert set(sub.produces) == {"x", "fraction", "z", "path"}

    def test_a_flag_raised_on_a_sibling_branch_is_not_raised_by_the_subgraph(self) -> None:
        # measured before: flags ('bright',), the check passed, and the gate ran without the flag
        branchy = {
            "s0": SgCount(label="s0"),
            "hb": {"op": SgRaiseBright(), "from": "s0"},
            "zb": {"op": SgCount(label="zb"), "from": "s0"},
        }
        sub = Subgraph(steps=dict(branchy), result="zb")
        assert sub.flags == ()
        refusal = "gate: SgGated is gated on the flag 'bright', which no node before it raises"
        with pytest.raises(ChainContractError, match=rf"^{refusal}"):
            Tracer(FlowGraph(flow={"sub": sub, "gate": SgGated(requires="bright")})).check(demo_record())
        # the same steps written flat are refused the same way
        with pytest.raises(ChainContractError, match=rf"^{refusal}"):
            Tracer(FlowGraph(flow={**branchy, "gate": SgGated(requires="bright")})).check(demo_record())

    def test_a_flag_on_the_result_lineage_is_raised(self) -> None:
        sub = Subgraph(steps={"s0": SgCount(label="s0"), "hb": SgRaiseBright(), "zb": SgCount(label="zb")})
        assert sub.flags == ("bright",)


class TestParsingNeverChangesTheStepsItIsGiven:
    """A bare marker's ``from:`` is read from a COPY — the steps mapping is the same after any number of parses."""

    TEXT = f"""steps:
  first: !class:{OPS}.SgPut {{key: first}}
  second: !class:{OPS}.SgPut {{key: second}}
  third: !class:{OPS}.SgPut {{key: third, from: first}}
"""

    def _steps(self) -> Dict[str, Any]:
        return confluid.load(self.TEXT, until="settled")["steps"]

    def test_a_result_changed_and_changed_back_parses_the_same_graph(self) -> None:
        # measured before: the re-parse wired third to second — path ('first', 'second', 'third')
        sub = Subgraph(steps=self._steps())
        first = [(step.name, step.from_) for step in sub.flow_steps]
        assert first == [("first", None), ("second", None), ("third", "first")]
        assert sub({})["path"] == ("first", "third")
        sub.result = "second"
        sub.flow_steps
        sub.result = ""
        assert [(step.name, step.from_) for step in sub.flow_steps] == first
        assert sub({})["path"] == ("first", "third")

    def test_parse_flow_leaves_the_marker_as_it_was_written(self) -> None:
        steps = self._steps()
        parse_flow(steps)
        assert steps["third"].kwargs["from"] == "first"
        again, _ = parse_flow(steps)
        assert again[2].from_ == "first"

    def test_an_unbuilt_parse_leaves_it_too(self) -> None:
        steps = self._steps()
        parsed, _ = parse_flow(steps, build=False)
        assert parsed[2].from_ == "first" and "from" not in parsed[2].op.kwargs
        assert steps["third"].kwargs["from"] == "first"

    def test_a_tracer_rerun_of_the_subgraph_keeps_the_wiring(self) -> None:
        tracer = Tracer(FlowGraph(flow={"a": Subgraph(steps=self._steps())})).run({})
        assert tracer.result[0]["path"] == ("first", "third")
        tracer.rerun_from("a", result="third")
        assert tracer.result[0]["path"] == ("first", "third")


NESTED = f"""flow:
  a:
    op: !class:{OPS}.SgPut {{key: a}}
  outer:
    op: {SUBGRAPH}
      steps:
        b:
          op: !class:{OPS}.SgPut {{key: b}}
        middle:
          op: {SUBGRAPH}
            steps:
              c:
                op: !class:{OPS}.SgPut {{key: c}}
              deep:
                op: {SUBGRAPH}
                  steps:
                    d:
                      op: !class:{OPS}.SgPut {{key: d}}
                  result: nosuch
outputs: outer
"""


class TestANestedRefusalPointsAtItsOwnSubgraph:
    def test_the_location_is_the_marker_of_the_subgraph_that_refuses(self, tmp_path: Path) -> None:
        # measured before: located at the OUTERMOST marker, nested_bad.yaml:5:9
        path = _write(tmp_path, NESTED)
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(path)).steps
        assert str(refused.value) == (
            "flow step 'outer' (a subgraph): flow step 'middle' (a subgraph): flow step 'deep' (a subgraph): "
            f"Subgraph: result 'nosuch' does not name an inner step (the steps are: ['d']) (at {path}:15:21)"
        )

    def test_an_inner_from_two_levels_deep_points_at_the_middle_subgraph(self, tmp_path: Path) -> None:
        step_c = "              c:\n                op: !class:tests._subgraph_ops.SgPut {key: c}\n"
        text = NESTED.replace("                  result: nosuch\n", "").replace(
            step_c, step_c + "                from: a\n"
        )
        path = _write(tmp_path, text)
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(path)).steps
        message = str(refused.value)
        assert message.startswith(
            "flow step 'outer' (a subgraph): flow step 'middle' (a subgraph): Subgraph: step 'c' reads 'a'"
        )
        assert message.endswith(f"(at {path}:10:15)") and message.count("(at ") == 1

    def test_a_refusal_of_the_outer_subgraph_itself_still_points_at_it(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            # appended at six spaces, 'result: zz' is a key of the OUTER subgraph's marker
            NESTED.replace("result: nosuch", "result: d").replace("outputs: outer\n", "") + "      result: zz\n",
        )
        with pytest.raises(ValueError) as refused:
            FlowGraph.from_yaml(str(path)).steps
        assert str(refused.value).startswith("flow step 'outer' (a subgraph): Subgraph: result 'zz'")
        assert str(refused.value).endswith(f"(at {path}:5:9)")


class TestAReferenceRefusalIsAPublicValueError:
    def test_the_traceback_header_is_a_public_name(self) -> None:
        # measured before: recordstream.flow.steps._UnknownReference
        with pytest.raises(ValueError) as refused:
            FlowGraph(flow={"a": SgCount(), "m": {"op": SgCount(), "from": "zz"}}).steps
        assert type(refused.value) is StepReferenceError
        assert not type(refused.value).__name__.startswith("_")
        assert issubclass(StepReferenceError, ValueError) and "StepReferenceError" in flow_pkg.__all__

    @pytest.mark.parametrize("key", ["from", "merge_from"])
    def test_an_outer_step_reading_a_step_inside_a_subgraph(self, key: str) -> None:
        with pytest.raises(ValueError) as refused:
            parse_flow({"prep": Subgraph(steps=_prep_steps()), "m": {"op": SgCount(), key: "prep/grey"}})
        assert str(refused.value) == (
            f"flow step 'm': {key}: 'prep/grey' reads the inner step 'grey' of 'prep', which is a subgraph — a step "
            "outside a subgraph cannot read a step inside it; move that step out of the subgraph"
        )


class TestTheStreamedRouteNamesTheMember:
    """With a stream-level op in the list, the refusal names the member's own index — before anything runs."""

    @pytest.mark.parametrize("where", [1, 2])
    def test_the_index_is_the_members_place_in_ops(self, where: int) -> None:
        from recordstream.ops.parallel import Parallel

        bad = Subgraph(steps=_prep_steps(), result="nosuch")
        ops: list = [SgCount(label="head"), Parallel(ops=[SgCount(label="par")], workers=1)]
        ops.insert(where, bad)
        with pytest.raises(ValueError, match=rf"^Stream.ops\[{where}\] \(a subgraph\): Subgraph: result 'nosuch'"):
            list(Stream(source=[demo_record(), demo_record()], ops=ops))
        assert SgCount.counts == {}
