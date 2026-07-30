"""FlowGraph over dict records — merge_from fan-in, step[key]/bare-step bind, lowering parity."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from recordstream import FlowGraph, Image, Label, Mask, Pipeline, Record, Stream, Transform
from recordstream.flow import parse_flow
from recordstream.ops.structure import RenameField, SelectFields


class _AddOffset(Transform):
    """Adds a configurable offset to every Image payload (bind target)."""

    handles = (Image,)

    def __init__(self, offset: float = 0.0, field: Optional[str] = None) -> None:
        super().__init__(field=field)
        self.offset = offset

    def __call__(self, record: Record) -> Record:
        out = dict(record)
        for key, item in record.items():
            if isinstance(item, Image) and (self.field is None or key == self.field):
                out[key] = Image(np.asarray(item) + self.offset, layout=item.layout)
        return out


class _MakeMask(Transform):
    """Derives a Mask entry from the first Image (a branch producer)."""

    def __call__(self, record: Record) -> Record:
        image = next(item for item in record.values() if isinstance(item, Image))
        return {**record, "mask": Mask(np.asarray(image)[..., 0] > 0.5)}


def _seed(value: float = 0.0) -> Record:
    return {"image": Image(np.full((2, 3, 3), value, dtype=np.float32)), "label": Label("x")}


class TestFlowGraph:
    def test_linear_flow(self) -> None:
        graph = FlowGraph(source=[_seed(1.0)], flow={"plus": _AddOffset(offset=2.0)})
        (out,) = list(graph)
        assert isinstance(out, dict) and np.allclose(np.asarray(out["image"]), 3.0)

    def test_merge_from_union(self) -> None:
        # Fork: derive a mask on a branch, SELECT the new entry, union it back into the main
        # stream. (Selecting is the idiom — a full branch record would also carry its own
        # 'image', and last-wins would overwrite the boosted one.)
        flow = {
            "start": {},
            "masked": {"op": _MakeMask(), "from": "start"},
            "mask_only": {"op": SelectFields(keys=["mask"]), "from": "masked"},
            "boosted": {"op": _AddOffset(offset=1.0), "from": "start"},
            "out": {"from": "boosted", "merge_from": ["mask_only"]},
        }
        graph = FlowGraph(source=[_seed(0.75)], flow=flow, outputs="out")
        (out,) = list(graph)
        assert np.allclose(np.asarray(out["image"]), 1.75)  # the boosted branch's image survives
        assert "mask" in out and isinstance(out["mask"], Mask)  # the selected branch entry
        assert out["label"].value == "x"

    def test_merge_collision_last_wins(self) -> None:
        # Both branches carry 'image'; the merge source is listed LAST -> its image wins
        # (dict-union semantics, listed order).
        flow = {
            "start": {},
            "a": {"op": _AddOffset(offset=1.0), "from": "start"},
            "b": {"op": _AddOffset(offset=5.0), "from": "start"},
            "out": {"from": "a", "merge_from": ["b"]},
        }
        (out,) = list(FlowGraph(source=[_seed(0.0)], flow=flow, outputs="out"))
        assert np.allclose(np.asarray(out["image"]), 5.0)  # b (last) wins over a

    def test_rename_avoids_collision(self) -> None:
        flow = {
            "start": {},
            "a": {"op": _AddOffset(offset=1.0), "from": "start"},
            "b_renamed": {"op": RenameField(src="image", dst="image_b"), "from": "start"},
            "out": {"from": "a", "merge_from": ["b_renamed"]},
        }
        (out,) = list(FlowGraph(source=[_seed(0.0)], flow=flow, outputs="out"))
        assert np.allclose(np.asarray(out["image"]), 1.0)  # branch a intact
        assert "image_b" in out  # branch b united under its renamed key

    def test_step_key_bind(self) -> None:
        # step[key] binds the NAMED ENTRY of the bound step's record result.
        class _OffsetFromItem(Transform):
            def __init__(self, item: Any = None) -> None:
                super().__init__()
                self.item = item

            def __call__(self, record: Record) -> Record:
                offset = float(np.asarray(self.item).mean())
                out = dict(record)
                for key, value in record.items():
                    if isinstance(value, Image):
                        out[key] = Image(np.asarray(value) + offset, layout=value.layout)
                return out

        flow = {
            "start": {},
            "probe": {"op": _AddOffset(offset=2.0), "from": "start"},  # image becomes 2.0
            "final": {"op": _OffsetFromItem(), "from": "start", "bind": {"item": "probe[image]"}},
        }
        (out,) = list(FlowGraph(source=[_seed(0.0)], flow=flow, outputs="final"))
        assert np.allclose(np.asarray(out["image"]), 2.0)  # 0.0 + mean(2.0)

    def test_bare_step_bind_is_whole_record(self) -> None:
        class _CaptureWhole(Transform):
            def __init__(self, item: Any = None) -> None:
                super().__init__()
                self.item = item

            def __call__(self, record: Record) -> Record:
                # bare "probe" bind = the step's WHOLE result record dict.
                assert isinstance(self.item, dict) and isinstance(self.item["image"], Image)
                return record

        flow = {
            "start": {},
            "probe": {"op": _AddOffset(offset=1.0), "from": "start"},
            "final": {"op": _CaptureWhole(), "from": "start", "bind": {"item": "probe"}},
        }
        (out,) = list(FlowGraph(source=[_seed(0.0)], flow=flow, outputs="final"))
        assert isinstance(out, dict)

    def test_legacy_fanin_key_removed(self) -> None:
        # target_from / metadata_from (the legacy role fan-in) were purged; they are now
        # unknown step keys — a flow document using one fails loudly at parse.
        flow = {
            "start": {},
            "a": {"op": _AddOffset(offset=1.0), "from": "start"},
            "out": {"from": "a", "target_from": "start"},
        }
        graph = FlowGraph(source=[_seed(0.0)], flow=flow, outputs="out")
        with pytest.raises(ValueError, match="unknown step key"):
            list(graph)

    def test_metadata_from_key_removed(self) -> None:
        with pytest.raises(ValueError, match="unknown step key"):
            parse_flow({"a": {}, "b": {"from": "a", "merge_from": ["a"], "metadata_from": "a"}})

    def test_merge_from_forward_ref_raises(self) -> None:
        with pytest.raises(ValueError, match="EARLIER step"):
            parse_flow({"a": {"merge_from": ["b"]}, "b": {}})


class TestRecordsThroughStream:
    def test_stream_carries_record_dicts_verbatim(self) -> None:
        stream = Stream(source=[_seed(1.0)], ops=[_AddOffset(offset=1.0)])
        (out,) = list(stream)
        assert isinstance(out, dict) and np.allclose(np.asarray(out["image"]), 2.0)

    def test_getitem(self) -> None:
        stream = Stream(source=[_seed(1.0), _seed(2.0)], ops=[RenameField(src="label", dst="klass")])
        out = stream[1]
        assert "klass" in out and np.allclose(np.asarray(out["image"]), 2.0)

    def test_compose_ops_route_records(self) -> None:
        # Pipeline (the compose-group grouping op — TransformChain's replacement).
        stream = Stream(
            source=[_seed(1.0)], ops=[Pipeline(transforms=[_AddOffset(offset=1.0), _AddOffset(offset=2.0)])]
        )
        (out,) = list(stream)
        assert np.allclose(np.asarray(out["image"]), 4.0)


# --------------------------------------------------------------------------- #
# YAML flow documents end-to-end (docs/graph.md's exact spellings) — closes the
# gap where bind: was only ever exercised through Python-built flow dicts.
# --------------------------------------------------------------------------- #
class TestFlowYaml:
    def _doc(self) -> str:
        return """
flow:
  spec:    {}
  masked:  !class:recordstream.ops.numpy.Threshold {low_level: 0.5, from: spec}
  thresh:  !class:recordstream.ops.formula.FormulaOp {formula: "amax(a) * 0.6", field: image, from: spec}
  gated:
    op: !class:recordstream.ops.numpy.Threshold {output: gated_mask}
    from: spec
    bind:
      low_level: thresh[image]
  out: {from: gated, merge_from: [masked]}
outputs: out
"""

    def _record(self) -> Record:
        return {"image": Image(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0)}

    def test_yaml_bind_via_plain_mapping_step(self, tmp_path: Path) -> None:
        # Scalar reserved keys ride in the marker mapping; the nested bind: mapping MUST use
        # the plain-mapping (op:) step form — a nested mapping under a !class: marker is
        # consumed by confluid as addressed configuration and never reaches parse_flow.
        path = tmp_path / "graph.yaml"
        path.write_text(self._doc())
        out = list(FlowGraph.from_yaml(str(path), source=[self._record()]))[0]
        assert set(out) == {"image", "mask", "gated_mask"}
        assert int(np.asarray(out["mask"]).sum()) == 8  # fixed 0.5 threshold
        assert int(np.asarray(out["gated_mask"]).sum()) == 6  # per-record amax(a)*0.6 bind

    def test_yaml_bind_runs_the_same_from_either_loader(self, tmp_path: Path) -> None:
        # FlowGraph.from_yaml and a hand-parsed flow are the same graph on the same engine.
        path = tmp_path / "graph.yaml"
        path.write_text(self._doc())
        record = self._record()

        from_yaml = list(FlowGraph.from_yaml(str(path), source=[dict(record)]))
        import confluid

        doc = confluid.resolve(str(path))
        parsed = list(FlowGraph(source=[dict(record)], flow=doc["flow"], outputs=str(doc.get("outputs", ""))))
        assert len(from_yaml) == len(parsed) == 1
        assert set(from_yaml[0]) == set(parsed[0])

    def test_nested_bind_under_marker_is_consumed_not_parsed(self, tmp_path: Path) -> None:
        # Pin the confluid behavior that makes the op:-form MANDATORY for bind — if this
        # ever starts surviving in marker kwargs, the doc rule can be relaxed.
        path = tmp_path / "graph.yaml"
        path.write_text(
            """
flow:
  spec: {}
  gated: !class:recordstream.ops.numpy.Threshold
    from: spec
    bind:
      low_level: spec[image]
"""
        )
        import confluid

        marker = confluid.resolve(str(path))["flow"]["gated"]
        assert "bind" not in marker.kwargs  # consumed as addressed configuration


class TestNativeExecution:
    """The graph engine runs on its OWN executor — it never lowers to run (2026-07-29)."""

    def test_reader_accounting_is_computed_once_per_graph(self, monkeypatch: Any) -> None:
        # _result_readers depends only on (steps, outputs); recomputing it per record was an
        # O(steps^2) tax measured at ~half the graph engine's overhead over a flat op list.
        import recordstream.flow as flow_mod

        calls = {"n": 0}
        real = flow_mod._result_readers

        def counting(steps: Any, outputs: str) -> Any:
            calls["n"] += 1
            return real(steps, outputs)

        monkeypatch.setattr(flow_mod, "_result_readers", counting)
        graph = FlowGraph(source=[_seed(1.0) for _ in range(25)], flow={"plus": _AddOffset(offset=2.0)})
        assert len(list(graph)) == 25
        assert calls["n"] == 1

    def test_there_is_no_lowering_pass_left_to_call(self) -> None:
        # The delegation this replaced built Stream(ops=to_ops(...)). Both converters are gone
        # with the context ops they targeted; a reintroduced one would be a second executor.
        import recordstream
        import recordstream.flow as flow_mod

        for gone in ("to_ops", "from_ops", "flow_yaml_to_stream"):
            assert not hasattr(flow_mod, gone), f"{gone} is back — the lowering pass has returned"
            assert not hasattr(recordstream, gone)
        assert not hasattr(Stream, "from_flow_yaml")

    def test_parallel_runs_the_graph_natively(self) -> None:
        source = [_seed(float(i)) for i in range(6)]
        serial = list(FlowGraph(source=source, flow={"plus": _AddOffset(offset=2.0)}))
        parallel = list(FlowGraph(source=source, flow={"plus": _AddOffset(offset=2.0)}).parallel(2))

        assert len(parallel) == len(serial) == 6
        for got, want in zip(parallel, serial):
            assert np.allclose(np.asarray(got["image"]), np.asarray(want["image"]))

    def test_parallel_preserves_source_order(self) -> None:
        source = [_seed(float(i)) for i in range(8)]
        out = list(FlowGraph(source=source, flow={"plus": _AddOffset(offset=1.0)}).parallel(3))
        assert [float(np.asarray(r["image"]).flat[0]) for r in out] == [float(i) + 1.0 for i in range(8)]


class _SplitChannels(Transform):
    """A 1→N EXPANDING step: one record per channel of the image."""

    EXPANDS = True

    def __call__(self, record: Record) -> Any:  # type: ignore[override]
        image = record["image"]
        return [{**record, "image": Image(np.asarray(image)[..., c : c + 1]), "channel": c} for c in range(3)]


class _Tag(Transform):
    """Marks the record so a post-expansion step is observable."""

    def __init__(self, tag: str = "") -> None:
        super().__init__()
        self.tag = tag

    def __call__(self, record: Record) -> Record:
        return {**record, "tag": self.tag}


class TestExpandingSteps:
    """A 1→N step forks the REMAINING subgraph, one branch per child (2026-07-29)."""

    def test_expansion_yields_one_record_per_child(self) -> None:
        graph = FlowGraph(source=[_seed(1.0)], flow={"split": _SplitChannels()})
        out = list(graph)
        assert [r["channel"] for r in out] == [0, 1, 2]

    def test_downstream_steps_run_once_per_child(self) -> None:
        graph = FlowGraph(source=[_seed(1.0)], flow={"split": _SplitChannels(), "tagged": _Tag(tag="t")})
        out = list(graph)
        assert [r["channel"] for r in out] == [0, 1, 2]
        assert all(r["tag"] == "t" for r in out)

    def test_depth_first_sibling_order_across_chained_expansions(self) -> None:
        # Nested-loop order (the flat engine's documented contract): the INNER expansion
        # varies fastest. Two 3-way splits => 9 branches, the second split's channel cycling
        # 0,1,2 within each child of the first.
        graph = FlowGraph(source=[_seed(1.0)], flow={"a": _SplitChannels(), "b": _SplitChannels()})
        out = list(graph)
        assert len(out) == 9
        assert [r["channel"] for r in out] == [0, 1, 2] * 3

    def test_branches_do_not_share_env_state(self) -> None:
        # Each child gets its OWN shallow copy of the step environment (the graph twin of
        # Context.copy()): a later fan-in must not see a sibling's result.
        flow = {
            "src": {},
            "split": _SplitChannels(),
            "out": {"from": "split", "merge_from": ["src"]},
        }
        out = list(FlowGraph(source=[_seed(1.0)], flow=flow, outputs="out"))
        assert [r["channel"] for r in out] == [0, 1, 2]

    def test_len_and_getitem_raise_for_an_expanding_graph(self) -> None:
        graph = FlowGraph(source=[_seed(1.0)], flow={"split": _SplitChannels()})
        with pytest.raises(TypeError, match="EXPANDING"):
            len(graph)
        with pytest.raises(TypeError, match="EXPANDING"):
            graph[0]

    def test_empty_expansion_drops_the_branch(self) -> None:
        class _Drop(Transform):
            EXPANDS = True

            def __call__(self, record: Record) -> Any:  # type: ignore[override]
                return []

        assert list(FlowGraph(source=[_seed(1.0)], flow={"gone": _Drop()})) == []

    def test_expansion_survives_the_spawn_boundary(self) -> None:
        # The worker returns a LIST precisely so one seed can yield several records.
        source = [_seed(1.0), _seed(2.0)]
        out = list(FlowGraph(source=source, flow={"split": _SplitChannels()}).parallel(2))
        assert len(out) == 6
        assert [r["channel"] for r in out] == [0, 1, 2, 0, 1, 2]


class TestOneExecutor:
    """`ops:` is the LINEAR SPELLING of a graph — both forms run the same kernel (2026-07-29)."""

    def test_an_ops_list_compiles_to_a_linear_step_graph(self) -> None:
        from recordstream.core import linear_steps

        ops = [_AddOffset(offset=1.0), _AddOffset(offset=2.0)]
        steps, outputs = linear_steps(ops)
        assert [s.name for s in steps] == ["s0", "s1"]
        assert outputs == "s1"
        # Positional names, so the SAME op twice is two steps (a name-keyed mapping would collapse them).
        assert all(s.from_ is None and not s.bind and not s.merge_from for s in steps)

    def test_a_repeated_op_stays_two_distinct_steps(self) -> None:
        from recordstream.core import linear_steps

        op = _AddOffset(offset=1.0)
        steps, _ = linear_steps([op, op])
        assert len(steps) == 2
        (out,) = list(Stream(source=[_seed(0.0)], ops=[op, op]))
        assert np.allclose(np.asarray(out["image"]), 2.0)  # applied twice, not once

    def test_stream_and_flowgraph_agree_on_the_same_linear_chain(self) -> None:
        ops = [_AddOffset(offset=1.0), _MakeMask()]
        flat = list(Stream(source=[_seed(0.75)], ops=ops))
        graph = list(FlowGraph(source=[_seed(0.75)], flow={f"s{i}": op for i, op in enumerate(ops)}))
        assert len(flat) == len(graph) == 1
        assert np.allclose(np.asarray(flat[0]["image"]), np.asarray(graph[0]["image"]))
        assert np.array_equal(np.asarray(flat[0]["mask"]), np.asarray(graph[0]["mask"]))

    def test_an_empty_ops_list_is_the_identity(self) -> None:
        # The kernel treats "no steps" as the identity graph; a bare Stream must still yield.
        source = [_seed(1.0), _seed(2.0)]
        assert len(list(Stream(source=source, ops=[]))) == 2
        assert len(list(Stream(source=source))) == 2

    def test_a_linear_chain_takes_the_env_free_path(self) -> None:
        from recordstream.core import linear_steps
        from recordstream.flow import is_linear

        steps, outputs = linear_steps([_AddOffset(offset=1.0), _AddOffset(offset=2.0)])
        assert is_linear(steps, outputs)
        # A fan-out graph must NOT qualify — it needs the step environment.
        branchy, out = parse_flow({"start": {}, "a": {"op": _AddOffset(offset=1.0), "from": "start"}}, "a")
        assert not is_linear(branchy, out)

    def test_stream_expansion_still_yields_every_child(self) -> None:
        # The flat engine's 1→N contract, now served by the shared kernel's linear path.
        out = list(Stream(source=[_seed(1.0)], ops=[_SplitChannels(), _Tag(tag="t")]))
        assert [r["channel"] for r in out] == [0, 1, 2]
        assert all(r["tag"] == "t" for r in out)
