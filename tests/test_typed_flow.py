"""FlowGraph over dict records — merge_from fan-in, step[key]/bare-step bind, lowering parity."""

from typing import Any, Dict, Optional

import numpy as np
import pytest

from sampleflux import FlowGraph, Flux, Image, Label, Mask, Pipeline, Record, Transform, to_ops
from sampleflux.flow import from_ops, parse_flow
from sampleflux.ops.context import MergeFields
from sampleflux.ops.structure import RenameField, SelectFields


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


class TestLoweringParity:
    def _flow(self) -> Dict[str, Any]:
        return {
            "start": {},
            "masked": {"op": _MakeMask(), "from": "start"},
            "mask_only": {"op": SelectFields(keys=["mask"]), "from": "masked"},
            "boosted": {"op": _AddOffset(offset=1.0), "from": "start"},
            "out": {"from": "boosted", "merge_from": ["mask_only"]},
        }

    def test_to_ops_runs_on_flux(self) -> None:
        # The lowered flat op list (MergeFields wiring) matches the native FlowGraph result.
        steps, outputs = parse_flow(self._flow())
        native = list(FlowGraph(source=[_seed(0.25)], flow=self._flow(), outputs="out"))
        lowered = list(Flux(source=[_seed(0.25)], ops=to_ops(steps, outputs)))
        assert len(native) == len(lowered) == 1
        assert list(native[0].keys()) == list(lowered[0].keys())
        assert np.array_equal(np.asarray(native[0]["image"]), np.asarray(lowered[0]["image"]))
        assert np.array_equal(np.asarray(native[0]["mask"]), np.asarray(lowered[0]["mask"]))

    def test_round_trip_from_ops(self) -> None:
        steps, outputs = parse_flow(self._flow())
        ops = to_ops(steps, outputs)
        assert any(isinstance(op, MergeFields) for op in ops)
        lifted, lifted_out = from_ops(ops)
        relowered = to_ops(*parse_flow(lifted, lifted_out))
        native = list(Flux(source=[_seed(0.5)], ops=relowered))
        assert len(native) == 1 and "mask" in native[0]

    def test_key_bind_round_trip(self) -> None:
        class _Reader(Transform):
            def __init__(self, item: Any = None) -> None:
                super().__init__()
                self.item = item

            def __call__(self, record: Record) -> Record:
                return {**record, "echo": self.item}

        flow = {
            "start": {},
            "probe": {"op": _AddOffset(offset=3.0), "from": "start"},
            "final": {"op": _Reader(), "from": "start", "bind": {"item": "probe[image]"}},
        }
        steps, outputs = parse_flow(flow)
        ops = to_ops(steps, outputs)
        lifted, _ = from_ops(ops)
        # the key-bind grammar survives the round trip
        final_step = lifted["final"] if "final" in lifted else list(lifted.values())[-1]
        assert isinstance(final_step, dict) and final_step["bind"]["item"].endswith("[image]")
        (out,) = list(Flux(source=[_seed(0.0)], ops=ops))
        assert np.allclose(np.asarray(out["echo"]), 3.0)


class TestRecordsThroughFlux:
    def test_flux_carries_record_dicts_verbatim(self) -> None:
        flux = Flux(source=[_seed(1.0)], ops=[_AddOffset(offset=1.0)])
        (out,) = list(flux)
        assert isinstance(out, dict) and np.allclose(np.asarray(out["image"]), 2.0)

    def test_getitem(self) -> None:
        flux = Flux(source=[_seed(1.0), _seed(2.0)], ops=[RenameField(src="label", dst="klass")])
        out = flux[1]
        assert "klass" in out and np.allclose(np.asarray(out["image"]), 2.0)

    def test_compose_ops_route_records(self) -> None:
        # Pipeline (the compose-group grouping op — TransformChain's replacement).
        flux = Flux(source=[_seed(1.0)], ops=[Pipeline(transforms=[_AddOffset(offset=1.0), _AddOffset(offset=2.0)])])
        (out,) = list(flux)
        assert np.allclose(np.asarray(out["image"]), 4.0)
