"""Typed FlowGraph — merge_from fan-in, step[key] bind, typed carriers through Flux, parity."""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from sampleflux import FlowGraph, Flux, Image, Label, Mask, Transform, TypedSample, to_ops
from sampleflux.flow import from_ops, parse_flow
from sampleflux.ops.context import MergeFields
from sampleflux.ops.structure import RenameField, SetRole


class _AddOffset(Transform):
    """Adds a configurable offset to every Image payload (bind target)."""

    handles = (Image,)

    def __init__(self, offset: float = 0.0, only: Optional[List[str]] = None) -> None:
        super().__init__(only=only)
        self.offset = offset

    def __call__(self, sample: TypedSample) -> TypedSample:
        out = sample
        for key, item in sample.items():
            if isinstance(item, Image) and (self.only is None or key in self.only):
                out = out.replace_field(key, Image(np.asarray(item) + self.offset, layout=item.layout))
        return out


class _MakeMask(Transform):
    """Derives a Mask field from the first Image (a branch producer)."""

    def __call__(self, sample: TypedSample) -> TypedSample:
        image = next(item for item in sample.fields.values() if isinstance(item, Image))
        out = sample.replace_field("mask", Mask(np.asarray(image)[..., 0] > 0.5))
        return out.set_role("mask", "target")


def _seed(value: float = 0.0) -> TypedSample:
    return TypedSample(
        {"image": Image(np.full((2, 3, 3), value, dtype=np.float32)), "label": Label("x")},
        roles={"label": "target"},
    )


class TestTypedFlowGraph:
    def test_linear_typed_flow(self) -> None:
        graph = FlowGraph(source=[_seed(1.0)], flow={"plus": _AddOffset(offset=2.0)})
        (out,) = list(graph)
        assert isinstance(out, TypedSample) and np.allclose(np.asarray(out["image"]), 3.0)

    def test_merge_from_union(self) -> None:
        # Fork: derive a mask on a branch, SELECT the new field, union it back into the main
        # stream. (Selecting is the idiom — a full branch bag would also carry its own
        # 'image', and last-wins would overwrite the boosted one.)
        from sampleflux.ops.structure import SelectFields

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
        assert "mask" in out and out.role_of("mask") == "target"  # the selected branch field
        assert out["label"].value == "x"

    def test_merge_collision_last_wins(self) -> None:
        # Both branches carry 'image'; the merge source is listed LAST -> its image wins.
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
        # bind offset := the 'probe' step's image payload mean is NOT expressible without a
        # value op — bind the FIELD instead and let the op read it: offset receives the
        # Image item from probe via step[image].
        class _OffsetFromItem(Transform):
            def __init__(self, item: Any = None) -> None:
                super().__init__()
                self.item = item

            def __call__(self, sample: TypedSample) -> TypedSample:
                offset = float(np.asarray(self.item).mean())
                out = sample
                for key, value in sample.items():
                    if isinstance(value, Image):
                        out = out.replace_field(key, Image(np.asarray(value) + offset, layout=value.layout))
                return out

        flow = {
            "start": {},
            "probe": {"op": _AddOffset(offset=2.0), "from": "start"},  # image becomes 2.0
            "final": {"op": _OffsetFromItem(), "from": "start", "bind": {"item": "probe[image]"}},
        }
        (out,) = list(FlowGraph(source=[_seed(0.0)], flow=flow, outputs="final"))
        assert np.allclose(np.asarray(out["image"]), 2.0)  # 0.0 + mean(2.0)

    def test_bare_step_bind_is_primary(self) -> None:
        class _CapturePrimary(Transform):
            def __init__(self, item: Any = None) -> None:
                super().__init__()
                self.item = item

            def __call__(self, sample: TypedSample) -> TypedSample:
                assert isinstance(self.item, Image)  # primary input-role field of the bound step
                return sample

        flow = {
            "start": {},
            "probe": {"op": _AddOffset(offset=1.0), "from": "start"},
            "final": {"op": _CapturePrimary(), "from": "start", "bind": {"item": "probe"}},
        }
        (out,) = list(FlowGraph(source=[_seed(0.0)], flow=flow, outputs="final"))
        assert isinstance(out, TypedSample)

    def test_typed_step_with_legacy_fanin_raises(self) -> None:
        flow = {
            "start": {},
            "a": {"op": _AddOffset(offset=1.0), "from": "start"},
            "out": {"from": "a", "target_from": "start"},
        }
        graph = FlowGraph(source=[_seed(0.0)], flow=flow, outputs="out")
        with pytest.raises(TypeError, match="LEGACY fan-in"):
            list(graph)

    def test_merge_and_legacy_fanin_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            parse_flow({"a": {}, "b": {"from": "a", "merge_from": ["a"], "target_from": "a"}})

    def test_merge_from_forward_ref_raises(self) -> None:
        with pytest.raises(ValueError, match="EARLIER step"):
            parse_flow({"a": {"merge_from": ["b"]}, "b": {}})


class TestTypedLoweringParity:
    def _flow(self) -> Dict[str, Any]:
        from sampleflux.ops.structure import SelectFields

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
        assert native[0] == lowered[0]

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

            def __call__(self, sample: TypedSample) -> TypedSample:
                return sample.replace_field("echo", self.item)

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


class TestTypedThroughFlux:
    def test_default_flux_carries_typed_verbatim(self) -> None:
        # No native=True needed: a TypedSample source item is NEVER coerced to legacy Sample.
        flux = Flux(source=[_seed(1.0)], ops=[_AddOffset(offset=1.0)])
        (out,) = list(flux)
        assert isinstance(out, TypedSample) and np.allclose(np.asarray(out["image"]), 2.0)

    def test_getitem_typed(self) -> None:
        flux = Flux(source=[_seed(1.0), _seed(2.0)], ops=[SetRole(key="image", role="aux")])
        assert flux[1].role_of("image") == "aux"

    def test_compose_ops_route_typed(self) -> None:
        from sampleflux.ops.transform_chain import TransformChain

        flux = Flux(source=[_seed(1.0)], ops=[TransformChain(ops=[_AddOffset(offset=1.0), _AddOffset(offset=2.0)])])
        (out,) = list(flux)
        assert np.allclose(np.asarray(out["image"]), 4.0)
