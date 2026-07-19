"""Tests for op-kind introspection (`sampleflux.kinds`) and the native multi-type engine."""

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, cast

import numpy as np
import pytest
from confluid import configurable

from sampleflux.core import Flux
from sampleflux.kinds import SAMPLE_KINDS, Input, OpContract, Target, classify_carrier, op_contract
from sampleflux.sample import InputMeta, Pair, Sample, TargetMeta

# ---------------------------------------------------------------------------
# Fixture ops (module-level so they pickle for spawn parity)
# ---------------------------------------------------------------------------


@configurable
class SampleOp:
    """A classic annotated Sample op."""

    def __call__(self, sample: Sample) -> Optional[Sample]:
        return sample._replace(input=sample.input + 1)


@configurable
class PairOp:
    """A metadata-free pair op: works on (input, target) tuples."""

    def __call__(self, pair: Tuple[Any, Any]) -> Tuple[Any, Any]:
        data, label = pair
        return data * 2, label


@configurable
class UntypedOp:
    """No annotations at all — works on anything (today's behavior)."""

    def __call__(self, sample):  # type: ignore[no-untyped-def]
        return sample


@configurable
class ExpandingOp:
    """A 1→N op, detected from the Iterator return annotation."""

    def __call__(self, sample: Sample) -> Iterator[Sample]:
        yield sample
        yield sample


@configurable
class ExpandingIterableOp:
    """A 1→N op via Iterable[...]."""

    def __call__(self, sample: Sample) -> Iterable[Sample]:
        return [sample, sample]


@configurable
class OverriddenOp:
    """Introspection-opaque op relying on explicit class-attr overrides."""

    SAMPLE_KIND_IN = "pair"
    SAMPLE_KIND_OUT = "pair"
    EXPANDS = False

    def __call__(self, *args):  # type: ignore[no-untyped-def]
        return args[0]


class StringAnnotatedOp:
    """PEP-563-style string annotations must resolve (get_type_hints)."""

    def __call__(self, sample: "Sample") -> "Sample":
        return sample


# ---------------------------------------------------------------------------
# classify_carrier / op_contract
# ---------------------------------------------------------------------------


class TestClassify:
    def test_kinds_taxonomy_is_closed(self) -> None:
        assert SAMPLE_KINDS == (
            "sample",
            "pair",
            "input",
            "target",
            "metadata",
            "input_meta",
            "target_meta",
            "value",
            "any",
        )

    def test_classify_carrier(self) -> None:
        assert classify_carrier(Sample(1)) == "sample"
        assert classify_carrier((np.zeros(3), 7)) == "pair"
        assert classify_carrier(np.zeros(3)) == "value"
        assert classify_carrier((1, 2, 3)) == "value"  # only 2-tuples are pairs


class TestOpContract:
    def test_sample_op(self) -> None:
        assert op_contract(SampleOp()) == OpContract("sample", "sample", False)

    def test_pair_op(self) -> None:
        assert op_contract(PairOp()) == OpContract("pair", "pair", False)

    def test_untyped_op_is_any(self) -> None:
        assert op_contract(UntypedOp()) == OpContract("any", "any", False)

    def test_expanding_iterator_and_iterable(self) -> None:
        assert op_contract(ExpandingOp()) == OpContract("sample", "sample", True)
        assert op_contract(ExpandingIterableOp()) == OpContract("sample", "sample", True)

    def test_class_attr_overrides(self) -> None:
        assert op_contract(OverriddenOp()) == OpContract("pair", "pair", False)

    def test_string_annotations_resolve(self) -> None:
        assert op_contract(StringAnnotatedOp()).accepts == "sample"

    def test_pair_return_is_not_expansion(self) -> None:
        # A Tuple return is a PAIR carrier, never a 1→N expansion.
        contract = op_contract(PairOp())
        assert contract.produces == "pair" and contract.expands is False

    def test_introspection_failure_degrades_to_any(self) -> None:
        class Broken:
            pass

        # Inject unresolvable string annotations dynamically (mypy-safe: no fake name in source).
        def _call(self, x):  # type: ignore[no-untyped-def]
            return x

        _call.__annotations__ = {"x": "NoSuchType", "return": "NoSuchType"}
        Broken.__call__ = _call  # type: ignore[method-assign, assignment]
        assert op_contract(Broken()) == OpContract("any", "any", False)


# ---------------------------------------------------------------------------
# Native multi-type engine
# ---------------------------------------------------------------------------


class TestNativeFlux:
    def test_pair_source_through_pair_op_stays_pairs(self) -> None:
        pairs = [(np.full(2, float(i)), i) for i in range(3)]
        flux = Flux(source=pairs, ops=[PairOp()], native=True)
        out = list(flux)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in out)
        assert out[1][0][0] == 2.0 and out[1][1] == 1

    def test_pair_source_promoted_for_sample_op_sticky(self) -> None:
        pairs = [(float(i), i) for i in range(3)]
        flux = Flux(source=pairs, ops=[SampleOp()], native=True)
        out = list(flux)
        assert all(isinstance(item, Sample) for item in out)  # promotion is sticky
        assert [s.input for s in out] == [1.0, 2.0, 3.0]
        assert all(s.meta == {} for s in out)

    def test_mixed_chain_pair_then_sample_op(self) -> None:
        pairs = [(float(i), i) for i in range(3)]
        flux = Flux(source=pairs, ops=[PairOp(), SampleOp()], native=True)
        out = list(flux)
        # PairOp doubled the value natively, then SampleOp promoted and added 1.
        assert [s.input for s in out] == [1.0, 3.0, 5.0]

    def test_pair_op_on_sample_carrier_preserves_metadata(self) -> None:
        samples = [Sample(input=float(i), target=i, metadata={"idx": i}) for i in range(3)]
        flux = Flux(source=samples, ops=[PairOp()], native=True)
        out = list(flux)
        assert [s.input for s in out] == [0.0, 2.0, 4.0]
        assert [s.meta["idx"] for s in out] == [0, 1, 2]  # metadata rides through the pair view

    def test_untyped_op_receives_carrier_verbatim(self) -> None:
        seen: list = []

        @configurable
        class Probe:
            def __call__(self, x):  # type: ignore[no-untyped-def]
                seen.append(type(x).__name__)
                return x

        list(Flux(source=[(1.0, 2)], ops=[Probe()], native=True))
        assert seen == ["tuple"]  # NOT coerced

    def test_default_mode_unchanged(self) -> None:
        # native=False (the default): 2-tuples coerce to Samples exactly as before.
        out = list(Flux(source=[(1.0, 2)], ops=[SampleOp()]))
        assert isinstance(out[0], Sample) and out[0].input == 2.0

    def test_native_spawn_parallel_parity(self) -> None:
        pairs = [(float(i), i) for i in range(4)]
        seq = list(Flux(source=list(pairs), ops=[PairOp()], native=True))
        par = list(Flux(source=list(pairs), ops=[PairOp()], native=True).parallel(2))
        assert [(a[0], a[1]) for a in seq] == [(b[0], b[1]) for b in par]

    def test_native_getitem(self) -> None:
        pairs = [(float(i), i) for i in range(4)]
        flux = Flux(source=pairs, ops=[PairOp()], native=True)
        item = flux[2]
        assert item[0] == 4.0 and item[1] == 2

    def test_native_filter_drop(self) -> None:
        @configurable
        class DropEven:
            def __call__(self, pair: Tuple[Any, Any]) -> Optional[Tuple[Any, Any]]:
                return None if pair[1] % 2 == 0 else pair

        out = list(Flux(source=[(0.0, 0), (1.0, 1), (2.0, 2)], ops=[DropEven()], native=True))
        assert [p[1] for p in out] == [1]


# ---------------------------------------------------------------------------
# Collate registry
# ---------------------------------------------------------------------------


class TestCollate:
    def test_sample_default_list_form_metadata(self) -> None:
        import torch

        from sampleflux.collate import collate

        batch = [Sample(input=torch.ones(2) * i, target=torch.tensor(i), metadata={"i": i}) for i in range(3)]
        out = collate(batch)
        assert isinstance(out, Sample) and out.is_batched
        assert out.input.shape == (3, 2) and out.batch_meta[2]["i"] == 2

    def test_pair_default(self) -> None:
        from sampleflux.collate import collate

        data, labels = collate([(np.ones(2), 1), (np.zeros(2), 0)])
        assert data.shape == (2, 2) and list(labels) == [1, 0]

    def test_value_default(self) -> None:
        from sampleflux.collate import collate

        out = collate([np.ones(2), np.zeros(2)])
        assert out.shape == (2, 2)

    def test_explicit_key_and_registration(self) -> None:
        from sampleflux.collate import collate, get_collate, register_collate, registered_collates

        @register_collate("yolo_test")
        def yolo_collate(items):  # type: ignore[no-untyped-def]
            return list(items)

        assert "yolo_test" in registered_collates()
        assert get_collate("yolo_test") is yolo_collate
        assert collate([(1, 2)], key="yolo_test") == [(1, 2)]

    def test_unknown_key_names_known(self) -> None:
        from sampleflux.collate import get_collate

        with pytest.raises(KeyError, match="known:"):
            get_collate("nope_nothing")

    def test_empty_batch_raises(self) -> None:
        from sampleflux.collate import collate

        with pytest.raises(ValueError, match="empty"):
            collate([])

    def test_stack_fallback_to_list(self) -> None:
        from sampleflux.collate import collate

        out = collate(["a", "b"], key="value")
        assert out == ["a", "b"]


# ---------------------------------------------------------------------------
# The field-scope grid + call styles (the (input, target, metadata) taxonomy)
# ---------------------------------------------------------------------------


@configurable
class BareInputOp:
    """Processes ONLY the input value (any array/tensor/dict), declared via the Input alias."""

    def __call__(self, x: Input):  # type: ignore[no-untyped-def]
        return x * 2


@configurable
class BareTargetOp:
    """Processes ONLY the target value."""

    def __call__(self, t: Target):  # type: ignore[no-untyped-def]
        return t + 100


@configurable
class UnpackedPairOp:
    """transform(input, target) — the classic AI signature, unpacked."""

    def __call__(self, input, target):  # type: ignore[no-untyped-def]
        return input * 2, target + 1


@configurable
class UnpackedInputMetaOp:
    """transform(input, metadata) — input with its metadata, unpacked."""

    def __call__(self, input, metadata):  # type: ignore[no-untyped-def]
        metadata["seen"] = True
        return input + 1, metadata


@configurable
class UnpackedTargetMetaOp:
    """transform(target, metadata) — target side selected by the first param name."""

    def __call__(self, target, metadata):  # type: ignore[no-untyped-def]
        return target * 10, {**metadata, "t": True}


@configurable
class UnpackedSampleOp:
    """transform(input, target, metadata) — the full triple, unpacked."""

    def __call__(self, input, target, metadata):  # type: ignore[no-untyped-def]
        return input + 1, target + 1, {**metadata, "s": True}


@configurable
class PackedInputMetaOp:
    """Packed InputMeta view — the op receives a named (input, metadata) object."""

    def __call__(self, view: InputMeta) -> InputMeta:
        meta = cast(Dict[str, Any], view.metadata)  # per-sample ops always see the dict form
        return InputMeta(view.input * 3, {**meta, "packed": True})


@configurable
class PackedTargetMetaOp:
    """Packed TargetMeta view."""

    def __call__(self, view: TargetMeta) -> TargetMeta:
        return TargetMeta(view.target - 1, view.metadata)


@configurable
class PackedNamedPairOp:
    """Packed Pair view (the named 2-tuple form)."""

    def __call__(self, p: Pair) -> Pair:
        return Pair(p.input + 0.5, p.target)


@configurable
class OptionalExtraArgOp:
    """One REQUIRED param + optional extras — must stay single-argument (packed/any)."""

    def __call__(self, sample, extra=None):  # type: ignore[no-untyped-def]
        return sample


class TestGridContracts:
    def test_bare_field_marks(self) -> None:
        assert op_contract(BareInputOp()) == OpContract("input", "any", False, "packed")
        assert op_contract(BareTargetOp()).accepts == "target"

    def test_unpacked_pair(self) -> None:
        assert op_contract(UnpackedPairOp()) == OpContract("pair", "any", False, "unpacked", ("input", "target"))

    def test_unpacked_meta_variants_by_param_names(self) -> None:
        assert op_contract(UnpackedInputMetaOp()) == OpContract(
            "input_meta", "any", False, "unpacked", ("input", "metadata")
        )
        assert op_contract(UnpackedTargetMetaOp()).accepts == "target_meta"

    def test_unpacked_sample_triple(self) -> None:
        assert op_contract(UnpackedSampleOp()) == OpContract(
            "sample", "any", False, "unpacked", ("input", "target", "metadata")
        )

    def test_packed_views(self) -> None:
        assert op_contract(PackedInputMetaOp()).accepts == "input_meta"
        assert op_contract(PackedInputMetaOp()).style == "packed"
        assert op_contract(PackedTargetMetaOp()).accepts == "target_meta"
        assert op_contract(PackedNamedPairOp()).accepts == "pair"

    def test_optional_extras_keep_single_arg_semantics(self) -> None:
        # Required arity 1 -> packed/any: op(sample) exactly as today.
        assert op_contract(OptionalExtraArgOp()) == OpContract("any", "any", False, "packed")

    def test_call_style_override(self) -> None:
        class Opaque:
            SAMPLE_KIND_IN = "pair"
            CALL_STYLE = "unpacked"

            def __call__(self, *args):  # type: ignore[no-untyped-def]
                return args[0], args[1]

        contract = op_contract(Opaque())
        assert contract.accepts == "pair" and contract.style == "unpacked"

    def test_classify_carrier_views_before_tuple_rule(self) -> None:
        assert classify_carrier(InputMeta(1, {"a": 1})) == "input_meta"
        assert classify_carrier(TargetMeta(1, {})) == "target_meta"
        assert classify_carrier(Pair(1, 2)) == "pair"
        assert classify_carrier((1, 2)) == "pair"  # the plain tuple stays a pair


class TestGridEngineBinding:
    def _samples(self, n: int = 2) -> list:
        return [Sample(input=float(i), target=i, metadata={"idx": i}) for i in range(n)]

    def test_bare_input_op_preserves_target_and_meta(self) -> None:
        out = list(Flux(source=self._samples(), ops=[BareInputOp()]))
        assert [s.input for s in out] == [0.0, 2.0]
        assert [s.target for s in out] == [0, 1]
        assert [s.meta["idx"] for s in out] == [0, 1]

    def test_bare_target_op(self) -> None:
        out = list(Flux(source=self._samples(), ops=[BareTargetOp()]))
        assert [s.target for s in out] == [100, 101]
        assert [s.input for s in out] == [0.0, 1.0]

    def test_unpacked_pair_op_merges_back(self) -> None:
        out = list(Flux(source=self._samples(), ops=[UnpackedPairOp()]))
        assert [(s.input, s.target) for s in out] == [(0.0, 1), (2.0, 2)]
        assert [s.meta["idx"] for s in out] == [0, 1]  # metadata preserved

    def test_unpacked_input_meta_op(self) -> None:
        out = list(Flux(source=self._samples(), ops=[UnpackedInputMetaOp()]))
        assert [s.input for s in out] == [1.0, 2.0]
        assert all(s.meta["seen"] is True for s in out)
        assert [s.target for s in out] == [0, 1]  # target untouched

    def test_unpacked_target_meta_op(self) -> None:
        out = list(Flux(source=self._samples(), ops=[UnpackedTargetMetaOp()]))
        assert [s.target for s in out] == [0, 10]
        assert all(s.meta["t"] is True for s in out)
        assert [s.input for s in out] == [0.0, 1.0]

    def test_unpacked_sample_op(self) -> None:
        out = list(Flux(source=self._samples(), ops=[UnpackedSampleOp()]))
        assert [(s.input, s.target) for s in out] == [(1.0, 1), (2.0, 2)]
        assert all(s.meta["s"] is True and "idx" in s.meta for s in out)

    def test_packed_views_merge_back(self) -> None:
        out = list(Flux(source=self._samples(), ops=[PackedInputMetaOp(), PackedTargetMetaOp()]))
        assert [s.input for s in out] == [0.0, 3.0]
        assert [s.target for s in out] == [-1, 0]
        assert all(s.meta["packed"] is True for s in out)

    def test_packed_named_pair(self) -> None:
        out = list(Flux(source=self._samples(), ops=[PackedNamedPairOp()]))
        assert [s.input for s in out] == [0.5, 1.5]
        assert [s.meta["idx"] for s in out] == [0, 1]

    def test_grid_chain_mixes_all_styles(self) -> None:
        ops = [BareInputOp(), UnpackedPairOp(), PackedInputMetaOp(), SampleOp()]
        out = list(Flux(source=self._samples(1), ops=ops))
        # 0.0 -> *2=0.0 -> pair(*2, +1)=(0.0, 1) -> *3=0.0 -> SampleOp(+1 input)=1.0
        assert out[0].input == 1.0 and out[0].target == 1
        assert out[0].meta["packed"] is True and out[0].meta["idx"] == 0

    def test_none_drops_in_every_scope(self) -> None:
        @configurable
        class DropPair:
            def __call__(self, input, target):  # type: ignore[no-untyped-def]
                return None

        assert list(Flux(source=self._samples(), ops=[DropPair()])) == []

    def test_pair_scope_single_return_is_a_loud_error(self) -> None:
        @configurable
        class BadPair:
            def __call__(self, input, target):  # type: ignore[no-untyped-def]
                return input  # ambiguous — must be a 2-tuple / Sample / None

        with pytest.raises(TypeError, match="same arity"):
            list(Flux(source=self._samples(1), ops=[BadPair()]))

    def test_native_bare_input_on_value_carrier_stays_value(self) -> None:
        out = list(Flux(source=[1.0, 2.0], ops=[BareInputOp()], native=True))
        assert out == [2.0, 4.0]  # no promotion — bare values stay bare

    def test_native_unpacked_pair_on_pair_carrier_stays_pair(self) -> None:
        out = list(Flux(source=[(1.0, 1), (2.0, 2)], ops=[UnpackedPairOp()], native=True))
        assert out == [(2.0, 2), (4.0, 3)]

    def test_native_view_carrier_promotes_field_correct(self) -> None:
        # An InputMeta carrier + a sample-op: from_any must NOT misread metadata as target.
        out = list(Flux(source=[InputMeta(1.0, {"m": 1})], ops=[SampleOp()], native=True))
        assert out[0].input == 2.0 and out[0].target is None and out[0].meta == {"m": 1}

    def test_view_collate_defaults(self) -> None:
        from sampleflux.collate import collate

        batch = collate([InputMeta(np.ones(2), {"i": 0}), InputMeta(np.zeros(2), {"i": 1})])
        assert isinstance(batch, InputMeta) and batch.input.shape == (2, 2)
        metas = cast(List[Dict[str, Any]], batch.metadata)  # batched form: list of per-item dicts
        assert metas[1] == {"i": 1}


# ---------------------------------------------------------------------------
# Combination bindings — mixed views as separate arguments
# ---------------------------------------------------------------------------


@configurable
class DualViewOp:
    """transform(InputMeta, TargetMeta) — both fields, each WITH its metadata."""

    def __call__(self, im: InputMeta, tm: TargetMeta):  # type: ignore[no-untyped-def]
        meta = cast(Dict[str, Any], im.metadata)
        return InputMeta(im.input * 2, {**meta, "im": True}), TargetMeta(
            tm.target + 1, {**meta, "im": True, "tm": True}
        )


@configurable
class MixedMarkViewOp:
    """transform(Input, TargetMeta) — a bare input value + the target with metadata."""

    def __call__(self, x: Input, tm: TargetMeta):  # type: ignore[no-untyped-def]
        return x + 0.5, tm


@configurable
class MetadataOnlyOp:
    """transform(metadata) — the metadata-only cell of the grid, declared by dict annotation."""

    def __call__(self, m: dict) -> dict:
        return {**m, "canonical": True}


class TestCombinationBindings:
    def _samples(self, n: int = 2) -> list:
        return [Sample(input=float(i), target=i, metadata={"idx": i}) for i in range(n)]

    def test_dual_view_contract(self) -> None:
        contract = op_contract(DualViewOp())
        assert contract.bindings == ("input_meta", "target_meta")
        assert contract.accepts == "sample"  # covers input+target+metadata — the grid summary
        assert contract.style == "unpacked"

    def test_dual_view_execution_merges_all_fields(self) -> None:
        out = list(Flux(source=self._samples(), ops=[DualViewOp()]))
        assert [s.input for s in out] == [0.0, 2.0]
        assert [s.target for s in out] == [1, 2]
        # last metadata-bearing element wins (it layered im's write too)
        assert all(s.meta["im"] is True and s.meta["tm"] is True and "idx" in s.meta for s in out)

    def test_mixed_mark_and_view(self) -> None:
        contract = op_contract(MixedMarkViewOp())
        assert contract.bindings == ("input", "target_meta")
        out = list(Flux(source=self._samples(1), ops=[MixedMarkViewOp()]))
        assert out[0].input == 0.5 and out[0].target == 0 and out[0].meta == {"idx": 0}

    def test_metadata_only_scope(self) -> None:
        contract = op_contract(MetadataOnlyOp())
        assert contract.accepts == "metadata" and contract.style == "packed"
        out = list(Flux(source=self._samples(1), ops=[MetadataOnlyOp()]))
        assert out[0].meta == {"idx": 0, "canonical": True}
        assert out[0].input == 0.0 and out[0].target == 0  # untouched

    def test_wrong_arity_return_is_a_loud_error(self) -> None:
        @configurable
        class Bad:
            def __call__(self, im: InputMeta, tm: TargetMeta):  # type: ignore[no-untyped-def]
                return im  # must be a 2-tuple / Sample / None

        with pytest.raises(TypeError, match="same arity"):
            list(Flux(source=self._samples(1), ops=[Bad()]))

    def test_binding_names_override_positions(self) -> None:
        @configurable
        class TargetFirst:
            def __call__(self, target, input):  # type: ignore[no-untyped-def]
                return target, input

        assert op_contract(TargetFirst()).bindings == ("target", "input")
