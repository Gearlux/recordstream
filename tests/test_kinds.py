"""Tests for op-kind introspection (`sampleflux.kinds`) and the native multi-type engine."""

from typing import Any, Iterable, Iterator, Optional, Tuple

import numpy as np
import pytest
from confluid import configurable

from sampleflux.core import Flux
from sampleflux.kinds import SAMPLE_KINDS, OpContract, classify_carrier, op_contract
from sampleflux.sample import Sample

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
        assert SAMPLE_KINDS == ("sample", "pair", "value", "any")

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
