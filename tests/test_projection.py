"""Tests for the field-projection primitive and the num_classes helper."""

import itertools
from typing import Collection, Iterator, List, get_args

import numpy as np
import pytest
import torch

from dataflux.core import Flux
from dataflux.projection import (
    _FIELDS,
    INPUT,
    TARGET,
    ProjectionField,
    SupportsProjection,
    _to_int,
    iter_inputs,
    iter_targets,
    num_classes,
    project,
)
from dataflux.sample import Sample

# --------------------------------------------------------------------------- #
# ProjectionField is a closed Literal a UI / form-spec can enumerate
# --------------------------------------------------------------------------- #


def test_projection_field_literal_enumerates_the_field_set() -> None:
    # The whole point of the Literal (vs a bare ``str``): callers — UIs, MCP
    # schemas, form-spec builders — read the allowed values from the annotation.
    assert get_args(ProjectionField) == ("input", "target", "metadata")


def test_fields_constant_is_derived_from_the_literal() -> None:
    # Single source of truth: the runtime-validation tuple comes FROM the Literal,
    # so the two can never drift.
    assert _FIELDS == get_args(ProjectionField)


# --------------------------------------------------------------------------- #
# Fallback path (sources that do NOT implement SupportsProjection)
# --------------------------------------------------------------------------- #


def _plain_source() -> List[Sample]:
    return [
        Sample(input=np.array([1, 2]), target=0, metadata={"i": 0}),
        Sample(input=np.array([3, 4]), target=2, metadata={"i": 1}),
    ]


def test_project_fallback_nulls_unrequested_fields() -> None:
    out = list(project(_plain_source(), (TARGET,)))
    assert [s.target for s in out] == [0, 2]
    assert all(s.input is None for s in out)
    assert all(s.metadata == {} for s in out)


def test_project_fallback_input_only() -> None:
    out = list(project(_plain_source(), (INPUT,)))
    assert all(s.target is None for s in out)
    assert np.array_equal(out[0].input, np.array([1, 2]))


def test_project_rejects_unknown_field() -> None:
    # An off-type value reaches the runtime guard (the Literal is a static hint,
    # not a runtime gate). mypy rightly objects — ignore it; that's the point.
    with pytest.raises(ValueError, match="Unknown projection field"):
        list(project(_plain_source(), ("bogus",)))  # type: ignore[arg-type]


def test_iter_helpers() -> None:
    assert list(iter_targets(_plain_source())) == [0, 2]
    inputs = list(iter_inputs(_plain_source()))
    assert np.array_equal(inputs[1], np.array([3, 4]))


# --------------------------------------------------------------------------- #
# Efficient path (sources that DO implement SupportsProjection)
# --------------------------------------------------------------------------- #


class _ProjectableSource:
    """A source that records which fields were requested and only builds those.

    Building the input increments ``input_builds`` — the test asserts a
    target-only walk never touches it, proving the efficient path skips
    unrequested-field construction.
    """

    def __init__(self, targets: List[int]) -> None:
        self.targets = targets
        self.input_builds = 0

    def _build_input(self, i: int) -> np.ndarray:
        self.input_builds += 1
        return np.full((2, 2), i)

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self) -> Iterator[Sample]:
        for i, t in enumerate(self.targets):
            yield Sample(input=self._build_input(i), target=t, metadata={})

    def project(self, fields: Collection[ProjectionField]) -> Iterator[Sample]:
        want = frozenset(fields)
        for i, t in enumerate(self.targets):
            yield Sample(
                input=self._build_input(i) if "input" in want else None,
                target=t if "target" in want else None,
                metadata={} if "metadata" not in want else {"i": i},
            )


def test_projectable_source_is_recognized_by_protocol() -> None:
    src = _ProjectableSource([0, 1])
    assert isinstance(src, SupportsProjection)


def test_efficient_target_only_skips_input_construction() -> None:
    src = _ProjectableSource([0, 1, 2])
    targets = list(iter_targets(src))
    assert targets == [0, 1, 2]
    assert src.input_builds == 0  # never decoded an input


# --------------------------------------------------------------------------- #
# num_classes
# --------------------------------------------------------------------------- #


def test_num_classes_int_targets_is_max_plus_one() -> None:
    # max id 2 even though id 1 absent from this "split" -> head size 3.
    assert num_classes(_ProjectableSource([0, 2, 0, 2])) == 3


def test_num_classes_walks_via_projection_without_inputs() -> None:
    src = _ProjectableSource([0, 1, 2, 3])
    assert num_classes(src) == 4
    assert src.input_builds == 0


def test_num_classes_torch_scalar_tensor_targets() -> None:
    src = [Sample(input=None, target=torch.tensor(k, dtype=torch.int64)) for k in (0, 4, 1)]
    assert num_classes(src) == 5


def test_num_classes_numpy_scalar_targets() -> None:
    src = [Sample(input=None, target=np.int64(k)) for k in (0, 1, 2)]
    assert num_classes(src) == 3


def test_num_classes_empty_source_raises() -> None:
    with pytest.raises(ValueError, match="no targets"):
        num_classes([])


def test_num_classes_none_target_raises() -> None:
    with pytest.raises(ValueError, match="no target"):
        num_classes([Sample(input=np.array([1]), target=None)])


# --------------------------------------------------------------------------- #
# _to_int coercion
# --------------------------------------------------------------------------- #


def test_to_int_rejects_bool() -> None:
    with pytest.raises(TypeError, match="bool"):
        _to_int(True)


def test_to_int_rejects_non_scalar() -> None:
    with pytest.raises(TypeError):
        _to_int("3")
    with pytest.raises(TypeError):
        _to_int(torch.tensor([1, 2, 3]))  # .item() on a multi-element tensor raises


def test_to_int_accepts_integer_valued_float_tensor() -> None:
    assert _to_int(torch.tensor(2.0)) == 2


# --------------------------------------------------------------------------- #
# Laziness
# --------------------------------------------------------------------------- #


def test_project_is_lazy() -> None:
    def infinite() -> Iterator[Sample]:
        for i in itertools.count():
            yield Sample(input=np.array([i]), target=i)

    first_two = list(itertools.islice(iter_targets(infinite()), 2))
    assert first_two == [0, 1]  # never exhausts the infinite source


# --------------------------------------------------------------------------- #
# Flux.project
# --------------------------------------------------------------------------- #


def test_flux_project_runs_pipeline_then_drops_fields() -> None:
    flux = Flux([Sample(input=np.array([1]), target=7, metadata={"k": "v"})])
    assert isinstance(flux, SupportsProjection)
    out = list(flux.project((TARGET,)))
    assert out[0].target == 7
    assert out[0].input is None
    # routed through the module-level project() too
    assert list(iter_targets(flux)) == [7]


def test_flux_num_classes_via_helper() -> None:
    flux = Flux([Sample(input=np.array([1]), target=t) for t in (0, 1, 2, 1)])
    assert num_classes(flux) == 3
