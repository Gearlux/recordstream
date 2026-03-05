from typing import Any, Iterator, List

import confluid  # type: ignore[import-not-found]

from dataflux.core import Flux
from dataflux.sample import Sample


@confluid.configurable
class MockSource:
    """A configurable data source for testing that doesn't dump the full payload."""

    def __init__(self, name: str = "test") -> None:
        self.name = name
        self._data: List[Any] = []

    def set_data(self, data: List[Any]) -> None:
        self._data = data

    def __iter__(self) -> Iterator[Sample]:
        for item in self._data:
            yield Sample.from_any(item)

    def __len__(self) -> int:
        return len(self._data)


def multiply(data: Any, factor: float = 1.0) -> Any:
    return data * factor


def add(data: Any, val: float = 0.0) -> Any:
    return data + val


def test_joint_flux_logic() -> None:
    """Verify that JointFlux aggregates streams and preserves per-source ops."""
    src_a = MockSource(name="src_a")
    src_a.set_data([1.0, 2.0])
    flux_a = Flux(src_a).map(multiply, factor=10.0)

    src_b = MockSource(name="src_b")
    src_b.set_data([3.0, 4.0])
    flux_b = Flux(src_b).map(add, val=100.0)

    joint = Flux.joint([flux_a, flux_b])

    results = joint.collect()
    assert len(results) == 4
    assert results[0].input == 10.0
    assert results[1].input == 20.0
    assert results[2].input == 103.0
    assert results[3].input == 104.0


def test_joint_serialization() -> None:
    """Verify that a hierarchical JointFlux tree is fully serializable via DataSource definitions."""
    src_a = MockSource(name="source_a")
    flux_a = Flux(src_a).map(multiply, factor=10.0)

    src_b = MockSource(name="source_b")
    flux_b = Flux(src_b).map(add, val=5.0)

    # Wrap in a global pipeline
    pipeline = Flux.joint([flux_a, flux_b]).map(multiply, factor=2.0)

    # 1. Serialize
    yaml_state = confluid.dump(pipeline)
    assert "!class:MockSource" in yaml_state
    assert "name: source_a" in yaml_state

    # 2. Reconstruct
    new_pipeline = confluid.load(yaml_state)

    # 3. Manually provide data to the reconstructed sources
    new_pipeline.source.fluxes[0].source.set_data([1.0])
    new_pipeline.source.fluxes[1].source.set_data([2.0])

    results = list(new_pipeline)
    # (1 * 10) * 2 = 20
    # (2 + 5) * 2 = 14
    assert results[0].input == 20.0
    assert results[1].input == 14.0


def test_joint_parallel_execution() -> None:
    """Verify that JointFlux works with the parallel engine."""
    src_a = MockSource(name="a")
    src_a.set_data([1.0] * 5)
    flux_a = Flux(src_a).map(multiply, factor=10.0)

    src_b = MockSource(name="b")
    src_b.set_data([2.0] * 5)
    flux_b = Flux(src_b).map(add, val=5.0)

    # Run joint stream in parallel
    pipeline = Flux.joint([flux_a, flux_b]).parallel(workers=2)

    results = pipeline.collect()
    assert len(results) == 10
    # First 5: 1.0 * 10 = 10.0
    for i in range(5):
        assert results[i].input == 10.0
    # Next 5: 2.0 + 5 = 7.0
    for i in range(5, 10):
        assert results[i].input == 7.0
