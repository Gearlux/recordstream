"""Tests for the generic intake → DataFlux adapter."""

from pathlib import Path
from typing import Any

import confluid
import intake
import intake.source.base
import numpy as np
import pytest
import torch
import xarray as xr

from dataflux.sample import Sample
from dataflux.storage.intake import IntakeSource


class _XArrayPartitionedSource(intake.source.base.DataSource):
    """Minimal in-process intake DataSource for testing.

    Yields ``n_partitions`` tiny 1-D xarray.DataArray objects with attrs.
    """

    container = "xarray"
    name = "test_xarray_partitioned"
    version = "0.1"
    partition_access = True

    def __init__(self, n_partitions: int = 3, label: str = "alpha", metadata: Any = None) -> None:
        super().__init__(metadata=metadata or {})
        self._n = n_partitions
        self._label = label

    def _get_schema(self) -> Any:
        return intake.source.base.Schema(
            datashape=None,
            dtype=str(np.dtype(np.float32)),
            shape=(8,),
            npartitions=self._n,
            extra_metadata={},
        )

    def _get_partition(self, i: int) -> xr.DataArray:
        arr = np.arange(8, dtype=np.float32) + (i * 100)
        return xr.DataArray(
            arr,
            dims=["x"],
            attrs={"label": self._label, "partition_index": i, "samplerate": 100.0},
        )

    def _close(self) -> None:
        pass


def test_intake_source_iterates_xarray_partitions_as_samples() -> None:
    src = _XArrayPartitionedSource(n_partitions=4)
    df_src = IntakeSource(source=src)
    samples = list(df_src)
    assert len(samples) == 4
    assert all(isinstance(s, Sample) for s in samples)
    assert all(isinstance(s.input, torch.Tensor) for s in samples)
    assert all(s.input.shape == (8,) for s in samples)
    # Metadata propagates from xarray attrs.
    assert samples[0].metadata["label"] == "alpha"
    assert samples[0].metadata["samplerate"] == 100.0
    assert samples[2].metadata["partition_index"] == 2


def test_intake_source_uses_target_attr() -> None:
    src = _XArrayPartitionedSource(n_partitions=2, label="bravo")
    df_src = IntakeSource(source=src, target_attr="label")
    samples = list(df_src)
    assert all(s.target == "bravo" for s in samples)


def test_intake_source_target_none_when_attr_missing() -> None:
    src = _XArrayPartitionedSource(n_partitions=1)
    df_src = IntakeSource(source=src, target_attr="not_a_real_attr")
    s = next(iter(df_src))
    assert s.target is None
    # The metadata still includes the actual attrs; it's only target lookup that misses.
    assert "label" in s.metadata


def test_intake_source_len_matches_npartitions() -> None:
    src = _XArrayPartitionedSource(n_partitions=7)
    df_src = IntakeSource(source=src)
    assert len(df_src) == 7


def test_intake_source_handles_numpy_partition() -> None:
    class NDArraySource(intake.source.base.DataSource):
        container = "ndarray"
        name = "test_ndarray"
        version = "0.1"
        partition_access = True

        def _get_schema(self) -> Any:
            return intake.source.base.Schema(
                datashape=None, dtype=str(np.dtype(np.float64)), shape=(3,), npartitions=2, extra_metadata={}
            )

        def _get_partition(self, i: int) -> np.ndarray:
            return np.array([i, i + 1, i + 2], dtype=np.float64)

        def _close(self) -> None:
            pass

    df_src = IntakeSource(source=NDArraySource())
    samples = list(df_src)
    assert len(samples) == 2
    assert isinstance(samples[0].input, torch.Tensor)
    assert samples[0].input.tolist() == [0.0, 1.0, 2.0]
    assert samples[0].metadata == {}


def test_intake_source_requires_either_catalog_or_source() -> None:
    with pytest.raises(ValueError, match="catalog_path"):
        IntakeSource()


def test_intake_source_rejects_both_catalog_and_source() -> None:
    src = _XArrayPartitionedSource()
    with pytest.raises(ValueError, match="not both"):
        IntakeSource(catalog_path="a.yml", source_name="x", source=src)


def test_intake_source_partial_pair_raises() -> None:
    with pytest.raises(ValueError):
        IntakeSource(catalog_path="a.yml")
    with pytest.raises(ValueError):
        IntakeSource(source_name="x")


def test_intake_source_configurable_yaml_roundtrip() -> None:
    df_src = IntakeSource(catalog_path="cat.yml", source_name="entry", target_attr="label")
    state = confluid.dump(df_src)
    restored = confluid.load(state)
    assert isinstance(restored, IntakeSource)
    assert restored.catalog_path == "cat.yml"
    assert restored.source_name == "entry"
    assert restored.target_attr == "label"


def test_intake_source_reset_after_close(tmp_path: Path) -> None:
    src = _XArrayPartitionedSource(n_partitions=2)
    df_src = IntakeSource(source=src)
    samples_a = list(df_src)
    df_src.close()
    # User-supplied source is preserved; iterating again still works.
    samples_b = list(df_src)
    assert len(samples_a) == len(samples_b) == 2
