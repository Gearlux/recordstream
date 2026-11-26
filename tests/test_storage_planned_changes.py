"""The storage sinks that REPLACE what they hold say so before a run — their ``planned_changes()``.

``HDF5Sink(overwrite=True)`` opens its file in mode ``w`` and ``ZarrBatchSink(overwrite=True)`` its array: what was
there is gone at the first record. Each says so with one ``delete …`` line, which a processor's plan lists as an
action (``recordstream run --dry_run`` prints it). Asking changes nothing on disk. A sink that appends, or whose file
is not there yet, has nothing to say.

Measured before: both sinks emptied their store when a run opened them, and the plan read ``sink: HDF5Sink`` only.
"""

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from recordstream.core import Stream
from recordstream.items import Image, Record
from recordstream.processing import DatasetProcessor
from recordstream.storage.base import size_text
from recordstream.storage.hdf5 import HDF5Sink
from recordstream.storage.zarr import ZarrBatchSink


class Three:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> Record:
        if not 0 <= index < 3:
            raise IndexError(index)
        return {"image": Image(np.full((2, 2), float(index), dtype=np.float32))}

    def __iter__(self) -> Iterator[Record]:
        return (self[index] for index in range(3))


def _written_h5(path: Path) -> Path:
    sink = HDF5Sink(path, overwrite=True)
    for record in Three():
        sink.write(record)
    sink.flush()
    sink.close()
    return path


def _written_zarr(path: Path) -> Path:
    sink = ZarrBatchSink(path, shape=[2, 2], overwrite=True)
    for record in Three():
        sink.write(record)
    sink.flush()
    return path


def _files(folder: Path) -> dict[str, bytes]:
    return {str(each.relative_to(folder)): each.read_bytes() for each in sorted(folder.rglob("*")) if each.is_file()}


class TestTheSizeText:
    @pytest.mark.parametrize(
        ("size", "text"),
        [(0, "0B"), (803, "803B"), (1024, "1.0K"), (317_440, "310.0K"), (1_288_490_189, "1.2G"), (2 * 1024**4, "2.0T")],
    )
    def test_a_byte_count_reads_as_du_writes_it(self, size: int, text: str) -> None:
        assert size_text(size) == text


class TestTheHDF5Sink:
    def test_overwrite_over_an_existing_file_says_it_deletes_the_file(self, tmp_path: Path) -> None:
        path = _written_h5(tmp_path / "train.h5")
        assert HDF5Sink(path, overwrite=True).planned_changes() == [f"delete {path} ({size_text(path.stat().st_size)})"]

    def test_asking_leaves_the_file_as_it_was(self, tmp_path: Path) -> None:
        path = _written_h5(tmp_path / "train.h5")
        before = path.read_bytes()
        HDF5Sink(path, overwrite=True).planned_changes()
        assert path.read_bytes() == before

    def test_an_appending_sink_says_nothing(self, tmp_path: Path) -> None:
        assert HDF5Sink(_written_h5(tmp_path / "train.h5")).planned_changes() == []

    def test_a_file_that_is_not_there_yet_is_nothing_to_delete(self, tmp_path: Path) -> None:
        assert HDF5Sink(tmp_path / "train.h5", overwrite=True).planned_changes() == []
        assert not (tmp_path / "train.h5").exists()

    def test_the_processors_plan_lists_it_and_changes_nothing(self, tmp_path: Path) -> None:
        path = _written_h5(tmp_path / "train.h5")
        before = path.read_bytes()
        plan = DatasetProcessor(stream=Stream(source=Three()), sink=HDF5Sink(path, overwrite=True)).plan()
        assert plan.actions == [f"delete {path} ({size_text(len(before))})"]
        assert plan.would_stop is False
        assert path.read_bytes() == before


class TestTheZarrBatchSink:
    def test_overwrite_over_an_existing_array_says_it_deletes_the_array(self, tmp_path: Path) -> None:
        path = _written_zarr(tmp_path / "train.zarr")
        held = _files(path / "data")
        count = "1 file" if len(held) == 1 else f"{len(held)} files"
        assert ZarrBatchSink(path, shape=[2, 2], overwrite=True).planned_changes() == [
            f"delete {path}/data ({count}, {size_text(sum(len(each) for each in held.values()))})"
        ]

    def test_asking_leaves_the_array_as_it_was(self, tmp_path: Path) -> None:
        path = _written_zarr(tmp_path / "train.zarr")
        before = _files(path)
        ZarrBatchSink(path, shape=[2, 2], overwrite=True).planned_changes()
        assert _files(path) == before

    def test_an_appending_sink_says_nothing(self, tmp_path: Path) -> None:
        path = _written_zarr(tmp_path / "train.zarr")
        assert ZarrBatchSink(path, shape=[2, 2]).planned_changes() == []

    def test_an_array_that_is_not_there_yet_is_nothing_to_delete(self, tmp_path: Path) -> None:
        assert ZarrBatchSink(tmp_path / "train.zarr", shape=[2, 2], overwrite=True).planned_changes() == []
        assert not (tmp_path / "train.zarr").exists()

    def test_the_processors_plan_lists_it_and_changes_nothing(self, tmp_path: Path) -> None:
        path = _written_zarr(tmp_path / "train.zarr")
        before = _files(path)
        sink = ZarrBatchSink(path, shape=[2, 2], overwrite=True)
        plan = DatasetProcessor(stream=Stream(source=Three()), sink=sink).plan()
        assert len(plan.actions) == 1 and plan.actions[0].startswith(f"delete {path}/data (")
        assert _files(path) == before
