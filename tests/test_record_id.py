"""Record identity — ``recordstream.record_id`` and ``IdentifiedSource``."""

from pathlib import Path
from typing import Any, List

import pytest
from confluid import load

from recordstream import (
    PARENT_ID,
    RECORD_ID,
    ConcatSource,
    DatasetSplit,
    IdentifiedSource,
    RangeSource,
    Stream,
    SupportsRecordKeys,
    derive,
    derived_id,
    has_record_keys,
    project,
    record_ids,
)
from recordstream.sources import FilesSource
from recordstream.storage.query import MetadataFilterSource


class _Places:
    """An indexable source with no name for its records."""

    def __init__(self, size: int = 5, uri: str = "") -> None:
        self._size = size
        self.dataset_uri = uri
        self.reads: List[int] = []

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Any:
        if not 0 <= index < self._size:
            raise IndexError(index)
        self.reads.append(index)
        return {"value": index}


class _Keyed(_Places):
    """The same, with a key of its own per record."""

    def record_keys(self) -> List[str]:
        return [f"k{index}" for index in range(self._size)]


class _Drop:
    def __call__(self, record: Any) -> Any:
        return record if record["value"] % 2 else None


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "a").mkdir()
    for name in ("a/1.txt", "a/2.txt", "b.txt"):
        (tmp_path / name).write_text("x")
    return tmp_path


class TestAnIdNamesTheDatasetAndTheRecord:
    def test_a_file_is_named_by_its_path_under_the_root(self, folder: Path) -> None:
        source = FilesSource(root=str(folder), pattern="**/*.txt", name="captures")
        there = folder.resolve().as_uri()  # the folder it reads names the dataset, before the source's name
        assert record_ids(source) == [f"{there}#a/1.txt", f"{there}#a/2.txt", f"{there}#b.txt"]
        assert has_record_keys(source)
        assert isinstance(source, SupportsRecordKeys)

    def test_a_listed_file_is_named_by_the_path_given(self, folder: Path) -> None:
        files = [str(folder / "b.txt"), str(folder / "a" / "1.txt")]
        assert record_ids(FilesSource(files=files, name="dropped")) == [f"dropped#{path}" for path in files]

    def test_two_folders_read_under_one_default_name_share_no_id(self, folder: Path, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        (other / "b.txt").write_text("x")
        first, second = FilesSource(root=str(folder)), FilesSource(root=str(other))
        assert first.name == second.name and not set(record_ids(first)) & set(record_ids(second))

    def test_a_dataset_uri_comes_before_a_name(self) -> None:
        source = _Keyed(size=2, uri="hf://datasets/some/set?split=train")
        source.name = "ignored"  # type: ignore[attr-defined]
        assert record_ids(source) == ["hf://datasets/some/set?split=train#k0", "hf://datasets/some/set?split=train#k1"]

    def test_a_source_without_keys_is_named_by_place_and_says_so(self) -> None:
        source = _Places(size=3, uri="mem://set")
        assert record_ids(source) == ["mem://set#0", "mem://set#1", "mem://set#2"]
        assert not has_record_keys(source)

    def test_asking_reads_no_record(self) -> None:
        source = _Places(size=4)
        record_ids(RangeSource(source, start=1))
        assert source.reads == []

    def test_a_source_that_cannot_be_counted_is_refused(self) -> None:
        with pytest.raises(TypeError, match="no record keys and no length"):
            record_ids(iter([{"value": 1}]))


class TestAViewKeepsItsSourcesIds:
    def test_a_range(self) -> None:
        assert record_ids(RangeSource(_Keyed(uri="d"), start=1, stop=3)) == ["d#k1", "d#k2"]

    def test_a_split_and_each_of_its_views(self) -> None:
        split = DatasetSplit(_Keyed(size=10, uri="d"), val_fraction=0.3, seed=7, split="val")
        every = set(record_ids(_Keyed(size=10, uri="d")))
        val, train = record_ids(split), record_ids(split.train)
        assert val == record_ids(split.val)
        assert set(val) | set(train) == every and not set(val) & set(train)
        assert [f"d#k{record['value']}" for record in split] == val

    def test_a_metadata_filter(self) -> None:
        kept = MetadataFilterSource(source=_Keyed(uri="d"), where="value.value >= 3")
        assert record_ids(kept) == ["d#k3", "d#k4"]

    def test_a_concatenation_gives_each_members_ids(self) -> None:
        both = ConcatSource([_Keyed(size=2, uri="a"), _Places(size=2, uri="b")])
        assert record_ids(both) == ["a#k0", "a#k1", "b#0", "b#1"]
        assert not has_record_keys(both)

    def test_a_stream_without_ops_is_its_source(self) -> None:
        assert record_ids(Stream(source=RangeSource(_Keyed(uri="d"), start=4))) == ["d#k4"]

    def test_a_view_over_a_view(self) -> None:
        inner = RangeSource(_Keyed(size=10, uri="d"), start=2)
        assert record_ids(RangeSource(inner, start=3, stop=5)) == ["d#k5", "d#k6"]


class TestIdentifiedSourceWritesTheIdIntoTheRecord:
    def test_each_record_carries_its_id_and_nothing_else_changes(self) -> None:
        source = IdentifiedSource(RangeSource(_Keyed(uri="d"), start=3))
        assert list(source) == [{"value": 3, RECORD_ID: "d#k3"}, {"value": 4, RECORD_ID: "d#k4"}]
        assert source[1][RECORD_ID] == "d#k4" and len(source) == 2

    def test_the_id_survives_ops_that_drop_records(self) -> None:
        stream = Stream(source=IdentifiedSource(_Keyed(uri="d")), ops=[_Drop()])
        assert [record[RECORD_ID] for record in stream] == ["d#k1", "d#k3"]

    def test_the_ids_alone_are_projected_without_a_read(self) -> None:
        inner = _Keyed(size=3, uri="d")
        source = IdentifiedSource(inner)
        assert list(project(source, [RECORD_ID])) == [{RECORD_ID: f"d#k{i}"} for i in range(3)]
        assert inner.reads == []

    def test_other_keys_are_projected_through_the_source(self) -> None:
        source = IdentifiedSource(_Keyed(size=2, uri="d"))
        assert list(project(source, ["value", RECORD_ID])) == [
            {"value": 0, RECORD_ID: "d#k0"},
            {"value": 1, RECORD_ID: "d#k1"},
        ]
        assert list(project(source, ["value"])) == [{"value": 0}, {"value": 1}]

    def test_a_view_over_it_keeps_the_ids(self) -> None:
        source = IdentifiedSource(_Keyed(uri="d"))
        assert record_ids(RangeSource(source, start=4)) == ["d#k4"]
        assert has_record_keys(source) and not has_record_keys(IdentifiedSource(_Places()))

    def test_it_declares_the_entry_it_writes(self, folder: Path) -> None:
        mapped = IdentifiedSource(_Keyed())
        mapped.source.produces = {"value": "int"}
        assert mapped.produces == {"value": "int", RECORD_ID: "str"}
        assert RECORD_ID in IdentifiedSource(FilesSource(root=str(folder))).produces

    def test_it_builds_from_yaml_and_lists_lazily(self, folder: Path) -> None:
        document = f"""
        source: !class:recordstream.sources.identified.IdentifiedSource
          source: !class:recordstream.sources.files.FilesSource
            root: {folder}
            name: captures
        """
        source = load(document)["source"]
        assert source._ids is None
        assert [record[RECORD_ID] for record in source] == [f"{folder.resolve().as_uri()}#b.txt"]

    def test_a_missing_source_is_refused_on_first_use(self) -> None:
        with pytest.raises(TypeError, match="IdentifiedSource requires a source"):
            len(IdentifiedSource())


class TestARecordMadeFromAnother:
    def test_it_is_named_by_its_parent_and_its_region(self) -> None:
        parent = {RECORD_ID: "d#k1", "image": "pixels"}
        child = derive(parent, "box3", {"crop": "pixels"})
        assert child == {"crop": "pixels", RECORD_ID: "d#k1@box3", PARENT_ID: "d#k1"}
        assert derive(child, 0)[PARENT_ID] == derived_id("d#k1", "box3")

    def test_a_parent_without_an_id_is_refused(self) -> None:
        with pytest.raises(KeyError, match="IdentifiedSource"):
            derive({"image": "pixels"}, "box3")
