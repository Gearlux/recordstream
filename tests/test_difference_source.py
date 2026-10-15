"""``DifferenceSource`` — the records of one source whose key no record of another carries."""

from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List

import pytest
from confluid import get_registry

from recordstream.sources import DifferenceSource, FilesSource
from recordstream.sources.difference import CompareBy


class _Projecting:
    """An indexable source counting how often it is read whole and how often projected."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records
        self.full_reads = 0
        self.projections = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self.full_reads += 1
        return self.records[index]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def project(self, keys: Collection[str]) -> Iterator[Dict[str, Any]]:
        self.projections += 1
        for record in self.records:
            yield {k: v for k, v in record.items() if k in keys}


def test_the_records_whose_key_the_other_source_carries_are_left_out() -> None:
    source = [{"file": "a", "x": 1}, {"file": "b", "x": 2}, {"file": "c", "x": 3}]
    difference = DifferenceSource(source=source, remove=[{"file": "b"}, {"file": "z"}])
    assert list(difference) == [{"file": "a", "x": 1}, {"file": "c", "x": 3}]
    assert len(difference) == 2
    assert difference[1] == {"file": "c", "x": 3}
    assert difference.removed == 1


def test_every_copy_of_a_removed_key_is_left_out() -> None:
    source = [{"file": "a"}, {"file": "b"}, {"file": "a"}]
    assert list(DifferenceSource(source=source, remove=[{"file": "a"}])) == [{"file": "b"}]


def test_paths_compare_whole_by_default() -> None:
    source = [{"file": "/data/test/a.json"}, {"file": "/data/train/a.json"}]
    difference = DifferenceSource(source=source, remove=[{"file": "/data/test/a.json"}])
    assert list(difference) == [{"file": "/data/train/a.json"}]


def test_name_compares_the_file_name_wherever_the_folder() -> None:
    source = [{"file": "/data/train/a.json"}, {"file": "/data/train/a.mat"}, {"file": "/data/train/b.json"}]
    difference = DifferenceSource(source=source, remove=[{"file": "/hub/tier1/a.json"}], compare="name")
    assert list(difference) == [{"file": "/data/train/a.mat"}, {"file": "/data/train/b.json"}]


def test_stem_compares_the_file_name_without_its_extension() -> None:
    # a capture converted on its way into the other set (a pickle become a sidecar) keeps its stem
    source = [{"file": "/rolland/x.pickle"}, {"file": "/rolland/y.pickle"}, {"file": "/rolland/x.tar.gz"}]
    difference = DifferenceSource(source=source, remove=[{"file": "/hub/x.json"}], compare="stem")
    # only the LAST extension is dropped: x.tar.gz's stem is x.tar
    assert list(difference) == [{"file": "/rolland/y.pickle"}, {"file": "/rolland/x.tar.gz"}]


def test_paths_may_be_path_objects() -> None:
    source = [{"file": Path("/a/x.json")}, {"file": Path("/a/y.json")}]
    difference = DifferenceSource(source=source, remove=[{"file": "/b/x.json"}], compare="name")
    assert list(difference) == [{"file": Path("/a/y.json")}]


def test_any_key_may_be_compared() -> None:
    source = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert list(DifferenceSource(source=source, remove=[{"id": 2}], key="id")) == [{"id": 1}, {"id": 3}]


def test_nothing_in_common_keeps_every_record() -> None:
    difference = DifferenceSource(source=[{"file": "a"}], remove=[{"file": "b"}])
    assert list(difference) == [{"file": "a"}]
    assert difference.removed == 0


def test_an_empty_remove_keeps_every_record() -> None:
    assert list(DifferenceSource(source=[{"file": "a"}], remove=[])) == [{"file": "a"}]


def test_the_keys_are_read_through_each_sources_projection() -> None:
    source = _Projecting([{"file": "a", "signal": "heavy"}, {"file": "b", "signal": "heavy"}])
    remove = _Projecting([{"file": "a", "signal": "heavy"}])
    difference = DifferenceSource(source=source, remove=remove)
    assert len(difference) == 1
    assert (source.full_reads, source.projections) == (0, 1)
    assert (remove.full_reads, remove.projections) == (0, 1)
    assert difference[0] == {"file": "b", "signal": "heavy"}
    assert source.full_reads == 1


def test_a_projection_walks_only_the_records_kept() -> None:
    source = _Projecting([{"file": "a", "x": 1}, {"file": "b", "x": 2}])
    difference = DifferenceSource(source=source, remove=[{"file": "a"}])
    assert list(difference.project({"x"})) == [{"x": 2}]
    assert source.full_reads == 0


def test_two_folders_listed_by_files_source(tmp_path: Path) -> None:
    for folder, names in {"train": ["a.json", "b.json", "c.json"], "hub/tier1": ["b.json"]}.items():
        (tmp_path / folder).mkdir(parents=True)
        for name in names:
            (tmp_path / folder / name).write_text("{}")
    difference = DifferenceSource(
        source=FilesSource(root=str(tmp_path / "train"), pattern="*.json", formats=[]),
        remove=FilesSource(root=str(tmp_path / "hub"), pattern="**/*.json", formats=[]),
        compare="name",
    )
    assert [Path(r["file"]).name for r in difference] == ["a.json", "c.json"]


def test_a_source_record_without_the_key_is_refused_by_its_index() -> None:
    difference = DifferenceSource(source=[{"file": "a"}, {"path": "b"}], remove=[{"file": "a"}])
    with pytest.raises(KeyError, match=r"DifferenceSource: source record 1 has no 'file' entry to compare"):
        len(difference)


def test_a_remove_record_without_the_key_is_refused_by_its_index() -> None:
    difference = DifferenceSource(source=[{"file": "a"}], remove=[{"file": "a"}, {}])
    with pytest.raises(KeyError, match=r"DifferenceSource: remove record 1 has no 'file' entry to compare"):
        len(difference)


def test_a_name_comparison_of_a_value_that_is_no_path_is_refused() -> None:
    difference = DifferenceSource(source=[{"file": 3}], remove=[], compare="stem")
    with pytest.raises(TypeError, match=r"DifferenceSource: source record 0's 'file' is int 3 — compare='stem'"):
        len(difference)


def test_an_unhashable_value_is_refused() -> None:
    difference = DifferenceSource(source=[{"file": ["a"]}], remove=[])
    with pytest.raises(TypeError, match=r"DifferenceSource: source record 0's 'file' is list \['a'\]"):
        len(difference)


def test_an_unknown_comparison_is_refused() -> None:
    with pytest.raises(ValueError, match=r"compare\n  Input should be 'value', 'name' or 'stem'"):
        DifferenceSource(source=[], remove=[], compare="suffix")  # type: ignore[arg-type]


def test_a_missing_source_or_remove_is_refused_on_first_use() -> None:
    with pytest.raises(TypeError, match=r"DifferenceSource\.source must be an indexable source.*got NoneType"):
        len(DifferenceSource(remove=[]))
    with pytest.raises(TypeError, match=r"DifferenceSource\.remove must be a source.*got NoneType"):
        len(DifferenceSource(source=[]))


def test_zero_arg_construction_does_no_work() -> None:
    difference = DifferenceSource()
    assert difference.source is None and difference.remove is None
    assert (difference.key, difference.compare) == ("file", "value")


def test_the_comparisons_are_the_literal_a_gui_lists() -> None:
    from typing import get_args

    assert get_args(CompareBy) == ("value", "name", "stem")


def test_it_is_registered_as_a_source() -> None:
    assert get_registry().get_class("DifferenceSource") is DifferenceSource
