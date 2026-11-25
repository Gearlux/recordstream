"""Groups as one record: as folders (``GroupedFilesSink`` / ``GroupedFilesSource``) or as a groups file
(``GroupsFileSink`` / ``GroupedSource``)."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pytest
from confluid import get_registry

from recordstream.sources import GroupedFilesSource, GroupedSource
from recordstream.sources.grouped import PICKS
from recordstream.storage.grouped_files import GroupedFilesSink, GroupsFileSink


def _files(folder: Path, *names: str) -> List[Path]:
    paths = []
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        paths.append(path)
    return paths


def _tree(folder: Path) -> List[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


@pytest.fixture
def grouped(tmp_path: Path) -> Path:
    """Two files of their own, a group of three and a group of two."""
    _files(tmp_path, "a.png", "z.png", "group_0001/b.png", "group_0001/c.png", "group_0001/d.png")
    _files(tmp_path, "group_0002/e.png", "group_0002/f.png")
    return tmp_path


class TestTheSinkWritesAFolderPerGroup:
    def test_a_grouped_file_goes_into_its_groups_folder_and_another_stays_where_it_is(self, tmp_path: Path) -> None:
        one, two, alone = _files(tmp_path / "in", "one.png", "two.png", "alone.png")
        sink = GroupedFilesSink(path=str(tmp_path / "out"))
        for record in ({"file": str(one), "group": 1}, {"file": str(two), "group": 1}, {"file": str(alone)}):
            sink.write(record)
        sink.flush()
        assert _tree(tmp_path / "out") == ["alone.png", "group_0001/one.png", "group_0001/two.png"]
        assert one.exists() and (tmp_path / "out" / "group_0001" / "one.png").read_text() == "one.png"  # a copy

    def test_the_place_under_root_is_kept_so_a_group_stays_inside_its_class_folder(self, tmp_path: Path) -> None:
        cat, dog, other = _files(tmp_path / "in", "cat/1.png", "dog/1.png", "cat/2.png")
        sink = GroupedFilesSink(path=str(tmp_path / "out"), root=str(tmp_path / "in"))
        for path, group in ((cat, 3), (dog, 3), (other, None)):
            sink.write({"file": str(path), "group": group})
        assert _tree(tmp_path / "out") == ["cat/2.png", "cat/group_0003/1.png", "dog/group_0003/1.png"]

    def test_the_entries_and_the_folder_prefix_are_settings(self, tmp_path: Path) -> None:
        (path,) = _files(tmp_path / "in", "x.png")
        sink = GroupedFilesSink(path=str(tmp_path / "out"), file="picture", group="duplicate_group", prefix="copies_")
        sink.write({"picture": str(path), "duplicate_group": "left"})
        assert _tree(tmp_path / "out") == ["copies_left/x.png"]

    def test_a_file_that_would_land_on_another_is_refused(self, tmp_path: Path) -> None:
        first, second = _files(tmp_path / "in", "cat/1.png", "dog/1.png")
        sink = GroupedFilesSink(path=str(tmp_path / "out"))
        sink.write({"file": str(first)})
        with pytest.raises(FileExistsError, match="set `root`"):
            sink.write({"file": str(second)})

    def test_a_file_outside_root_and_a_record_without_a_file_are_refused(self, tmp_path: Path) -> None:
        (path,) = _files(tmp_path / "elsewhere", "x.png")
        sink = GroupedFilesSink(path=str(tmp_path / "out"), root=str(tmp_path / "in"))
        with pytest.raises(ValueError, match="is not under root"):
            sink.write({"file": str(path)})
        with pytest.raises(KeyError, match="no 'file' entry"):
            sink.write({"group": 1})

    def test_it_is_a_sink_node_that_builds_without_arguments(self) -> None:
        assert getattr(GroupedFilesSink, "__confluid_category__") == "sink"
        assert "GroupedFilesSink" in get_registry().list_classes(category="sink")
        GroupedFilesSink()


class TestTheSourceServesOneFilePerGroup:
    def test_a_group_folder_is_one_record(self, grouped: Path) -> None:
        source = GroupedFilesSource(root=str(grouped), pick="first")
        records = list(source)
        assert len(source) == len(records) == 4
        assert [Path(r["file"]).relative_to(grouped).as_posix() for r in records] == [
            "a.png",
            "group_0001/b.png",
            "group_0002/e.png",
            "z.png",
        ]
        assert [r["group_size"] for r in records] == [1, 3, 2, 1]

    def test_next_takes_another_file_of_the_group_at_every_read(self, grouped: Path) -> None:
        source = GroupedFilesSource(root=str(grouped))
        epochs = [[Path(r["file"]).name for r in source] for _ in range(4)]
        assert [epoch[1] for epoch in epochs] == ["b.png", "c.png", "d.png", "b.png"]
        assert [epoch[2] for epoch in epochs] == ["e.png", "f.png", "e.png", "f.png"]
        assert {epoch[0] for epoch in epochs} == {"a.png"}

    def test_first_is_the_same_file_at_every_read(self, grouped: Path) -> None:
        source = GroupedFilesSource(root=str(grouped), pick="first")
        assert {Path(source[1]["file"]).name for _ in range(5)} == {"b.png"}

    def test_seeded_is_one_file_per_group_for_good_and_another_seed_picks_anew(self, grouped: Path) -> None:
        def picked(seed: int) -> List[str]:
            return [Path(r["file"]).name for r in GroupedFilesSource(root=str(grouped), pick="seeded", seed=seed)]

        assert picked(0) == picked(0)
        assert len({tuple(picked(seed)) for seed in range(20)}) > 1
        source = GroupedFilesSource(root=str(grouped), pick="seeded", seed=3)
        assert len({source[1]["file"] for _ in range(5)}) == 1

    def test_groups_inside_class_folders_are_found_by_the_pattern(self, tmp_path: Path) -> None:
        _files(tmp_path, "cat/1.png", "cat/group_0001/2.png", "cat/group_0001/3.png", "dog/group_0001/4.png")
        source = GroupedFilesSource(root=str(tmp_path), pattern="**/*.png", pick="first")
        assert [Path(r["file"]).relative_to(tmp_path).as_posix() for r in source] == [
            "cat/1.png",
            "cat/group_0001/2.png",
            "dog/group_0001/4.png",
        ]

    def test_what_the_sink_wrote_is_read_back_as_one_record_per_group(self, tmp_path: Path) -> None:
        files = _files(tmp_path / "in", "1.png", "2.png", "3.png", "4.png")
        sink = GroupedFilesSink(path=str(tmp_path / "out"))
        for path, group in zip(files, (1, 1, None, 2)):
            sink.write({"file": str(path), "group": group})
        assert len(GroupedFilesSource(root=str(tmp_path / "out"))) == 3

    def test_a_folder_that_is_missing_or_holds_no_match_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is not a directory"):
            len(GroupedFilesSource(root=str(tmp_path / "nowhere")))
        with pytest.raises(ValueError, match="matched no files"):
            len(GroupedFilesSource(root=str(tmp_path), pattern="*.png"))

    def test_an_unknown_pick_is_refused_when_built(self) -> None:
        with pytest.raises(ValueError, match="'next', 'first' or 'seeded'"):
            GroupedFilesSource(pick="last")  # type: ignore[arg-type]
        assert PICKS == ("next", "first", "seeded")

    def test_it_is_a_source_node_that_builds_without_arguments_and_declares_its_entries(self) -> None:
        assert getattr(GroupedFilesSource, "__confluid_category__") == "source"
        assert GroupedFilesSource().produces == ("file", "group_size")


def _read_all(source: Any, queue: Any) -> None:
    queue.put([Path(record["file"]).name for record in source])


class TestTheCountIsSharedByEveryWorker:
    def test_a_worker_started_with_the_source_goes_on_where_the_others_are(self, grouped: Path) -> None:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        source = GroupedFilesSource(root=str(grouped))
        seen: Dict[int, str] = {}
        for epoch in range(3):  # a new worker per epoch, as a loader that restarts its workers
            queue = context.Queue()
            worker = context.Process(target=_read_all, args=(source, queue))
            worker.start()
            seen[epoch] = queue.get(timeout=60)[1]
            worker.join(timeout=60)
        assert [seen[epoch] for epoch in range(3)] == ["b.png", "c.png", "d.png"]

    def test_a_copy_made_any_other_way_counts_on_its_own(self, grouped: Path) -> None:
        source = GroupedFilesSource(root=str(grouped))
        assert Path(source[1]["file"]).name == "b.png"
        copy = pickle.loads(pickle.dumps(source))
        assert Path(copy[1]["file"]).name == "b.png"
        assert Path(source[1]["file"]).name == "c.png"


class _Numbers:
    """An indexable source of ``{"n": index}`` records that counts its full reads."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.reads = 0

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self.reads += 1
        return {"n": index}


@pytest.fixture
def groups_file(tmp_path: Path) -> str:
    """Six records: 1, 3 and 4 stand for each other, 2 and 5 too, 0 is alone."""
    path = tmp_path / "groups.json"
    path.write_text(json.dumps({"records": 6, "groups": [[1, 3, 4], [2, 5]]}))
    return str(path)


class TestTheSinkWritesAGroupsFile:
    def test_it_writes_how_many_records_and_each_group_of_two_or_more(self, tmp_path: Path) -> None:
        sink = GroupsFileSink(path=str(tmp_path / "sub" / "groups.json"), group="duplicate_group")
        for group in (None, 7, 2, 7, 7, 2, 9):
            sink.write({"duplicate_group": group} if group is not None else {"other": 1})
        sink.flush()
        stored = json.loads((tmp_path / "sub" / "groups.json").read_text())
        assert stored == {"records": 7, "groups": [[1, 3, 4], [2, 5]]}  # the group of one (9) is no group

    def test_it_is_a_sink_node_that_builds_without_arguments_and_needs_a_path_to_write(self) -> None:
        assert getattr(GroupsFileSink, "__confluid_category__") == "sink"
        with pytest.raises(ValueError, match="`path` is empty"):
            GroupsFileSink().flush()


class TestAGroupOfRecordsIsOneRecord:
    def test_the_view_serves_one_record_per_group_in_the_order_of_first_indices(self, groups_file: str) -> None:
        view = GroupedSource(source=_Numbers(6), groups=groups_file, pick="first")
        assert len(view) == 3
        assert [record["n"] for record in view] == [0, 1, 2]

    def test_next_takes_another_record_of_the_group_at_every_read(self, groups_file: str) -> None:
        view = GroupedSource(source=_Numbers(6), groups=groups_file)
        epochs = [[record["n"] for record in view] for _ in range(4)]
        assert epochs == [[0, 1, 2], [0, 3, 5], [0, 4, 2], [0, 1, 5]]

    def test_seeded_is_one_record_per_group_for_good(self, groups_file: str) -> None:
        def picked(seed: int) -> List[int]:
            return [r["n"] for r in GroupedSource(source=_Numbers(6), groups=groups_file, pick="seeded", seed=seed)]

        assert picked(4) == picked(4)
        assert len({tuple(picked(seed)) for seed in range(20)}) > 1

    def test_a_key_restricted_walk_answers_with_each_groups_first_record_and_counts_nothing(
        self, groups_file: str
    ) -> None:
        from recordstream.projection import project

        view = GroupedSource(source=_Numbers(6), groups=groups_file)
        assert [record["n"] for record in project(view, ["n"])] == [0, 1, 2]
        assert [record["n"] for record in view] == [0, 1, 2]  # the walk moved no group on

    def test_what_the_sink_wrote_is_read_back_over_the_same_source(self, tmp_path: Path) -> None:
        sink = GroupsFileSink(path=str(tmp_path / "groups.json"))
        for group in (1, None, 1, 2, 2):
            sink.write({"group": group})
        sink.flush()
        view = GroupedSource(source=_Numbers(5), groups=str(tmp_path / "groups.json"), pick="first")
        assert [record["n"] for record in view] == [0, 1, 3]

    def test_a_source_of_another_length_is_refused(self, groups_file: str) -> None:
        with pytest.raises(ValueError, match="written for 6 records and the source has 7"):
            len(GroupedSource(source=_Numbers(7), groups=groups_file))

    def test_a_missing_file_a_record_named_twice_and_a_source_without_an_index_are_refused(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            len(GroupedSource(source=_Numbers(2), groups=str(tmp_path / "none.json")))
        twice = tmp_path / "twice.json"
        twice.write_text(json.dumps({"records": 3, "groups": [[0, 1], [1, 2]]}))
        with pytest.raises(ValueError, match="names record 1 twice"):
            len(GroupedSource(source=_Numbers(3), groups=str(twice)))
        with pytest.raises(TypeError, match="__len__ and __getitem__"):
            len(GroupedSource(source=iter(()), groups=str(twice)))

    def test_a_worker_started_with_the_view_goes_on_where_the_others_are(self, groups_file: str) -> None:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        view = GroupedSource(source=_Numbers(6), groups=groups_file)
        seen = []
        for _ in range(3):
            queue = context.Queue()
            worker = context.Process(target=_read_numbers, args=(view, queue))
            worker.start()
            seen.append(queue.get(timeout=60)[1])
            worker.join(timeout=60)
        assert seen == [1, 3, 4]

    def test_it_is_a_source_node_that_builds_without_arguments(self) -> None:
        assert getattr(GroupedSource, "__confluid_category__") == "source"
        GroupedSource()


def _read_numbers(view: Any, queue: Any) -> None:
    queue.put([record["n"] for record in view])
