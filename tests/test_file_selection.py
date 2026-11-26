"""``FileSelection`` — the files of several folders ("sets") as records, limited the same way wherever they are read.

The one place a run says WHICH files of its datasets it reads: the globs, the files left out, how deep it looks, how
many folders of each set and how many files of each folder, and how many in all.
"""

import os
from pathlib import Path
from typing import Any, List

import pytest

from recordstream.sources import FileSelection


def _tree(root: Path, *files: str) -> Path:
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    return root


def _names(selection: FileSelection) -> List[str]:
    return [record["relative"] for record in selection]


@pytest.fixture
def two_sets(tmp_path: Path) -> List[str]:
    a = _tree(
        tmp_path / "data" / "A" / "1.0", "f1/x.json", "f1/y.json", "f1/z.json", "f2/x.json", "f3/x.json", "top.json"
    )
    b = _tree(tmp_path / "data" / "B" / "2.0", "g1/x.json", "g1/y.json", "g2/deep/x.json")
    return [str(a), str(b)]


class TestWhatIsListed:
    def test_every_file_of_each_set_in_turn_each_in_name_order(self, two_sets: List[str]) -> None:
        selection = FileSelection(sets=two_sets, pattern="**/*.json")
        assert _names(selection) == [
            "1.0/f1/x",
            "1.0/f1/y",
            "1.0/f1/z",
            "1.0/f2/x",
            "1.0/f3/x",
            "1.0/top",
            "2.0/g1/x",
            "2.0/g1/y",
            "2.0/g2/deep/x",
        ]
        assert len(selection) == 9
        assert selection[0] == {
            "file": str(Path(two_sets[0]) / "f1" / "x.json"),
            "set": two_sets[0],
            "relative": "1.0/f1/x",
        }

    def test_the_relative_name_starts_where_relative_to_says(self, two_sets: List[str], tmp_path: Path) -> None:
        selection = FileSelection(sets=two_sets[1:], pattern="**/*.json", relative_to=str(tmp_path / "data"))
        assert _names(selection) == ["B/2.0/g1/x", "B/2.0/g1/y", "B/2.0/g2/deep/x"]

    def test_several_globs_are_one_listing_without_a_file_twice(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "s", "a.json", "a.mat", "b.mat", "c.png")
        selection = FileSelection(sets=[str(root)], pattern=["*.json", "*.mat", "a.*"], formats=[])
        assert [Path(r["file"]).name for r in selection] == ["a.json", "a.mat", "b.mat"]

    def test_a_glob_without_a_slash_leaves_out_by_name_and_one_with_by_place(self, two_sets: List[str]) -> None:
        assert _names(FileSelection(sets=two_sets[:1], pattern="**/*.json", exclude="y.*")) == [
            "1.0/f1/x",
            "1.0/f1/z",
            "1.0/f2/x",
            "1.0/f3/x",
            "1.0/top",
        ]
        assert _names(FileSelection(sets=two_sets[:1], pattern="**/*.json", exclude="f1/*")) == [
            "1.0/f2/x",
            "1.0/f3/x",
            "1.0/top",
        ]

    def test_hidden_files_and_everything_in_a_hidden_folder_are_never_listed(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "s", "a.json", ".DS_Store", ".sync/manifest.json", "f/.hidden.json")
        assert [Path(r["file"]).name for r in FileSelection(sets=[str(root)], formats=[])] == ["a.json"]

    def test_a_companion_a_format_consumes_is_left_out(self, tmp_path: Path) -> None:
        class Pairs:
            name = "pairs"

            def matches(self, path: Path) -> bool:
                return path.suffix in (".meta", ".data")

            def consumes(self, path: Path) -> bool:
                return path.suffix == ".data"

            def read(self, path: Path, *, mmap: bool = True) -> Any:
                raise AssertionError("the listing never reads a file")

        root = _tree(tmp_path / "s", "a.meta", "a.data", "b.meta", "b.data")
        selection = FileSelection(sets=[str(root)], formats=[Pairs()], max_files_per_folder=2)
        assert [Path(r["file"]).name for r in selection] == ["a.meta", "b.meta"], "a pair counts once"


class TestTheLimits:
    @pytest.mark.parametrize(
        "depth, expected",
        [
            (0, ["1.0/top"]),
            (1, ["1.0/f1/x", "1.0/f1/y", "1.0/f1/z", "1.0/f2/x", "1.0/f3/x", "1.0/top", "2.0/g1/x", "2.0/g1/y"]),
            (
                None,
                [
                    "1.0/f1/x",
                    "1.0/f1/y",
                    "1.0/f1/z",
                    "1.0/f2/x",
                    "1.0/f3/x",
                    "1.0/top",
                    "2.0/g1/x",
                    "2.0/g1/y",
                    "2.0/g2/deep/x",
                ],
            ),
        ],
    )
    def test_max_depth_is_the_most_folders_between_the_set_and_a_file(
        self, two_sets: List[str], depth: Any, expected: List[str]
    ) -> None:
        assert _names(FileSelection(sets=two_sets, pattern="**/*.json", max_depth=depth)) == expected

    def test_max_folders_per_set_keeps_each_sets_first_folders_in_name_order(self, two_sets: List[str]) -> None:
        """The set's own folder is one of its folders, first in name order; a set without files of its own starts at
        its first folder that has some."""
        selection = FileSelection(sets=two_sets, pattern="**/*.json", max_folders_per_set=2)
        assert _names(selection) == [
            "1.0/f1/x",
            "1.0/f1/y",
            "1.0/f1/z",
            "1.0/top",
            "2.0/g1/x",
            "2.0/g1/y",
            "2.0/g2/deep/x",
        ]
        selection = FileSelection(sets=two_sets, pattern="**/*.json", exclude="top.json", max_folders_per_set=2)
        assert _names(selection) == [
            "1.0/f1/x",
            "1.0/f1/y",
            "1.0/f1/z",
            "1.0/f2/x",
            "2.0/g1/x",
            "2.0/g1/y",
            "2.0/g2/deep/x",
        ]

    def test_max_subfolders_per_folder_keeps_each_folders_first_subfolders_at_every_level(
        self, two_sets: List[str]
    ) -> None:
        """A folder's own files are always kept; of its subfolders holding a listed file below them, the first in name
        order — and the same again inside each one kept."""
        selection = FileSelection(sets=two_sets, pattern="**/*.json", max_subfolders_per_folder=2)
        assert _names(selection) == [
            "1.0/f1/x",
            "1.0/f1/y",
            "1.0/f1/z",
            "1.0/f2/x",
            "1.0/top",
            "2.0/g1/x",
            "2.0/g1/y",
            "2.0/g2/deep/x",
        ]
        selection = FileSelection(sets=two_sets, pattern="**/*.json", max_subfolders_per_folder=1)
        assert _names(selection) == ["1.0/f1/x", "1.0/f1/y", "1.0/f1/z", "1.0/top", "2.0/g1/x", "2.0/g1/y"]

    def test_a_subfolder_holding_no_listed_file_takes_no_place(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "set", "a/notes.txt", "b/x.json", "c/x.json", "d/x.json")
        selection = FileSelection(sets=[str(root)], pattern="**/*.json", max_subfolders_per_folder=2)
        assert _names(selection) == ["set/b/x", "set/c/x"]

    def test_max_subfolders_per_folder_reaches_every_branch_where_max_folders_per_set_stops_at_the_first(
        self, tmp_path: Path
    ) -> None:
        """A dataset of three families, each a few folders: the folder limit of the set takes its first five folders,
        all of the first two families; the subfolder limit takes folders of every family."""
        root = _tree(
            tmp_path / "1.3",
            "BTE_training_data/BTE/a.json",
            "BTE_training_data/BTE_2/a.json",
            "BTE_training_data/BTE_Long/Bluetooth/a.json",
            "LTE_training_data/LTE/LTE/a.json",
            "LTE_training_data/LTE_MIMO_Dataset/a.json",
            "LTE_training_data/LTE_MIMO_Dataset_2/a.json",
            "WLAN_training_data/WLAN/a.json",
            "WLAN_training_data/WLAN_2/a.json",
        )
        per_set = FileSelection(sets=[str(root)], pattern="**/*.json", max_folders_per_set=5)
        assert [name.split("/")[1] for name in _names(per_set)] == ["BTE_training_data"] * 3 + ["LTE_training_data"] * 2
        per_folder = FileSelection(sets=[str(root)], pattern="**/*.json", max_subfolders_per_folder=2)
        assert _names(per_folder) == [
            "1.3/BTE_training_data/BTE/a",
            "1.3/BTE_training_data/BTE_2/a",
            "1.3/LTE_training_data/LTE/LTE/a",
            "1.3/LTE_training_data/LTE_MIMO_Dataset/a",
        ], "the set's own folder keeps its first two subfolders too"
        per_folder = FileSelection(sets=[str(root)], pattern="**/*.json", max_subfolders_per_folder=3)
        assert [name.split("/")[1] for name in _names(per_folder)] == [
            "BTE_training_data",
            "BTE_training_data",
            "BTE_training_data",
            "LTE_training_data",
            "LTE_training_data",
            "LTE_training_data",
            "WLAN_training_data",
            "WLAN_training_data",
        ]

    def test_max_files_per_folder_keeps_each_folders_first_files(self, two_sets: List[str]) -> None:
        selection = FileSelection(sets=two_sets, pattern="**/*.json", max_files_per_folder=1)
        assert _names(selection) == ["1.0/f1/x", "1.0/f2/x", "1.0/f3/x", "1.0/top", "2.0/g1/x", "2.0/g2/deep/x"]

    def test_max_files_keeps_the_first_files_of_the_whole_selection(self, two_sets: List[str]) -> None:
        assert _names(FileSelection(sets=two_sets, pattern="**/*.json", max_files=4)) == [
            "1.0/f1/x",
            "1.0/f1/y",
            "1.0/f1/z",
            "1.0/f2/x",
        ]

    def test_the_limits_apply_subfolders_then_folders_then_files_per_folder_then_the_total(
        self, two_sets: List[str]
    ) -> None:
        # subfolders first: f1 and top's folder stay in the first set, so the folder limit of 2 then keeps both
        selection = FileSelection(
            sets=two_sets, pattern="**/*.json", max_subfolders_per_folder=1, max_folders_per_set=2
        )
        assert _names(selection) == ["1.0/f1/x", "1.0/f1/y", "1.0/f1/z", "1.0/top", "2.0/g1/x", "2.0/g1/y"]
        selection = FileSelection(
            sets=two_sets,
            pattern="**/*.json",
            exclude="top.json",
            max_folders_per_set=1,
            max_files_per_folder=2,
            max_files=3,
        )
        assert _names(selection) == ["1.0/f1/x", "1.0/f1/y", "2.0/g1/x"]

    def test_select_answers_each_file_with_its_set(self, two_sets: List[str]) -> None:
        pairs = FileSelection(sets=two_sets, pattern="**/*.json", max_files_per_folder=1, max_depth=1).select()
        assert [(file.relative_to(root).as_posix(), str(root)) for file, root in pairs] == [
            ("f1/x.json", two_sets[0]),
            ("f2/x.json", two_sets[0]),
            ("f3/x.json", two_sets[0]),
            ("top.json", two_sets[0]),
            ("g1/x.json", two_sets[1]),
        ]

    def test_select_reads_the_sets_it_is_given_instead(self, two_sets: List[str]) -> None:
        selection = FileSelection(sets=two_sets, pattern="**/*.json", max_files=1)
        assert [str(root) for _file, root in selection.select(sets=[two_sets[1]])] == [two_sets[1]]


class TestASet:
    def test_a_set_may_be_any_object_naming_its_folder_asked_only_when_listed(self, two_sets: List[str]) -> None:
        asked: List[str] = []

        class Folder:
            def __fspath__(self) -> str:
                asked.append("asked")
                return two_sets[1]

        selection = FileSelection(sets=[Folder()], pattern="**/*.json")
        assert asked == [], "the constructor does no I/O"
        assert _names(selection) == ["2.0/g1/x", "2.0/g1/y", "2.0/g2/deep/x"]
        assert selection[0]["set"] == two_sets[1]

    def test_a_set_folder_is_expanded(self, two_sets: List[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SETS_HERE", str(Path(two_sets[1]).parent))
        assert len(FileSelection(sets=["$SETS_HERE/2.0"], pattern="**/*.json")) == 3


class TestRefusals:
    def test_no_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"FileSelection: no sets — name the folders to read"):
            len(FileSelection())

    def test_an_empty_set_is_refused_never_read_as_the_current_folder(self) -> None:
        with pytest.raises(ValueError, match=r"FileSelection: a set is empty — name its folder"):
            len(FileSelection(sets=[""]))

    def test_a_set_that_is_no_folder_is_refused_naming_it(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"FileSelection: set '.*nowhere' is not a folder"):
            len(FileSelection(sets=[str(tmp_path / "nowhere")]))

    def test_a_selection_of_nothing_is_refused_naming_the_sets_and_globs(self, two_sets: List[str]) -> None:
        with pytest.raises(ValueError, match=r"FileSelection: \['\*\*/\*\.mat'\] matched no files under \["):
            len(FileSelection(sets=two_sets, pattern="**/*.mat"))

    @pytest.mark.parametrize(
        "setting",
        ["max_depth", "max_subfolders_per_folder", "max_folders_per_set", "max_files_per_folder", "max_files"],
    )
    def test_a_negative_limit_is_refused_by_its_range_mark(self, setting: str) -> None:
        with pytest.raises(ValueError, match=rf"{setting}\n  Input should be greater than or equal to 0"):
            FileSelection(**{setting: -1})  # type: ignore[arg-type]  # the value refused

    def test_it_builds_with_no_arguments_and_is_a_source(self) -> None:
        FileSelection()
        assert getattr(FileSelection, "__confluid_category__") == "source"


def test_the_listing_is_made_once_for_its_settings(two_sets: List[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """``source[i]`` per record must not list the sets again (a dataset has thousands of files)."""
    selection = FileSelection(sets=two_sets, pattern="**/*.json", formats=[])
    calls: List[str] = []
    original = Path.glob

    def counted(self: Path, pattern: str) -> Any:
        calls.append(pattern)
        return original(self, pattern)

    monkeypatch.setattr(Path, "glob", counted)
    [selection[i] for i in range(len(selection))]
    assert len(calls) == 2, "one glob a set"
    selection.max_files = 2
    assert len(selection) == 2 and len(calls) == 4, "a changed setting lists again"
    selection.max_files = 0
    selection.max_subfolders_per_folder = 1
    assert len(selection) == 6 and len(calls) == 6, "the subfolder limit is one of the settings the listing follows"


def test_a_path_is_given_back_as_the_operating_system_writes_it(two_sets: List[str]) -> None:
    file = FileSelection(sets=two_sets, pattern="**/*.json")[0]["file"]
    assert file == os.path.join(two_sets[0], "f1", "x.json")
