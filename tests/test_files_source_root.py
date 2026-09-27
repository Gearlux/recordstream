"""``FilesSource(root=…, pattern=…)`` — listing a FOLDER instead of passing every path.

The list form serves "someone handed me files"; the folder form serves "review this
directory tree", and the glob pattern IS the scope control: ``*`` a folder's own files,
``*/*`` its immediate subdirectories, ``**/*`` everything below, ``one_dir/*`` a single
named subdirectory. There is deliberately no second knob meaning the same thing.
"""

from pathlib import Path
from typing import List

import pytest

from recordstream.sources.files import FilesSource


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """root/{a,b}/*.dat plus a stray file and a dotfile at the top, as a real capture tree has."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for name in ("one.dat", "two.dat"):
        (tmp_path / "a" / name).write_text(name)
    (tmp_path / "b" / "three.dat").write_text("three")
    (tmp_path / "b" / "notes.txt").write_text("notes")
    (tmp_path / "top.dat").write_text("top")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / "b" / ".hidden").write_text("hidden")
    return tmp_path


def _names(source: FilesSource) -> List[str]:
    return [Path(source[i]["file"]).name for i in range(len(source))]


class TestTheFolderForm:
    def test_the_default_pattern_lists_the_folder_s_own_files(self, tree: Path) -> None:
        assert _names(FilesSource(root=str(tree), formats=[])) == ["top.dat"]

    def test_one_level_down_lists_every_subdirectory(self, tree: Path) -> None:
        """Sorted by full path, so a/ comes before b/ and the order is stable across runs."""
        source = FilesSource(root=str(tree), pattern="*/*", formats=[])
        assert _names(source) == ["one.dat", "two.dat", "notes.txt", "three.dat"]

    def test_a_named_subdirectory_is_the_one_directory_at_a_time_scope(self, tree: Path) -> None:
        assert _names(FilesSource(root=str(tree), pattern="a/*", formats=[])) == ["one.dat", "two.dat"]

    def test_a_suffix_in_the_pattern_selects_the_file_kind(self, tree: Path) -> None:
        assert _names(FilesSource(root=str(tree), pattern="*/*.dat", formats=[])) == [
            "one.dat",
            "two.dat",
            "three.dat",
        ]

    def test_everything_below_the_root(self, tree: Path) -> None:
        assert len(FilesSource(root=str(tree), pattern="**/*.dat", formats=[])) == 4

    def test_dotfiles_are_never_listed(self, tree: Path) -> None:
        """A folder glob picks up .DS_Store, which is not data and fails to decode."""
        for pattern in ("*", "*/*", "**/*"):
            names = _names(FilesSource(root=str(tree), pattern=pattern, formats=[]))
            assert not any(n.startswith(".") for n in names), pattern

    def test_directories_matched_by_the_glob_are_not_records(self, tree: Path) -> None:
        assert _names(FilesSource(root=str(tree), pattern="*", formats=[])) == ["top.dat"]

    def test_exclude_still_applies_on_top_of_the_scan(self, tree: Path) -> None:
        source = FilesSource(root=str(tree), pattern="*/*", exclude="*.txt", formats=[])
        assert "notes.txt" not in _names(source)

    def test_the_listing_is_sorted_so_reruns_agree(self, tree: Path) -> None:
        source = FilesSource(root=str(tree), pattern="**/*.dat", formats=[])
        assert [Path(source[i]["file"]).as_posix() for i in range(len(source))] == sorted(
            Path(source[i]["file"]).as_posix() for i in range(len(source))
        )


class TestUserAndEnvironmentPaths:
    def test_an_environment_variable_in_the_root_is_expanded(self, tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_DATA_ROOT", str(tree))
        assert _names(FilesSource(root="$TEST_DATA_ROOT", pattern="a/*", formats=[])) == [
            "one.dat",
            "two.dat",
        ]


class TestWhatItRefuses:
    def test_files_and_root_together_are_refused_at_construction(self, tree: Path) -> None:
        with pytest.raises(ValueError, match="not both"):
            FilesSource(files=[str(tree / "top.dat")], root=str(tree))

    def test_a_root_that_is_not_a_directory_says_so(self, tree: Path) -> None:
        source = FilesSource(root=str(tree / "top.dat"), formats=[])
        with pytest.raises(ValueError, match="is not a directory"):
            len(source)

    def test_a_pattern_matching_nothing_names_the_folder_and_the_pattern(self, tree: Path) -> None:
        """A silent empty listing is the failure that looks like a broken pipeline three ops later."""
        source = FilesSource(root=str(tree), pattern="*.nope", formats=[])
        with pytest.raises(ValueError, match=r"matched no files under"):
            len(source)


class TestTheListFormIsUnchanged:
    def test_an_explicit_list_still_works(self, tree: Path) -> None:
        source = FilesSource(files=[str(tree / "top.dat")], formats=[])
        assert _names(source) == ["top.dat"]

    def test_no_root_and_no_files_is_an_empty_source_as_before(self) -> None:
        assert len(FilesSource(formats=[])) == 0

    def test_the_constructor_does_no_io(self, tmp_path: Path) -> None:
        """Zero-arg / lazy: a root that does not exist is only reported on first use."""
        source = FilesSource(root=str(tmp_path / "not-here-yet"), formats=[])
        with pytest.raises(ValueError, match="is not a directory"):
            len(source)


class TestAFileListAssignedAfterConstruction:
    """A consuming workspace may fill ``files`` on a LIVE source — annotaide does exactly
    this to hand a drag-and-drop to the graph's file source. The constructor never sees it,
    so the conflict has to be caught at read time or the drop silently does nothing."""

    def test_assigning_files_to_a_folder_source_is_refused_at_read_time(self, tree: Path) -> None:
        source = FilesSource(root=str(tree), formats=[])
        assert len(source) == 1  # the folder listing, before anything is assigned
        source.files = [str(tree / "a" / "one.dat")]  # what _fill_files_socket does
        with pytest.raises(ValueError, match="cannot also be given a file list"):
            len(source)

    def test_the_message_names_both_halves_of_the_conflict(self, tree: Path) -> None:
        source = FilesSource(root=str(tree), formats=[])
        source.files = [str(tree / "a" / "one.dat"), str(tree / "a" / "two.dat")]
        with pytest.raises(ValueError, match=r"root=.*2 file\(s\) were assigned"):
            len(source)

    def test_clearing_the_root_lets_the_assigned_list_serve(self, tree: Path) -> None:
        source = FilesSource(root=str(tree), formats=[])
        source.files = [str(tree / "a" / "one.dat")]
        source.root = ""
        assert _names(source) == ["one.dat"]
