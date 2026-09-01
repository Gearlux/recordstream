"""``FilesSource`` delivers paths; ``ReadImage`` turns a path into pixels — two jobs, two nodes."""

from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image as PILImage

from recordstream.items import Image
from recordstream.ops.image import ReadImage
from recordstream.sources.files import FilesSource


def _png(path: Path, value: int) -> str:
    PILImage.fromarray(np.full((4, 5, 3), value, dtype=np.uint8)).save(path)
    return str(path)


@pytest.fixture()
def images(tmp_path: Path) -> List[str]:
    return [_png(tmp_path / f"img_{i}.png", value=10 * i) for i in range(3)]


class TestFilesSource:
    """The source knows NOTHING about file contents — each path is one record, verbatim."""

    def test_each_file_is_one_record_of_its_path(self, images: List[str]) -> None:
        source = FilesSource(files=images)
        assert len(source) == 3
        assert source[1] == {"file": images[1]}

    def test_the_order_is_the_files_order(self, images: List[str]) -> None:
        assert [r["file"] for r in FilesSource(files=list(reversed(images)))] == list(reversed(images))

    def test_a_non_image_file_is_a_record_too(self, tmp_path: Path) -> None:
        """The CON case of the old fused design: the source no longer judges readability —
        what a file MEANS is the graph's business (a ReadImage, a ReadAudio, …)."""
        stray = tmp_path / "notes.txt"
        stray.write_text("not an image")
        assert FilesSource(files=[str(stray)])[0] == {"file": str(stray)}

    def test_zero_arg_construction_is_the_legal_empty_source(self) -> None:
        assert len(FilesSource()) == 0 and list(FilesSource()) == []

    def test_the_list_can_be_grown_after_construction(self, images: List[str]) -> None:
        source = FilesSource(files=images[:1])
        source.files.append(images[2])
        assert len(source) == 2


class TestReadImage:
    """The op that reads the file: ``{file}`` in, ``{file, image}`` out."""

    def test_the_path_becomes_an_image_item(self, images: List[str]) -> None:
        out = ReadImage()({"file": images[1]})
        assert isinstance(out["image"], Image) and out["image"].shape == (4, 5, 3)
        assert int(out["image"][0, 0, 0]) == 10
        assert out["file"] == images[1], "the path stays as provenance"

    def test_a_grayscale_file_yields_three_channels(self, tmp_path: Path) -> None:
        path = tmp_path / "gray.png"
        PILImage.fromarray(np.full((4, 5), 7, dtype=np.uint8), mode="L").save(path)
        assert ReadImage()({"file": str(path)})["image"].shape == (4, 5, 3)

    def test_an_unreadable_file_passes_through_UNCHANGED(self, tmp_path: Path) -> None:
        """Never fatal, never dropped: the record keeps its row (a viewer shows it as
        unreadable) and one stray text file cannot cost the run around it."""
        stray = tmp_path / "notes.txt"
        stray.write_text("nope")
        out = ReadImage()({"file": str(stray)})
        assert out == {"file": str(stray)} and "image" not in out

    def test_a_missing_file_passes_through_too(self, tmp_path: Path) -> None:
        record = {"file": str(tmp_path / "gone.png")}
        assert ReadImage()(record) == record

    def test_a_record_without_the_field_passes_through(self, images: List[str]) -> None:
        record = {"note": "no file here"}
        assert ReadImage()(record) == record

    def test_field_and_output_are_knobs(self, images: List[str]) -> None:
        out = ReadImage(field="path", output="picture")({"path": images[0]})
        assert isinstance(out["picture"], Image)

    def test_composes_over_the_source(self, images: List[str], tmp_path: Path) -> None:
        stray = tmp_path / "notes.txt"
        stray.write_text("x")
        records = [ReadImage()(r) for r in FilesSource(files=[images[0], str(stray)])]
        assert "image" in records[0] and "image" not in records[1]

    def test_zero_arg_construction_works(self) -> None:
        assert ReadImage().field == "file"


class TestTheExcludePattern:
    def test_excluded_names_are_not_records(self) -> None:
        """A PAIRED format's companion file (SigMF's .sigmf-data) is not a record of its own —
        the pair's reader consumes it via its sibling. Filtering NAMES is free (no file reads),
        so the count and the ids stay honest without walking anything."""
        from recordstream.sources.files import FilesSource

        source = FilesSource(files=["a.sigmf-meta", "a.sigmf-data", "b.png"], exclude="*.sigmf-data")
        assert len(source) == 2
        assert [r["file"] for r in source] == ["a.sigmf-meta", "b.png"]
        assert source[1] == {"file": "b.png"}

    def test_no_pattern_serves_everything(self) -> None:
        from recordstream.sources.files import FilesSource

        assert len(FilesSource(files=["a", "b"])) == 2
