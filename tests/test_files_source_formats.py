"""``FilesSource`` and the file-format registry — the pair-aware listing.

The listing rule: a paired format's companion half is excluded at the SOURCE so
``len()``/ids count one record per pair, without reading anything — decided by the
registry's ``consumes()`` (name + sibling stat), because a static exclude glob cannot
say "exclude the data file only when its sidecar exists" (a self-describing lone half
must stay listed). With no format packages installed the listing is the plain file
list, byte-identical to the pre-registry behaviour.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from recordstream.formats import sibling
from recordstream.sources.files import FilesSource


class _PairedFormat:
    """A fake paired format: ``.prime`` is primary, ``.shadow`` is its companion."""

    name = "paired"

    def matches(self, path: Path) -> bool:
        return path.name.endswith(".prime")

    def consumes(self, path: Path) -> bool:
        return path.name.endswith(".shadow") and sibling(path, ".shadow", ".prime") is not None

    def read(self, path: Path, *, mmap: bool = True) -> Dict[str, Any]:
        raise AssertionError("the LISTING must never read a file")


class TestThePairAwareListing:
    def test_the_companion_half_of_a_pair_is_not_served(self, tmp_path: Path) -> None:
        (tmp_path / "a.prime").write_bytes(b"\x00")
        (tmp_path / "a.shadow").write_bytes(b"\x00")
        source = FilesSource(files=[str(tmp_path / "a.prime"), str(tmp_path / "a.shadow")], formats=[_PairedFormat()])
        assert len(source) == 1
        assert source[0] == {"file": str(tmp_path / "a.prime")}

    def test_a_lone_companion_stays_listed(self, tmp_path: Path) -> None:
        """``consumes`` gates only the companion of a PRESENT pair — whether a lone
        half is readable is the FORMAT's decision at read time, so the listing must
        keep it (a self-describing lone half decodes; an undecodable one refuses with
        a located message instead of silently vanishing)."""
        (tmp_path / "b.shadow").write_bytes(b"\x00")
        source = FilesSource(files=[str(tmp_path / "b.shadow")], formats=[_PairedFormat()])
        assert len(source) == 1

    def test_listing_never_reads_a_file(self, tmp_path: Path) -> None:
        (tmp_path / "a.prime").write_bytes(b"\x00")
        (tmp_path / "a.shadow").write_bytes(b"\x00")
        source = FilesSource(files=[str(tmp_path / "a.prime"), str(tmp_path / "a.shadow")], formats=[_PairedFormat()])
        # _PairedFormat.read raises AssertionError — len/getitem/iter must not trip it.
        assert list(source) == [{"file": str(tmp_path / "a.prime")}]

    def test_the_manual_exclude_glob_still_composes(self, tmp_path: Path) -> None:
        (tmp_path / "keep.prime").write_bytes(b"\x00")
        (tmp_path / "noise.tmp").write_bytes(b"\x00")
        source = FilesSource(
            files=[str(tmp_path / "keep.prime"), str(tmp_path / "noise.tmp")],
            exclude="*.tmp",
            formats=[_PairedFormat()],
        )
        assert [source[i]["file"] for i in range(len(source))] == [str(tmp_path / "keep.prime")]

    def test_an_empty_format_list_is_the_plain_listing(self, tmp_path: Path) -> None:
        (tmp_path / "a.prime").write_bytes(b"\x00")
        (tmp_path / "a.shadow").write_bytes(b"\x00")
        files = [str(tmp_path / "a.prime"), str(tmp_path / "a.shadow")]
        assert len(FilesSource(files=files, formats=[])) == 2

    def test_no_installed_formats_means_the_plain_listing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import recordstream.formats as formats_module

        monkeypatch.setattr(formats_module, "_FORMATS", None)
        monkeypatch.setattr(formats_module, "_iter_entries", lambda: [])
        (tmp_path / "a.prime").write_bytes(b"\x00")
        (tmp_path / "a.shadow").write_bytes(b"\x00")
        files = [str(tmp_path / "a.prime"), str(tmp_path / "a.shadow")]
        assert len(FilesSource(files=files)) == 2

    def test_zero_arg_construction_touches_no_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import recordstream.formats as formats_module

        def exploding_entries() -> List[Any]:
            raise AssertionError("the constructor must not scan entry points")

        monkeypatch.setattr(formats_module, "_FORMATS", None)
        monkeypatch.setattr(formats_module, "_iter_entries", exploding_entries)
        source = FilesSource()
        assert source.files == [] and source.name == "files"
