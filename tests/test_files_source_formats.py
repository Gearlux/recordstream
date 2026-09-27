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

    def __init__(self) -> None:
        self.consumes_calls = 0

    def matches(self, path: Path) -> bool:
        return path.name.endswith(".prime")

    def consumes(self, path: Path) -> bool:
        self.consumes_calls += 1
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


class TestTheServedListingIsResolvedOncePerFileList:
    """Deciding what a listing SERVES costs a `consumes` call per file per format, and those
    calls stat the filesystem (a paired format asks whether a sibling exists). Re-deciding it
    on every index access therefore makes indexing O(n) with I/O and iterating O(n²) — measured
    on a real 1430-file library, **13.4 ms per `source[i]`**, so a walk that touched each file
    once spent 19 seconds inside the listing it had already computed.

    The resolved list is memoised against the file list itself, which is what keeps it correct:
    a consumer FILLS `files` in place (a drop adds to the listing it is looking at), and the
    answer must follow that without anyone having to invalidate anything.
    """

    def test_indexing_does_not_re_decide_the_listing_every_time(self, tmp_path: Path) -> None:
        fmt = _PairedFormat()
        for stem in "abcde":
            (tmp_path / f"{stem}.prime").touch()
            (tmp_path / f"{stem}.shadow").touch()
        source = FilesSource(files=sorted(str(p) for p in tmp_path.iterdir()), formats=[fmt])

        len(source)
        settled = fmt.consumes_calls
        for index in range(len(source)):
            source[index]

        assert fmt.consumes_calls == settled, "the listing was re-decided per access"

    def test_a_file_list_FILLED_afterwards_is_still_served(self, tmp_path: Path) -> None:
        """The reason the memo keys on the list and not on a flag: a drop appends to the very
        list the source is serving, and nothing calls an invalidate."""
        (tmp_path / "one.prime").touch()
        (tmp_path / "two.prime").touch()
        source = FilesSource(files=[str(tmp_path / "one.prime")], formats=[_PairedFormat()])
        assert len(source) == 1

        source.files.append(str(tmp_path / "two.prime"))

        assert len(source) == 2
        assert source[1] == {"file": str(tmp_path / "two.prime")}

    def test_a_changed_exclude_is_still_honoured(self, tmp_path: Path) -> None:
        (tmp_path / "keep.prime").touch()
        (tmp_path / "drop.prime").touch()
        source = FilesSource(files=sorted(str(p) for p in tmp_path.iterdir()), formats=[])
        assert len(source) == 2

        source.exclude = "drop.*"

        assert len(source) == 1

    def test_a_ROOT_source_still_sees_a_file_that_appears_afterwards(self, tmp_path: Path) -> None:
        """The hazard the memo's key is chosen to avoid. `_listed` is documented to scan the
        folder NOW, and a root source's `files` is EMPTY — so keying the memo on the file list
        would freeze the listing at its first access and a file dropped into the folder would
        never appear. Keying on what was actually listed keeps the scan as live as it was."""
        (tmp_path / "first.prime").touch()
        source = FilesSource(root=str(tmp_path), formats=[_PairedFormat()])
        assert len(source) == 1

        (tmp_path / "second.prime").touch()

        assert len(source) == 2
        assert source[1] == {"file": str(tmp_path / "second.prime")}

    def test_a_ROOT_source_still_stops_re_deciding_what_it_serves(self, tmp_path: Path) -> None:
        """And it still gets the fix: the glob runs per access as it always did, but the
        `consumes` storm behind it does not."""
        fmt = _PairedFormat()
        for stem in "abcde":
            (tmp_path / f"{stem}.prime").touch()
            (tmp_path / f"{stem}.shadow").touch()
        source = FilesSource(root=str(tmp_path), formats=[fmt])

        len(source)
        settled = fmt.consumes_calls
        for index in range(len(source)):
            source[index]

        assert fmt.consumes_calls == settled
