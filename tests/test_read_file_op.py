"""``ReadFile`` — the registry-dispatching file decoder.

Per registered format, in registry order: ``consumes(path)`` → ``None`` (the paired
format's companion half — a drop surface's "consumed" verdict), else ``matches(path)``
→ ``{**record, **fmt.read(path, mmap=self.mmap)}`` (a raise rides out as that file's
refusal); a file NO format knows is REFUSED naming the file and the installed formats —
the measured alternative was a broken ``{"file": …}`` row with no error anywhere. A
record without the ``field`` key passes through untouched (op composability), and
``formats=None`` resolves the registry lazily so construction stays zero-arg.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from recordstream.ops import ReadFile


class _Fake:
    """A scripted format: fixed answers, recorded calls."""

    def __init__(self, name: str, matches: bool = False, consumes: bool = False) -> None:
        self.name = name
        self._matches = matches
        self._consumes = consumes
        self.read_calls: List[Tuple[Path, bool]] = []

    def matches(self, path: Path) -> bool:
        return self._matches

    def consumes(self, path: Path) -> bool:
        return self._consumes

    def read(self, path: Path, *, mmap: bool = True) -> Dict[str, Any]:
        self.read_calls.append((path, mmap))
        return {"decoded_by": self.name}


class TestDispatch:
    def test_consumes_wins_within_a_format(self, tmp_path: Path) -> None:
        fmt = _Fake("both", matches=True, consumes=True)
        assert ReadFile(formats=[fmt])({"file": str(tmp_path / "x.any")}) is None
        assert fmt.read_calls == []

    def test_the_first_matching_format_reads_and_later_ones_never_run(self, tmp_path: Path) -> None:
        first = _Fake("first", matches=True)
        second = _Fake("second", matches=True)
        out = ReadFile(formats=[first, second])({"file": str(tmp_path / "x.any")})
        assert out is not None and out["decoded_by"] == "first"
        assert second.read_calls == []

    def test_an_earlier_match_beats_a_later_consume(self, tmp_path: Path) -> None:
        """Registry ORDER is the tie-break for overlapping claims — the loop is
        per-format, not consumes-across-all-then-matches."""
        matcher = _Fake("matcher", matches=True)
        consumer = _Fake("consumer", consumes=True)
        out = ReadFile(formats=[matcher, consumer])({"file": str(tmp_path / "x.any")})
        assert out is not None and out["decoded_by"] == "matcher"

    def test_the_decoded_entries_merge_over_the_dropped_record(self, tmp_path: Path) -> None:
        fmt = _Fake("fake", matches=True)
        out = ReadFile(formats=[fmt])({"file": str(tmp_path / "x.any"), "kept": 7})
        assert out is not None and out["kept"] == 7 and out["decoded_by"] == "fake"

    def test_the_mmap_knob_is_forwarded_to_the_format(self, tmp_path: Path) -> None:
        fmt = _Fake("fake", matches=True)
        ReadFile(formats=[fmt], mmap=False)({"file": str(tmp_path / "x.any")})
        ReadFile(formats=[fmt])({"file": str(tmp_path / "x.any")})
        assert [mmap for _, mmap in fmt.read_calls] == [False, True]

    def test_a_read_error_propagates_for_the_refused_verdict(self, tmp_path: Path) -> None:
        class _Broken(_Fake):
            def read(self, path: Path, *, mmap: bool = True) -> Dict[str, Any]:
                raise ValueError(f"{path.name}: truncated payload")

        with pytest.raises(ValueError, match="truncated payload"):
            ReadFile(formats=[_Broken("broken", matches=True)])({"file": str(tmp_path / "x.any")})

    def test_a_record_without_the_field_passes_through(self) -> None:
        record: Dict[str, Any] = {"other": 1}
        assert ReadFile(formats=[_Fake("fake", matches=True)])(record) == record


class TestUnknownFilesAreRefused:
    def test_the_refusal_names_the_file_and_the_installed_formats(self, tmp_path: Path) -> None:
        fmt = _Fake("some-format")
        with pytest.raises(ValueError) as caught:
            ReadFile(formats=[fmt])({"file": str(tmp_path / "notes.txt")})
        message = str(caught.value)
        assert "notes.txt" in message and "some-format" in message

    def test_an_empty_format_list_still_refuses_readably(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="notes.txt"):
            ReadFile(formats=[])({"file": str(tmp_path / "notes.txt")})


class TestDiscoveryPlumbing:
    def test_the_op_is_a_categorised_canvas_node(self) -> None:
        assert getattr(ReadFile, "__confluid_category__") == "op"
        assert getattr(ReadFile, "__confluid_group__") == "formats"

    def test_zero_arg_construction_touches_no_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import recordstream.formats as formats_module

        def exploding_entries() -> list:
            raise AssertionError("the constructor must not scan entry points")

        monkeypatch.setattr(formats_module, "_FORMATS", None)
        monkeypatch.setattr(formats_module, "_iter_entries", exploding_entries)
        op = ReadFile()
        assert op.field == "file" and op.mmap is True
