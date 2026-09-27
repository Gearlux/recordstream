"""The file-format registry — `recordstream.formats`.

The contract:

* ``FileFormat`` is a runtime-checkable Protocol: ``name``, ``matches(path)``
  (name-only), ``consumes(path)`` (name + sibling stat: "is this a paired format's
  companion half, consumed via its sibling?"), ``read(path, *, mmap=True) -> Record``.
* ``file_formats()`` scans the ``recordstream.formats`` entry-point group, each module
  contributing through a ``formats() -> Iterable[FileFormat]`` hook. The engine ships
  NO formats of its own — domain packages register theirs. A failing entry is a
  warning and a skip, NEVER fatal — one half-installed format package must not blank
  the whole drop path. The scan is cached in ``_FORMATS``; the entry iterator is the
  module-level ``_iter_entries`` so tests can stand entries in.
* ``sibling(path, own_tail, want_tail)`` is the ONE prefix-tolerant pairing helper:
  explicit name concatenation (never ``with_suffix``), a drop store's ``NNNN_``
  ordering prefix stripped, and several same-named copies resolved by RANK in drop
  order.
"""

import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import recordstream.formats as formats_module
from recordstream.formats import FileFormat, file_formats, sibling


class _FakeFormat:
    """A registry-shaped format for structural tests — never touches file contents."""

    def __init__(self, name: str = "fake", primary: str = ".prime", companion: str = ".shadow") -> None:
        self.name = name
        self.primary = primary
        self.companion = companion
        self.read_calls: List[Tuple[Path, bool]] = []

    def matches(self, path: Path) -> bool:
        return path.name.endswith(self.primary)

    def consumes(self, path: Path) -> bool:
        return path.name.endswith(self.companion) and sibling(path, self.companion, self.primary) is not None

    def read(self, path: Path, *, mmap: bool = True) -> Dict[str, Any]:
        self.read_calls.append((path, mmap))
        return {"decoded_by": self.name}


def _entry(name: str, payload: Any) -> Any:
    """An entry-point stand-in: ``.name`` + ``.load()`` returning the module (or raising)."""

    class _Entry:
        def __init__(self) -> None:
            self.name = name

        def load(self) -> Any:
            if isinstance(payload, Exception):
                raise payload
            return payload

    return _Entry()


def _module_with_formats(*fmts: Any) -> types.ModuleType:
    module = types.ModuleType("fake_formats_module")
    module.formats = lambda: fmts  # type: ignore[attr-defined]
    return module


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with an empty cache and NO installed entry points."""
    monkeypatch.setattr(formats_module, "_FORMATS", None)
    monkeypatch.setattr(formats_module, "_iter_entries", lambda: [])


class TestTheRegistry:
    def test_the_engine_ships_no_formats_of_its_own(self) -> None:
        """Modality-neutrality: which formats exist is domain knowledge — with no
        format packages installed the registry is empty and every listing is plain."""
        assert file_formats() == ()

    def test_entries_contribute_in_group_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = _module_with_formats(_FakeFormat(name="alpha"))
        second = _module_with_formats(_FakeFormat(name="beta"), _FakeFormat(name="gamma"))
        monkeypatch.setattr(formats_module, "_iter_entries", lambda: [_entry("a", first), _entry("b", second)])
        assert [fmt.name for fmt in file_formats()] == ["alpha", "beta", "gamma"]

    def test_a_broken_entry_is_skipped_and_the_rest_survive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        good = _module_with_formats(_FakeFormat(name="survivor"))
        monkeypatch.setattr(
            formats_module,
            "_iter_entries",
            lambda: [_entry("broken", ImportError("no such backend")), _entry("good", good)],
        )
        assert [fmt.name for fmt in file_formats()] == ["survivor"]

    def test_a_module_without_the_hook_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hookless = types.ModuleType("no_hook")
        good = _module_with_formats(_FakeFormat(name="survivor"))
        monkeypatch.setattr(
            formats_module, "_iter_entries", lambda: [_entry("no-hook", hookless), _entry("good", good)]
        )
        assert [fmt.name for fmt in file_formats()] == ["survivor"]

    def test_the_scan_runs_once_and_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def counting_entries() -> List[Any]:
            calls["n"] += 1
            return []

        monkeypatch.setattr(formats_module, "_iter_entries", counting_entries)
        file_formats()
        file_formats()
        assert calls["n"] == 1

    def test_a_registry_shaped_object_satisfies_the_protocol(self) -> None:
        assert isinstance(_FakeFormat(), FileFormat)


class TestSiblingPairing:
    """The shared pairing helper. A drop store numbers files as they were dropped
    (``NNNN_`` prefixes), so the k-th own-tail copy of a name pairs the k-th want-tail
    copy."""

    def test_an_exact_sibling_wins(self, tmp_path: Path) -> None:
        (tmp_path / "capture.meta").write_text("{}")
        (tmp_path / "capture.data").write_bytes(b"\x00")
        assert sibling(tmp_path / "capture.meta", ".meta", ".data") == tmp_path / "capture.data"

    def test_a_prefixed_copy_still_pairs(self, tmp_path: Path) -> None:
        (tmp_path / "0001_capture.meta").write_text("{}")
        (tmp_path / "0002_capture.data").write_bytes(b"\x00")
        assert sibling(tmp_path / "0001_capture.meta", ".meta", ".data") == tmp_path / "0002_capture.data"

    def test_no_sibling_answers_none(self, tmp_path: Path) -> None:
        (tmp_path / "capture.meta").write_text("{}")
        assert sibling(tmp_path / "capture.meta", ".meta", ".data") is None

    def test_tails_concatenate_so_a_dotted_stem_survives(self, tmp_path: Path) -> None:
        """Explicit name concatenation, never ``with_suffix`` — a stem containing a dot
        loses its tail under ``with_suffix``."""
        (tmp_path / "rec.v1.meta").write_text("{}")
        (tmp_path / "rec.v1.data").write_bytes(b"\x00")
        assert sibling(tmp_path / "rec.v1.meta", ".meta", ".data") == tmp_path / "rec.v1.data"

    def test_a_bare_tail_pairs_too(self, tmp_path: Path) -> None:
        """A tail need not start at a dot — ``<stem>p0.bin`` beside ``<stem>.scp`` is a
        real paired-format convention."""
        (tmp_path / "capture.scp").write_text("x")
        (tmp_path / "capturep0.bin").write_bytes(b"\x00")
        assert sibling(tmp_path / "capturep0.bin", "p0.bin", ".scp") == tmp_path / "capture.scp"

    def test_several_copies_resolve_by_rank_in_drop_order(self, tmp_path: Path) -> None:
        for name in ("0001_x.meta", "0009_x.meta", "0000_x.data", "0008_x.data"):
            (tmp_path / name).write_bytes(b"\x00")
        assert sibling(tmp_path / "0001_x.meta", ".meta", ".data") == tmp_path / "0000_x.data"
        assert sibling(tmp_path / "0009_x.meta", ".meta", ".data") == tmp_path / "0008_x.data"

    def test_a_back_to_back_re_drop_pairs_by_rank_not_distance(self, tmp_path: Path) -> None:
        """m,d,m,d interleave: the second meta is EQUIDISTANT to both datas — only rank
        answers."""
        for name in ("0000_x.meta", "0001_x.data", "0002_x.meta", "0003_x.data"):
            (tmp_path / name).write_bytes(b"\x00")
        assert sibling(tmp_path / "0002_x.meta", ".meta", ".data") == tmp_path / "0003_x.data"

    def test_a_copy_ranked_past_the_last_twin_shares_the_final_one(self, tmp_path: Path) -> None:
        for name in ("0000_x.meta", "0001_x.data", "0002_x.meta"):
            (tmp_path / name).write_bytes(b"\x00")
        assert sibling(tmp_path / "0002_x.meta", ".meta", ".data") == tmp_path / "0001_x.data"

    def test_a_copy_without_a_counter_cannot_rank_and_refuses(self, tmp_path: Path) -> None:
        for name in ("x.meta", "0000_x.data", "0008_x.data"):
            (tmp_path / name).write_bytes(b"\x00")
        with pytest.raises(ValueError, match="ambiguous"):
            sibling(tmp_path / "x.meta", ".meta", ".data")


class _ScanningFormat(_FakeFormat):
    """A format that can answer from a file's METADATA alone — the optional capability."""

    def __init__(self, name: str = "scanner", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.scan_calls: List[Path] = []

    def scan(self, path: Path) -> Dict[str, Any]:
        self.scan_calls.append(path)
        return {"scanned_by": self.name}


class _StandInFormat(_ScanningFormat):
    """A format whose scan keeps the payload's key with a PLACEHOLDER behind it.

    The real shape, measured on two shipped formats: the rate and the centre live ON the
    payload item, so a survey wants that entry present — and its array is empty, reporting 0
    samples where a read reports a million. Which makes the entry present but not ANSWERED.
    """

    scan_stands_in = ("payload",)

    def scan(self, path: Path) -> Dict[str, Any]:
        return {**super().scan(path), "payload": "STAND-IN"}


class TestScanningAFileWithoutDecodingIt:
    """``scan`` is the registry's OPTIONAL capability for "what does this file's metadata
    say?" — the sidecar or header only, never the payload.

    It exists because the two questions have costs that differ by orders of magnitude for
    the formats that pair a small metadata file with a large one: a consumer surveying a
    whole listing (where was each recording made? how long is each one?) cannot afford
    ``read`` per file, and has no business knowing which files even have a sidecar. So a
    format that can answer cheaply says so by implementing this, and a format that cannot
    simply does not — the dispatcher then answers ``None`` rather than quietly falling back
    to a full decode, because a silent fallback is exactly the cost the caller was avoiding.
    """

    def test_the_claiming_format_answers_from_its_metadata(self, tmp_path: Path) -> None:
        fmt = _ScanningFormat()
        path = tmp_path / "a.prime"
        path.touch()

        assert formats_module.scan_file(path, formats=[fmt]) == {"scanned_by": "scanner"}
        assert fmt.scan_calls == [path]
        assert fmt.read_calls == [], "a scan must never decode the payload"

    def test_a_format_without_the_capability_answers_None_rather_than_reading(self, tmp_path: Path) -> None:
        """The whole point is the cost; a quiet full read would defeat it."""
        fmt = _FakeFormat()
        path = tmp_path / "a.prime"
        path.touch()

        assert formats_module.scan_file(path, formats=[fmt]) is None
        assert fmt.read_calls == []

    def test_a_companion_half_is_dropped_like_it_is_for_a_read(self, tmp_path: Path) -> None:
        """Its content rides with the sibling's record — the same rule ``ReadFile`` follows."""
        fmt = _ScanningFormat()
        (tmp_path / "a.prime").touch()
        companion = tmp_path / "a.shadow"
        companion.touch()

        assert formats_module.scan_file(companion, formats=[fmt]) is None
        assert fmt.scan_calls == []

    def test_a_file_no_format_claims_answers_None(self, tmp_path: Path) -> None:
        """Unlike ``ReadFile``, an unknown file is NOT an error here: a survey walks whatever
        the listing holds, and one unreadable file must not refuse the whole answer."""
        path = tmp_path / "a.unknown"
        path.touch()

        assert formats_module.scan_file(path, formats=[_ScanningFormat()]) is None

    def test_the_first_claiming_format_wins_in_registry_order(self, tmp_path: Path) -> None:
        first, second = _ScanningFormat(name="first"), _ScanningFormat(name="second")
        path = tmp_path / "a.prime"
        path.touch()

        assert formats_module.scan_file(path, formats=[first, second]) == {"scanned_by": "first"}
        assert second.scan_calls == []

    def test_it_falls_back_to_the_installed_registry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fmt = _ScanningFormat()
        monkeypatch.setattr(formats_module, "_FORMATS", (fmt,))
        path = tmp_path / "a.prime"
        path.touch()

        assert formats_module.scan_file(path) == {"scanned_by": "scanner"}

    def test_a_scan_that_raises_is_the_callers_to_see(self, tmp_path: Path) -> None:
        """A malformed sidecar is a located error, exactly as a malformed file is for ``read``;
        the survey above decides whether to skip it, this layer does not decide for it."""

        class _Broken(_ScanningFormat):
            def scan(self, path: Path) -> Dict[str, Any]:
                raise ValueError("sidecar is not valid JSON")

        path = tmp_path / "a.prime"
        path.touch()
        with pytest.raises(ValueError, match="not valid JSON"):
            formats_module.scan_file(path, formats=[_Broken()])


class TestWhatAScanGENUINELYAnswers:
    """``scan_stands_in`` — the entries a cheap answer fills with a placeholder.

    A format may keep the payload's key and empty its array so a SURVEY can read the physics
    that lives on that item (measured on two shipped formats: ``scan`` reports 0 samples where
    ``read`` reports 1 000 001 and 100 000 000). A caller answering a QUESTION about that entry
    must not be handed it — a filter given the stand-in tests an empty array and answers
    confidently wrong — so ``scan_answered`` is the same dispatch with those entries removed.
    """

    def test_a_survey_still_gets_the_stand_in(self, tmp_path: Path) -> None:
        """``scan_file`` is unchanged: the entry is what makes the physics reachable."""
        path = tmp_path / "a.prime"
        path.touch()
        assert formats_module.scan_file(path, formats=[_StandInFormat()]) == {
            "scanned_by": "scanner",
            "payload": "STAND-IN",
        }

    def test_answering_a_question_gets_the_entry_MARKED(self, tmp_path: Path) -> None:
        path = tmp_path / "a.prime"
        path.touch()
        answered = formats_module.scan_answered(path, formats=[_StandInFormat()])
        assert answered == {"scanned_by": "scanner", "payload": formats_module.STAND_IN}
        assert formats_module.answers(answered, ("scanned_by",))
        assert not formats_module.answers(answered, ("payload",))
        assert not formats_module.answers(answered, ("absent",))

    def test_the_entry_is_MARKED_rather_than_removed(self, tmp_path: Path) -> None:
        """Removing it broke the next op in a real chain (`RenameField: unknown key 'signal'`)
        — a chain is written against the record a full READ produces. And a key is not a
        stable handle: the same chain renames that entry two steps later, so the marker has
        to ride the VALUE."""
        path = tmp_path / "a.prime"
        path.touch()
        answered = formats_module.scan_answered(path, formats=[_StandInFormat()]) or {}
        assert "payload" in answered

    def test_the_marker_survives_a_rename(self, tmp_path: Path) -> None:
        from recordstream.ops.structure import RenameField

        path = tmp_path / "a.prime"
        path.touch()
        answered = formats_module.scan_answered(path, formats=[_StandInFormat()]) or {}
        renamed = RenameField(src="payload", dst="input")(answered)
        assert not formats_module.answers(renamed, ("input",))

    def test_a_format_declaring_none_keeps_everything(self, tmp_path: Path) -> None:
        """Absent means the cheap answer stands in for nothing — the default."""
        path = tmp_path / "a.prime"
        path.touch()
        assert formats_module.scan_answered(path, formats=[_ScanningFormat()]) == {"scanned_by": "scanner"}

    def test_it_answers_None_exactly_where_scan_file_does(self, tmp_path: Path) -> None:
        (tmp_path / "a.prime").touch()
        companion = tmp_path / "a.shadow"
        companion.touch()
        unknown = tmp_path / "a.unknown"
        unknown.touch()
        formats = [_StandInFormat()]
        assert formats_module.scan_answered(companion, formats=formats) is None  # a companion half
        assert formats_module.scan_answered(unknown, formats=formats) is None  # nothing claims it
        assert formats_module.scan_answered(tmp_path / "a.prime", formats=[_FakeFormat()]) is None  # no scan


class TestReadFileCanAnswerFromTheMetadata:
    """``ReadFile.for_projection()`` — the cheap twin a key-restricted walk runs instead."""

    def _record(self, path: Path) -> Dict[str, Any]:
        return {"file": str(path)}

    def test_the_cheap_twin_answers_from_the_scan(self, tmp_path: Path) -> None:
        from recordstream.ops.formats import ReadFile

        fmt = _ScanningFormat()
        path = tmp_path / "a.prime"
        path.touch()
        cheap = ReadFile(formats=[fmt]).for_projection()
        assert cheap(self._record(path)) == {"file": str(path), "scanned_by": "scanner"}
        assert fmt.read_calls == []

    def test_it_marks_what_the_format_only_stands_in_for(self, tmp_path: Path) -> None:
        from recordstream.ops.formats import ReadFile

        path = tmp_path / "a.prime"
        path.touch()
        cheap = ReadFile(formats=[_StandInFormat()]).for_projection()
        out = cheap(self._record(path))
        assert out == {"file": str(path), "scanned_by": "scanner", "payload": formats_module.STAND_IN}
        assert not formats_module.answers(out, ("payload",))

    def test_a_companion_half_drops_exactly_as_a_read_does(self, tmp_path: Path) -> None:
        """Or a projected walk yields fewer records than the source has ids."""
        from recordstream.ops.formats import ReadFile

        fmt = _ScanningFormat()
        (tmp_path / "a.prime").touch()
        companion = tmp_path / "a.shadow"
        companion.touch()
        assert ReadFile(formats=[fmt]).for_projection()(self._record(companion)) is None
        assert ReadFile(formats=[fmt])(self._record(companion)) is None

    def test_a_format_that_cannot_answer_cheaply_returns_the_record_UNCHANGED(self, tmp_path: Path) -> None:
        """Not an error and not a decode: the caller sees its keys are missing and re-runs."""
        from recordstream.ops.formats import ReadFile

        fmt = _FakeFormat()
        path = tmp_path / "a.prime"
        path.touch()
        assert ReadFile(formats=[fmt]).for_projection()(self._record(path)) == {"file": str(path)}
        assert fmt.read_calls == []

    def test_an_unknown_file_is_left_for_the_real_op_to_refuse(self, tmp_path: Path) -> None:
        """The located refusal must keep coming from ONE place — the real op."""
        from recordstream.ops.formats import ReadFile

        path = tmp_path / "a.unknown"
        path.touch()
        record = self._record(path)
        assert ReadFile(formats=[_ScanningFormat()]).for_projection()(record) == record
        with pytest.raises(ValueError, match="no file format matches"):
            ReadFile(formats=[_ScanningFormat()])(record)
