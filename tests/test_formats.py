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
