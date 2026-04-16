"""Tests for the generic disk cache primitive."""

import os
import threading
import time
from pathlib import Path
from typing import Callable

import pytest

from dataflux.storage.cache import CacheBudgetExceeded, DiskCache


def _write(payload: bytes) -> Callable[[Path], None]:
    """Return an extract_fn that writes ``payload`` to its target path."""

    def _ex(target: Path) -> None:
        target.write_bytes(payload)

    return _ex


def test_get_or_extract_calls_function_only_on_first_access(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)
    calls = {"n": 0}

    def extract(target: Path) -> None:
        calls["n"] += 1
        target.write_bytes(b"hello")

    p1 = cache.get_or_extract("a/b.bin", extract)
    p2 = cache.get_or_extract("a/b.bin", extract)
    assert p1 == p2
    assert p1.read_bytes() == b"hello"
    assert calls["n"] == 1


def test_get_or_extract_atomic_partial_then_rename(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)
    cache.get_or_extract("entry.bin", _write(b"x" * 100))
    # No .partial nor .lock should remain alongside the final file.
    contents = sorted(p.name for p in (tmp_path / "cache").rglob("*"))
    assert "entry.bin" in contents
    assert not any(name.endswith(".partial") for name in contents)


def test_extract_failure_leaves_no_partial_on_disk(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)

    def boom(target: Path) -> None:
        target.write_bytes(b"half")
        raise RuntimeError("simulated extraction failure")

    with pytest.raises(RuntimeError, match="simulated"):
        cache.get_or_extract("broken.bin", boom)
    leftover = list((tmp_path / "cache").rglob("*.partial"))
    assert leftover == []
    assert not (tmp_path / "cache" / "broken.bin").exists()


def test_lru_eviction_respects_max_bytes(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=300)
    # Three entries of 100 bytes fit exactly.
    cache.get_or_extract("a.bin", _write(b"a" * 100))
    time.sleep(0.01)
    cache.get_or_extract("b.bin", _write(b"b" * 100))
    time.sleep(0.01)
    cache.get_or_extract("c.bin", _write(b"c" * 100))
    # Adding a fourth must evict the oldest (a).
    time.sleep(0.01)
    cache.get_or_extract("d.bin", _write(b"d" * 100))
    names = sorted(p.name for p in (tmp_path / "cache").iterdir() if p.is_file() and not p.name.endswith(".lock"))
    assert names == ["b.bin", "c.bin", "d.bin"]


def test_touch_updates_lru_order(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=300)
    cache.get_or_extract("a.bin", _write(b"a" * 100))
    time.sleep(0.01)
    cache.get_or_extract("b.bin", _write(b"b" * 100))
    time.sleep(0.01)
    cache.get_or_extract("c.bin", _write(b"c" * 100))
    time.sleep(0.01)
    # Touch a so it's no longer the oldest.
    cache.touch("a.bin")
    time.sleep(0.01)
    cache.get_or_extract("d.bin", _write(b"d" * 100))
    names = sorted(p.name for p in (tmp_path / "cache").iterdir() if p.is_file() and not p.name.endswith(".lock"))
    assert "a.bin" in names
    assert "b.bin" not in names  # b was the oldest after touch


def test_single_entry_exceeds_budget_raises(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=10)
    with pytest.raises(CacheBudgetExceeded):
        cache.get_or_extract("big.bin", _write(b"x" * 100))
    assert not (tmp_path / "cache" / "big.bin").exists()
    assert list((tmp_path / "cache").rglob("*.partial")) == []


def test_rejects_absolute_keys(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)
    with pytest.raises(ValueError):
        cache.get_or_extract("/etc/passwd", _write(b""))


def test_rejects_dotdot_keys(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)
    with pytest.raises(ValueError):
        cache.get_or_extract("../escape.bin", _write(b""))


def test_concurrent_get_or_extract_serialized_with_file_lock(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=10_000)
    calls: list[float] = []
    barrier = threading.Barrier(4)

    def extract(target: Path) -> None:
        # Simulate a slow extraction; if not serialized, multiple threads would
        # arrive here near-simultaneously.
        calls.append(time.time())
        time.sleep(0.05)
        target.write_bytes(b"shared payload")

    def worker() -> None:
        barrier.wait()
        cache.get_or_extract("shared.bin", extract)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # extract() must have been called exactly once despite 4 simultaneous requests.
    assert len(calls) == 1
    assert (tmp_path / "cache" / "shared.bin").read_bytes() == b"shared payload"


def test_max_bytes_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DiskCache(tmp_path / "cache", max_bytes=0)


def test_evict_until_handles_already_under_budget(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)
    cache.get_or_extract("only.bin", _write(b"x" * 50))
    cache.evict_until(1024)  # already under budget; should be a no-op
    assert (tmp_path / "cache" / "only.bin").exists()


def test_extract_fn_must_produce_file(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)

    def no_op(target: Path) -> None:
        pass  # forgets to write

    with pytest.raises(RuntimeError, match="did not produce"):
        cache.get_or_extract("absent.bin", no_op)


def test_touch_on_missing_key_is_safe(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)
    cache.touch("never_made.bin")  # must not raise


def test_get_or_extract_returns_existing_file_without_calling_extract(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache", max_bytes=1024)
    target = tmp_path / "cache" / "preset.bin"
    target.write_bytes(b"already here")
    original_mtime = target.stat().st_mtime

    def boom(p: Path) -> None:
        raise AssertionError("extract must not be called for an existing entry")

    time.sleep(0.01)
    out = cache.get_or_extract("preset.bin", boom)
    assert out == target
    # touch() should have bumped mtime.
    assert os.path.getmtime(target) >= original_mtime
