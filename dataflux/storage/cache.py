"""Generic LRU disk cache for materialized files.

Used by sources that must first extract data from compressed archives or remote
storage before it can be memory-mapped or randomly accessed. The cache primitive
itself is dataset-agnostic — only the ``extract_fn`` passed to ``get_or_extract``
is caller-specific.
"""

import fcntl
import os
from pathlib import Path
from typing import Callable, List, Tuple, Union

from logflow import get_logger

logger = get_logger(__name__)


class CacheBudgetExceeded(Exception):
    """Raised when a single entry would exceed the entire cache budget on its own."""


class DiskCache:
    """LRU disk cache for materialized files.

    Behavior:
      - ``get_or_extract(key, extract_fn)`` calls ``extract_fn`` only on a miss,
        atomically (writes to a ``<key>.partial`` sibling, then renames on
        success). On failure the partial is removed.
      - LRU is tracked via mtime; ``evict_until(target_bytes)`` removes oldest
        files first.
      - Entries larger than ``max_bytes`` raise :class:`CacheBudgetExceeded`
        rather than silently thrashing.
      - Concurrent ``get_or_extract`` calls on the same key (across processes)
        are serialized via an ``fcntl.flock`` on a sidecar ``<key>.lock`` file,
        so multiple processes can safely share the same cache root.

    Args:
        root: Directory to use as the cache root. Created if missing.
        max_bytes: Total budget in bytes. Defaults to 50 GiB.
    """

    def __init__(self, root: Union[str, Path], max_bytes: int = 50 * 2**30) -> None:
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive; got {max_bytes!r}")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        rel = Path(key)
        if rel.is_absolute() or any(part == ".." for part in rel.parts):
            raise ValueError(f"Cache key must be a relative path without '..': {key!r}")
        return self.root / rel

    def _lock_path_for(self, key: str) -> Path:
        target = self._path_for(key)
        return target.parent / (target.name + ".lock")

    def _partial_path_for(self, key: str) -> Path:
        target = self._path_for(key)
        return target.parent / (target.name + ".partial")

    def get_or_extract(self, key: str, extract_fn: Callable[[Path], None]) -> Path:
        """Return the cached path for ``key``, calling ``extract_fn(partial_path)`` on miss.

        ``extract_fn`` MUST write the materialized file to the path it receives
        and either return normally on success or raise on failure. The path it
        receives is a ``<key>.partial`` sibling; on success the cache renames it
        atomically to the final ``key`` path.
        """
        target = self._path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            self.touch(key)
            return target

        lock_path = self._lock_path_for(key)
        partial = self._partial_path_for(key)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                # Re-check after acquiring the lock — another process may have populated it.
                if target.exists():
                    self.touch(key)
                    return target

                if partial.exists():
                    partial.unlink()

                try:
                    extract_fn(partial)
                except BaseException:
                    if partial.exists():
                        partial.unlink()
                    raise

                if not partial.exists():
                    raise RuntimeError(f"extract_fn for key {key!r} did not produce a file at {partial}")

                size = partial.stat().st_size
                if size > self.max_bytes:
                    partial.unlink()
                    raise CacheBudgetExceeded(
                        f"Cache entry for {key!r} ({size} bytes) exceeds the "
                        f"total cache budget ({self.max_bytes} bytes); refusing to cache."
                    )

                # Make room for the new entry before promoting it.
                self.evict_until(self.max_bytes - size, exclude={target})

                partial.rename(target)
                logger.info(f"DiskCache: cached {key} ({size} bytes) at {target}")
                return target
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def touch(self, key: str) -> None:
        """Mark ``key`` as most-recently-used by bumping its mtime."""
        target = self._path_for(key)
        if target.exists():
            os.utime(target, None)

    def evict_until(self, target_bytes: int, exclude: "set[Path] | None" = None) -> None:
        """Evict least-recently-used entries until total bytes <= ``target_bytes``."""
        target_bytes = max(0, target_bytes)
        exclude = exclude or set()
        entries: List[Tuple[float, int, Path]] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.endswith(".partial") or path.name.endswith(".lock"):
                continue
            if path in exclude:
                continue
            stat = path.stat()
            entries.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in entries)
        if total <= target_bytes:
            return
        entries.sort(key=lambda x: x[0])  # oldest first
        for _, size, path in entries:
            if total <= target_bytes:
                break
            try:
                path.unlink()
                total -= size
                logger.info(f"DiskCache: evicted {path} ({size} bytes)")
            except OSError as e:  # pragma: no cover - environmental
                logger.warning(f"DiskCache: failed to evict {path}: {e}")
