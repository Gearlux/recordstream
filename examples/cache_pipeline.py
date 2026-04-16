"""DiskCache standalone demo.

Shows the three things you need to know about the disk-cache primitive:
1. ``get_or_extract`` calls your extract function only on a miss.
2. Subsequent calls return the cached path without re-running extraction.
3. The cache enforces an LRU budget — the oldest entry is evicted when a new
   one would push total usage past ``max_bytes``.

Runs end-to-end with no external data and no DataFlux pipeline.
"""

import tempfile
import time
from pathlib import Path
from typing import Callable

from dataflux.storage.cache import CacheBudgetExceeded, DiskCache


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dataflux-cache-demo-") as tmp:
        cache = DiskCache(Path(tmp), max_bytes=300)
        print(f"Cache root: {cache.root}  (max_bytes={cache.max_bytes})")

        # 1. First access: extract is called.
        invocations = {"n": 0}

        def expensive_extract(target: Path) -> None:
            invocations["n"] += 1
            print(f"  extract_fn invoked (#{invocations['n']}) → writing to {target.name}")
            target.write_bytes(b"alpha-payload" * 8)  # 104 bytes

        path_a = cache.get_or_extract("alpha.bin", expensive_extract)
        print(f"First call:  cached at {path_a}")

        # 2. Second access: extract is NOT called.
        path_a2 = cache.get_or_extract("alpha.bin", expensive_extract)
        assert path_a2 == path_a
        print(f"Second call: returned same path; extract invocations={invocations['n']}")

        # 3. Fill past the budget — observe LRU eviction. The first three
        # entries (104 + 100 + 100 = 304 bytes) already nudge past the 300-byte
        # budget on the third write, so alpha (the oldest) is evicted.
        def _writer(payload: bytes) -> Callable[[Path], None]:
            def _ex(target: Path) -> None:
                target.write_bytes(payload)

            return _ex

        time.sleep(0.01)
        cache.get_or_extract("beta.bin", _writer(b"beta" * 25))  # 100 bytes
        time.sleep(0.01)
        cache.get_or_extract("gamma.bin", _writer(b"gamma" * 20))  # 100 bytes
        time.sleep(0.01)
        cache.get_or_extract("delta.bin", _writer(b"delta" * 20))  # 100 bytes

        present = sorted(p.name for p in cache.root.iterdir() if p.is_file() and not p.name.endswith(".lock"))
        print(f"After eviction: {present}")
        assert "alpha.bin" not in present, "expected alpha to be evicted as the oldest"

        # 4. Single oversized entries are refused, not silently thrashing.
        try:
            cache.get_or_extract("oversized.bin", _writer(b"x" * 10_000))
        except CacheBudgetExceeded as e:
            print(f"Refused oversized entry as expected: {e}")

    print("OK — cache_pipeline.py finished.")


if __name__ == "__main__":
    main()
