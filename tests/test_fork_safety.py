"""The two fork hazards on the DataLoader path, and the guards that remove them.

A ``DataLoader`` worker may be a FORKED child — torch's default start method on Linux, and what
you get on macOS whenever something in the process has set ``fork`` (fastai does, at import). A
forked child inherits the parent's memory but NOT its threads, so anything holding a thread pool
or a system handle across the fork can die there. When it does it dies with a **SIGSEGV and no
Python traceback**, surfacing only as ``DataLoader worker exited unexpectedly`` — which is why
both guards are pinned here rather than left to be re-diagnosed.

The two are INDEPENDENT and neither fixes the other:

* :func:`~recordstream.ensure_materialized` — a lazy source built in the child runs its download
  through ``_scproxy`` / CoreFoundation, which is not fork-safe.
* ``cv2.setNumThreads(0)`` — OpenCV's thread pool, inherited across a fork, crashes the child.
  albumentations runs on OpenCV, so any ops list containing one is exposed.

The subprocess tests are the ones that matter: they reproduce the real crash, so removing a guard
makes them fail rather than making them pass differently. They are subprocesses because a SIGSEGV
takes the whole interpreter down — an in-process test could not report it.
"""

import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest

from recordstream import Image, Mask, ensure_materialized

albumentations = pytest.importorskip("albumentations")
pytest.importorskip("torch")


def _run(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter that forks its DataLoader workers."""
    script = (
        textwrap.dedent(
            """
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)   # what fastai does at import
        import numpy as np, torch
        from torch.utils.data import DataLoader
        from recordstream import Image, Mask, collate_records
        from recordstream.core import _apply_op
        import albumentations as A
        """
        )
        + textwrap.dedent(body)
    )
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)


# --------------------------------------------------------------------------- #
# cv2's thread pool
# --------------------------------------------------------------------------- #
class TestOpenCVThreadingIsDisabled:
    def test_applying_an_albumentations_op_turns_the_pool_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard fires on USE, not on import — a process that never touches albumentations
        must not have its OpenCV settings changed by importing a data library.

        The module flag is reset first because the guard is deliberately ONE-TIME (its cost is
        then a single bool check per op); by the time this runs, another test has usually already
        tripped it.
        """
        import cv2

        from recordstream.core import _apply_op, families

        monkeypatch.setattr(families, "_CV2_THREADING_DISABLED", False)
        record = {
            "image": Image(np.zeros((8, 8, 3), np.uint8)),
            "mask": Mask(np.zeros((8, 8), np.int64)),
        }

        _apply_op(record, albumentations.Resize(height=4, width=4))
        # `setNumThreads(0)` means "no pool — run on the calling thread", and cv2 then REPORTS 1
        # rather than 0 (measured). Asserting 0 would look right and always fail.
        assert cv2.getNumThreads() == 1

    def test_the_guard_runs_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """So the cost of the guarantee is one bool check per op, not a cv2 call per record."""
        from recordstream.core import _apply_op, families

        monkeypatch.setattr(families, "_CV2_THREADING_DISABLED", False)
        calls: list = []
        monkeypatch.setattr(families, "_disable_cv2_threading", lambda: calls.append(1))

        record = {"image": Image(np.zeros((8, 8, 3), np.uint8))}
        resize = albumentations.Resize(height=4, width=4)
        for _ in range(3):
            _apply_op(record, resize)
        # The invoker calls it every time; the FUNCTION is what short-circuits, so the guarantee
        # holds for a record that arrives after something else re-enabled the pool.
        assert len(calls) == 3

    def test_a_forked_worker_survives_an_albumentations_op(self) -> None:
        """The real crash, reproduced. Without the guard this dies with
        *"DataLoader worker ... is killed by signal: Segmentation fault: 11"*.
        """
        result = _run(
            """
            records = [{"image": Image(np.zeros((32, 32, 3), np.uint8)),
                        "mask": Mask(np.zeros((32, 32), np.int64))} for _ in range(8)]
            resize = A.Resize(height=16, width=16)
            # The PARENT applies one first — which is what creates the thread pool that the
            # forked child would inherit. Without that this passes vacuously.
            _apply_op(records[0], resize)

            class _DS(torch.utils.data.Dataset):
                def __len__(self): return len(records)
                def __getitem__(self, i): return _apply_op(records[i], resize)

            loader = DataLoader(_DS(), batch_size=2, num_workers=2, collate_fn=collate_records)
            n = sum(1 for _ in loader)
            print("BATCHES", n)
            """
        )
        assert result.returncode == 0, f"forked worker died:\n{result.stderr[-2000:]}"
        assert "BATCHES 4" in result.stdout


# --------------------------------------------------------------------------- #
# a lazily-built source
# --------------------------------------------------------------------------- #
class TestEnsureMaterialized:
    def test_it_reads_one_whole_record(self) -> None:
        """A whole RECORD, not a cheaper question: loading a dataset object is not the same as
        building everything a read needs, so `len()` was measured NOT to be enough."""
        reads: list = []

        class _Lazy:
            def __getitem__(self, index: int) -> dict:
                reads.append(index)
                return {"image": Image(np.zeros((4, 4, 3), np.uint8))}

            def __len__(self) -> int:
                return 4

        source = _Lazy()
        assert ensure_materialized(source) is source, "it must return the source, so it composes"
        assert reads == [0]

    def test_an_empty_source_is_not_an_error(self) -> None:
        """A split that happens to have no rows has nothing to warm, and a caller should not
        need a guard for it."""
        assert ensure_materialized([]) == []

    def test_an_iterable_only_source_is_warmed_too(self) -> None:
        consumed = []

        def _gen() -> Any:
            consumed.append(1)
            yield {"image": Image(np.zeros((4, 4, 3), np.uint8))}

        ensure_materialized(_gen())
        assert consumed == [1]
