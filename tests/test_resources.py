"""``recordstream.resources``: what a record costs in its worker, and how many records the machine can build at once.

The planning tests give the governor stated numbers (a machine of 18 GB, records of 4 GB) and check the count it
plans; the measuring tests build records in a fresh spawn process, where nothing else has raised the peak.
"""

import json
import multiprocessing
import os
import re
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any, Tuple

import numpy as np
import pytest

from recordstream import resources
from recordstream.resources import (
    MemoryReading,
    RecordCost,
    WorkerGovernor,
    WorkerPlan,
    measure,
    read_memory,
    reserve_for,
)

GB = 2**30


def _machine(available: float, total: float = 18, processes: int = 0) -> MemoryReading:
    return MemoryReading(available=int(available * GB), total=int(total * GB), processes=processes)


def _cost(memory: float = 4, footprint: float = 0.5, cores: float = 1.0, **kwargs: Any) -> RecordCost:
    return RecordCost(memory=int(memory * GB), footprint=int(footprint * GB), cores=cores, **kwargs)


def _governor(ceiling: int = 12, auto: bool = True, cores: int = 12, *costs: RecordCost) -> WorkerGovernor:
    governor = WorkerGovernor(ceiling=ceiling, auto=auto, cores=cores)
    for cost in costs:
        governor.observe(cost)
    return governor


class TestTheReserve:
    def test_a_small_machine_keeps_two_gigabytes(self) -> None:
        assert reserve_for(18 * GB) == 2 * GB

    def test_a_large_machine_keeps_a_tenth(self) -> None:
        assert reserve_for(64 * GB) == int(6.4 * GB)


class TestBeforeARecordIsMeasured:
    def test_auto_builds_one_record_to_measure_it(self) -> None:
        governor = _governor()
        plan = governor.plan(_machine(13))
        assert (plan.workers, plan.limit, plan.fits) == (1, "measuring", True)
        assert governor.admit(0, _machine(13))
        assert not governor.admit(1, _machine(13))

    def test_a_fixed_count_starts_as_many(self) -> None:
        governor = _governor(8, False)
        assert (governor.plan(_machine(13)).workers, governor.plan(_machine(13)).limit) == (8, "ceiling")
        assert governor.admit(7, _machine(13))
        assert not governor.admit(8, _machine(13))


class TestTheMemoryDecides:
    def test_as_many_records_as_the_free_memory_holds(self) -> None:
        """13 GB free − 2 GB reserve = 11 GB: two records of 4 GB × 1.25 and the one worker not started yet (0.5 GB)
        fit, three do not."""
        plan = _governor(12, True, 12, _cost()).plan(_machine(13, processes=1))
        assert (plan.workers, plan.limit, plan.fits) == (2, "memory", True)
        assert plan.record_memory == 5 * GB

    def test_what_the_started_workers_hold_is_not_charged_again(self) -> None:
        """The bluetooth quickstart (2026-10-23): seven workers started, one of them holding 0.5 GB after its record,
        4.4 GB free. What they hold is already missing from the free memory, so 2.4 GB over the reserve holds four
        records of 0.4 GB × 1.25 — it read one, every worker charged the largest footprint."""
        governor = _governor(12, True, 12, _cost(memory=0.4, footprint=0.5))
        plan = governor.plan(_machine(4.4, processes=7))
        assert (plan.workers, plan.limit, plan.fits) == (4, "memory", True)

    def test_a_worker_not_started_yet_costs_its_footprint(self) -> None:
        """11 GB over the reserve, records of 5 GB, workers of 1 GB: two workers started hold two records (10 GB);
        none started, the second worker's start-up does not fit beside them (12 GB)."""
        governor = _governor(12, True, 12, _cost(footprint=1.0))
        assert governor.plan(_machine(13, processes=2)).workers == 2
        assert governor.plan(_machine(13, processes=0)).workers == 1

    def test_the_largest_record_measured_counts(self) -> None:
        governor = _governor(12, True, 12, _cost(memory=4), _cost(memory=1))
        assert governor.plan(_machine(13, processes=1)).record_memory == 5 * GB

    def test_a_fixed_count_is_lowered_by_the_memory(self) -> None:
        plan = _governor(8, False, 12, _cost()).plan(_machine(13, processes=1))
        assert (plan.workers, plan.limit) == (2, "memory")

    def test_a_record_that_does_not_fit_still_builds_alone(self) -> None:
        plan = _governor(12, True, 12, _cost()).plan(_machine(3.8, processes=1))
        assert (plan.workers, plan.limit, plan.fits) == (1, "memory", False)


class TestTheCoresDecide:
    def test_a_record_that_keeps_four_cores_busy_counts_as_four(self) -> None:
        plan = _governor(12, True, 12, _cost(memory=0.1, cores=4.0)).plan(_machine(60, total=64))
        assert (plan.workers, plan.limit) == (3, "cores")

    def test_a_record_that_waits_counts_as_one_core(self) -> None:
        plan = _governor(12, True, 12, _cost(memory=0.1, cores=0.2)).plan(_machine(60, total=64))
        assert (plan.workers, plan.limit) == (12, "cores")

    def test_a_fixed_count_is_not_lowered_by_the_cores(self) -> None:
        plan = _governor(8, False, 12, _cost(memory=0.1, cores=4.0)).plan(_machine(60, total=64))
        assert (plan.workers, plan.limit) == (8, "ceiling")


class TestTheGpuDecides:
    def test_records_share_the_gpus_free_memory(self) -> None:
        """16 GB on the GPU (12 free + the record's 4): 90 % of it holds two records of 4 GB × 1.25."""
        cost = _cost(memory=0.1, gpu_memory=4 * GB, gpu_free=12 * GB)
        plan = _governor(12, True, 12, cost).plan(_machine(60, total=64))
        assert (plan.workers, plan.limit, plan.record_gpu_memory) == (2, "gpu", 5 * GB)

    def test_a_fixed_count_is_not_lowered_by_the_gpu(self) -> None:
        cost = _cost(memory=0.1, gpu_memory=4 * GB, gpu_free=12 * GB)
        assert _governor(8, False, 12, cost).plan(_machine(60, total=64)).limit == "ceiling"


class TestTheReserveIsKept:
    def test_no_record_starts_under_the_reserve(self) -> None:
        governor = _governor(8, False)
        assert not governor.admit(0, _machine(1.5))
        assert not governor.admit(3, _machine(1.5))

    def test_the_run_must_stop_with_at_most_one_record_building(self) -> None:
        governor = _governor(8, False)
        assert governor.starved(0, _machine(1.5))
        assert governor.starved(1, _machine(1.5))
        assert not governor.starved(2, _machine(1.5))  # one of them finishing frees memory
        assert not governor.starved(1, _machine(2.5))


class TestThePlanReads:
    def test_in_one_line(self) -> None:
        plan = _governor(12, True, 12, _cost()).plan(_machine(13, processes=1))
        assert plan.describe() == (
            "2 (memory: 13.0 GB free − 2.0 GB reserve, 5.0 GB a record + 0.5 GB a worker; cores: 12 at 1.0 each)"
        )

    def test_with_its_gpu(self) -> None:
        cost = _cost(memory=0.1, gpu_memory=4 * GB, gpu_free=12 * GB)
        assert _governor(12, True, 12, cost).plan(_machine(60, total=64)).describe().endswith("; GPU: 5.0 GB a record)")

    def test_while_measuring(self) -> None:
        assert _governor().plan(_machine(13)).describe() == "1 (measuring a record; 13.0 GB free)"

    def test_as_json(self) -> None:
        plan = _governor(12, True, 12, _cost()).plan(_machine(13, processes=1))
        assert WorkerPlan(**json.loads(json.dumps(asdict(plan)))) == plan


def test_a_ceiling_under_one_is_refused() -> None:
    with pytest.raises(ValueError, match="^WorkerGovernor: ceiling must be >= 1; got 0$"):
        WorkerGovernor(ceiling=0, auto=True)


def test_the_cores_default_to_the_machines() -> None:
    assert WorkerGovernor(ceiling=1, auto=True).cores == os.cpu_count()


# --- measuring, in a fresh spawn process ---------------------------------------------------------------------------


def _allocate_200_mb() -> int:
    block = np.ones(200 * 2**20, dtype=np.uint8)  # touched, so resident
    return int(block.sum() > 0)


def _busy(seconds: float) -> None:
    stop = time.process_time() + seconds
    while time.process_time() < stop:
        pass


def _measure_in_fresh_process(task: str) -> Tuple[Any, RecordCost]:
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(_measured, task).result(timeout=60)


def _measured(task: str) -> Tuple[Any, RecordCost]:
    if task == "allocate":
        return measure(_allocate_200_mb)
    if task == "busy":
        return measure(lambda: _busy(0.3))
    if task == "sleep":
        return measure(lambda: time.sleep(0.3))
    return measure(_on_mps)


def _on_mps() -> int:
    import torch

    block = torch.ones(64 * 2**20, device="mps")  # 256 MB on the GPU
    torch.mps.synchronize()
    return int(block.numel())


class TestARecordIsMeasured:
    def test_its_peak_memory_above_what_the_worker_held(self) -> None:
        result, cost = _measure_in_fresh_process("allocate")
        assert result == 1
        assert 190 * 2**20 <= cost.memory <= 400 * 2**20
        assert cost.footprint > 0

    def test_a_busy_record_keeps_one_core(self) -> None:
        _, cost = _measure_in_fresh_process("busy")
        assert 0.7 <= cost.cores <= 1.3
        assert cost.seconds >= 0.3

    def test_a_waiting_record_keeps_almost_none(self) -> None:
        _, cost = _measure_in_fresh_process("sleep")
        assert cost.cores < 0.2

    def test_the_gpus_share_of_the_memory_counts(self) -> None:
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("a GPU that shares the machine's memory (Apple MPS) is not on this machine")
        _, cost = _measure_in_fresh_process("mps")
        assert cost.memory >= 256 * 2**20

    def test_a_process_without_torch_has_no_gpu_share(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        assert resources._unified_gpu_memory() == 0
        assert not resources._cuda_initialized()

    def test_a_record_on_cuda_reports_its_peak_and_the_free_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset = []
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            reset_peak_memory_stats=lambda: reset.append(True),
            max_memory_allocated=lambda: 3 * GB,
            mem_get_info=lambda: (10 * GB, 16 * GB),
        )
        mps = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(
            sys.modules, "torch", types.SimpleNamespace(cuda=cuda, backends=types.SimpleNamespace(mps=mps))
        )
        _, cost = measure(lambda: None)
        assert (cost.gpu_memory, cost.gpu_free, reset) == (3 * GB, 10 * GB, [True])


class TestThePeakIsReadPerPlatform:
    def test_linux_counts_kilobytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(resources.resource, "getrusage", lambda who: types.SimpleNamespace(ru_maxrss=1000))
        assert resources._peak_rss() == 1024000

    def test_macos_counts_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(resources.resource, "getrusage", lambda who: types.SimpleNamespace(ru_maxrss=1000))
        assert resources._peak_rss() == 1000

    def test_windows_reads_the_peak_working_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        info = types.SimpleNamespace(peak_wset=5000)
        monkeypatch.setattr(resources.psutil, "Process", lambda: types.SimpleNamespace(memory_info=lambda: info))
        assert resources._peak_rss() == 5000


class TestTheMachineIsRead:
    def test_the_free_memory_and_the_workers_alive(self) -> None:
        reading = read_memory([os.getpid()])
        assert 0 < reading.available <= reading.total
        assert reading.processes == 1

    def test_a_worker_that_has_ended_is_left_out(self) -> None:
        process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(0,))
        process.start()
        process.join(timeout=30)
        assert process.pid is not None
        assert read_memory([process.pid]).processes == 0

    def test_without_workers(self) -> None:
        reading = read_memory()
        assert reading.processes == 0
        assert re.fullmatch(r"\d+", str(reading.total))
