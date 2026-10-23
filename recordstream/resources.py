"""How many records a dataset run builds at once on THIS machine — measured on the records, held to while it runs.

:class:`~recordstream.processing.DatasetProcessor` builds its records in spawn workers when ``workers`` is above 1
or ``auto``. Each worker measures what building a record took (:func:`measure`): its peak memory above what it held
before the record (the GPU's share added on a machine whose GPU shares the memory), the cores it kept busy, the CUDA
memory it used. The running process keeps a :class:`WorkerGovernor`, which reads the machine's free memory
(:func:`read_memory`) before each record is handed out and admits the record only when it fits:

- the free memory over a reserve left to the machine covers every building record's planned peak (the largest peak
  measured, times :data:`PEAK_MARGIN`) and the footprint of every worker not started yet — what the started workers
  hold is already missing from the free memory, so it is never charged again;
- with ``workers: auto``, the records building at once also keep within the machine's cores (a record that keeps 4
  cores busy counts as 4) and, when the records use CUDA, within the GPU's free memory;
- a fixed count is a ceiling the memory check may lower — never raise.

Under the reserve, no record starts; under the reserve with at most one record building, the run cannot make room by
waiting, and the processor stops it (:class:`OutOfMemory`) rather than drive the machine into swap.

A plan is a :class:`WorkerPlan` — plain numbers, so a tool running the config can report it as JSON
(``dataclasses.asdict``).
"""

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Tuple, TypeVar

import psutil

if sys.platform != "win32":
    import resource

#: Free memory a run leaves the machine: at least this many bytes ...
RESERVE_BYTES = 2 * 2**30
#: ... or this share of the machine's memory, when that is more.
RESERVE_SHARE = 0.10
#: A record is planned at its measured peak times this: records drawn from the same settings differ in size.
PEAK_MARGIN = 1.25
#: The share of the GPU's free memory the records may use together.
GPU_SHARE = 0.9

#: What decided a plan's count: no record measured yet, the count asked for, the memory, the cores or the GPU.
Limit = Literal["measuring", "ceiling", "memory", "cores", "gpu"]

T = TypeVar("T")


class OutOfMemory(MemoryError):
    """The machine's free memory fell under the run's reserve with no other record to wait for."""


@dataclass(frozen=True)
class RecordCost:
    """What building one record took in its worker.

    Args:
        memory: Bytes the worker's peak rose above ``footprint`` while it built the record, plus what the GPU took
            on a machine whose GPU shares the memory (Apple silicon).
        footprint: Bytes the worker held before the record — what an idle worker holds.
        cores: CPU seconds over wall seconds: 1.0 for one busy core, 4.0 for a record whose libraries kept four.
        gpu_memory: Peak CUDA bytes; 0 when the record used no CUDA.
        gpu_free: CUDA bytes free when the record was built; 0 when it used no CUDA.
        seconds: How long the record took.
    """

    memory: int
    footprint: int
    cores: float
    gpu_memory: int = 0
    gpu_free: int = 0
    seconds: float = 0.0


@dataclass(frozen=True)
class MemoryReading:
    """The machine's memory now, as the run sees it.

    Args:
        available: Bytes the machine can give without swapping.
        total: The machine's memory.
        processes: How many of the run's worker processes are alive.
    """

    available: int
    total: int
    processes: int = 0


@dataclass(frozen=True)
class WorkerPlan:
    """How many records may build at once now, and the numbers that decided it.

    Args:
        workers: Records building at once — never under 1, so a run always moves.
        limit: What decided ``workers`` (:data:`Limit`).
        fits: ``False`` when not even one record fits the memory: the run builds one at a time and warns.
        available: Free bytes, as read.
        reserve: Bytes the run leaves the machine.
        record_memory: Bytes planned per record (the largest measured peak times :data:`PEAK_MARGIN`); 0 before a
            record is measured.
        footprint: Bytes an idle worker holds; 0 before a record is measured.
        cores: The machine's cores.
        record_cores: The most cores a record kept busy; 0 before a record is measured.
        record_gpu_memory: Planned CUDA bytes per record; 0 without CUDA.
    """

    workers: int
    limit: Limit
    fits: bool
    available: int
    reserve: int
    record_memory: int
    footprint: int
    cores: int
    record_cores: float
    record_gpu_memory: int = 0

    def describe(self) -> str:
        """The plan in one line: ``2 (memory: 13.1 GB free − 2.0 GB reserve, 5.1 GB a record; cores: 12 at 1.0
        each)``."""
        if self.limit == "measuring":
            return f"1 (measuring a record; {gigabytes(self.available)} free)"
        text = (
            f"{self.workers} ({self.limit}: {gigabytes(self.available)} free − {gigabytes(self.reserve)} reserve, "
            f"{gigabytes(self.record_memory)} a record + {gigabytes(self.footprint)} a worker; "
            f"cores: {self.cores} at {self.record_cores:.1f} each"
        )
        if self.record_gpu_memory:
            text += f"; GPU: {gigabytes(self.record_gpu_memory)} a record"
        return text + ")"


def gigabytes(count: int) -> str:
    """``count`` bytes as a person reads them: ``4.1 GB``."""
    return f"{count / 2**30:.1f} GB"


def reserve_for(total: int) -> int:
    """The free memory a run leaves a machine of ``total`` bytes: :data:`RESERVE_BYTES`, or :data:`RESERVE_SHARE`
    of it when that is more."""
    return max(RESERVE_BYTES, int(RESERVE_SHARE * total))


def read_memory(pids: Iterable[int] = ()) -> MemoryReading:
    """The machine's free memory, and how many of the processes ``pids`` (the run's workers) are alive."""
    memory = psutil.virtual_memory()
    processes = sum(1 for pid in pids if psutil.pid_exists(pid))
    return MemoryReading(available=int(memory.available), total=int(memory.total), processes=processes)


def _peak_rss() -> int:
    """This process's peak resident memory, in bytes."""
    if sys.platform == "win32":
        return int(psutil.Process().memory_info().peak_wset)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # macOS counts bytes, Linux kilobytes


def _unified_gpu_memory() -> int:
    """Bytes the GPU holds for this process where it shares the machine's memory (Apple's MPS) — memory a resident
    size does not show (1 GB on the GPU adds about 80 MB to it). 0 when the process has not imported torch: torch is
    never imported here, a record that uses the GPU has imported it."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.backends.mps.is_available():
        return 0
    return int(torch.mps.driver_allocated_memory())


def _cuda_initialized() -> bool:
    torch = sys.modules.get("torch")
    return bool(torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized())


def measure(build: Callable[[], T]) -> Tuple[T, RecordCost]:
    """``build()`` and what it took, read in the process that builds (a worker)."""
    footprint = int(psutil.Process().memory_info().rss)
    unified = _unified_gpu_memory()
    torch = sys.modules.get("torch")
    if _cuda_initialized():
        torch.cuda.reset_peak_memory_stats()  # type: ignore[union-attr]  # _cuda_initialized: torch is imported
    wall, cpu = time.perf_counter(), time.process_time()
    result = build()
    seconds = time.perf_counter() - wall
    cores = (time.process_time() - cpu) / seconds if seconds > 0 else 0.0
    memory = max(0, _peak_rss() - footprint) + max(0, _unified_gpu_memory() - unified)
    gpu_memory = gpu_free = 0
    if _cuda_initialized():  # the record may have started CUDA itself
        torch = sys.modules["torch"]
        gpu_memory = int(torch.cuda.max_memory_allocated())
        gpu_free = int(torch.cuda.mem_get_info()[0])
    cost = RecordCost(
        memory=memory, footprint=footprint, cores=cores, gpu_memory=gpu_memory, gpu_free=gpu_free, seconds=seconds
    )
    return result, cost


class WorkerGovernor:
    """Decides, before each record is handed out, whether one more may start building.

    Args:
        ceiling: The most records building at once: the count asked for, or the machine's cores for ``auto``.
        auto: ``True`` for ``workers: auto`` — one record first, to measure it, then as many as the memory, the
            cores and the GPU allow. ``False`` for a fixed count, which only the memory may lower.
        cores: The machine's cores; ``os.cpu_count()`` when not given.
    """

    def __init__(self, ceiling: int, auto: bool, cores: Optional[int] = None) -> None:
        if ceiling < 1:
            raise ValueError(f"WorkerGovernor: ceiling must be >= 1; got {ceiling}")
        self.ceiling = ceiling
        self.auto = auto
        self.cores = cores or os.cpu_count() or 1
        self.measured = 0
        self.memory = 0
        self.footprint = 0
        self.record_cores = 0.0
        self.gpu_memory = 0
        self.gpu_budget = 0

    def observe(self, cost: RecordCost) -> None:
        """Take a built record's cost into the plan: the largest of each measure counts."""
        self.measured += 1
        self.memory = max(self.memory, cost.memory)
        self.footprint = max(self.footprint, cost.footprint)
        self.record_cores = max(self.record_cores, cost.cores)
        if cost.gpu_memory:
            self.gpu_memory = max(self.gpu_memory, cost.gpu_memory)
            self.gpu_budget = max(self.gpu_budget, cost.gpu_free + cost.gpu_memory)

    def plan(self, reading: MemoryReading) -> WorkerPlan:
        """How many records may build at once with the memory ``reading`` shows."""
        reserve = reserve_for(reading.total)
        record_memory = math.ceil(self.memory * PEAK_MARGIN)
        numbers = dict(
            available=reading.available,
            reserve=reserve,
            record_memory=record_memory,
            footprint=self.footprint,
            cores=self.cores,
            record_cores=self.record_cores,
            record_gpu_memory=math.ceil(self.gpu_memory * PEAK_MARGIN),
        )
        if not self.measured:
            if self.auto:
                return WorkerPlan(workers=1, limit="measuring", fits=True, **numbers)  # type: ignore[arg-type]
            return WorkerPlan(workers=self.ceiling, limit="ceiling", fits=True, **numbers)  # type: ignore[arg-type]
        # What the started workers hold — idle, or part way through a record — is already missing from the free
        # memory: never charge it again. Each building record is planned at its whole peak (a record part way counts
        # in full: the plan errs large), each worker still to start at its footprint.
        budget = reading.available - reserve
        by_memory = 0
        for count in range(1, self.ceiling + 2):  # one past the ceiling: room for more is the ceiling's limit
            if count * record_memory + max(0, count - reading.processes) * self.footprint > budget:
                break
            by_memory = count
        limits: list[Tuple[int, Limit]] = [(by_memory, "memory")]
        if self.auto:
            limits.append((max(1, int(self.cores // max(1.0, self.record_cores))), "cores"))
            if self.gpu_memory:
                gpu = int(GPU_SHARE * self.gpu_budget // math.ceil(self.gpu_memory * PEAK_MARGIN))
                limits.append((max(1, gpu), "gpu"))
        limits.append((self.ceiling, "ceiling"))
        count, limit = min(limits, key=lambda pair: pair[0])  # the first of equal limits names it
        return WorkerPlan(workers=max(1, count), limit=limit, fits=by_memory >= 1, **numbers)  # type: ignore[arg-type]

    def admit(self, running: int, reading: MemoryReading) -> bool:
        """Whether one more record may start while ``running`` build: never under the reserve; always the first
        one otherwise, so a run moves; the rest as the plan allows."""
        if reading.available < reserve_for(reading.total):
            return False
        return running == 0 or running < self.plan(reading).workers

    def starved(self, running: int, reading: MemoryReading) -> bool:
        """Whether the run must stop: free memory under the reserve and at most one record building, so waiting for
        records to finish frees nothing that would let the next one start."""
        return running <= 1 and reading.available < reserve_for(reading.total)
