import gc
import time
import weakref
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, cast

import numpy as np
import pytest
from confluid import to_pydantic

from recordstream import FlowGraph, Image, Record, Transform, item_data
from recordstream.core import Stream
from recordstream.core.wrappers import WrappedOp
from recordstream.ops.parallel import Parallel


def heavy_op(x: np.ndarray) -> np.ndarray:
    time.sleep(0.1)  # simulated per-record WORKLOAD (not a synchronization wait)
    return x * 2


def plus_one(x: np.ndarray) -> np.ndarray:
    return x + 1


class _Copies(Transform):
    """A 1→N op: seed ``i`` becomes ``i % 3`` records, so every third seed yields none."""

    EXPANDS = True

    def __call__(self, record: Record) -> Any:  # type: ignore[override]
        seed = int(record["x"][0])
        return [{**record, "copy": k} for k in range(seed % 3)]


class CountingSource:
    """Makes each record only when asked for it and counts how many it has handed out."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.read = 0

    def __iter__(self) -> Iterator[Record]:
        for i in range(self.n):
            self.read += 1
            yield {"x": np.full(4, float(i))}


def _stream(source: Iterable[Record], ops: List[Any], **parallel: int) -> Stream:
    return Stream(source, ops=ops).parallel(**parallel)


def _graph(source: Iterable[Record], ops: List[Any], **parallel: int) -> FlowGraph:
    return FlowGraph(source=source, flow={f"step{i}": op for i, op in enumerate(ops)}).parallel(**parallel)


def _parallel_op(source: Iterable[Record], ops: List[Any], **parallel: int) -> Stream:
    return Stream(source, ops=[Parallel(ops=ops, **parallel)])


ENGINES: Dict[str, Callable[..., Iterable[Record]]] = {"Stream": _stream, "FlowGraph": _graph}
# Every route that runs records in workers. The Parallel op runs 1→1 ops only and checks its settings when it
# starts, so it sits out the expansion and `.parallel()` refusal tests.
ROUTES: Dict[str, Callable[..., Iterable[Record]]] = {**ENGINES, "Parallel": _parallel_op}


def _open(route: str, source: Iterable[Record], ops: List[Any], **parallel: int) -> Generator[Record, None, None]:
    """Start iterating ``route`` — a generator, so a test can stop it early with ``close()``."""
    return cast(Generator[Record, None, None], iter(ROUTES[route](source, ops, **parallel)))


def test_parallel_execution() -> None:
    source = [{"x": Image(np.array([i]))} for i in range(10)]

    start = time.time()
    # Use a real top-level function for pickling; key= targets the record entry's payload.
    pipeline = Stream(source).map(heavy_op, key="x").parallel(workers=4)
    results = pipeline.collect()
    duration = time.time() - start

    assert len(results) == 10
    assert isinstance(results[0]["x"], Image)  # item type survives the spawn round-trip
    assert int(item_data(results[0]["x"])[0]) == 0
    assert int(item_data(results[9]["x"])[0]) == 18
    # We only assert the pipeline completes — this isn't a benchmark.
    assert duration < 15.0


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("window, in_flight", [({}, 4), ({"window": 3}, 6), ({"window": 1}, 2)])
def test_the_source_is_read_at_most_window_records_per_worker_ahead(
    route: str, window: Dict[str, int], in_flight: int
) -> None:
    # Submitting the whole source first read all 50 records before yielding one, and a stop then
    # waited for every one of them to be processed.
    source = CountingSource(50)
    records = _open(route, source, [WrappedOp(plus_one, key="x")], workers=2, **window)
    next(records)
    assert source.read == in_flight
    records.close()
    assert source.read == in_flight


@pytest.mark.parametrize("route", ROUTES)
def test_a_yielded_record_is_released_once_the_next_ones_are_taken(route: str) -> None:
    # The list of every future kept every result alive until the last record: memory grew with the source.
    records = _open(route, CountingSource(50), [WrappedOp(plus_one, key="x")], workers=2)
    first = weakref.ref(next(records)["x"])
    for _ in range(10):
        next(records)
    gc.collect()
    assert first() is None
    records.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_expansion_keeps_source_order_past_the_window(engine: str) -> None:
    # 20 seeds is more than the 4 in flight: the window refills while expanded records are yielded.
    out = list(ENGINES[engine](CountingSource(20), [_Copies()], workers=2))
    assert [(int(r["x"][0]), r["copy"]) for r in out] == [(i, k) for i in range(20) for k in range(i % 3)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_window_below_one_is_refused(engine: str) -> None:
    with pytest.raises(ValueError, match=rf"{engine}\.parallel\(window=0\): must be >= 1"):
        ENGINES[engine]([], [], workers=2, window=0)


@pytest.mark.parametrize("setting", ["workers", "window"])
def test_the_parallel_op_refuses_a_setting_below_one(setting: str) -> None:
    # The range mark is checked when the op is built (from YAML the refusal names the file and line) ...
    below_one: Dict[str, Any] = {setting: 0}
    with pytest.raises(ValueError, match=rf"{setting}\s+Input should be greater than or equal to 1"):
        Parallel(**below_one)
    # ... and a value set on the attribute afterwards is refused when the op starts.
    op = Parallel(ops=[WrappedOp(plus_one, key="x")])
    setattr(op, setting, 0)
    with pytest.raises(ValueError, match=rf"Parallel\({setting}=0\): must be >= 1"):
        list(Stream([{"x": np.zeros(1)}], ops=[op]))


@pytest.mark.parametrize("setting, default", [("workers", 4), ("window", 2)])
def test_the_parallel_op_offers_its_settings_from_one_up(setting: str, default: int) -> None:
    # The schema a node palette and an MCP form are built from.
    field = to_pydantic(Parallel).model_json_schema()["properties"][setting]
    assert (field["default"], field["minimum"]) == (default, 1)
