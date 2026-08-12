"""``FlowGraph`` — the engine facade over a ``flow:`` document.

The named-step twin of ``Stream``: same kernel (:mod:`recordstream.flow.execute`), different
authoring form. It earns its own module by size and by owning a document grammar of its own
(``docs/architecture.md`` §5); ``Stream`` reaches it only through body-local imports, which
is what keeps the core→flow direction one-way.
"""

import concurrent.futures
import multiprocessing
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from confluid import configurable
from confluid import resolve as _confluid_resolve
from loggair import get_logger

from recordstream.core.families import _extra_op_families, _op_expands
from recordstream.flow.execute import _graph_worker_task, _result_readers, is_linear, run_steps, run_steps_multi
from recordstream.flow.parse import parse_flow
from recordstream.flow.steps import FlowStep
from recordstream.items import Record

logger = get_logger(__name__)


@configurable(category="engine")
class FlowGraph:
    """Named-step graph engine — executes a ``flow:`` document natively.

    The named-step twin of :class:`~recordstream.core.Stream`, over the SAME kernel: steps run
    in document order against a per-record environment of named results, with fan-out isolation
    (copy-on-read, move on last read) and automatic result lifetimes. A LINEAR graph converts
    to a Stream (:meth:`to_stream`); a branchy one has no flat spelling by design.

    Args:
        source: Any iterable or indexable dataset (duck-typed) yielding record dicts; ``None`` = empty stream.
        flow: The flow mapping (step-name -> op / marker / step mapping) or a parsed list of FlowStep.
        outputs: Name of the step whose result is yielded. Blank (default) = the last step.
        chunk_size: Batch size for chunked iteration; ``0`` (the default) yields single records.
    """

    def __init__(
        self,
        source: Optional[Any] = None,
        flow: Optional[Union[Dict[str, Any], List[FlowStep]]] = None,
        outputs: str = "",
        chunk_size: int = 0,
    ) -> None:
        # Partial / zero-arg: store config only; parsing/validation happen in the cached property.
        self.source = source
        self.flow = flow
        self.outputs = str(outputs)
        self._chunk_size = int(chunk_size)
        self._workers = 1
        self._parsed: Optional[Tuple[List[FlowStep], str]] = None
        self._readers: Optional[Dict[str, List[Tuple[int, str]]]] = None

    # -- parsing -----------------------------------------------------------

    @property
    def steps(self) -> List[FlowStep]:
        """The parsed, validated steps (cached; recomputed only if ``flow`` is reassigned)."""
        return self._ensure_parsed()[0]

    @property
    def output_step(self) -> str:
        """The resolved output step name."""
        return self._ensure_parsed()[1]

    def _ensure_parsed(self) -> Tuple[List[FlowStep], str]:
        if self._parsed is None:
            if self.flow is None:
                raise ValueError("FlowGraph.flow is not set — provide a flow mapping or FlowStep list.")
            if isinstance(self.flow, list) and all(isinstance(s, FlowStep) for s in self.flow):
                names = [s.name for s in self.flow]
                out = self.outputs or (names[-1] if names else "")
                if out not in names:
                    raise ValueError(f"FlowGraph: outputs {out!r} does not name a step ({names!r})")
                self._parsed = (list(self.flow), out)
            else:
                self._parsed = parse_flow(cast(Dict[str, Any], self.flow), self.outputs)
        return self._parsed

    def _ensure_readers(self) -> Dict[str, List[Tuple[int, str]]]:
        """The reader accounting, computed ONCE per graph (see :func:`run_steps`)."""
        if self._readers is None:
            steps, outputs = self._ensure_parsed()
            self._readers = _result_readers(steps, outputs)
        return self._readers

    @classmethod
    def from_yaml(cls, path: str, source: Optional[Any] = None) -> "FlowGraph":
        """Build a FlowGraph from a ``{flow: {...}, outputs: ...}`` YAML document (or inline string).

        Uses ``confluid.resolve`` so step markers stay UNbuilt until :func:`parse_flow`
        pops the reserved step keys and flows each op itself.
        """
        doc = _confluid_resolve(path)
        if not isinstance(doc, dict) or "flow" not in doc:
            raise ValueError(f"FlowGraph.from_yaml: {path!r} has no 'flow:' mapping")
        return cls(source=source, flow=doc["flow"], outputs=str(doc.get("outputs", "") or ""))

    @classmethod
    def from_ops_yaml(cls, path: str, source: Optional[Any] = None) -> "FlowGraph":
        """Load a flat ``{ops: [...]}`` YAML document as a LINEAR step graph.

        No lifting is involved: a sequence IS a graph, so the op list becomes positional
        steps (``recordstream.core.linear_steps``) — the same compilation a ``Stream``'s
        ``ops`` list goes through, because they are the same thing spelled two ways.
        """
        from recordstream.core import Stream, linear_steps

        stream = Stream.from_ops_yaml(path, source=source)
        steps, outputs = linear_steps(stream.ops)
        return cls(source=source, flow=steps, outputs=outputs)

    # -- execution ---------------------------------------------------------

    def _run(self, seed: Any) -> Optional[Any]:
        """Run one record through the steps; ``None`` = filtered (an op returned None)."""
        steps, outputs = self._ensure_parsed()
        return run_steps(seed, steps, outputs, self._ensure_readers())

    def __iter__(self) -> Iterator[Any]:
        if self.source is None:
            return
        it = self._iter_records()
        if self._chunk_size > 0:
            batch: List[Record] = []
            for record in it:
                batch.append(record)
                if len(batch) == self._chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            yield from it

    def _iter_records(self) -> Iterator[Record]:
        if self._workers > 1:
            yield from self._iter_parallel()
            return
        assert self.source is not None
        steps, outputs = self._ensure_parsed()
        readers = self._ensure_readers()
        for item in self.source:
            yield from run_steps_multi(item, steps, outputs, readers)

    def _iter_parallel(self) -> Iterator[Record]:
        """Multiprocess execution — the graph's OWN spawn pool, one future per source record.

        Mirrors :meth:`recordstream.core.Stream._iter_parallel`: ``spawn`` (consistent with
        Loggair, no CI deadlocks), third-party op families shipped to the workers by
        reference. The steps pickle because their ops already must; the source never crosses
        the boundary (only the seed record does).
        """
        assert self.source is not None
        steps, outputs = self._ensure_parsed()
        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._workers, mp_context=ctx) as executor:
            extra_families = _extra_op_families()
            futures = [
                executor.submit(_graph_worker_task, item, steps, outputs, extra_families) for item in self.source
            ]
            for future in futures:
                yield from future.result()

    @property
    def _expands(self) -> bool:
        """True when any step op is 1→N — the length/index map is then unknowable."""
        return any(step.op is not None and _op_expands(step.op) for step in self._ensure_parsed()[0])

    def _guard_not_expanding(self, operation: str) -> None:
        if self._expands:
            raise TypeError(
                f"FlowGraph.{operation} is unavailable: a step op is 1→N EXPANDING, so the "
                "expanded length/index map is unknowable. Iterate the graph, wrap it in a torch "
                "IterableDataset, window at the SOURCE for random access, or call .collect()."
            )

    def __len__(self) -> int:
        from collections.abc import Sized

        self._guard_not_expanding("__len__")
        if isinstance(self.source, Sized):
            return len(self.source)
        return 0

    def __getitem__(self, index: int) -> Any:
        if self.source is None:
            raise TypeError("FlowGraph source is None — cannot index.")
        self._guard_not_expanding("__getitem__")
        if hasattr(self.source, "__getitem__"):
            raw = self.source[index]
        else:
            raise TypeError(
                f"FlowGraph source {type(self.source).__name__} does not support indexing; "
                "wrap it in a list or use iteration."
            )
        result = self._run(raw)
        if result is None:
            raise IndexError(f"Record {index} filtered out by the flow")
        return result

    def parallel(self, workers: int = 4) -> "FlowGraph":
        """Enable multiprocess execution on the graph's own spawn pool."""
        self._workers = workers
        return self

    def batch(self, chunk_size: int) -> "FlowGraph":
        """Group yielded records into lists of ``chunk_size``."""
        self._chunk_size = chunk_size
        return self

    def collect(self) -> List[Any]:
        """Materialize the full stream into a list."""
        return list(self)

    def to_stream(self) -> Any:
        """The ``Stream`` twin of a LINEAR graph — same source, same ops, same engine.

        Only a straight chain converts: a `Stream` carries an op LIST, which cannot express
        fan-out. A branchy graph has no flat spelling (that is what the deleted lowering pass
        manufactured, at the cost of destroying the structure), so it raises.
        """
        from recordstream.core import Stream

        steps, outputs = self._ensure_parsed()
        if not is_linear(steps, outputs):
            raise TypeError(
                "FlowGraph.to_stream: this graph is not a straight chain (it forks or merges), "
                "and a Stream's ops list cannot express that. Iterate the FlowGraph directly — "
                "it is the same engine."
            )
        return Stream(source=self.source, ops=[s.op for s in steps if s.op is not None])
