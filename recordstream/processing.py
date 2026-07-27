"""Generic source→sink pipeline runner.

:class:`DatasetProcessor` orchestrates a :class:`~recordstream.core.Stream` from source
to sink — a runnable that drives whole-dataset processing (windowing, format
conversion, data acquisition) with an optional console progress bar. It is the
generic, modality-neutral data-pipeline runner: it iterates the stream and writes
each item to the sink, carrier-agnostic (it never inspects item internals), so it
works for any ``Stream`` regardless of what flows through it.

Wired as the ``runnable:`` object of a config and run via ``recordstream run``, or
docked into a visual-editor canvas as a runnable node.
"""

from contextlib import nullcontext
from typing import Any, Iterable, Iterator, Optional

from confluid import configurable
from loggair import get_logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from recordstream.core import Stream
from recordstream.runnable import ProgressReporting
from recordstream.storage.base import Storage

logger = get_logger(__name__)


def _stream_total(stream: Stream) -> Optional[int]:
    """``len(stream.source)`` when the source is sized, else ``None`` (a glob-based / streaming source)."""
    try:
        return len(stream.source)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


@configurable
class DatasetProcessor(ProgressReporting):
    """Orchestrate a RecordStream pipeline from source to sink.

    Args:
        stream: The :class:`~recordstream.core.Stream` to execute. Required to run;
            defaulted to ``None`` for zero-arg construction (validated in
            :meth:`run`, the workspace lazy-construction rule).
        sink: Optional sink; when absent, records are materialized to a list.
        show_progress: If ``True``, wrap iteration with a ``rich.progress`` bar.
            The total is derived from ``len(stream.source)`` when available; an
            unsized source falls back to a count-only bar. Default ``False``.
        progress_desc: Optional label for the progress bar (defaults to
            ``"DatasetProcessor"``). Ignored when ``show_progress`` is ``False``.
    """

    def __init__(
        self,
        stream: Optional[Stream] = None,
        sink: Optional[Any] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> None:
        self.stream = stream
        self.sink = sink
        self.show_progress = show_progress
        self.progress_desc = progress_desc

    def run(self) -> None:
        logger.info("Starting DatasetProcessor...")
        if self.stream is None:
            raise ValueError("DatasetProcessor.run() requires a 'stream' — none was configured.")
        # Confluid keeps Class kwargs deferred (post-construction paradigm) so
        # when the processor was loaded from YAML, ``self.stream``, its source,
        # its ops, and ``self.sink`` may all be Fluid stubs. Materialize them
        # here so callers don't need to know.
        from confluid import flow
        from confluid.fluid import Fluid

        stream = flow(self.stream) if isinstance(self.stream, Fluid) else self.stream
        if isinstance(stream.source, Fluid):
            stream.source = flow(stream.source)
        stream.ops = [flow(op) if isinstance(op, Fluid) else op for op in stream.ops]
        self.stream = stream
        sink = flow(self.sink) if isinstance(self.sink, Fluid) else self.sink

        iterator = self._wrap_progress(stream)
        # Drive an executor's progress bar (a GUI canvas) per item — independent of the console
        # ``show_progress`` rich bar; a no-op when no progress callback was injected.
        total = _stream_total(stream)
        desc = self.progress_desc or "DatasetProcessor"

        if sink:
            logger.info(f"Streaming data to sink: {sink.__class__.__name__}")
            # Replicates recordstream.core.Stream.to_sink so we can iterate through
            # our progress wrapper while preserving the Storage context + flush.
            sink_ctx: Any = sink if isinstance(sink, Storage) else nullcontext()
            count = 0
            with sink_ctx:
                for record in iterator:
                    sink.write(record)
                    count += 1
                    self._report_progress(count, total, desc)
                sink.flush()
            logger.info(f"Streamed {count} record(s) to sink.")
        else:
            logger.info("No sink provided. Materializing data in-memory.")
            results = []
            for count, record in enumerate(iterator, start=1):
                results.append(record)
                self._report_progress(count, total, desc)
            logger.info(f"Processed {len(results)} records.")

        logger.info("Processing complete.")

    def _wrap_progress(self, stream: Stream) -> Iterable[Any]:
        """Wrap the stream iterator in rich.progress when ``show_progress`` is enabled.

        Source length is probed defensively — not every DataSource implements
        ``__len__`` (e.g. glob-based streaming sources). Missing lengths
        degrade to a count-only bar instead of breaking the run.
        """
        if not self.show_progress:
            return stream
        desc = self.progress_desc or "DatasetProcessor"
        return _ProgressIter(stream, total=_stream_total(stream), desc=desc)


class _ProgressIter:
    """Iterable wrapper that opens a ``rich.progress`` bar at ``__iter__`` time.

    Kept as a class (not a generator) so ``DatasetProcessor._wrap_progress``
    can return a value that is truthy to ``bool()`` even when empty, matching
    the contract of raw ``Stream`` which behaves like a ``Sized`` (``Stream``
    subclasses ``torch.utils.data.Dataset``).
    """

    def __init__(self, stream: Stream, total: Optional[int], desc: str) -> None:
        self._stream = stream
        self._total = total
        self._desc = desc

    def __iter__(self) -> Iterator[Any]:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(self._desc, total=self._total)
            for record in self._stream:
                yield record
                progress.update(task, advance=1)


__all__ = ["DatasetProcessor"]
