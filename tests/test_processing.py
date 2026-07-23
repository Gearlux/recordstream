"""Tests for :class:`sampleflux.processing.DatasetProcessor` — progress-bar toggle."""

from typing import Any, Iterator, List
from unittest.mock import patch

import confluid
import pytest
from confluid import configurable

from sampleflux.core import Flux
from sampleflux.processing import DatasetProcessor
from sampleflux.sample import Sample


@configurable
class _SizedSource:
    """Minimal Sized source yielding ``n`` trivial Samples."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n):
            yield Sample(input=i, target=None, metadata={})


@configurable
class _UnsizedSource:
    """Source that supports __iter__ but not __len__ — e.g. a streaming glob."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n):
            yield Sample(input=i, target=None, metadata={})


class _RecordingSink:
    """DataSink stand-in: records writes and flush, so we can assert order."""

    def __init__(self) -> None:
        self.written: List[Sample] = []
        self.flushed = False

    def write(self, sample: Sample) -> None:
        self.written.append(sample)

    def flush(self) -> None:
        self.flushed = True


def test_run_without_progress_uses_raw_flux() -> None:
    flux = Flux(source=_SizedSource(3))
    sink = _RecordingSink()
    with patch("sampleflux.processing.Progress") as mock_progress:
        DatasetProcessor(flux=flux, sink=sink).run()
    mock_progress.assert_not_called()
    assert len(sink.written) == 3
    assert sink.flushed


def test_run_with_progress_sized_source_sets_total() -> None:
    flux = Flux(source=_SizedSource(5))
    sink = _RecordingSink()
    with patch("sampleflux.processing.Progress") as mock_progress:
        DatasetProcessor(flux=flux, sink=sink, show_progress=True).run()
    mock_progress.assert_called_once()
    progress = mock_progress.return_value.__enter__.return_value
    progress.add_task.assert_called_once()
    call = progress.add_task.call_args
    assert call.args[0] == "DatasetProcessor"
    assert call.kwargs["total"] == 5
    assert len(sink.written) == 5


def test_run_with_progress_unsized_source_falls_back_to_none() -> None:
    flux = Flux(source=_UnsizedSource(4))
    sink = _RecordingSink()
    with patch("sampleflux.processing.Progress") as mock_progress:
        DatasetProcessor(flux=flux, sink=sink, show_progress=True).run()
    progress = mock_progress.return_value.__enter__.return_value
    progress.add_task.assert_called_once()
    assert progress.add_task.call_args.kwargs["total"] is None


def test_progress_desc_overrides_default() -> None:
    flux = Flux(source=_SizedSource(1))
    sink = _RecordingSink()
    with patch("sampleflux.processing.Progress") as mock_progress:
        DatasetProcessor(flux=flux, sink=sink, show_progress=True, progress_desc="my-run").run()
    progress = mock_progress.return_value.__enter__.return_value
    assert progress.add_task.call_args.args[0] == "my-run"


def test_no_sink_materializes_in_memory() -> None:
    """Progress wrapper also exercised on the sinkless path."""
    flux = Flux(source=_SizedSource(2))
    with patch("sampleflux.processing.Progress") as mock_progress:
        DatasetProcessor(flux=flux, show_progress=True).run()
    progress = mock_progress.return_value.__enter__.return_value
    assert progress.add_task.call_args.kwargs["total"] == 2


def test_progress_bar_updates_per_sample() -> None:
    """Asserts progress.update() fires exactly once per emitted sample."""
    flux = Flux(source=_SizedSource(3))
    sink = _RecordingSink()
    with patch("sampleflux.processing.Progress") as mock_progress:
        progress = mock_progress.return_value.__enter__.return_value
        DatasetProcessor(flux=flux, sink=sink, show_progress=True).run()
    assert progress.update.call_count == 3


def test_yaml_roundtrip_preserves_progress_flags() -> None:
    flux = Flux(source=_SizedSource(1))
    proc = DatasetProcessor(flux=flux, show_progress=True, progress_desc="from-yaml")
    state = confluid.dump(proc)
    restored: Any = confluid.load(state)
    assert restored.show_progress is True
    assert restored.progress_desc == "from-yaml"


def test_yaml_roundtrip_default_is_off() -> None:
    proc = DatasetProcessor(flux=Flux(source=_SizedSource(1)))
    restored: Any = confluid.load(confluid.dump(proc))
    assert restored.show_progress is False
    assert restored.progress_desc is None


def test_progress_callback_fires_per_sample_without_console_bar() -> None:
    """The executor's (FluxStudio) progress callback fires per sample even when show_progress is OFF.

    The native ComfyUI bar is independent of the rich console bar — set_progress_callback drives it
    regardless of ``show_progress``.
    """
    flux = Flux(source=_SizedSource(3))
    sink = _RecordingSink()
    reports: List[tuple] = []
    proc = DatasetProcessor(flux=flux, sink=sink)  # show_progress defaults to False
    proc.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
    proc.run()
    # One report per emitted sample, with a monotonically increasing value and the sized total.
    assert [v for v, _, _ in reports] == [1.0, 2.0, 3.0]
    assert all(total == 3.0 for _, total, _ in reports)
    assert all(desc == "DatasetProcessor" for *_, desc in reports)


def test_progress_callback_uses_progress_desc() -> None:
    flux = Flux(source=_SizedSource(1))
    reports: List[tuple] = []
    proc = DatasetProcessor(flux=flux, progress_desc="my-run")  # sinkless path
    proc.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
    proc.run()
    assert reports == [(1.0, 1.0, "my-run")]


def test_progress_callback_noop_for_unsized_source() -> None:
    """An unsized source has no total — the executor bar stays indeterminate (callback never fires)."""
    flux = Flux(source=_UnsizedSource(4))
    reports: List[tuple] = []
    proc = DatasetProcessor(flux=flux)
    proc.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
    proc.run()
    assert reports == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
