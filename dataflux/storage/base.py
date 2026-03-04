from typing import Iterator, Protocol, runtime_checkable

from dataflux.sample import Sample


@runtime_checkable
class DataSource(Protocol):
    """Minimum contract for a DataFlux data source."""

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples in the source."""
        ...

    def __len__(self) -> int:
        """Total number of samples available."""
        ...


@runtime_checkable
class DataSink(Protocol):
    """Minimum contract for a DataFlux data sink."""

    def write(self, sample: Sample) -> None:
        """Write a single sample to the sink."""
        ...

    def flush(self) -> None:
        """Ensure all pending writes are committed to storage."""
        ...


class Storage:
    """Base class for storage backends providing context manager support."""

    def open(self) -> "Storage":
        return self

    def close(self) -> None:
        pass

    def __enter__(self) -> "Storage":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
