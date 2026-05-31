from typing import Any, Iterator, Protocol, runtime_checkable

import torch

from dataflux.sample import Sample


def to_numpy(data: Any) -> Any:
    """Convert a torch tensor to a numpy array for array-storage backends (HDF5 / Zarr).

    Detaches and moves to CPU first so tensors carrying grad or living on a GPU
    convert cleanly. Non-tensor values pass through unchanged.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


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
        pass  # pragma: no cover

    def __enter__(self) -> "Storage":
        return self.open()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
