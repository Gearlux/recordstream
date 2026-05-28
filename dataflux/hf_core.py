"""Hugging Face Datasets based implementation of DataFlux core."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import datasets
from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)


def _encode_complex(val: Any) -> Any:
    """Recursively encode complex types into Arrow-compatible dicts."""
    # Use np.iscomplexobj to catch arrays and scalars (numpy or python)
    if np.iscomplexobj(val) or (isinstance(val, torch.Tensor) and val.is_complex()):
        kind = "numpy" if not isinstance(val, torch.Tensor) else "torch"
        if np.isscalar(val):
            return {
                "_complex_": True,
                "real": float(val.real),
                "imag": float(val.imag),
                "dtype": str(getattr(val, "dtype", "complex128")),
                "kind": "scalar",
            }

        real = val.real
        imag = val.imag
        if kind == "torch":
            real = real.numpy()
            imag = imag.numpy()
            dtype = str(val.dtype)
        else:
            dtype = str(val.dtype)

        return {
            "_complex_": True,
            "real": real,
            "imag": imag,
            "dtype": dtype,
            "kind": kind,
        }

    if isinstance(val, dict):
        return {k: _encode_complex(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_encode_complex(v) for v in val]
    return val


def _decode_complex(val: Any) -> Any:
    """Recursively decode complex types from Arrow-compatible dicts."""
    if isinstance(val, dict) and val.get("_complex_"):
        real = val["real"]
        imag = val["imag"]
        dtype_str = val.get("dtype", "complex128")
        kind = val.get("kind", "numpy")

        # When coming back from Arrow/HF, arrays are often lists
        if isinstance(real, list):
            real = np.array(real)
            imag = np.array(imag)

        if kind == "scalar":
            return complex(real, imag)

        # Build intermediate numpy complex array
        res_np = real + 1j * imag
        if kind == "torch":
            torch_dtype = torch.complex64 if "complex64" in dtype_str else torch.complex128
            # Convert to appropriate numpy complex first to ensure precision, then to torch
            np_dtype = np.complex64 if "complex64" in dtype_str else np.complex128
            return torch.from_numpy(res_np.astype(np_dtype)).to(torch_dtype)

        return res_np.astype(dtype_str)

    if isinstance(val, dict):
        return {k: _decode_complex(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_decode_complex(v) for v in val]
    return val


def _sample_to_dict(sample: Sample) -> Dict[str, Any]:
    """Flatten Sample triplet into a dictionary for HF datasets."""
    d = {
        "input": _encode_complex(sample.input),
        "target": _encode_complex(sample.target),
    }
    # Flatten metadata
    if sample.metadata:
        for k, v in sample.metadata.items():
            # Avoid collisions with reserved keys
            if k in ("input", "target"):
                k = f"meta_{k}"
            d[k] = _encode_complex(v)
    return d


def _dict_to_sample(d: Dict[str, Any]) -> Sample:
    """Reconstruct Sample triplet from an HF dataset row."""
    input_val = _decode_complex(d.get("input"))
    target_val = _decode_complex(d.get("target"))
    metadata = {k: _decode_complex(v) for k, v in d.items() if k not in ("input", "target")}
    # Unflatten meta_ collisions
    if "meta_input" in metadata:
        metadata["input"] = metadata.pop("meta_input")
    if "meta_target" in metadata:
        metadata["target"] = metadata.pop("meta_target")
    return Sample(input=input_val, target=target_val, metadata=metadata)


def wrap_op(op: Callable) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Wrap a legacy Sample-based op so it can be used with HF datasets.map()."""
    def wrapper(row: Dict[str, Any]) -> Dict[str, Any]:
        sample = _dict_to_sample(row)
        result = op(sample)
        if result is None:
            # For HF filter ops or flat maps
            return row # Filtering is separate in HF
        return _sample_to_dict(result)
    return wrapper


@configurable
class HFFlux:
    """
    Flux engine backed by Hugging Face `datasets`.
    Provides a high-performance implementation of the DataFlux API.
    """

    def __init__(self, dataset: Union[datasets.Dataset, datasets.IterableDataset]) -> None:
        self._dataset = dataset

    @property
    def dataset(self) -> Union[datasets.Dataset, datasets.IterableDataset]:
        return self._dataset

    @classmethod
    def from_source(cls, source: Any) -> "HFFlux":
        """Create an HFFlux from any iterable source."""
        if hasattr(source, "_dataset") and isinstance(source._dataset, (datasets.Dataset, datasets.IterableDataset)):
            return cls(source._dataset)

        def gen() -> Iterator[Dict[str, Any]]:
            for item in source:
                yield _sample_to_dict(Sample.from_any(item))

        # Use IterableDataset by default to avoid large cache files and permission issues
        # on external drives (like /Volumes/Store).
        ds = datasets.IterableDataset.from_generator(gen)
        return cls(ds)

    def map(self, func: Callable, **kwargs: Any) -> "HFFlux":
        """
        Apply a transformation to the dataset.
        If `func` expects a Sample, it should be wrapped.
        """
        # Default behavior: assume func handles dicts (HF style)
        # We can add a 'wrapper' mode if needed for compatibility with old ops.
        return HFFlux(self._dataset.map(func, **kwargs))

    def filter(self, predicate: Callable, **kwargs: Any) -> "HFFlux":
        """Filter the dataset."""
        return HFFlux(self._dataset.filter(predicate, **kwargs))

    def parallel(self, workers: int = 4) -> "HFFlux":
        """Set default number of processes for subsequent operations."""
        # datasets.map uses `num_proc` parameter.
        # We can store this or apply it to the next operation.
        # For now, this is just for API compatibility.
        return self

    def __iter__(self) -> Iterator[Sample]:
        for row in self._dataset:
            yield _dict_to_sample(row)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Sample:
        return _dict_to_sample(self._dataset[index])

    def collect(self) -> List[Sample]:
        return list(self)

    def info(self) -> Any:
        """Expose HF dataset info (schema, metadata)."""
        return self._dataset.info
