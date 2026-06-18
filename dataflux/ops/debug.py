"""Sample inspection / debug ops."""

from typing import Any, Literal, Optional

from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)

# Per-sample output is DIAGNOSTIC, so the logger level is restricted to trace/debug (the workspace
# "Diagnostic Log Levels" mandate — never info/warning for per-iteration events). Console visibility
# comes from ``to_console`` (a plain ``print``), independent of the log level.
LogLevel = Literal["trace", "debug"]

_MAX_VALUE_REPR = 200


def _summarize(value: Any) -> str:
    """A compact one-line description: array shape/dtype + a length-capped value preview (large
    arrays elided by numpy), else a length-capped ``repr``."""
    if value is None:
        return "None"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        values = _cap(_array_values_repr(value))
        return f"{type(value).__name__}(shape={tuple(shape)}, dtype={dtype}, values={values})"
    return _cap(repr(value))


def _cap(text: str) -> str:
    """Truncate a repr to ``_MAX_VALUE_REPR`` chars with an ellipsis marker."""
    return text if len(text) <= _MAX_VALUE_REPR else text[:_MAX_VALUE_REPR] + "…"


def _array_values_repr(value: Any) -> str:
    """Compact one-line repr of an array's values (numpy / torch / anything array-like)."""
    try:
        import numpy as np

        return np.array2string(np.asarray(value), threshold=20, separator=", ").replace("\n", " ")
    except Exception:  # noqa: BLE001 - best-effort preview; fall back to plain repr
        return repr(value)


def _summarize_metadata(metadata: Any) -> str:
    """Summarise a metadata payload — each VALUE compacted so a large array shows shape/dtype."""
    if isinstance(metadata, dict):
        return "{" + ", ".join(f"{key!r}: {_summarize(value)}" for key, value in metadata.items()) + "}"
    if isinstance(metadata, list):
        return f"[batch of {len(metadata)} metadata dicts]"
    return _summarize(metadata)


@configurable(category="op", group="debug")
class PrintSampleOp:
    """Log / print a summary of each sample passing through (a pass-through ``Sample -> Sample`` op).

    A pipeline probe: emits a compact description of the sample — ``input`` / ``target`` shape+dtype
    plus a length-capped value preview (large arrays elided), and the ``metadata`` (values
    summarised the same way) — to the LogFlow logger (the LOG file + console) and, by default, to stdout via
    ``print`` (so it shows in a terminal / the FluxStudio node output panel regardless of log
    level). The sample is returned UNCHANGED.

    Args:
        label: A prefix identifying this probe in the output (e.g. "after-impairments").
        level: LogFlow level for the logged line — "trace" or "debug" (per-sample output is
            diagnostic, so info/warning are deliberately not offered; use ``to_console`` to see it).
        include_data: Include an ``input`` / ``target`` shape+dtype + value preview.
        include_metadata: Include the sample's metadata (values summarised).
        to_console: Also ``print`` the line to stdout — guaranteed console / node-panel visibility,
            independent of the log level. Set False to log only.
        limit: Stop emitting after this many samples (None = every sample) — avoids flooding on a
            large dataset; the op still passes EVERY sample through unchanged.
    """

    def __init__(
        self,
        label: str = "sample",
        level: LogLevel = "debug",
        include_data: bool = True,
        include_metadata: bool = True,
        to_console: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        self.label = label
        self.level = level
        self.include_data = include_data
        self.include_metadata = include_metadata
        self.to_console = to_console
        self.limit = limit
        self._count = 0  # runtime probe counter — NOT config (per-instance, per-process)

    def __call__(self, sample: Sample) -> Sample:
        if self.limit is None or self._count < self.limit:
            message = self._format(sample)
            getattr(logger, self.level)(message)  # level is a closed Literal, so this method exists
            if self.to_console:
                print(message)
        self._count += 1
        return sample

    def _format(self, sample: Sample) -> str:
        parts = [f"[{self.label} #{self._count}]"]
        if self.include_data:
            parts.append(f"input={_summarize(sample.input)}")
            parts.append(f"target={_summarize(sample.target)}")
        if self.include_metadata:
            parts.append(f"metadata={_summarize_metadata(sample.metadata)}")
        return " ".join(parts)
