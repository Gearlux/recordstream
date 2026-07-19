"""Context ops — move data between the per-sample :class:`~sampleflux.context.Context` and the stream.

The six flat-list building blocks of graph-shaped pipelines: ``Save`` (fork snapshot),
``Use`` (branch start), ``Drop`` (cell hygiene), ``Apply`` (per-sample parameter from a
cell), ``Capture`` (an op's ``@output`` into a cell), and ``Mix`` (fan-in). A branchy
canvas graph or ``flow:`` document lowers to a plain sequential op list containing these
(``sampleflux.flow.to_ops``), executable by the ordinary ``Flux`` engine — and lifts back
(``from_ops``).

Unlike the stash family these NEVER touch ``sample.metadata``: graph wiring lives on the
engine-created Context data plane, so the metadata bus stays byte-identical to a linear
run. Cells are stored by reference (ops are copy-on-write by convention); ``Use`` copies
on read unless it drops the cell — mirroring ``UnstashInputOp(copy=True, remove=True)``.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, cast

from confluid import configurable, flow
from confluid.fluid import Fluid

from sampleflux.context import require
from sampleflux.sample import Sample

_MISSING = object()


def _flow_if_fluid(value: Any) -> Any:
    """Materialize a still-deferred confluid marker (nested op values need per-item flow)."""
    return flow(value) if isinstance(value, Fluid) else value


def _read_output(op: Any, name: str) -> Any:
    """Read attribute ``name`` off ``op``, looking through ``target``/``op`` wrapper chains.

    Reads a live ``@output`` attribute through wrapper chains (incl. our own ``Apply.op`` slot) so
    ``Capture(op=Apply(op=X, …))`` reaches X's ``@output``. Returns ``_MISSING`` when absent.
    """
    cur, seen = op, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        value = getattr(cur, name, _MISSING)
        if value is not _MISSING:
            return value
        cur = getattr(cur, "target", None) or getattr(cur, "op", None)
    return _MISSING


def _cell_field(value: Any, field: str) -> Any:
    """A cell's contribution to a Sample field: the Sample's own field, or the raw value verbatim."""
    if isinstance(value, Sample):
        return getattr(value, field)
    return value


@configurable(category="op", group="structure")
class Save:
    """Snapshot the stream sample into a Context cell (pass-through).

    The sample continues down the linear stream unchanged AND becomes readable by later
    ``Use`` / ``Apply`` / ``Mix`` steps — the fork point of a fan-out. Stored by
    reference (readers copy); ops are copy-on-write by convention, so the snapshot stays
    intact as the stream continues (insert ``CopySampleOp`` before an in-place op).

    Args:
        name: Context cell to store the sample under; required at call time, validated lazily.
    """

    def __init__(self, name: str = "") -> None:
        # Lazy / zero-arg: store config only; the cell name is validated at first call.
        self.name = str(name)

    def __call__(self, sample: Sample) -> Sample:
        if not self.name:
            raise ValueError("Save: 'name' (the context cell to write) is required")
        require("Save").put(self.name, sample)
        return sample


@configurable(category="op", group="structure")
class Use:
    """Replace the stream sample with a Context cell's value (a branch start).

    The incoming sample is discarded; the cell's value becomes the stream sample
    (``Sample.from_any`` coerces a raw cell value). Reads a DEEP COPY so two branches
    reading one fork stay independent — unless ``drop`` frees the cell, which skips the
    copy (move semantics, the right choice for a cell's LAST reader).

    Args:
        name: Context cell to read; required at call time, validated lazily.
        drop: When True, free the cell after reading and skip the defensive copy (move semantics).
    """

    def __init__(self, name: str = "", drop: bool = False) -> None:
        # Lazy / zero-arg: store config only; the cell name is validated at first call.
        self.name = str(name)
        self.drop = bool(drop)

    def __call__(self, sample: Sample) -> Sample:
        if not self.name:
            raise ValueError("Use: 'name' (the context cell to read) is required")
        ctx = require("Use")
        value = ctx.get(self.name)
        if self.drop:
            ctx.delete(self.name)
        else:
            value = deepcopy(value)
        return Sample.from_any(value)


@configurable(category="op", group="structure")
class Drop:
    """Free Context cells (pass-through) — the explicit liveness hygiene step.

    Deleting a missing cell raises: in a compiled graph that means the liveness pass and
    the op order disagree, which should fail loudly rather than leak.

    Args:
        names: Context cells to delete after this point; an empty list (default) is a no-op.
    """

    def __init__(self, names: Optional[List[str]] = None) -> None:
        # Lazy / zero-arg: store config only.
        self.names = list(names) if names else []

    def __call__(self, sample: Sample) -> Sample:
        if self.names:
            ctx = require("Drop")
            for name in self.names:
                ctx.delete(name)
        return sample


@configurable(category="op", group="structure")
class Apply:
    """Set a wrapped op's parameter from a Context cell, then apply the op.

    The declarative per-sample-parameter step (``ConfigureOp`` with the value coming from
    a cell instead of an inline compute chain): the cell holds a prior branch's result —
    a Sample cell contributes its ``input``, a raw cell value (e.g. a ``Capture``\\ d
    ``@output``) is used as-is. The value is ``setattr``'d as ``param`` on ``op``
    post-construction (the confluid paradigm), then ``op`` runs on the incoming sample.

    Confluid ``!class:`` / ``!lazy:`` markers in ``op`` are flowed lazily at first call
    (like ``ConfigureOp``), so an ``Apply()`` built from YAML costs nothing.

    Args:
        op: The op to configure and apply; required at call time, validated lazily.
        param: Attribute name on ``op`` to set with the cell value; required at call time.
        source: Context cell holding the value; required at call time, validated lazily.
        drop: When True, free the source cell after reading it.
    """

    def __init__(
        self,
        op: Optional[object] = None,
        param: str = "",
        source: str = "",
        drop: bool = False,
    ) -> None:
        # Lazy / zero-arg: store config only; op/param/source are validated at first call.
        self.op = op
        self.param = str(param)
        self.source = str(source)
        self.drop = bool(drop)

    def __call__(self, sample: Sample) -> Optional[Sample]:
        if self.op is None:
            raise ValueError("Apply: an 'op' to configure and apply is required")
        if not self.param:
            raise ValueError("Apply: 'param' (the op attribute to set) is required")
        if not self.source:
            raise ValueError("Apply: 'source' (the context cell holding the value) is required")
        self.op = _flow_if_fluid(self.op)
        ctx = require("Apply")
        value = ctx.get(self.source)
        if self.drop:
            ctx.delete(self.source)
        value = _cell_field(value, "input")
        op = cast(Any, self.op)
        setattr(op, self.param, value)
        # _apply_op = the engine's contract-aware chokepoint, so a field-scoped wrapped op
        # (e.g. a pair-scoped op from the kinds grid) applies exactly as in a bare ops list.
        from sampleflux.core import _apply_op

        return _apply_op(sample, op)

    def close(self) -> None:
        """Propagate close() to the wrapped op if it owns resources."""
        close_fn = getattr(self.op, "close", None)
        if callable(close_fn):
            close_fn()


@configurable(category="op", group="structure")
class Capture:
    """Apply an op, then record its ``@output`` attribute(s) into Context cells.

    Records a wrapped op's live ``@output``: the wrapped op runs once (stochastic-correct
    — the value is read from the actual run, never recomputed) and each requested
    ``@output`` is stored as a raw cell value for a later ``Apply``/``Mix`` to read. The
    returned sample is ``op(sample)`` — transformations are kept.

    Confluid ``!class:`` / ``!lazy:`` markers in ``op`` are flowed lazily at first call,
    so a ``Capture()`` built from YAML costs nothing.

    Args:
        op: The op to apply; its ``@output`` attributes are read after it runs. Required at call time.
        output: A single ``@output`` attribute name to capture. Blank = capture only the ``captures`` entries.
        name: Context cell for the ``output`` value. Blank (default) = the ``output`` name itself.
        captures: Mapping of ``@output`` attribute name -> context cell, for capturing several outputs in one apply.
    """

    def __init__(
        self,
        op: Optional[object] = None,
        output: str = "",
        name: str = "",
        captures: Optional[Dict[str, str]] = None,
    ) -> None:
        # Lazy / zero-arg: store config only; op/outputs are validated at first call.
        self.op = op
        self.output = str(output)
        self.name = str(name)
        self.captures = dict(captures) if captures else {}

    def _items(self) -> Dict[str, str]:
        """The full ``{output_name: cell_name}`` map — ``captures`` plus the single-output form."""
        items = dict(self.captures)
        if self.output:
            items.setdefault(self.output, self.name or self.output)
        return items

    def __call__(self, sample: Sample) -> Optional[Sample]:
        if self.op is None:
            raise ValueError("Capture: an 'op' to apply is required")
        items = self._items()
        if not items:
            raise ValueError("Capture: nothing to capture — set 'output' (and 'name') or 'captures'")
        self.op = _flow_if_fluid(self.op)
        ctx = require("Capture")
        op = cast(Any, self.op)
        # _apply_op = the engine's contract-aware chokepoint (field-scoped ops capture too).
        from sampleflux.core import _apply_op

        result = _apply_op(sample, op)
        if result is None:
            return None  # the wrapped op filtered the sample (FilterOp semantics)
        for attr, cell in items.items():
            value = _read_output(op, attr)
            if value is _MISSING:
                raise AttributeError(f"Capture: {type(op).__name__!r} has no @output attribute {attr!r} to capture")
            ctx.put(cell, value)
        return cast(Optional[Sample], result)

    def close(self) -> None:
        """Propagate close() to the wrapped op if it owns resources."""
        close_fn = getattr(self.op, "close", None)
        if callable(close_fn):
            close_fn()


@configurable(category="op", group="structure")
class Mix:
    """Fan-in: compose one sample from Context cells and the incoming stream sample.

    Each named slot reads its cell — a Sample cell contributes its corresponding field, a
    raw cell value is used verbatim — while an unnamed slot keeps the incoming sample's
    field. Metadata merges incoming-first, then each named cell's metadata in slot order
    (``input_from``, ``target_from``, ``metadata_from`` — last write wins), so branch
    traceability survives the merge and ``metadata_from`` has the final say.

    Args:
        input_from: Context cell providing the mixed ``input``. Blank (default) = keep the incoming input.
        target_from: Context cell providing the mixed ``target``. Blank (default) = keep the incoming target.
        metadata_from: Context cell whose metadata merges LAST (wins conflicts). Blank (default) = none.
        drop: Context cells to free after mixing (defaults to none).
    """

    def __init__(
        self,
        input_from: str = "",
        target_from: str = "",
        metadata_from: str = "",
        drop: Optional[List[str]] = None,
    ) -> None:
        # Lazy / zero-arg: store config only; cell names are resolved at first call.
        self.input_from = str(input_from)
        self.target_from = str(target_from)
        self.metadata_from = str(metadata_from)
        self.drop = list(drop) if drop else []

    def __call__(self, sample: Sample) -> Sample:
        ctx = require("Mix")

        mixed_input = sample.input
        mixed_target = sample.target
        metadata: Dict[str, Any] = dict(sample.meta)

        for cell_name, field in ((self.input_from, "input"), (self.target_from, "target")):
            if not cell_name:
                continue
            value = ctx.get(cell_name)
            if field == "input":
                mixed_input = _cell_field(value, "input")
            else:
                mixed_target = _cell_field(value, "target")
            if isinstance(value, Sample):
                metadata.update(value.meta)
        if self.metadata_from:
            value = ctx.get(self.metadata_from)
            extra = value.meta if isinstance(value, Sample) else value
            if not isinstance(extra, dict):
                raise TypeError(
                    f"Mix: metadata_from cell {self.metadata_from!r} holds {type(extra).__name__}, "
                    "expected a Sample or a dict"
                )
            metadata.update(extra)

        for cell_name in self.drop:
            ctx.delete(cell_name)

        return Sample(input=mixed_input, target=mixed_target, metadata=metadata)
