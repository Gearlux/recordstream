"""``ConfigureOp`` — per-record parameter injection.

Some op parameters are only known per record (a threshold derived from the record's own
max, a crop length derived from its duration). ``ConfigureOp`` runs a ``compute`` op-chain
that derives the value FROM the record, injects it as a constructor attribute of the
``target`` op (post-construction configuration — the confluid paradigm), then applies
``target`` to the ORIGINAL record.

Modality-neutral — it threads any record through any ops — so it lives in core
sampleflux (compose group, alongside ``Pipeline`` / ``Enable`` / ``RandomApply``).
"""

from typing import Any, List, Optional, cast

from confluid import configurable, flow
from confluid.fluid import Fluid

from sampleflux.items import Record, item_data


@configurable(category="op", group="compose")
class ConfigureOp:
    """Compute a value from the record and inject it as a parameter of a target op.

    The ``ops`` chain runs on the incoming record as a SIDE branch — its transformations
    are discarded (the original record continues). The ``source``-keyed entry of the
    chain's final record becomes the VALUE (payload-unwrapped via ``item_data``): it is
    set as the ``param`` attribute of ``target``, then ``target`` is applied to the
    original record.

    Confluid ``!class:`` / ``!lazy:`` markers in ``ops`` / ``target`` are flowed lazily at
    first call (like ``Pipeline``), so a ``ConfigureOp()`` built from YAML costs nothing.

    YAML — a per-record threshold derived from the record's own statistics:

    .. code-block:: yaml

        - !class:sampleflux.ops.configure.ConfigureOp
          ops:
            - !class:sampleflux.ops.formula.FormulaOp {field: image, formula: "amax(a) * 0.5"}
          source: image
          target: !class:sampleflux.ops.numpy.Threshold
            low_op: ">="
          param: low_level

    Args:
        ops: Value-computing op-chain run on a side-branch copy of the record. Empty = the incoming record.
        target: The op to configure and apply; required at call time, validated lazily.
        param: Target attribute name to set with the computed value (e.g. ``low_level``).
        source: Record key of the side-branch result holding the computed value; required at call time.
    """

    def __init__(
        self,
        ops: Optional[List[Any]] = None,
        target: Optional[object] = None,
        param: str = "",
        source: str = "",
    ) -> None:
        # Lazy / zero-arg: store config only; target/param/source are validated at first call.
        self.ops = list(ops) if ops else []
        self.target = target
        self.param = str(param)
        self.source = str(source)

    def __call__(self, record: Record) -> Optional[Record]:
        if self.target is None:
            raise ValueError("ConfigureOp: a 'target' op is required")
        if not self.param:
            raise ValueError("ConfigureOp: 'param' (the target attribute to set) is required")
        if not self.source:
            raise ValueError("ConfigureOp: 'source' (the record key holding the computed value) is required")
        if isinstance(self.target, Fluid):
            self.target = flow(self.target)
        # _apply_op = the engine's op-family dispatch, so bare library transforms
        # work in the compute chain and as target exactly as in a bare ops list.
        from sampleflux.core import _apply_op

        current: Record = record
        for i, op in enumerate(self.ops):
            if isinstance(op, Fluid):
                op = flow(op)
                self.ops[i] = op
            if op is None:
                continue
            result = _apply_op(current, op)
            if result is None:
                return None  # the compute chain filtered the record (FilterOp semantics)
            current = result
        if self.source not in current:
            raise KeyError(f"ConfigureOp: side-branch result has no key {self.source!r} (keys: {list(current)})")
        value = item_data(current[self.source])
        target = cast(Any, self.target)
        setattr(target, self.param, value)
        return _apply_op(record, target)

    def close(self) -> None:
        """Propagate close() to inner ops that own resources."""
        for op in [*self.ops, self.target]:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()
