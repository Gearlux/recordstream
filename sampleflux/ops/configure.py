"""``ConfigureOp`` — per-sample parameter injection (the helios ``Configure`` pattern).

Some op parameters are only known per sample (a threshold derived from the sample's own
max, a crop length derived from its duration). ``ConfigureOp`` is the taidal port of the
legacy helios ``Configure`` transform (``Split`` → ``ToMetadata`` → ``Config``): a
``compute`` op-chain derives the value FROM the sample, the value is written to
``metadata[key]`` (traceability) and injected as a constructor attribute of the ``target``
op (post-construction configuration — the confluid paradigm), then ``target`` is applied
to the ORIGINAL sample.

Modality-neutral — it threads any ``Sample`` through any ops — so it lives in core
sampleflux (compose group, alongside ``TransformChain`` / ``Enable`` / ``RandomApply``).
"""

from typing import Any, List, Optional, cast

from confluid import configurable, flow
from confluid.fluid import Fluid

from sampleflux.bag.items import item_data
from sampleflux.bag.sample import Sample, primary


@configurable(category="op", group="compose")
class ConfigureOp:
    """Compute a value from the sample and inject it as a parameter of a target op.

    The ``ops`` chain runs on the incoming sample as a SIDE branch — its input/target
    transformations are discarded (the original sample continues), while metadata writes
    survive (the shared metadata-bus convention). The chain's final primary input item
    becomes the VALUE: it is written to ``metadata[key]`` and set as the ``param``
    attribute of ``target``, then ``target`` is applied to the original sample.

    Confluid ``!class:`` / ``!lazy:`` markers in ``ops`` / ``target`` are flowed lazily at
    first call (like ``TransformChain``), so a ``ConfigureOp()`` built from YAML costs nothing.

    YAML — a per-sample threshold (the helios ``Configure(TimeInSamples, CropToSize)``
    shape, here deriving ``ThresholdOp.low_level`` from the sample's own statistics):

    .. code-block:: yaml

        - !class:sampleflux.ops.configure.ConfigureOp
          ops:
            - !class:sampleflux.ops.numpy.MaxOp {}
          target: !class:sampleflux.ops.numpy.ThresholdOp
            low_op: ">="
          param: low_level

    Args:
        ops: Value-computing op-chain; the chain's final primary input item is injected. Empty = the incoming input.
        target: The op to configure and apply; required at call time, validated lazily.
        param: Target attribute name to set with the computed value (e.g. ``low_level``).
        key: Metadata key the value is also written to. Blank (default) = ``param``.
    """

    def __init__(
        self,
        ops: Optional[List[Any]] = None,
        target: Optional[object] = None,
        param: str = "",
        key: str = "",
    ) -> None:
        # Lazy / zero-arg: store config only; target/param are validated at first call.
        self.ops = list(ops) if ops else []
        self.target = target
        self.param = str(param)
        self.key = str(key)

    def __call__(self, sample: Sample) -> Optional[Sample]:
        if self.target is None:
            raise ValueError("ConfigureOp: a 'target' op is required")
        if not self.param:
            raise ValueError("ConfigureOp: 'param' (the target attribute to set) is required")
        if isinstance(self.target, Fluid):
            self.target = flow(self.target)
        # _apply_op = the engine's contract-aware chokepoint, so field-scoped ops
        # (e.g. a pair-scoped op from the kinds grid) work in the compute chain and as target.
        from sampleflux.core import _apply_op

        current: Sample = sample
        for i, op in enumerate(self.ops):
            if isinstance(op, Fluid):
                op = flow(op)
                self.ops[i] = op
            if op is None:
                continue
            result = _apply_op(current, op)
            if result is None:
                return None  # the compute chain filtered the sample (FilterOp semantics)
            current = result
        # The computed value is the primary input item's payload of the side-branch result.
        value = item_data(primary(current)[1])
        target = cast(Any, self.target)
        setattr(target, self.param, value)
        return _apply_op(sample, target)

    def close(self) -> None:
        """Propagate close() to inner ops that own resources."""
        for op in [*self.ops, self.target]:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()
