"""``CaptureOutputOp`` — record an op's declared ``@output`` value into sample metadata.

Wraps a target op: applies it to the sample (so the op's ``@output`` properties take their
post-call values), then copies one or more of those ``@output`` attributes off the *live op
instance* into ``metadata[key]``, and returns the applied sample.

This is the capture half of FluxStudio's "wire one op's runtime ``@output`` into a LATER op's
parameter" feature. A canvas wire from e.g. ``NoiseFloorOp.applied_snr_db`` into another op's
parameter compiles to a ``CaptureOutputOp`` (records the producer's ACTUAL drawn value) followed
by a ``ConfigureOp(ops=[UnstashInputOp(key)], target=…, param=…)`` that injects it per sample.
The value MUST be captured from the real run — many ``@output``\\ s are stochastic
(``applied_snr_db`` is a random SNR draw) and so cannot be re-derived by re-running the op.

Modality-neutral — it threads any ``Sample`` through any op — so it lives in core sampleflux
(``compose`` group, alongside ``ConfigureOp`` / ``FormulaOp`` / the stash family).
"""

from typing import Any, Dict, Optional, cast

from confluid import configurable, flow
from confluid.fluid import Fluid

from sampleflux.sample import Sample

_MISSING = object()


@configurable(category="op", group="compose")
class CaptureOutputOp:
    """Apply an op, then copy its ``@output`` attribute(s) into the sample metadata.

    The wrapped ``op`` is applied to the incoming sample (its input/target transformations are
    KEPT — the returned sample is ``op(sample)``), then each requested ``@output`` attribute is
    read off the live ``op`` instance and written to ``metadata[<its key>]``. Use ``captures`` to
    record SEVERAL outputs from ONE application (so a stochastic op runs exactly once); ``output``
    / ``key`` are the single-output convenience form.

    Confluid ``!class:`` / ``!lazy:`` markers in ``op`` are flowed lazily at first call (like
    ``ConfigureOp``), so a ``CaptureOutputOp()`` built from YAML costs nothing.

    YAML — capture ``NoiseFloorOp``'s drawn SNR so a later op can read it back:

    .. code-block:: yaml

        - !class:sampleflux.ops.capture.CaptureOutputOp
          op: !class:waivefront.torchsig.processing.NoiseFloorOp {}
          output: applied_snr_db
          key: __captured_snr

    Args:
        op: The op to apply; its ``@output`` attributes are read after it runs. Required at call time, validated lazily.
        output: A single ``@output`` attribute name to capture. Blank = capture only the ``captures`` entries.
        key: Metadata key for the ``output`` value. Blank (default) = the ``output`` name itself.
        captures: Mapping of ``@output`` attribute name -> metadata key, for capturing several outputs in one apply.
    """

    def __init__(
        self,
        op: Optional[object] = None,
        output: str = "",
        key: str = "",
        captures: Optional[Dict[str, str]] = None,
    ) -> None:
        # Lazy / zero-arg: store config only; op/outputs are validated at first call.
        self.op = op
        self.output = str(output)
        self.key = str(key)
        self.captures = dict(captures) if captures else {}

    def _items(self) -> Dict[str, str]:
        """The full ``{output_name: metadata_key}`` map — ``captures`` plus the single-output form."""
        items = dict(self.captures)
        if self.output:
            items.setdefault(self.output, self.key or self.output)
        return items

    @staticmethod
    def _read_output(op: Any, name: str) -> Any:
        """Read the ``@output`` attribute ``name`` off ``op``, looking THROUGH a ``target`` chain.

        The op may be wrapped (e.g. by a ``ConfigureOp``, which exposes the configured op as
        ``.target``) when its own params are also configured — so the ``@output`` lives on the
        innermost wrapped op. Walk ``.target`` to the first level that declares ``name``; returns
        ``_MISSING`` if no level has it.
        """
        cur, seen = op, set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            value = getattr(cur, name, _MISSING)
            if value is not _MISSING:
                return value
            cur = getattr(cur, "target", None)
        return _MISSING

    def __call__(self, sample: Sample) -> Optional[Sample]:
        if self.op is None:
            raise ValueError("CaptureOutputOp: an 'op' to apply is required")
        items = self._items()
        if not items:
            raise ValueError("CaptureOutputOp: nothing to capture — set 'output' (and 'key') or 'captures'")
        if isinstance(self.op, Fluid):
            self.op = flow(self.op)
        op = cast(Any, self.op)
        result = op(sample)
        if result is None:
            return None  # the wrapped op filtered the sample (FilterOp semantics)
        for name, meta_key in items.items():
            value = self._read_output(op, name)
            if value is _MISSING:
                raise AttributeError(
                    f"CaptureOutputOp: {type(op).__name__!r} has no @output attribute {name!r} to capture"
                )
            result.meta[meta_key] = value
        return cast(Optional[Sample], result)

    def close(self) -> None:
        """Propagate close() to the wrapped op if it owns resources."""
        close_fn = getattr(self.op, "close", None)
        if callable(close_fn):
            close_fn()
