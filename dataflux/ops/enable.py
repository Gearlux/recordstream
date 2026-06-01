"""``Enable`` — toggle one or more ops on/off via a single named CLI flag.

A compose-group op (alongside ``Tee`` / ``Parallel``): wrap an inner op-list
so the whole chain can be switched on or off from one boolean attribute whose
name becomes the CLI flag. Modality-neutral — it threads any ``Sample``
through any ops — so it lives in core dataflux, not a domain package.
"""

from typing import List, Optional, Tuple

from confluid import configurable
from logflow import get_logger

from dataflux.sample import Sample

logger = get_logger(__name__)


@configurable(category="op", group="compose")
class Enable:
    """Wrap one or more ops so they can be toggled on/off via a single named CLI flag.

    ``ops`` is a list; even a single-op guard uses ``ops: [op]``. The wrapper
    threads each sample through every op in sequence — same semantics as
    listing them inline in ``Flux.ops`` — so visualization chains like
    ``ConvertToImageOp`` → ``SaveImageOp`` share one toggle instead
    of needing a wrapper per op.

    The toggle flag is supplied in YAML as an *extra* kwarg whose name becomes
    the CLI hook — Confluid's post-construction setattr promotes it to an
    instance attribute, and Liquify's ``--<key> <value>`` overrides match
    Fluid kwargs by name (see
    :func:`liquifai.core._merge_overrides_into_fluids`).

    YAML:

    .. code-block:: yaml

        - !class:dataflux.ops.enable.Enable
          visualize: false        # ← any boolean attribute name works; this name IS the CLI flag
          ops:
            - !class:dataflux.ops.image.ConvertToImageOp {}
            - !class:waivefront.visualizers.SaveImageOp
              output_dir: ./segments_png

    CLI:

    .. code-block:: bash

        marainer process pipeline.yaml --visualize true
        marainer process pipeline.yaml --visualize+        # polarity shorthand → True
        marainer process pipeline.yaml --visualize-        # polarity shorthand → False

    Inner ops stay deferred (not materialized) until the wrapper actually
    fires for the first time, so guarding expensive-to-construct ops with
    ``Enable(..., visualize=False)`` costs nothing at startup.

    Disambiguating multiple wrappers
    --------------------------------
    When two or more ``Enable`` instances live in the same pipeline, give
    each a ``name:`` in YAML. That name becomes the preferred identifier
    in Confluid's hierarchy (``--help``) and Liquify's override matcher,
    so you can toggle them independently:

    .. code-block:: yaml

        - !class:dataflux.ops.enable.Enable
          name: overlay                 # dotted-override key
          visualize: false              # same attr name is fine — name scopes it
          ops: [render-with-overlays, save-to ./debug_png]
        - !class:dataflux.ops.enable.Enable
          name: labelstudio
          visualize: false
          ops: [render-clean, save-to ./ls_png]

    CLI:

    .. code-block:: bash

        # Targeted — only the overlay chain fires.
        marainer process pipeline.yaml --overlay.visualize true

        # Broadcast — every Fluid with a `visualize` kwarg flips.
        marainer process pipeline.yaml --visualize true

    ``name`` is a plain string on the instance; Confluid's post-construction
    paradigm setattr's it automatically from YAML with no ctor change.

    Constraints:
      * ``ops`` is required and must be a non-empty list — validated **lazily**
        on first call (zero-arg construction stays valid per the dataflux
        "Lazy Initialization & Zero-Arg Construction" convention).
      * Exactly one boolean attribute (other than ``ops`` / ``name`` and
        dunders) may be set on the wrapper — that's the toggle.
        ``RuntimeError`` is raised on first call if zero or multiple are
        present.

    Args:
        ops: Non-empty list of callables ``Sample -> Sample`` gated by the toggle.
    """

    def __init__(self, ops: Optional[List] = None) -> None:
        # Lazy / zero-arg: store config only; the non-empty requirement is enforced lazily in __call__.
        self.ops: List = list(ops) if ops else []

    def _toggle(self) -> Tuple[str, bool]:
        candidates = [
            (k, v)
            for k, v in vars(self).items()
            if k != "ops" and not k.startswith("_") and not k.startswith("__confluid_") and isinstance(v, bool)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Enable requires exactly one boolean toggle attribute (the CLI flag name); "
                f"found {len(candidates)}: {[k for k, _ in candidates]}"
            )
        return candidates[0]

    @property
    def enabled(self) -> bool:
        _, value = self._toggle()
        return value

    @property
    def flag_name(self) -> str:
        name, _ = self._toggle()
        return name

    def __call__(self, sample: Sample) -> Sample:
        if not self.ops:
            raise ValueError("Enable requires a non-empty 'ops' list.")
        if not self.enabled:
            return sample
        from confluid import flow
        from confluid.fluid import Fluid

        for i, op in enumerate(self.ops):
            if isinstance(op, Fluid):
                op = flow(op)
                self.ops[i] = op
            if op is None:
                continue
            sample = op(sample)
        return sample

    def close(self) -> None:
        """Propagate close to inner ops that own resources (e.g. SampleSinkOp)."""
        for op in self.ops:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["Enable"]
