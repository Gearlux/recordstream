"""``Enable`` — toggle one or more ops on/off from a single declared flag.

A compose-group op (alongside ``Pipeline`` / ``Parallel``): wrap an inner op-list
so the whole chain can be switched on or off from ONE boolean, ``enabled``.
Modality-neutral — it threads any record through any ops — so it lives in core
recordstream, not a domain package.
"""

from typing import Any, List, Optional

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record

logger = get_logger(__name__)


@configurable(category="op", group="compose")
class Enable:
    """Wrap one or more ops so the whole chain can be switched on or off.

    ``ops`` is a list; even a single-op guard uses ``ops: [op]``. The wrapper
    threads each record through every op in sequence — same semantics as
    listing them inline in ``Stream.ops`` — so a whole visualization chain
    shares one toggle instead of needing a wrapper per op.

    The toggle is ``enabled``: a DECLARED constructor parameter exposed as a
    settable property. Being declared is what makes it reachable from every
    front-end — a YAML key, a CLI override, a Python kwarg, a generated
    tool/form schema, a canvas widget — through the same introspection every
    other ``@configurable`` parameter uses. There is no dynamic toggle-attribute
    naming: an unrecognised boolean key on this class is an error, not a flag.

    Several wrappers in one pipeline are told apart by ``name``, which scopes
    the CLI flag to that instance (``--<name>.enabled``); a bare ``--enabled``
    still broadcasts to every wrapper at once.

    YAML:

    .. code-block:: yaml

        - !class:recordstream.ops.enable.Enable
          name: visualize             # ← names THIS instance; scopes its CLI flag
          enabled: false
          ops:
            - !class:recordstream.ops.image.ConvertToImage {}
            - !class:recordstream.ops.debug.PrintRecordOp {}

    CLI:

    .. code-block:: bash

        # Targeted — only the wrapper named `visualize` flips.
        recordstream run pipeline.yaml --visualize.enabled true
        recordstream run pipeline.yaml --visualize.enabled+   # polarity shorthand → True
        recordstream run pipeline.yaml --visualize.enabled-   # polarity shorthand → False

        # Broadcast — every Enable in the config flips.
        recordstream run pipeline.yaml --enabled false

    Python:

    .. code-block:: python

        op = Enable(ops=[convert, save], name="visualize", enabled=False)
        op.enabled = True     # plain attribute write (validated: must be a bool)

    Inner ops stay deferred (not materialized) until the wrapper actually
    fires for the first time, so guarding expensive-to-construct ops with
    ``enabled: false`` costs nothing at startup.

    Constraints:
      * ``ops`` is required and must be a non-empty list — validated **lazily**
        on first call (zero-arg construction stays valid per the recordstream
        "Partial Initialization & Zero-Arg Construction" convention).
      * ``enabled`` must be a ``bool``; a non-bool raises ``TypeError`` at set time.
      * Any OTHER boolean attribute set on the wrapper raises ``ValueError`` on
        first call. That is the migration guard for the retired dynamic-toggle
        form (``visualize: false`` as a bare kwarg), which would otherwise be
        accepted silently by the post-construction paradigm and never read.

    Args:
        ops: Non-empty list of ops (native or bare library transforms) gated by the toggle.
        enabled: Whether the wrapped ops fire. Settable post-construction, from YAML,
            and from the CLI (``--enabled`` / ``--<name>.enabled``).
        name: Identifier for THIS instance — scopes its CLI flag to ``--<name>.enabled``
            and labels it in ``--help``. Empty (the default) leaves it unnamed, reachable
            only by the broadcast form.
    """

    def __init__(self, ops: Optional[List] = None, enabled: bool = True, name: str = "") -> None:
        # Partial / zero-arg: store config only; `ops` non-emptiness and stray-toggle
        # rejection are enforced lazily on first call.
        self.ops: List = list(ops) if ops else []
        self.name = name
        self.enabled = enabled
        self._checked = False

    @property
    def enabled(self) -> bool:
        """Whether the wrapped ops fire for each record (the one toggle)."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: Any) -> None:
        # A settable property — not a plain attribute — so `confluid.accepts_key`
        # reports it settable and the CLI/YAML override paths admit `enabled`.
        if not isinstance(value, bool):
            raise TypeError(f"Enable.enabled must be a bool; got {type(value).__name__} ({value!r}).")
        self._enabled = value

    def _check(self) -> None:
        """Partial one-time validation, run on the first record."""
        if not self.ops:
            raise ValueError("Enable requires a non-empty 'ops' list.")
        stray = [key for key, value in vars(self).items() if isinstance(value, bool) and not key.startswith("_")]
        if stray:
            raise ValueError(
                f"Enable: unexpected boolean attribute(s) {stray} — the toggle is 'enabled'. "
                f"Dynamic toggle names are retired: write `name: {stray[0]}` + `enabled: <bool>` "
                f"and toggle it with `--{stray[0]}.enabled true`."
            )

    def __call__(self, record: Record) -> Optional[Record]:
        if not self._checked:
            self._check()
            self._checked = True
        if not self.enabled:
            return record
        from confluid import flow
        from confluid.fluid import Fluid

        # _apply_op = the engine's op-family dispatch, so bare library transforms
        # run under the toggle exactly as in a bare ops list.
        from recordstream.core import _apply_op

        current: Optional[Record] = record
        for i, op in enumerate(self.ops):
            if current is None:
                return None
            if isinstance(op, Fluid):
                op = flow(op)
                self.ops[i] = op
            if op is None:
                continue
            current = _apply_op(current, op)
        return current

    def close(self) -> None:
        """Propagate close to inner ops that own resources (e.g. RecordSinkOp)."""
        for op in self.ops:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["Enable"]
