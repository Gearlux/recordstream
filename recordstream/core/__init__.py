"""
RecordStream's engine core — the ``Stream`` facade and the machinery it runs on.

One module per cohesive unit, re-exported here so ``from recordstream.core import Stream``
is unchanged; the canonical dotted path a config / form-spec / MCP schema spells out is the
SUBMODULE one (``!class:recordstream.core.stream.Stream``), because that is what
``cls.__module__`` says. Both resolve — ``confluid.resolve_class`` falls back to a
module-path import, and this package re-exports every name — but a GENERATED config uses the
submodule spelling.

Submodules, bottom of the layer first (imports run strictly one way):

    - recordstream.core.families: the op-family registry + ``_apply_op``, the single
      op-application chokepoint every composing op routes through, plus the ``EXPANDS``
      protocol. Imports nothing from the rest of core.
    - recordstream.core.mapstyle: the ``MapStyle`` Protocol + the ``RecordSource`` union —
      "a dataset", said structurally so the engine never imports a framework.
    - recordstream.core.wrappers: ``FilterOp`` / ``WrappedOp``, the targets of ``Stream``'s
      fluent ``.filter()`` / ``.map()`` (``docs/architecture.md`` §5).
    - recordstream.core.stream: ``Stream`` + ``JointStream``, the ops-list plumbing
      (``linear_steps`` / ``_worker_task``) and ``ensure_record_dataset``.

``__all__`` below is load-bearing, not decoration: a visual editor's node bridge scans this
module in two passes, and the first (``recordstream.discovery.scan_module``) filters on
``member.__module__``, so it sees NOTHING here now that the classes live in submodules. The
second pass — the one that surfaces ``Stream`` / ``JointStream`` as engine nodes — walks
exactly this ``__all__``.
"""

from recordstream.core.families import (  # noqa: F401  — see the internal-surface note below
    _ALB_KEYS,
    _OP_FAMILIES,
    OpInvoker,
    OpMatcher,
    _apply_op,
    _expand,
    _extra_op_families,
    _is_albumentations,
    _is_torchvision_v2,
    _op_expands,
    _sync_op_families,
    register_op_family,
    registered_op_families,
)
from recordstream.core.mapstyle import MapStyle, RecordSource
from recordstream.core.stream import (  # noqa: F401  — see the internal-surface note below
    JointStream,
    Stream,
    _check_ops_materialized,
    _worker_task,
    ensure_record_dataset,
    linear_steps,
)
from recordstream.core.wrappers import FilterOp, WrappedOp

__all__ = [
    "FilterOp",
    "JointStream",
    "MapStyle",
    "OpInvoker",
    "OpMatcher",
    "RecordSource",
    "Stream",
    "WrappedOp",
    "ensure_record_dataset",
    "linear_steps",
    "register_op_family",
    "registered_op_families",
]

# The private names re-exported above are the engine's INTERNAL cross-module surface: the op
# dispatch `_apply_op` that every composing op in `recordstream.ops` imports, the spawn-worker
# helpers, and the registry list / family predicates the suite snapshots. They stay OUT of
# `__all__` (a leading underscore already keeps them off a visual editor's palette), but
# `from recordstream.core import _apply_op` must keep working — so they are re-exported
# deliberately, with the `noqa` marking that as intent rather than a stray unused import.
#
# NOTE for test doubles: these are BOUND NAMES, not views of the defining module. Patching
# `recordstream.core._apply_op` does NOT affect the copy `flow.execute` already imported —
# patch the module that USES it. The one exception is `_OP_FAMILIES`, a mutable list whose
# identity is shared, so `core._OP_FAMILIES[:] = snapshot` still restores the real registry.
