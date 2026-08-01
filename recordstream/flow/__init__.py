"""The ``flow:`` document and the :class:`FlowGraph` engine.

A **flow document** is the named-step form of a pipeline: a mapping of ``step-name → op``,
where a step's name is how later steps reference its result. It is the spelling to reach for
when a pipeline BRANCHES; a straight chain is written as a plain ``ops:`` list, which the
engine compiles to positional steps (``recordstream.core.linear_steps``). Both parse to the
same :class:`FlowStep` list and run through the same per-record kernel — there is ONE
execution model, and no lowering pass between the two forms (the flow⇄ops converters and the
per-record context ops they emitted were deleted 2026-07-30; see ``docs/architecture.md`` §3).

.. code-block:: yaml

    flow:
      spec:     !class:mypkg.MakeSpectrogram()                # input: the source record
      rescaled: !class:recordstream.ops.numpy.Threshold()     # input: previous step
      masked:   !class:mypkg.Segment() {from: spec}           # 2nd reader of spec = fan-out
      out: {from: masked, merge_from: [rescaled]}             # fan-in (no op)
    outputs: out

Step grammar (the three RESERVED step keys, stripped before the op is built):

- ``from:`` — the step supplying this step's input record. Omitted = the previous step
  (the first step reads the source record). Must name an EARLIER step: document order is
  the schedule, so forward references are errors and cycles are inexpressible.
- ``merge_from:`` — fan-in: UNION the named steps' record entries into this step's incoming
  record before the op runs (listed order, last-write-wins on a key collision).
- ``bind:`` — ``{param: ref}`` per-record parameters: ``ref`` is a step name (the step's
  whole result record), ``step[key]`` (one entry of it), or ``step.attr`` (the step op's
  live ``@output`` after it ran — read through wrapper chains by ``_read_output``).

A step may be a plain mapping with no op (``out: {from: a, merge_from: [b]}``) — a pure
fan-in/identity step; ``{}`` is the identity (used to give the source a referable name).
``outputs:`` names the step whose result the pipeline yields (default: the last step).

Step results are freed automatically: ``_result_readers`` counts each step's readers
slot-granularly and the kernel drops a result after its last one. A straight chain needs no
environment at all — :func:`is_linear` routes it to ``_run_linear``.

Submodules, bottom of the layer first (imports run strictly one way):

    - recordstream.flow.steps: the step MODEL — ``FlowStep``, the ``bind:`` reference
      grammar, ``RESERVED_STEP_KEYS``. Pure data; imports nothing from its siblings.
    - recordstream.flow.parse: ``parse_flow`` — the only module that knows the DOCUMENT form.
    - recordstream.flow.execute: the per-record kernel (``run_steps_multi`` / ``run_steps`` /
      ``is_linear`` + the two routes) and the spawn-worker entry point.
    - recordstream.flow.graph: ``FlowGraph``, the engine facade.

The canonical dotted path for a config is the SUBMODULE one
(``!class:recordstream.flow.graph.FlowGraph``); the package re-export keeps
``from recordstream.flow import FlowGraph`` and the older spelling working. ``__all__`` is
load-bearing — a visual editor's node bridge surfaces ``FlowGraph`` through it, because
``scan_module``'s ``__module__`` filter no longer sees anything in this package.
"""

from recordstream.flow.execute import (  # noqa: F401  — see the internal-surface note below
    _graph_worker_task,
    _result_readers,
    _run_from,
    _run_linear,
    is_linear,
    run_steps,
    run_steps_multi,
)
from recordstream.flow.graph import FlowGraph
from recordstream.flow.parse import parse_flow
from recordstream.flow.steps import (  # noqa: F401  — see the internal-surface note below
    _MISSING,
    RESERVED_STEP_KEYS,
    FlowStep,
    _BindRef,
    _read_output,
    _split_bind_ref,
)

__all__ = ["FlowGraph", "FlowStep", "parse_flow", "run_steps", "run_steps_multi", "is_linear", "RESERVED_STEP_KEYS"]

# The private names re-exported above are the engine's INTERNAL cross-module surface: the
# kernel's two routes, the reader accounting `core.stream` imports, the spawn-worker entry
# point, and the bind-grammar helpers the suite pins. They stay OUT of `__all__` (a leading
# underscore already keeps them off a visual editor's palette), but
# `from recordstream.flow import _result_readers` must keep working — so they are re-exported
# deliberately, with the `noqa` marking that as intent rather than a stray unused import.
#
# NOTE for test doubles: these are BOUND NAMES, not views of the defining module. Patching
# `recordstream.flow._result_readers` does NOT affect the copy `flow.graph` already imported —
# patch the module that USES it (see `tests/test_typed_flow.py`).
