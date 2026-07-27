"""Higher-order runnables — compose other runnables into a workflow.

A *runnable* is any object exposing a no-arg ``run(self)`` (a trainer, an
evaluator, a :class:`~recordstream.processing.DatasetProcessor`). This module adds
Confluid-``@configurable`` *combinators* that HOLD other runnables and orchestrate
them — the runnable-level analogue of the composing ops (``Pipeline`` /
``Parallel`` / ``Enable``):

* :class:`Sequence` — run a list of runnables in order (the workflow itself).
* :class:`Conditional` — run one of two runnables depending on a condition.
* :class:`Switch` — run one of several runnables keyed by a select value.

Conditions are themselves Confluid-``@configurable`` *predicates* — a no-arg
``__call__(self) -> bool`` (:class:`PathExists` / :class:`Not` / :class:`AllOf`
/ :class:`AnyOf`) — so a whole workflow (steps, branches, AND the conditions
that pick them) serialises to ONE Confluid YAML document, runs via the generic
``recordstream run workflow.yaml`` runner, and — being plain ``@configurable``
classes — is surfaced by discovery / a visual editor with no bespoke glue.

Branches are held as INSTANCES and only ``run()`` when selected. Example::

    !class:recordstream.workflow.Sequence
    steps:
      - !lazy:DownloadData
      - !class:recordstream.workflow.Conditional
        condition: !class:recordstream.workflow.PathExists { path: $MODEL_ROOT/model.ckpt }
        if_false: !lazy:TrainModel        # cache miss -> train
        if_true:  null                    # cache hit  -> skip, fall through
      - !lazy:Evaluate

The guarantee: **only the selected branch's ``run()`` is ever called** — an
unchosen branch is never run (and, wired ``!lazy:``, never even built, so no
model / dataset is materialised). This is the *memoise-and-continue* answer to
"don't recompute, move on": the next ``steps:`` entry runs regardless, because
``Sequence`` drives them in order — no execution-blocking, no dead branches.
Runnable proof (the resume-safe train→evaluate pipeline, run twice, both
guarantees asserted): ``examples/workflow_pipeline.py``; usage: ``docs/workflow.md``.

All combinators are zero-arg constructible and do NO functional work in
``__init__`` (the workspace lazy-construction convention); branches and
conditions are flowed lazily inside ``run()`` (a Confluid ``!class:`` / ``!lazy:``
member arrives as a deferred ``Fluid`` stub and is materialised on demand).

The combinators inherit :class:`~recordstream.runnable.TorchRunner` and
:class:`~recordstream.runnable.ProgressReporting` so a workflow runs correctly on a
GUI canvas: a combinator may wrap a *trainer*, so it declares ``__torch_runner__``
(the executor re-enables autograd for the whole run — otherwise an inner
``loss.backward()`` dies under the executor's inference mode; restoring autograd is
harmless for an inner evaluator), and it FORWARDS the executor-injected progress
callback to whichever branch is running (:func:`_run`).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from confluid import configurable, flow
from confluid.fluid import Fluid
from loggair import get_logger

from recordstream.runnable import ProgressReporting, TorchRunner

logger = get_logger(__name__)


def _resolve(value: Any) -> Any:
    """Flow a possibly-deferred Confluid value to a live object (idempotent on live ones)."""
    return flow(value) if isinstance(value, Fluid) else value


def _evaluate_condition(condition: Any) -> bool:
    """Evaluate a workflow condition to a ``bool``.

    Accepts a ``@configurable`` predicate (no-arg ``__call__ -> bool``), any
    zero-arg callable, a plain ``bool``, or a deferred ``Fluid`` resolving to one
    of those. ``None`` is falsy.
    """
    condition = _resolve(condition)
    if condition is None:
        return False
    if callable(condition):
        return bool(condition())
    return bool(condition)


def _run(runnable: Any, progress_callback: Any = None) -> None:
    """Flow ``runnable`` (if deferred) and call its ``run()``; ``None`` is a no-op.

    When ``progress_callback`` is supplied (the combinator's OWN callback, injected by a GUI
    executor via :meth:`ProgressReporting.set_progress_callback`) it is FORWARDED to the branch
    first, so a canvas progress bar tracks whichever runnable is executing inside the workflow
    (the combinator itself has no loop to report from).
    """
    runnable = _resolve(runnable)
    if runnable is None:
        return
    run = getattr(runnable, "run", None)
    if not callable(run):
        raise TypeError(f"workflow branch {type(runnable).__name__!r} has no callable run() method")
    if progress_callback is not None:
        setter = getattr(runnable, "set_progress_callback", None)
        if callable(setter):
            setter(progress_callback)
    run()


@configurable
class Sequence(TorchRunner, ProgressReporting):
    """Run a list of runnables in order — the workflow itself.

    Args:
        steps: Runnables (or deferred ``!lazy:`` / ``!class:`` markers) to run in
            order. A ``None`` entry is skipped. An empty list is a no-op.
    """

    def __init__(self, steps: Optional[List[Any]] = None) -> None:
        # Lazy / zero-arg: store config only; branches are flowed in run().
        self.steps: List[Any] = list(steps) if steps else []

    def run(self) -> None:
        total = len(self.steps)
        for i, step in enumerate(self.steps):
            resolved = _resolve(step)
            self.steps[i] = resolved  # cache the flowed step so we only flow once
            if resolved is None:
                continue
            logger.info(f"Sequence step {i + 1}/{total}: {type(resolved).__name__}")
            _run(resolved, self._progress_callback)


@configurable
class Conditional(TorchRunner, ProgressReporting):
    """Run one of two runnables depending on a condition.

    The runnable-level ``if``/``else``: a runnable that, on a condition, triggers
    another runnable held as an instance.

    Args:
        condition: A predicate (no-arg ``__call__ -> bool``), zero-arg callable,
            ``bool``, or deferred ``Fluid`` resolving to one of those. ``None`` is
            falsy.
        if_true: Runnable to run when the condition holds. ``None`` = do nothing.
        if_false: Runnable to run otherwise. ``None`` = do nothing (skip and let
            an enclosing :class:`Sequence` continue to the next step).
    """

    def __init__(self, condition: Any = None, if_true: Any = None, if_false: Any = None) -> None:
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def run(self) -> None:
        chosen = self.if_true if _evaluate_condition(self.condition) else self.if_false
        branch = _resolve(chosen)
        if branch is None:
            logger.debug("Conditional: selected branch is None — nothing to run.")
            return
        logger.info(f"Conditional -> {type(branch).__name__}")
        _run(branch, self._progress_callback)


@configurable
class Switch(TorchRunner, ProgressReporting):
    """Run one of several runnables keyed by a select's value.

    Args:
        select: A no-arg callable / predicate / deferred value producing the
            case KEY (coerced to ``str``). ``None`` (or a ``None`` result) selects
            ``default``.
        cases: Mapping of key -> runnable. The runnable whose key matches the
            select runs; an unmatched key falls back to ``default``.
        default: Runnable to run when no case matches. ``None`` = no-op.
    """

    def __init__(
        self,
        select: Any = None,
        cases: Optional[Dict[str, Any]] = None,
        default: Any = None,
    ) -> None:
        self.select = select
        self.cases: Dict[str, Any] = dict(cases) if cases else {}
        self.default = default

    def run(self) -> None:
        key = self._select()
        chosen = self.cases.get(key, self.default) if key is not None else self.default
        branch = _resolve(chosen)
        if branch is None:
            logger.debug(f"Switch: no branch for key {key!r} and no default — nothing to run.")
            return
        logger.info(f"Switch[{key!r}] -> {type(branch).__name__}")
        _run(branch, self._progress_callback)

    def _select(self) -> Optional[str]:
        selector = _resolve(self.select)
        if selector is None:
            return None
        value = selector() if callable(selector) else selector
        return None if value is None else str(value)


@configurable
class PathExists:
    """Predicate: ``True`` iff ``path`` exists on disk.

    The canonical cache check — pair with :class:`Conditional` to skip a step
    whose output artifact (a checkpoint, a converted dataset) is already present.

    Args:
        path: Filesystem path to test. Empty / ``None`` -> ``False``.
    """

    def __init__(self, path: str = "") -> None:
        self.path = path

    def __call__(self) -> bool:
        return bool(self.path) and Path(self.path).exists()


@configurable
class Not:
    """Predicate: the negation of another condition.

    Args:
        condition: The condition to negate (predicate / callable / ``bool`` /
            ``Fluid``). ``None`` is falsy, so ``Not(None)`` is ``True``.
    """

    def __init__(self, condition: Any = None) -> None:
        self.condition = condition

    def __call__(self) -> bool:
        return not _evaluate_condition(self.condition)


@configurable
class AllOf:
    """Predicate: ``True`` iff EVERY sub-condition is truthy (logical AND).

    An empty list is ``True`` (vacuous truth).

    Args:
        conditions: Conditions to AND together (each a predicate / callable /
            ``bool`` / ``Fluid``).
    """

    def __init__(self, conditions: Optional[List[Any]] = None) -> None:
        self.conditions: List[Any] = list(conditions) if conditions else []

    def __call__(self) -> bool:
        return all(_evaluate_condition(c) for c in self.conditions)


@configurable
class AnyOf:
    """Predicate: ``True`` iff at least ONE sub-condition is truthy (logical OR).

    An empty list is ``False``.

    Args:
        conditions: Conditions to OR together (each a predicate / callable /
            ``bool`` / ``Fluid``).
    """

    def __init__(self, conditions: Optional[List[Any]] = None) -> None:
        self.conditions: List[Any] = list(conditions) if conditions else []

    def __call__(self) -> bool:
        return any(_evaluate_condition(c) for c in self.conditions)


__all__ = ["Sequence", "Conditional", "Switch", "PathExists", "Not", "AllOf", "AnyOf"]
