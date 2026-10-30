"""Higher-order runnables — compose other runnables into a workflow.

A *runnable* is any object exposing a no-arg ``run(self)`` (a trainer, an
evaluator, a :class:`~recordstream.processing.DatasetProcessor`). This module adds
Confluid-``@configurable`` *combinators* that HOLD other runnables and orchestrate
them — the runnable-level analogue of the composing ops (``Pipeline`` /
``Parallel`` / ``Enable``):

* :class:`Sequence` — run a list of runnables in order (the workflow itself).
* :class:`Conditional` — run one of two runnables depending on a condition.
* :class:`Switch` — run one of several runnables keyed by a select value.

The steps a shell script makes around its runs are runnables too:

* :class:`Require` — refuse before anything runs, unless a condition holds.
* :class:`ReplaceFolder` — empty a folder once; a marker in it makes a re-run resume.
* :class:`RunConfig` — run one config in its own process (``recordstream run <config> --scope s key=value``).

Every step can say what it would do without doing it — its ``plan()``, a :class:`Plan` —
and :class:`Sequence` makes every step's checks before its first step runs;
``recordstream run <workflow> --dry-run`` prints the plans and changes nothing.

Conditions are themselves Confluid-``@configurable`` *predicates* — a no-arg
``__call__(self) -> bool`` (:class:`PathExists` / :class:`Not` / :class:`AllOf`
/ :class:`AnyOf` / :class:`EnvSet` / :class:`FreeSpace` / :class:`HasFiles`), each
saying what it checks (``describe()``) — so a whole workflow (steps, branches, AND the conditions
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
GUI canvas: a combinator may wrap a *trainer*, so it declares ``__needs_autograd__``
(the executor re-enables autograd for the whole run — otherwise an inner
``loss.backward()`` dies under the executor's inference mode; restoring autograd is
harmless for an inner evaluator), and it FORWARDS the executor-injected progress
callback to whichever branch is running (:func:`_run`).
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from annotated_types import Interval
from confluid import configurable, flow
from confluid.fluid import Fluid
from loggair import get_logger
from typing_extensions import Annotated

from recordstream.runnable import ProgressReporting, TorchRunner

logger = get_logger(__name__)

#: How ``recordstream run --dry-run`` prints a plan: lines for a person, one JSON line for the step that planned it.
PlanFormat = Literal["text", "json"]

#: What :class:`RunConfig` runs: this interpreter's ``recordstream run``, so a workflow's steps run in the
#: environment the workflow itself runs in. ``-P`` keeps the run folder off the import path, as the ``recordstream``
#: command does: a folder named like a package (a project's checkout, its ``*.egg-info``) never shadows the installed
#: one, so ``pkg://`` and every import resolve wherever the workflow is started.
_RUNNER = [sys.executable, "-P", "-m", "recordstream.cli", "run"]

#: The flags a :class:`RunConfig` adds to plan its config instead of running it.
_PLAN_FLAGS = ["--dry-run", "--plan-format", "json"]


class WorkflowError(RuntimeError):
    """A workflow refused to go on: what it checks is not there, a step failed, a step is wrongly set up."""


class RequirementNotMet(WorkflowError):
    """A check a step makes before anything runs failed; the message says what to do."""


class StepFailed(WorkflowError):
    """A :class:`RunConfig` step's process ended with a non-zero status, which the workflow ends with too."""

    def __init__(self, message: str, returncode: int) -> None:
        super().__init__(message)
        self.returncode = returncode


@dataclass
class Check:
    """One thing a step checks before it runs: what, whether it holds, and — when it does not — what to do."""

    description: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"description": self.description, "ok": self.ok, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Check":
        return cls(description=str(data["description"]), ok=bool(data["ok"]), detail=str(data.get("detail", "")))


@dataclass
class Plan:
    """What a runnable would do, said without doing it — the answer of its ``plan()`` and of ``--dry-run``.

    JSON-safe (:meth:`to_dict`), so a parent step reads its child process's plan, and a tool returns it as is.

    Attributes:
        title: The step's name as a person reads it.
        checks: What it checks before anything runs.
        actions: What it would change, one line each (``delete /x (88G)``).
        commands: The processes it would start, each an argv.
        details: Numbers a person wants to see (``records``, ``start``, ``to_build``), JSON values.
        steps: The plans of the steps it holds (a :class:`Sequence`'s, a :class:`RunConfig`'s process).
        checked_first: Whether its checks run before the workflow's first step; ``False`` for a step whose inputs an
            earlier step writes, checked when its turn comes.
    """

    title: str
    checks: List[Check] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    commands: List[List[str]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    steps: List["Plan"] = field(default_factory=list)
    checked_first: bool = True

    @property
    def would_stop(self) -> bool:
        """Whether a real run would refuse: a failed check here, or in a step checked before the first runs."""
        return any(not check.ok for check in self.checks) or any(
            step.would_stop for step in self.steps if step.checked_first
        )

    @property
    def first_failure(self) -> Optional[str]:
        """What the first failed check says (its detail, else its description), searched as :attr:`would_stop`."""
        for check in self.checks:
            if not check.ok:
                return check.detail or check.description
        for step in self.steps:
            if step.checked_first and step.would_stop:
                return step.first_failure
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "checks": [check.to_dict() for check in self.checks],
            "actions": list(self.actions),
            "commands": [list(command) for command in self.commands],
            "details": dict(self.details),
            "steps": [step.to_dict() for step in self.steps],
            "checked_first": self.checked_first,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            title=str(data["title"]),
            checks=[Check.from_dict(check) for check in data.get("checks", [])],
            actions=[str(action) for action in data.get("actions", [])],
            commands=[[str(word) for word in command] for command in data.get("commands", [])],
            details=dict(data.get("details", {})),
            steps=[cls.from_dict(step) for step in data.get("steps", [])],
            checked_first=bool(data.get("checked_first", True)),
        )


def plan_of(runnable: Any) -> Plan:
    """``runnable``'s plan: its own ``plan()`` when it has one, else that it would run (all a runner can say)."""
    runnable = _resolve(runnable)
    planner = getattr(runnable, "plan", None)
    if callable(planner):
        plan = planner()
        if isinstance(plan, Plan):
            return plan
    name = type(runnable).__name__
    return Plan(title=name, actions=[f"run {name}"])


def display_command(argv: List[str]) -> str:
    """A command as a person types it: this interpreter's ``recordstream run`` written ``recordstream run``."""
    words = ["recordstream", "run", *argv[len(_RUNNER) :]] if argv[: len(_RUNNER)] == _RUNNER else list(argv)
    return shlex.join(words)


def plan_lines(plan: Plan) -> List[str]:
    """``plan`` as the lines ``--dry-run`` prints: each step numbered, then its checks, actions, commands, numbers
    and the plans it holds, indented beneath it."""
    if not plan.steps:
        return [plan.title, *_plan_body(plan, 6)]
    lines: List[str] = []
    for index, step in enumerate(plan.steps, start=1):
        lines.append(f"[{index}/{len(plan.steps)}] {step.title}")
        lines.extend(_plan_body(step, 6))
    return lines


def _plan_body(plan: Plan, indent: int, failures_said: bool = False) -> List[str]:
    """``plan``'s lines at ``indent``; ``failures_said``: a step above already printed this one's failure (a
    :class:`RunConfig` carries its process's refusal with its hint), so its failed checks are not printed again."""
    pad = " " * indent
    shown = [check for check in plan.checks if check.ok or not failures_said]
    lines = [f"{pad}{'ok' if check.ok else 'FAILED'}: {_check_text(check)}" for check in shown]
    failures_said = failures_said or any(not check.ok for check in plan.checks)
    if not plan.checked_first:
        lines.append(f"{pad}(checked when its turn comes: an earlier step writes what it reads)")
    lines.extend(f"{pad}{action}" for action in plan.actions)
    lines.extend(f"{pad}{display_command(command)}" for command in plan.commands)
    if plan.details:
        lines.append(pad + ", ".join(f"{key}: {value}" for key, value in plan.details.items()))
    for step in plan.steps:
        lines.append(f"{pad}{step.title}:")
        lines.extend(_plan_body(step, indent + 2, failures_said))
    return lines


def _check_text(check: Check) -> str:
    return f"{check.description} — {check.detail}" if check.detail and not check.ok else check.description


def _describe(condition: Any) -> str:
    """What a condition checks, in words: its ``describe()``, else its class name."""
    condition = _resolve(condition)
    describer = getattr(condition, "describe", None)
    if callable(describer):
        return str(describer())
    if condition is None or isinstance(condition, bool):
        return str(condition)
    return type(condition).__name__


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

    Before the first step runs, every step is planned (:func:`plan_of`) and its checks made: a failed one stops the
    workflow there, naming the step, so a long first step never runs ahead of a refusal a later one would make. A
    step whose inputs an earlier step writes is checked when its turn comes (its plan's ``checked_first`` is
    false — :class:`RunConfig`'s ``checked_first: false``). :meth:`plan` is the dry run: every step's plan, nothing
    run.

    Args:
        steps: Runnables (or deferred ``!lazy:`` / ``!class:`` markers) to run in
            order. A ``None`` entry is skipped. An empty list is a no-op.
        only: The names of the steps to run (a step's ``name``, as :class:`RunConfig` has); a step without a name
            always runs. Empty (default) runs every step. A name no step has is refused, listing the names there are.
    """

    def __init__(self, steps: Optional[List[Any]] = None, only: Optional[List[str]] = None) -> None:
        # Partial / zero-arg: store config only; branches are flowed in run().
        self.steps: List[Any] = list(steps) if steps else []
        self.only: List[str] = list(only) if only else []

    def run(self) -> None:
        selected = self._selected()
        total = len(selected)
        plans = [plan_of(step) for step in selected]
        for index, plan in enumerate(plans, start=1):
            if plan.checked_first and plan.would_stop:
                raise RequirementNotMet(f"[{index}/{total}] {plan.title}: {plan.first_failure}")
        for index, step in enumerate(selected, start=1):
            logger.info(f"Sequence step {index}/{total}: {plans[index - 1].title}")
            _run(step, self._progress_callback)

    def plan(self) -> Plan:
        """Every selected step's plan, nothing run — what ``recordstream run <workflow> --dry-run`` prints."""
        return Plan(title=type(self).__name__, steps=[plan_of(step) for step in self._selected()])

    def _selected(self) -> List[Any]:
        """The steps that run, flowed once and cached: every step but ``None`` entries and those ``only`` leaves out."""
        for index, step in enumerate(self.steps):
            self.steps[index] = _resolve(step)  # cache the flowed step so we only flow once
        steps = [step for step in self.steps if step is not None]
        if not self.only:
            return steps
        names = [str(getattr(step, "name", "") or "") for step in steps]
        unknown = [name for name in self.only if name not in names]
        if unknown:
            raise WorkflowError(
                f"Sequence: only names {', '.join(unknown)} — the steps are named: "
                f"{', '.join(name for name in names if name) or '(none)'}"
            )
        return [step for step, name in zip(steps, names) if not name or name in self.only]


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

    def describe(self) -> str:
        return f"{self.path} exists"


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

    def describe(self) -> str:
        return f"not: {_describe(self.condition)}"


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

    def describe(self) -> str:
        return "all of: " + "; ".join(_describe(c) for c in self.conditions)


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

    def describe(self) -> str:
        return "any of: " + "; ".join(_describe(c) for c in self.conditions)


@configurable
class EnvSet:
    """Predicate: ``True`` iff the environment variable ``name`` is set to some text (an empty value is unset).

    Args:
        name: The environment variable (``DATA_ROOT``).
    """

    def __init__(self, name: str = "") -> None:
        self.name = name

    def __call__(self) -> bool:
        return bool(self.name) and bool(os.environ.get(self.name))

    def describe(self) -> str:
        return f"{self.name} is set"


@configurable
class FreeSpace:
    """Predicate: ``True`` iff the disk ``path`` is on has at least ``gigabytes`` free.

    ``path`` need not exist yet — the folder a run will create — so the free space is read on the nearest folder
    above it that does.

    Args:
        path: The folder whose disk is measured.
        gigabytes: The free space needed, in GB (2^30 bytes).
    """

    def __init__(self, path: str = "", gigabytes: Annotated[float, Interval(ge=0)] = 0) -> None:
        self.path = path
        self.gigabytes = gigabytes

    @property
    def folder(self) -> Path:
        """The nearest folder at or above ``path`` that exists."""
        folder = Path(self.path)
        while not folder.exists() and folder.parent != folder:
            folder = folder.parent
        return folder

    @property
    def free_gigabytes(self) -> float:
        return shutil.disk_usage(self.folder).free / 2**30

    def __call__(self) -> bool:
        return self.free_gigabytes >= self.gigabytes

    def describe(self) -> str:
        return f"{self.gigabytes:g} GB free at {self.folder} ({self.free_gigabytes:.0f} GB free)"


@configurable
class HasFiles:
    """Predicate: ``True`` iff the folder ``path`` holds a file ``pattern`` matches — a run's inputs are there.

    Args:
        path: The folder.
        pattern: A glob, relative to ``path`` (``*.sigmf-meta``; ``**/*.json`` reaches into sub-folders).
    """

    def __init__(self, path: str = "", pattern: str = "*") -> None:
        self.path = path
        self.pattern = pattern

    @property
    def count(self) -> int:
        folder = Path(self.path)
        return sum(1 for found in folder.glob(self.pattern) if found.is_file()) if folder.is_dir() else 0

    def __call__(self) -> bool:
        return self.count > 0

    def describe(self) -> str:
        return f"{self.count} file(s) matching {self.pattern} in {self.path}"


@configurable
class Require:
    """Refuse to go on unless a condition holds — checked by the enclosing :class:`Sequence` before its first step.

    Args:
        condition: The predicate (:class:`EnvSet`, :class:`FreeSpace`, :class:`HasFiles`, :class:`PathExists`, …).
        message: What to do when it does not hold (``source project.bashrc``); empty says what was checked.
        title: The step's name in a plan; empty names it by what it checks.
    """

    def __init__(self, condition: Any = None, message: str = "", title: str = "") -> None:
        self.condition = condition
        self.message = message
        self.title = title

    def plan(self) -> Plan:
        description = _describe(self.condition)
        ok = _evaluate_condition(self.condition)
        return Plan(
            title=self.title or description,
            checks=[Check(description=description, ok=ok, detail="" if ok else self.message)],
        )

    def run(self) -> None:
        if not _evaluate_condition(self.condition):
            raise RequirementNotMet(self.message or _describe(self.condition))


@configurable
class ReplaceFolder:
    """Empty ``path`` once: a folder without ``marker`` is deleted and made again with the marker in it; a folder
    holding the marker is kept — so a run interrupted after this step resumes into what it wrote.

    The marker is what makes it safe to run twice; give each layout its own (``.store_v2``), so a folder of an older
    layout is replaced and never resumed into. A filesystem root, the home folder and an empty path are refused, and
    so is a folder whose parent does not exist (a mistyped path is not created) unless ``parents`` says to make it.

    Args:
        path: The folder.
        marker: The file inside it that says this step made it.
        parents: Make the folders above ``path`` that are missing — for a default folder the workflow owns; false
            (default) refuses a missing parent, so a mistyped path is not created.
    """

    def __init__(self, path: str = "", marker: str = ".replaced", parents: bool = False) -> None:
        self.path = path
        self.marker = marker
        self.parents = parents

    def plan(self) -> Plan:
        title = f"replace {self.path}"
        refusal = self._refusal()
        if refusal:
            return Plan(
                title=title, checks=[Check(description=f"{self.path} may be replaced", ok=False, detail=refusal)]
            )
        folder = Path(self.path)
        if (folder / self.marker).is_file():
            return Plan(title=title, actions=[f"keep {folder}: it holds its marker {self.marker}"])
        actions = [f"delete {folder} ({_size(folder)})"] if folder.exists() else []
        if not folder.parent.is_dir():
            actions.append(f"create {folder.parent}")
        return Plan(title=title, actions=[*actions, f"create {folder} and its marker {self.marker}"])

    def run(self) -> None:
        refusal = self._refusal()
        if refusal:
            raise WorkflowError(refusal)
        folder = Path(self.path)
        if (folder / self.marker).is_file():
            logger.info(f"Keeping {folder}: it holds its marker {self.marker}.")
            return
        if folder.exists():
            logger.info(f"Deleting {folder} ({_size(folder)}): it does not hold the marker {self.marker}.")
            shutil.rmtree(folder)
        folder.mkdir(parents=self.parents)
        (folder / self.marker).touch()

    def _refusal(self) -> Optional[str]:
        """Why this folder may not be replaced, or ``None``."""
        if not self.path or not self.marker:
            return f"ReplaceFolder: refusing to replace path {self.path!r} with marker {self.marker!r} — set both"
        if "${" in self.path:
            return (
                f"ReplaceFolder: refusing to replace {self.path}: it holds a placeholder nothing filled — a config key "
                "read in a string is dotted (${store.root}); a plain ${root} reads the environment variable root"
            )
        folder = Path(self.path)
        resolved = folder.expanduser().resolve()
        if resolved.parent == resolved or resolved == Path.home().resolve():
            return f"ReplaceFolder: refusing to empty {folder}: it is a filesystem root or the home folder"
        if folder.exists() and not folder.is_dir():
            return f"ReplaceFolder: refusing to replace {folder}: it is a file, not a folder"
        if not self.parents and not folder.parent.is_dir():
            return (
                f"ReplaceFolder: {folder}: its folder {folder.parent} does not exist — make it, set another path, "
                "or set parents: true"
            )
        return None


def _size(folder: Path) -> str:
    """A folder's size as ``du -sh`` writes it (``88G``)."""
    total = float(sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()))
    for unit in ("B", "K", "M", "G"):
        if total < 1024:
            return f"{total:.0f}{unit}" if unit == "B" else f"{total:.1f}{unit}"
        total /= 1024
    return f"{total:.1f}T"


def _execute(argv: List[str], capture: bool) -> Tuple[int, str]:
    """Run ``argv`` in the running folder and environment: its exit status, and — when ``capture`` — what it printed
    (stdout and stderr together, so a plan's refusal reaches the step that asked); otherwise it prints as it goes."""
    if capture:
        finished = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return finished.returncode, finished.stdout or ""
    return subprocess.run(argv, check=False).returncode, ""


def _setting_text(value: Any) -> str:
    """A setting's value as ``recordstream run`` reads it back: text as written, ``true``/``false``, ``[a, b]``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_setting_text(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_setting_text(item)}" for key, item in value.items()) + "}"
    return str(value)


def _what_went_wrong(output: str) -> str:
    """The line of a process's output that says why it stopped: its last ``Error:`` line, else the last exception
    line of a traceback (``ParserError: …``), else nothing."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("Error:"):
            return line[len("Error:") :].strip()
    for line in reversed(lines):
        head = line.split(":", 1)[0]
        if head.endswith(("Error", "Exception")) and " " not in head:
            return line
    return ""


def _plan_in(output: str) -> Optional[Plan]:
    """The plan a ``--plan-format json`` dry run printed: its last line that reads as one."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return Plan.from_dict(json.loads(line))
            except (ValueError, KeyError, TypeError):
                continue
    return None


@configurable
class RunConfig:
    """Run one config in its own process — ``recordstream run <config> --scope <s> key=value`` — a step of a workflow.

    A process of its own frees the step's memory when it ends, ends with the step's own status, and is the command a
    person could type: :meth:`plan` shows it, and asks the process's dry run what it would do. A failed run stops the
    workflow with its exit status (:class:`StepFailed`).

    Args:
        config: The config, handed to ``recordstream run`` as written: a relative path is the run folder's,
            ``pkg://<distribution>/<path>`` a package's.
        scopes: The scopes it runs with (``--scope`` each).
        settings: Values it runs with, ``key: value`` each (``store_root: /elsewhere``, ``runnable.workers: auto``).
        name: The name ``Sequence(only=...)`` picks it by; empty: always run.
        title: Its name in a plan; empty: ``name``, else ``config``.
        hint: The command that writes its inputs, added to a refusal (``signal-detection iq val wlan``).
        checked_first: Whether its inputs are checked before the workflow's first step; false for a step whose
            inputs an earlier step writes.
    """

    def __init__(
        self,
        config: str = "",
        scopes: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None,
        name: str = "",
        title: str = "",
        hint: str = "",
        checked_first: bool = True,
    ) -> None:
        self.config = config
        self.scopes = scopes
        self.settings = settings
        self.name = name
        self.title = title
        self.hint = hint
        self.checked_first = checked_first

    @property
    def command(self) -> List[str]:
        """The command this step runs."""
        scopes = [word for scope in (self.scopes or []) for word in ("--scope", str(scope))]
        settings = [f"{key}={_setting_text(value)}" for key, value in (self.settings or {}).items()]
        return [*_RUNNER, self.config, *scopes, *settings]

    @property
    def label(self) -> str:
        return self.title or self.name or self.config

    def plan(self) -> Plan:
        if not self.config:
            return Plan(title=self.label or "RunConfig", checks=[Check("a config is set", False, "set config")])
        command = self.command
        returncode, output = _execute([*command, *_PLAN_FLAGS], capture=True)
        child = _plan_in(output)
        plan = Plan(title=self.label, commands=[command], checked_first=self.checked_first)
        if child is None:
            said = _what_went_wrong(output)
            detail = f"its dry run ended with exit {returncode}{': ' + said if said else ''}"
            plan.checks.append(
                Check(f"{self.label} can be planned", False, f"{detail} (see: {display_command(command)} --dry-run)")
            )
            return plan
        plan.steps.append(child)
        if child.would_stop:
            hint = f" — run: {self.hint}" if self.hint else ""
            plan.checks.append(Check(f"{self.label}: its inputs", False, f"{child.first_failure}{hint}"))
        return plan

    def run(self) -> None:
        if not self.config:
            raise WorkflowError("RunConfig: no config to run — set config")
        command = self.command
        returncode, _ = _execute(command, capture=False)
        if returncode:
            raise StepFailed(f"{self.label}: {display_command(command)} ended with exit {returncode}", returncode)


__all__ = [
    "Sequence",
    "Conditional",
    "Switch",
    "PathExists",
    "Not",
    "AllOf",
    "AnyOf",
    "EnvSet",
    "FreeSpace",
    "HasFiles",
    "Require",
    "ReplaceFolder",
    "RunConfig",
    "Check",
    "Plan",
    "PlanFormat",
    "WorkflowError",
    "RequirementNotMet",
    "StepFailed",
    "plan_of",
    "plan_lines",
    "display_command",
]
