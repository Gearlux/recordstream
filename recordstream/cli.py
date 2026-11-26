"""The ``recordstream`` CLI — a generic runner for any Confluid-wired runnable.

``recordstream run <config.yaml>`` loads a Confluid YAML that binds a *runnable*
object (anything exposing a no-arg ``run()``) under the top-level ``runnable:``
key, materializes it against the whole document (so the flat config's top-level
keys broadcast into its constructor), and calls ``run()``. This is the single
entry point that replaces bespoke per-verb CLIs: a training run, an evaluation, a
dataset conversion, or a whole :mod:`~recordstream.workflow` are all just
runnables — the ``!class:`` the YAML roots on decides what happens.

Example::

    # convert.yaml
    runnable: !class:recordstream.processing.DatasetProcessor
      stream: !class:recordstream.Stream { source: !class:my.Source(), ops: [...] }
      sink: !class:recordstream.storage.HDF5Sink { path: out.h5 }

    recordstream convert.yaml        # run is the default command
    recordstream run convert.yaml    # the same run, the command named
"""

import inspect
import json
import sys
from typing import Any

from confluid import collect_report
from confluid.report import ConfigurationReport
from liquifai import LiquifyApp
from loggair import get_logger

from recordstream.workflow import Plan, PlanFormat, StepFailed, WorkflowError, plan_lines, plan_of

logger = get_logger(__name__)

app = LiquifyApp(name="recordstream")

__all__ = ["app", "main", "materialize_runnable", "run"]


def materialize_runnable(runnable: Any) -> Any:
    """Build a bound ``runnable:`` node **with the flat config's keys broadcast in**.

    Why this exists instead of a bare ``flow()``: broadcasting (a top-level YAML key
    injecting into the same-named constructor parameter) only happens when a Fluid is
    built *against the document it came from*. Liquifai's DI does that for a command
    parameter annotated with a **configurable class** — it materializes the block with
    ``context=<the loaded config>``. A generic runner cannot use such an annotation: its
    parameter is ``Any`` precisely because the runnable is polymorphic, so DI falls back
    to handing over the raw Fluid and deep-flowing it with no context, and every
    top-level sibling is silently dropped.

    Silently is the operative word — a dropped ``train_set:`` surfaces later as an empty
    dataset, and a dropped ``max_epochs: 3`` does not surface at all: the run proceeds on
    the constructor default and looks configured.

    So this reaches back to the loaded document through liquifai's context and calls
    ``load(node, context=document)``. Nested stubs still ride the normal deferred
    path — a ``!lazy:`` marker stays deferred for the runnable to flow at run time.

    **liquifai 0.1.1 fixes this at its own layer** (``di.deep_flow`` now takes the
    document), so once that release is on PyPI and the floors are raised, the runners can
    go back to ``flow_mode="auto"`` and this helper can shrink away. It stays until then
    because generated CI clones each local-sourced dependency's ``main`` — reverting early
    would silently regress to dropping every top-level key. Keeping it is harmless
    meanwhile: building against the document is correct under either liquifai.

    Args:
        runnable: The value liquifai bound to the command's ``runnable`` parameter —
            typically a :class:`~confluid.fluid.Fluid`, but a live object (already built)
            passes through untouched.

    Returns:
        The built runnable. Falls back to a plain ``flow()`` when there is no liquifai
        context or its config is not a mapping (a YAML whose root is a single ``!class:``
        document has no siblings to broadcast, so there is nothing to lose).

    Example::

        @app.script_command(flow_mode="manual")
        def run(runnable: Any) -> None:
            runnable = materialize_runnable(runnable)
            runnable.run()
    """
    from confluid import flow, load
    from confluid.fluid import Fluid
    from liquifai.context import get_context

    if not isinstance(runnable, Fluid):
        return runnable

    context = get_context()
    document = getattr(context, "config_data", None) if context is not None else None
    if isinstance(document, dict):
        return load(runnable, context=document)
    return flow(runnable)


# flow_mode="manual" because materialize_runnable() does the building itself: liquifai's auto deep-flow would
# bare-flow the node and drop every broadcast top-level key (see that function's docstring).
@app.script_command(flow_mode="manual", default=True)
def run(runnable: Any, dry_run: bool = False, plan_format: PlanFormat = "text") -> None:
    """Run the object a config binds under its top-level `runnable:` key.

        recordstream run <config.yaml> [key=value ...]
        recordstream run <config.yaml> --dry_run                      # what a run would do, nothing changed
        recordstream run <config.yaml> --dry_run --plan_format json   # the same plan as one JSON line

    `run` is the default command: `recordstream <config.yaml>` is the same run. A step that refuses or fails prints
    one `Error:` line and exits 1, or with the failed step's own status (`--debug` keeps the traceback).

    Args:
        runnable: The object bound under `runnable:`.
        dry_run: Run nothing; print each step's checks, what it would delete or create, and the commands it would
            start. Exits 1 when a real run would stop.
        plan_format: How --dry_run prints the plan: text (readable lines) or json (one line, for a script or a
            parent step). Only used with --dry_run: `--plan_format json` alone is refused, so asking for the
            plan never starts a real run.
    """
    if plan_format != "text" and not dry_run:
        print(
            f"Error: --plan_format only applies with --dry_run; add --dry_run to print the plan "
            f"(--plan_format {plan_format} alone ran nothing)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    if runnable is None:
        logger.error("'runnable' was not bound — provide one under 'runnable:' in your YAML.")
        return

    with collect_report() as report:
        runnable = materialize_runnable(runnable)
    _forward_unused_flags(runnable, report)

    label = runnable.__class__.__name__
    if dry_run:
        try:
            plan = plan_of(runnable)
        except WorkflowError as refused:  # a workflow that cannot be planned is refused as a real run is
            if _debugging():
                raise
            print(f"Error: {refused}", file=sys.stderr, flush=True)
            raise SystemExit(1) from refused
        _print_plan(plan, plan_format)
        return
    run_method = getattr(runnable, "run", None)
    if not callable(run_method):
        logger.error(f"The injected runnable ({label}) does not implement 'run()'.")
        return
    logger.info(f"recordstream running: {label}")
    try:
        run_method()
    except WorkflowError as stopped:
        if _debugging():
            raise
        print(f"Error: {stopped}", file=sys.stderr, flush=True)
        raise SystemExit(stopped.returncode if isinstance(stopped, StepFailed) else 1) from stopped


def _forward_unused_flags(runnable: Any, report: ConfigurationReport) -> None:
    """Say what became of the command-line keys the runnable did not take: hand them on, or warn.

    ``recordstream run workflow.yaml --show_progress`` types a key of the processor a step runs, not of the workflow:
    the workflow's own objects take none of it, so confluid's report (:func:`liquifai.overrides.unused_overrides`)
    says it matched nothing. A runnable with ``forward_flags`` (:class:`~recordstream.workflow.Sequence` and its
    kin) hands those keys to its steps, each run applying them with its own rule — a typo gets that run's own
    warning. Any other runnable has no run to hand them to: each is warned about, as liquifai warns for a command
    that builds its objects in its own body (``Override 'zzz' matched nothing and was ignored``).

    Two kinds of key are never judged. One whose head the document declares (``--standardize.workers``,
    ``--max_files`` beside a ``max_files:`` line) is the document's own knob, whatever the report says. One that is an
    option of this command (``--dry_run``) is the command's. A liquifai older than the verdict leaves every key as it
    was.
    """
    from liquifai.context import get_context

    try:
        from liquifai.overrides import unused_overrides, warn_unused_overrides
    except ImportError:  # a liquifai older than the verdict: nothing is handed on, nothing warned
        return

    context = get_context()
    if context is None or not context.cli_overrides:
        return
    # the keys that are not judged: the document's own, and this command's own (`--dry_run`)
    own = {
        *(context.raw_document if isinstance(context.raw_document, dict) else {}),
        *inspect.signature(run).parameters,
    }
    typed = context.cli_overrides
    flags = {key: typed[key] for key in unused_overrides(typed, report) if key.partition(".")[0] not in own}
    if not flags:
        return
    forward = getattr(runnable, "forward_flags", None)
    if callable(forward):
        logger.debug(f"Handing the runs of {runnable.__class__.__name__} the keys it does not use: {', '.join(flags)}")
        forward(flags)
    else:
        warn_unused_overrides(flags, report, context.config_path)


def _print_plan(plan: Plan, plan_format: PlanFormat) -> None:
    """Print a dry run's plan; exit 1 when a real run would stop (the plan says so)."""
    if plan_format == "json":
        print(json.dumps(plan.to_dict()), flush=True)
    else:
        verdict = (
            f"DRY RUN: a real run would stop — {plan.first_failure}"
            if plan.would_stop
            else "DRY RUN: every step would run"
        )
        print("\n".join(["DRY RUN: nothing is changed", *plan_lines(plan), verdict]), flush=True)
    if plan.would_stop:
        raise SystemExit(1)


def _debugging() -> bool:
    """Whether ``--debug`` is on the command line: then an error keeps its traceback."""
    from liquifai.context import get_context

    context = get_context()
    return bool(context is not None and getattr(context, "debug", False))


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
