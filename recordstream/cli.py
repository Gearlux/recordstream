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

    recordstream run convert.yaml
"""

from typing import Any

from liquifai import LiquifyApp
from loggair import get_logger

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
    ``materialize(node, context=document)``. Nested stubs still ride the normal deferred
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
    from confluid import flow, materialize
    from confluid.fluid import Fluid
    from liquifai.context import get_context

    if not isinstance(runnable, Fluid):
        return runnable

    context = get_context()
    document = getattr(context, "config_data", None) if context is not None else None
    if isinstance(document, dict):
        return materialize(runnable, context=document)
    return flow(runnable)


@app.script_command(flow_mode="manual")
def run(runnable: Any) -> None:
    """Run any Confluid-instantiated object that exposes ``.run()``.

    The YAML config binds the object under the top-level ``runnable:`` key.
    ``flow_mode="manual"`` because :func:`materialize_runnable` does the building
    itself — liquifai's auto deep-flow would bare-flow the node and drop every
    broadcast top-level key (see that function's docstring).
    """
    if runnable is None:
        logger.error("'runnable' was not bound — provide one under 'runnable:' in your YAML.")
        return

    runnable = materialize_runnable(runnable)

    label = runnable.__class__.__name__
    run_method = getattr(runnable, "run", None)
    if not callable(run_method):
        logger.error(f"The injected runnable ({label}) does not implement 'run()'.")
        return
    logger.info(f"recordstream running: {label}")
    run_method()


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
