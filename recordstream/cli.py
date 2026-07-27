"""The ``recordstream`` CLI — a generic runner for any Confluid-wired runnable.

``recordstream run <config.yaml>`` loads a Confluid YAML that binds a *runnable*
object (anything exposing a no-arg ``run()``) under the top-level ``runnable:``
key, flows it under the active context (so nested ``!ref:`` markers resolve), and
calls ``run()``. This is the single entry point that replaces bespoke per-verb
CLIs: a training run, an evaluation, a dataset conversion, or a whole
:mod:`~recordstream.workflow` are all just runnables — the ``!class:`` the YAML roots
on decides what happens.

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


@app.script_command(flow_mode="auto")
def run(runnable: Any) -> None:
    """Run any Confluid-instantiated object that exposes ``.run()``.

    The YAML config binds the object under the top-level ``runnable:`` key.
    ``flow_mode="auto"`` deep-flows it under Confluid's active context so nested
    ``!ref:`` markers resolve against the loaded YAML's top-level keys.
    """
    if runnable is None:
        logger.error("'runnable' was not bound — provide one under 'runnable:' in your YAML.")
        return

    from confluid import flow
    from confluid.fluid import Fluid

    if isinstance(runnable, Fluid):
        runnable = flow(runnable)

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
