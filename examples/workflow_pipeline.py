"""A resume-safe train → evaluate workflow — ONE Confluid document of runnables.

The compelling case for the workflow combinators: a pipeline you can re-run after a crash
(or a second `recordstream run`) that SKIPS the work whose artifact already exists and
carries on — *memoise and continue*, expressed declaratively:

1. ``Sequence`` drives the stages in order (the workflow itself).
2. ``Conditional`` + ``PathExists`` guard the expensive stage: train ONLY when the
   checkpoint is missing. On a cache hit the branch is not just skipped — wired
   ``!lazy:``, it is **never even constructed** (no model, no dataset materialised).
3. ``Switch`` picks the report format off a plain config value — one key a CLI override
   can flip (``--select text``) without touching the workflow shape.

This script runs the SAME document twice and proves both guarantees: the second pass
evaluates again but does not retrain, and the trainer class records exactly ONE
construction across both passes.

Standalone, zero-arg, exit 0 (CI runs every ``examples/*.py``).
"""

import tempfile
from pathlib import Path

import confluid
from confluid import configurable
from confluid.fluid import Fluid

# ---------------------------------------------------------------------------------------
# Three tiny stub runnables (a runnable = any object with a no-arg run()). Real pipelines
# put a trainer / DatasetProcessor here; stubs keep the example self-contained and fast.
# ---------------------------------------------------------------------------------------


@configurable
class TrainModel:
    """Stub trainer: 'trains' by writing the checkpoint artifact.

    Args:
        ckpt: Path the checkpoint artifact is written to.
    """

    #: Instrumentation for THIS example: counts constructions, to prove the ``!lazy:``
    #: guarantee (an unchosen branch is never built). Not a pattern for real runnables.
    builds = 0

    def __init__(self, ckpt: str = "") -> None:
        type(self).builds += 1
        self.ckpt = ckpt

    def run(self) -> None:
        Path(self.ckpt).write_text("weights")
        print(f"  [train]    wrote {Path(self.ckpt).name}")


@configurable
class Evaluate:
    """Stub evaluator: reads the checkpoint, writes a report in the given format.

    Args:
        ckpt: Checkpoint artifact to 'evaluate'.
        report: Path the report is written to.
        fmt: Report format tag written into the file.
    """

    def __init__(self, ckpt: str = "", report: str = "", fmt: str = "text") -> None:
        self.ckpt = ckpt
        self.report = report
        self.fmt = fmt

    def run(self) -> None:
        weights = Path(self.ckpt).read_text()
        Path(self.report).write_text(f"[{self.fmt}] accuracy of {weights!r}: 0.93")
        print(f"  [evaluate] wrote {Path(self.report).name} ({self.fmt})")


def workflow_yaml(work: Path) -> str:
    """The whole pipeline — stages, the cache guard, AND the format switch — as ONE document."""
    return f"""
runnable: !class:recordstream.workflow.Sequence
  steps:
    # Stage 1 — the expensive stage, guarded: train ONLY when the checkpoint is missing.
    # On a cache hit the !lazy: branch is never even BUILT (no model materialised).
    - !class:recordstream.workflow.Conditional
      condition: !class:recordstream.workflow.PathExists
        path: {work / "model.ckpt"}
      if_true: null                       # cache hit  -> skip, Sequence continues
      if_false: !lazy:TrainModel  # @configurable classes resolve by registered NAME
        ckpt: {work / "model.ckpt"}

    # Stage 2 — always runs; the report FORMAT is a Switch on a plain config value
    # (override from a CLI with --select text — the workflow shape never changes).
    - !class:recordstream.workflow.Switch
      select: json
      cases:
        json: !lazy:Evaluate
          ckpt: {work / "model.ckpt"}
          report: {work / "report.json"}
          fmt: json
        text: !lazy:Evaluate
          ckpt: {work / "model.ckpt"}
          report: {work / "report.txt"}
          fmt: text
"""


def run_document(path: Path) -> None:
    """What ``recordstream run <config>`` does: bind the top-level ``runnable:`` and run it."""
    loaded = confluid.load(str(path))
    runnable = loaded["runnable"]
    if isinstance(runnable, Fluid):
        runnable = confluid.flow(runnable)
    runnable.run()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        doc = work / "workflow.yaml"
        doc.write_text(workflow_yaml(work))

        print("--- pass 1: nothing cached -> trains, then evaluates ---")
        run_document(doc)
        assert (work / "model.ckpt").exists() and (work / "report.json").exists()
        assert TrainModel.builds == 1

        (work / "report.json").unlink()  # so pass 2 visibly re-evaluates

        print("--- pass 2: checkpoint exists -> SKIPS training, evaluates again ---")
        run_document(doc)
        assert (work / "report.json").exists()

        # The two guarantees: no retrain (artifact-guarded), and the unchosen !lazy:
        # branch was never CONSTRUCTED on pass 2 — builds stayed at one.
        assert TrainModel.builds == 1, f"trainer was rebuilt: {TrainModel.builds}"
        print("\ntrainer constructions across both passes:", TrainModel.builds)
        print("OK")


if __name__ == "__main__":
    main()
