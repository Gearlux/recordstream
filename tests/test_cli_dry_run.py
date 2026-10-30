"""``recordstream run <workflow.yaml>`` end to end: the steps run in order, each config in its own process, and
``--dry-run`` prints what a run would do, changes nothing and exits 1 when a run would stop.

Each test writes a small workflow into a temporary folder and runs the real command there. Its configs use only
recordstream's own steps: a step's process keeps the run folder off the import path (``python -P``), so a module
written beside the configs would not import — as for any installed command.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from recordstream.workflow import Plan

#: A process that hangs fails the test this long after it starts, naming the command.
_DEADLINE_S = 120


def _write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text)
    return path


def _run(folder: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    command: List[str] = [sys.executable, "-P", "-m", "recordstream.cli", "run", *args]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    try:
        return subprocess.run(
            command, cwd=folder, env=environment, capture_output=True, text=True, timeout=_DEADLINE_S, check=False
        )
    except subprocess.TimeoutExpired as expired:
        pytest.fail(f"{' '.join(command)} did not end within {_DEADLINE_S} s: {expired}")


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    # A run that leaves a trace: the folder it makes (a.done/ with its marker).
    _write(tmp_path, "touch_a.yaml", "runnable: !class:recordstream.workflow.ReplaceFolder { path: a.done }\n")
    _write(tmp_path, "touch_b.yaml", "runnable: !class:recordstream.workflow.ReplaceFolder { path: b.done }\n")
    _write(tmp_path, "fail.yaml", "runnable: !class:recordstream.workflow.Require { message: this step fails }\n")
    _write(
        tmp_path,
        "workflow.yaml",
        """
needed: needed.txt
runnable: !class:recordstream.workflow.Sequence
  steps:
    - !class:recordstream.workflow.Require
      condition: !class:recordstream.workflow.PathExists { path: !ref:needed }
      message: write needed.txt first
    - !class:recordstream.workflow.RunConfig { config: touch_a.yaml, name: a }
    - !class:recordstream.workflow.RunConfig { config: touch_b.yaml, name: b }
""",
    )
    return tmp_path


def test_a_workflow_runs_each_config_in_turn(folder: Path) -> None:
    (folder / "needed.txt").write_text("")
    finished = _run(folder, "workflow.yaml")
    assert finished.returncode == 0, finished.stderr
    assert (folder / "a.done").is_dir() and (folder / "b.done").is_dir()


def test_only_picks_the_steps_from_the_command_line(folder: Path) -> None:
    (folder / "needed.txt").write_text("")
    finished = _run(folder, "workflow.yaml", "runnable.only=[b]")
    assert finished.returncode == 0, finished.stderr
    assert not (folder / "a.done").exists() and (folder / "b.done").exists()


def test_a_refusal_is_one_error_line_exit_1_and_nothing_runs(folder: Path) -> None:
    finished = _run(folder, "workflow.yaml")
    assert finished.returncode == 1
    errors = [line for line in finished.stderr.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1 and "write needed.txt first" in errors[0]
    assert "Traceback" not in finished.stderr
    assert not (folder / "a.done").exists()


def test_a_step_that_fails_at_its_turn_stops_the_workflow_with_its_exit_status(folder: Path) -> None:
    _write(
        folder,
        "failing.yaml",
        """
runnable: !class:recordstream.workflow.Sequence
  steps:
    - !class:recordstream.workflow.RunConfig { config: fail.yaml, checked_first: false }
    - !class:recordstream.workflow.RunConfig { config: touch_a.yaml }
""",
    )
    finished = _run(folder, "failing.yaml")
    assert finished.returncode == 1
    assert "Error: this step fails" in finished.stderr  # the step's own refusal
    assert "fail.yaml ended with exit 1" in finished.stderr  # and the workflow's
    assert not (folder / "a.done").exists()


def test_the_dry_run_prints_every_step_and_changes_nothing(folder: Path) -> None:
    (folder / "needed.txt").write_text("")
    finished = _run(folder, "workflow.yaml", "--dry-run")
    assert finished.returncode == 0, finished.stderr
    assert "[1/3]" in finished.stdout and "[3/3] b" in finished.stdout
    assert "recordstream run touch_a.yaml" in finished.stdout
    lines = finished.stdout.splitlines()
    assert lines[0] == "DRY RUN: nothing is changed" and lines[-1] == "DRY RUN: every step would run"
    assert not (folder / "a.done").exists() and not (folder / "b.done").exists()


def test_the_dry_run_of_a_workflow_that_would_stop_exits_1_and_says_why(folder: Path) -> None:
    finished = _run(folder, "workflow.yaml", "--dry-run")
    assert finished.returncode == 1
    assert "FAILED: needed.txt exists — write needed.txt first" in finished.stdout
    assert "[2/3] a" in finished.stdout  # it goes on past the step that would stop
    assert finished.stdout.splitlines()[-1] == "DRY RUN: a real run would stop — write needed.txt first"


def test_with_debug_a_refusal_keeps_its_traceback(folder: Path) -> None:
    finished = _run(folder, "workflow.yaml", "--debug")
    assert finished.returncode == 1
    assert "Traceback" in finished.stderr and "RequirementNotMet" in finished.stderr


def test_the_dry_run_as_json_is_one_plan(folder: Path) -> None:
    (folder / "needed.txt").write_text("")
    finished = _run(folder, "workflow.yaml", "--dry-run", "--plan-format", "json")
    assert finished.returncode == 0, finished.stderr
    plan = Plan.from_dict(json.loads(finished.stdout.strip().splitlines()[-1]))
    assert [step.title for step in plan.steps] == ["needed.txt exists", "a", "b"]


def test_the_dry_run_of_a_dataset_run_over_an_empty_folder_names_the_folder(folder: Path) -> None:
    (folder / "recordings").mkdir()
    _write(
        folder,
        "process.yaml",
        """
runnable: !class:recordstream.processing.DatasetProcessor
  stream: !class:recordstream.core.stream.Stream
    source: !class:recordstream.sources.files.FilesSource { root: recordings, pattern: "*.json" }
""",
    )
    finished = _run(folder, "process.yaml", "--dry-run")
    assert finished.returncode == 1
    assert "recordings" in finished.stdout and "FAILED" in finished.stdout


@pytest.mark.parametrize("spelling", [["--plan_format", "json"], ["--plan-format", "json"]])
def test_a_plan_format_without_a_dry_run_is_refused_and_runs_nothing(folder: Path, spelling: List[str]) -> None:
    (folder / "needed.txt").write_text("")
    finished = _run(folder, "workflow.yaml", *spelling)
    assert finished.returncode == 1
    assert "Error: --plan_format only applies with --dry_run; add --dry_run to print the plan" in finished.stderr
    assert not (folder / "a.done").exists(), "the workflow ran"


def test_the_default_plan_format_without_a_dry_run_still_runs(folder: Path) -> None:
    (folder / "needed.txt").write_text("")
    finished = _run(folder, "workflow.yaml", "--plan_format", "text")
    assert finished.returncode == 0, finished.stderr
    assert (folder / "a.done").exists() and (folder / "b.done").exists()


def test_a_plan_format_outside_text_and_json_is_refused(folder: Path) -> None:
    (folder / "needed.txt").write_text("")
    finished = _run(folder, "workflow.yaml", "--dry_run", "--plan_format", "xml")
    assert finished.returncode == 1
    # liquifai prints its refusals on stdout
    said = " ".join((finished.stdout + finished.stderr).split())
    assert "Error: recordstream run: --plan_format must be one of text, json; got 'xml'" in said


@pytest.mark.parametrize("plan_format", ["text", "json"])
def test_the_dry_run_of_a_step_name_the_workflow_lacks_is_one_error_line(folder: Path, plan_format: str) -> None:
    """A workflow that cannot even be planned (``only:`` names a step it does not have) is refused as a real run is:
    one ``Error:`` line, exit 1, no traceback — never a planning traceback."""
    finished = _run(folder, "workflow.yaml", "runnable.only=[nope]", "--dry_run", "--plan_format", plan_format)
    assert finished.returncode == 1
    errors = [line for line in finished.stderr.splitlines() if line.startswith("Error:")]
    assert errors == ["Error: Sequence: only names nope — the steps are named: a, b"], finished.stderr
    assert "Traceback" not in finished.stderr
    assert finished.stdout == ""


def test_with_debug_the_dry_run_of_a_step_name_the_workflow_lacks_keeps_its_traceback(folder: Path) -> None:
    finished = _run(folder, "workflow.yaml", "runnable.only=[nope]", "--dry_run", "--debug")
    assert finished.returncode == 1
    assert "Traceback" in finished.stderr and "WorkflowError" in finished.stderr
