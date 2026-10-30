"""``recordstream run <config> --scope <name>`` refuses a name the config does not declare, before anything runs.

``--scope vla`` (meaning ``val``) used to run the config as if no scope were given — a typo wrote the train split.
confluid refuses the bare name (its architecture §1, amendment 2026-10-30); these tests hold the command to it: one
``Error:`` line naming the config and its scopes, exit 1, nothing done — and a workflow whose step passes a scope its
config lacks stops before its first step, its dry run saying why.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

#: A process that hangs fails the test this long after it starts, naming the command.
_DEADLINE_S = 120

_SPLIT = """\
folder: train.done
val: !scope:val
  folder: val.done
runnable: !class:recordstream.workflow.ReplaceFolder { path: !ref:folder }
"""


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


def _errors(finished: "subprocess.CompletedProcess[str]") -> List[str]:
    return [line for line in (finished.stdout + finished.stderr).splitlines() if line.startswith("Error:")]


def test_a_scope_the_config_does_not_declare_is_one_error_line_and_nothing_runs(tmp_path: Path) -> None:
    config = _write(tmp_path, "split.yaml", _SPLIT)
    finished = _run(tmp_path, "split.yaml", "--scope", "vla")
    assert finished.returncode == 1
    assert len(_errors(finished)) == 1
    expected = f"Error: No scope block declares 'vla' (the scopes of {config.resolve()}: val)"
    said = finished.stdout + finished.stderr
    assert "".join(expected.split()) in "".join(said.split())  # the console wraps a long path onto a second line
    assert "Traceback" not in finished.stderr
    assert not (tmp_path / "train.done").exists()


def test_a_declared_scope_runs_as_before(tmp_path: Path) -> None:
    _write(tmp_path, "split.yaml", _SPLIT)
    finished = _run(tmp_path, "split.yaml", "--scope", "val")
    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "val.done").is_dir() and not (tmp_path / "train.done").exists()


def test_a_keyed_scope_of_a_dimension_the_config_lacks_still_runs(tmp_path: Path) -> None:
    """What ``--framework <name>`` passes: a config with one backend has no ``framework`` blocks."""
    _write(tmp_path, "split.yaml", _SPLIT)
    finished = _run(tmp_path, "split.yaml", "--scope", "framework=torch")
    assert finished.returncode == 0, finished.stderr
    assert (tmp_path / "train.done").is_dir()


@pytest.fixture()
def workflow(tmp_path: Path) -> Path:
    _write(tmp_path, "split.yaml", _SPLIT)
    _write(tmp_path, "first.yaml", "runnable: !class:recordstream.workflow.ReplaceFolder { path: first.done }\n")
    _write(
        tmp_path,
        "workflow.yaml",
        """
runnable: !class:recordstream.workflow.Sequence
  steps:
    - !class:recordstream.workflow.RunConfig { config: first.yaml, name: first }
    - !class:recordstream.workflow.RunConfig { config: split.yaml, name: split, scopes: [vla] }
""",
    )
    return tmp_path


def test_a_workflow_step_passing_an_undeclared_scope_stops_the_workflow_before_its_first_step(workflow: Path) -> None:
    finished = _run(workflow, "workflow.yaml")
    assert finished.returncode == 1
    assert "No scope block declares 'vla'" in finished.stdout + finished.stderr
    assert not (workflow / "first.done").exists() and not (workflow / "train.done").exists()


def test_the_dry_run_of_that_workflow_says_the_step_would_stop(workflow: Path) -> None:
    finished = _run(workflow, "workflow.yaml", "--dry-run")
    assert finished.returncode == 1
    assert "No scope block declares 'vla'" in finished.stdout
    assert finished.stdout.splitlines()[-1].startswith("DRY RUN: a real run would stop")
    assert not (workflow / "first.done").exists()
