"""``run`` is recordstream's default command: ``recordstream <config.yaml>`` is ``recordstream run <config.yaml>``.

Each test writes a config into a temporary folder and runs the real command there, with and without the ``run``
token, so the two spellings are compared on what they do — the folder a run makes, the plan a dry run prints.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

#: A process that hangs fails the test this long after it starts, naming the command.
_DEADLINE_S = 120


def _recordstream(folder: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    command: List[str] = [sys.executable, "-P", "-m", "recordstream.cli", *args]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    try:
        return subprocess.run(
            command, cwd=folder, env=environment, capture_output=True, text=True, timeout=_DEADLINE_S, check=False
        )
    except subprocess.TimeoutExpired as expired:
        pytest.fail(f"{' '.join(command)} did not end within {_DEADLINE_S} s: {expired}")


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    # A run that leaves a trace: the folder it makes.
    (tmp_path / "touch.yaml").write_text("runnable: !class:recordstream.workflow.ReplaceFolder { path: a.done }\n")
    (tmp_path / "workflow.yaml").write_text(
        """
runnable: !class:recordstream.workflow.Sequence
  steps:
    - !class:recordstream.workflow.RunConfig { config: touch.yaml, name: a }
"""
    )
    return tmp_path


def test_a_config_runs_without_the_run_token(folder: Path) -> None:
    finished = _recordstream(folder, "touch.yaml")
    assert finished.returncode == 0, finished.stderr
    assert (folder / "a.done").is_dir()


def test_the_run_token_still_runs_the_config(folder: Path) -> None:
    finished = _recordstream(folder, "run", "touch.yaml")
    assert finished.returncode == 0, finished.stderr
    assert (folder / "a.done").is_dir()


def test_a_setting_on_the_command_line_reaches_the_runnable_without_the_run_token(folder: Path) -> None:
    finished = _recordstream(folder, "touch.yaml", "runnable.path=b.done")
    assert finished.returncode == 0, finished.stderr
    assert (folder / "b.done").is_dir() and not (folder / "a.done").exists()


def test_the_dry_run_prints_the_same_plan_with_and_without_the_run_token(folder: Path) -> None:
    bare = _recordstream(folder, "workflow.yaml", "--dry-run")
    named = _recordstream(folder, "run", "workflow.yaml", "--dry-run")
    assert bare.returncode == named.returncode == 0, bare.stderr + named.stderr
    assert bare.stdout == named.stdout
    assert bare.stdout.splitlines()[0] == "DRY RUN: nothing is changed"
    assert not (folder / "a.done").exists()


def test_a_config_named_run_is_run_when_written_with_its_suffix(folder: Path) -> None:
    (folder / "run.yaml").write_text("runnable: !class:recordstream.workflow.ReplaceFolder { path: r.done }\n")
    finished = _recordstream(folder, "run.yaml")
    assert finished.returncode == 0, finished.stderr
    assert (folder / "r.done").is_dir()


def test_no_arguments_prints_the_help_and_runs_nothing(folder: Path) -> None:
    finished = _recordstream(folder)
    assert finished.returncode == 0, finished.stderr
    assert "run" in finished.stdout and "Global Options" in finished.stdout
    assert "was not bound" not in finished.stdout + finished.stderr
