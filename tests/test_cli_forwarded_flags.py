"""``recordstream run <workflow.yaml> --key value``: a key the workflow does not use itself is handed to every run it
starts, so a setting of a run's processor (``--path`` of the folder step here, ``--show_progress`` of a dataset
processor) is typed once, on the workflow's command line.

Each test runs the real command in a temporary folder whose configs use only recordstream's own steps (the pattern of
``tests/test_cli_dry_run.py``): the folder step's ``path`` stands for any setting of the runnable a step runs.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

#: A process that hangs fails the test this long after it starts, naming the command.
_DEADLINE_S = 120


def _write(folder: Path, name: str, text: str) -> None:
    (folder / name).write_text(text)


def _run(folder: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    command: List[str] = [sys.executable, "-P", "-m", "recordstream.cli", "run", *args]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    try:
        return subprocess.run(
            command, cwd=folder, env=environment, capture_output=True, text=True, timeout=_DEADLINE_S, check=False
        )
    except subprocess.TimeoutExpired as expired:
        pytest.fail(f"{' '.join(command)} did not end within {_DEADLINE_S} s: {expired}")


def _commands(stdout: str) -> List[str]:
    """The ``recordstream run`` lines a dry run prints, one per run."""
    return [line.strip() for line in stdout.splitlines() if line.strip().startswith("recordstream run ")]


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    _write(tmp_path, "touch_a.yaml", "runnable: !class:recordstream.workflow.ReplaceFolder { path: a.done }\n")
    _write(tmp_path, "touch_b.yaml", "runnable: !class:recordstream.workflow.ReplaceFolder { path: b.done }\n")
    _write(
        tmp_path,
        "workflow.yaml",
        """
runnable: !class:recordstream.workflow.Sequence
  steps:
    - !class:recordstream.workflow.RunConfig { config: touch_a.yaml, name: a }
    - !class:recordstream.workflow.RunConfig { config: touch_b.yaml, name: b, forward: false }
    - !class:recordstream.workflow.RunConfig { config: touch_b.yaml, name: c }
""",
    )
    # a workflow with a knob of its own: a top-level mapping the steps are fed from
    _write(
        tmp_path,
        "knob.yaml",
        """
wf:
  only: []
  folder: out
runnable: !class:recordstream.workflow.Sequence
  only: !ref:wf.only
  steps:
    - !class:recordstream.workflow.RunConfig
      config: touch_a.yaml
      name: a
      settings: { path: "${wf.folder}/a.done" }
""",
    )
    for made in ("out", "zz"):  # the folders the knob can name: a folder step refuses one that is not there
        (tmp_path / made).mkdir()
    return tmp_path


def test_a_key_the_workflow_does_not_use_reaches_every_run_that_takes_forwards(folder: Path) -> None:
    finished = _run(folder, "workflow.yaml", "--path", "elsewhere.done", "--dry-run")
    assert finished.returncode == 0, finished.stderr
    assert _commands(finished.stdout) == [
        "recordstream run touch_a.yaml path=elsewhere.done",
        "recordstream run touch_b.yaml",  # `forward: false`
        "recordstream run touch_b.yaml path=elsewhere.done",
    ]


def test_the_run_gets_the_key_the_dry_run_showed(folder: Path) -> None:
    finished = _run(folder, "workflow.yaml", "--path", "elsewhere.done")
    assert finished.returncode == 0, finished.stderr
    assert (folder / "elsewhere.done").is_dir()
    assert (folder / "b.done").is_dir()  # the run that forwards nothing kept its own folder
    assert not (folder / "a.done").exists()


def test_a_typed_key_comes_after_the_steps_settings_and_wins(folder: Path) -> None:
    finished = _run(folder, "knob.yaml", "--path", "q.done", "--dry-run")
    assert finished.returncode == 0, finished.stderr
    assert _commands(finished.stdout) == ["recordstream run touch_a.yaml path=out/a.done path=q.done"]
    assert "replace q.done" in finished.stdout


def test_the_workflows_own_knob_is_the_workflows_and_is_not_handed_on(folder: Path) -> None:
    finished = _run(folder, "knob.yaml", "--wf.folder", "zz", "--dry-run")
    assert finished.returncode == 0, finished.stderr
    assert _commands(finished.stdout) == ["recordstream run touch_a.yaml path=zz/a.done"]


def test_the_commands_own_options_and_the_workflows_own_parameters_are_not_handed_on(folder: Path) -> None:
    finished = _run(folder, "knob.yaml", "--dry_run", "--plan_format", "text", "runnable.only=[a]")
    assert finished.returncode == 0, finished.stderr
    assert _commands(finished.stdout) == ["recordstream run touch_a.yaml path=out/a.done"]


def test_several_keys_are_handed_on_in_the_order_typed(folder: Path) -> None:
    finished = _run(folder, "workflow.yaml", "--path", "p.done", "--zzz", "1", "--aaa", "2", "--dry-run")
    # the folder step takes none of `zzz` and `aaa`: they ride along, in order, for the run to judge
    assert _commands(finished.stdout)[0] == "recordstream run touch_a.yaml path=p.done zzz=1 aaa=2"
