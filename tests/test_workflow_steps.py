"""The workflow's steps: what a shell script does around its runs, as ``@configurable`` runnables of one document.

* :class:`Require` refuses before anything runs (:class:`EnvSet`, :class:`FreeSpace`, :class:`HasFiles`,
  :class:`PathExists` say what they check);
* :class:`ReplaceFolder` empties a folder ONCE — a marker inside it makes a re-run after an interruption resume;
* :class:`RunConfig` runs one config in its own process, ``recordstream run <config> --scope s key=value``;
* :class:`Sequence` checks every step before the first runs, runs only the steps ``only`` names, and plans without
  running (:meth:`plan`, the dry run).

A :class:`RunConfig` process is replaced here by :func:`recordstream.workflow._execute` patched to a fake; the
processes themselves are ``tests/test_cli_dry_run.py``'s.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest
import yaml
from confluid import configurable

from recordstream import workflow
from recordstream.workflow import (
    Check,
    Conditional,
    EnvSet,
    FreeSpace,
    HasFiles,
    Not,
    PathExists,
    Plan,
    ReplaceFolder,
    Require,
    RequirementNotMet,
    RunConfig,
    Sequence,
    StepFailed,
    Switch,
    WorkflowError,
    plan_lines,
    plan_of,
)

_RUN_LOG: List[str] = []


@configurable
class _Step:
    """A runnable that logs its tag; ``name`` makes it selectable by ``Sequence(only=...)``."""

    def __init__(self, tag: str = "", name: str = "") -> None:
        self.tag = tag
        self.name = name

    def run(self) -> None:
        _RUN_LOG.append(self.tag)


@pytest.fixture(autouse=True)
def _clear_log() -> Iterator[None]:
    _RUN_LOG.clear()
    yield
    _RUN_LOG.clear()


class _FakeProcesses:
    """Stands in for the processes :class:`RunConfig` starts: records each command, answers with a code and stdout."""

    def __init__(self) -> None:
        self.commands: List[List[str]] = []
        self.environments: List[Optional[Dict[str, str]]] = []
        self.returncode = 0
        self.plan: Optional[Plan] = Plan(title="DatasetProcessor", details={"records": 3})
        self.output_without_plan = "no plan here\n"

    def __call__(self, argv: List[str], capture: bool, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
        self.commands.append(list(argv))
        self.environments.append(None if env is None else dict(env))
        if capture:
            if self.plan is None:
                return 1, self.output_without_plan
            return (1 if self.plan.would_stop else 0), "a line an op printed\n" + json.dumps(self.plan.to_dict()) + "\n"
        return self.returncode, ""


def _config_of(command: List[str]) -> str:
    """The config a ``recordstream run`` command line runs: the word after ``run``."""
    return command[command.index("run") + 1]


@pytest.fixture()
def processes(monkeypatch: pytest.MonkeyPatch) -> _FakeProcesses:
    fake = _FakeProcesses()
    monkeypatch.setattr(workflow, "_execute", fake)
    return fake


# --------------------------------------------------------------------------- #
# Conditions say what they check
# --------------------------------------------------------------------------- #
def test_env_set_is_true_only_for_a_variable_set_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKFLOW_PROBE", "/data")
    assert EnvSet(name="WORKFLOW_PROBE")() is True
    monkeypatch.setenv("WORKFLOW_PROBE", "")
    assert EnvSet(name="WORKFLOW_PROBE")() is False
    monkeypatch.delenv("WORKFLOW_PROBE")
    assert EnvSet(name="WORKFLOW_PROBE")() is False
    assert EnvSet(name="WORKFLOW_PROBE").describe() == "WORKFLOW_PROBE is set"


def test_free_space_is_measured_on_the_nearest_folder_that_exists(tmp_path: Path) -> None:
    not_yet = tmp_path / "a" / "b" / "store"
    assert FreeSpace(path=str(not_yet), gigabytes=0)() is True
    assert FreeSpace(path=str(not_yet), gigabytes=10**9)() is False
    described = FreeSpace(path=str(not_yet), gigabytes=170).describe()
    assert described.startswith(f"170 GB free at {tmp_path} (")
    assert described.endswith(" GB free)")


def test_has_files_counts_the_files_a_pattern_matches(tmp_path: Path) -> None:
    (tmp_path / "rec_00000.sigmf-meta").write_text("{}")
    (tmp_path / "notes.txt").write_text("")
    assert HasFiles(path=str(tmp_path), pattern="*.sigmf-meta")() is True
    assert HasFiles(path=str(tmp_path), pattern="*.npy")() is False
    assert HasFiles(path=str(tmp_path / "missing"))() is False
    assert (
        HasFiles(path=str(tmp_path), pattern="*.sigmf-meta").describe()
        == f"1 file(s) matching *.sigmf-meta in {tmp_path}"
    )


def test_path_exists_and_not_describe_themselves(tmp_path: Path) -> None:
    assert PathExists(path=str(tmp_path)).describe() == f"{tmp_path} exists"
    assert Not(condition=PathExists(path=str(tmp_path))).describe() == f"not: {tmp_path} exists"


# --------------------------------------------------------------------------- #
# Require
# --------------------------------------------------------------------------- #
def test_require_refuses_with_its_message(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    step = Require(condition=PathExists(path=str(missing)), message="run the download first")
    with pytest.raises(RequirementNotMet, match="run the download first"):
        step.run()
    assert isinstance(RequirementNotMet("x"), WorkflowError) and isinstance(RequirementNotMet("x"), RuntimeError)


def test_require_without_a_message_says_what_it_checked(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    with pytest.raises(RequirementNotMet, match=f"{missing} exists"):
        Require(condition=PathExists(path=str(missing))).run()


def test_a_met_requirement_runs_nothing_and_plans_one_passed_check(tmp_path: Path) -> None:
    step = Require(condition=PathExists(path=str(tmp_path)), message="never shown")
    step.run()
    plan = step.plan()
    assert plan.checks == [Check(description=f"{tmp_path} exists", ok=True, detail="")]
    assert plan.would_stop is False


def test_an_unmet_requirement_plans_a_failed_check_carrying_its_message(tmp_path: Path) -> None:
    plan = Require(condition=PathExists(path=str(tmp_path / "x")), message="make it").plan()
    assert plan.checks == [Check(description=f"{tmp_path / 'x'} exists", ok=False, detail="make it")]
    assert plan.would_stop is True


# --------------------------------------------------------------------------- #
# ReplaceFolder
# --------------------------------------------------------------------------- #
def test_replace_folder_makes_a_missing_folder_and_its_marker(tmp_path: Path) -> None:
    store = tmp_path / "store"
    ReplaceFolder(path=str(store), marker=".store_v2").run()
    assert (store / ".store_v2").is_file()


def test_replace_folder_empties_a_folder_without_its_marker(tmp_path: Path) -> None:
    store = tmp_path / "store"
    (store / "old").mkdir(parents=True)
    (store / "old" / "sample.npy").write_bytes(b"x")
    ReplaceFolder(path=str(store), marker=".store_v2").run()
    assert sorted(path.name for path in store.iterdir()) == [".store_v2"]


def test_replace_folder_leaves_a_folder_holding_its_marker_as_it_is(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / ".store_v2").touch()
    (store / "sample.npy").write_bytes(b"x")
    ReplaceFolder(path=str(store), marker=".store_v2").run()
    assert (store / "sample.npy").read_bytes() == b"x"


def test_replace_folder_refuses_a_folder_whose_parent_is_missing(tmp_path: Path) -> None:
    store = tmp_path / "missing" / "store"
    with pytest.raises(WorkflowError, match=f"{tmp_path / 'missing'} does not exist"):
        ReplaceFolder(path=str(store), marker=".m").run()
    assert ReplaceFolder(path=str(store), marker=".m").plan().would_stop is True


def test_replace_folder_with_parents_makes_the_missing_folders_above_it(tmp_path: Path) -> None:
    store = tmp_path / "a" / "b" / "store"
    plan = ReplaceFolder(path=str(store), marker=".m", parents=True).plan()
    assert plan.would_stop is False
    assert plan.actions == [f"create {store.parent}", f"create {store} and its marker .m"]
    assert not store.parent.exists()  # planning changes nothing
    ReplaceFolder(path=str(store), marker=".m", parents=True).run()
    assert (store / ".m").is_file()


@pytest.mark.parametrize("path", ["", "/", str(Path.home())])
def test_replace_folder_refuses_to_empty_a_root_a_home_or_nothing(path: str) -> None:
    with pytest.raises(WorkflowError, match="refusing"):
        ReplaceFolder(path=path, marker=".m").run()


def test_replace_folder_refuses_a_path_whose_placeholder_nothing_filled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``"${output}"`` reads the ENVIRONMENT variable ``output``; unset, the text stays — never a folder so named."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WorkflowError, match=r"placeholder .*dotted"):
        ReplaceFolder(path="${output}", marker=".m").run()
    assert list(tmp_path.iterdir()) == []
    assert ReplaceFolder(path="${output}", marker=".m").plan().would_stop is True


def test_replace_folder_plans_what_it_would_delete_and_create_and_changes_nothing(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "sample.npy").write_bytes(b"x" * 10)
    plan = ReplaceFolder(path=str(store), marker=".m").plan()
    assert plan.actions[0].startswith(f"delete {store} (")
    assert plan.actions[1] == f"create {store} and its marker .m"
    assert (store / "sample.npy").exists() and not (store / ".m").exists()
    (store / ".m").touch()
    assert ReplaceFolder(path=str(store), marker=".m").plan().actions == [f"keep {store}: it holds its marker .m"]


# --------------------------------------------------------------------------- #
# RunConfig
# --------------------------------------------------------------------------- #
def test_run_config_runs_the_config_in_its_own_process(processes: _FakeProcesses) -> None:
    RunConfig(
        config="sets/wlan.yaml",
        scopes=["val"],
        settings={"store_root": "/data/My Store/2.0", "runnable.workers": "auto", "sets": ["wlan", "nr"], "x": True},
    ).run()
    assert processes.commands == [
        [
            sys.executable,
            "-P",
            "-m",
            "recordstream.cli",
            "run",
            "sets/wlan.yaml",
            "--scope",
            "val",
            "store_root=/data/My Store/2.0",
            "runnable.workers=auto",
            "sets=[wlan, nr]",
            "x=true",
        ]
    ]


def test_a_text_inside_a_list_or_mapping_setting_reaches_the_process_as_written(processes: _FakeProcesses) -> None:
    """A text YAML would read otherwise inside ``[...]`` / ``{...}`` — a glob's ``*`` (an alias), a comma, an empty
    text — is quoted, so the process reads back the value the workflow holds."""
    settings = {
        "source": {"pattern": "*/*/val/*.sigmf-meta", "names": "a, b", "empty": "", "plain": "x"},
        "sets": ["wlan", "*x"],
    }
    RunConfig(config="a.yaml", settings=settings).run()
    words = processes.commands[0][processes.commands[0].index("a.yaml") + 1 :]
    assert words == [
        'source={pattern: "*/*/val/*.sigmf-meta", names: "a, b", empty: "", plain: x}',
        'sets=[wlan, "*x"]',
    ]
    assert [yaml.safe_load(word.split("=", 1)[1]) for word in words] == [settings["source"], settings["sets"]]


def test_a_failed_run_stops_with_its_exit_status(processes: _FakeProcesses) -> None:
    processes.returncode = 3
    with pytest.raises(StepFailed, match="sets/wlan.yaml .*exit 3") as failed:
        RunConfig(config="sets/wlan.yaml", title="val: wlan").run()
    assert failed.value.returncode == 3


def test_a_run_config_plans_through_the_dry_run_of_its_own_process(processes: _FakeProcesses) -> None:
    plan = RunConfig(config="sets/wlan.yaml", scopes=["val"], name="wlan").plan()
    assert processes.commands[-1][-3:] == ["--dry-run", "--plan-format", "json"]
    assert plan.title == "wlan"
    assert plan.commands == [processes.commands[-1][:-3]]
    assert plan.steps == [processes.plan]
    assert plan.would_stop is False


def test_a_run_config_whose_inputs_are_missing_names_the_command_that_makes_them(processes: _FakeProcesses) -> None:
    processes.plan = Plan(title="DatasetProcessor", checks=[Check("the source can be read", False, "no files in /r")])
    plan = RunConfig(config="sets/wlan.yaml", hint="recordstream run make_sets.yaml --scope val").plan()
    assert plan.would_stop is True
    (failed,) = [check for check in plan.checks if not check.ok]
    assert failed.detail == "no files in /r — run: recordstream run make_sets.yaml --scope val"


def test_a_process_that_answers_no_plan_is_a_failed_check(processes: _FakeProcesses) -> None:
    processes.plan = None
    plan = RunConfig(config="sets/wlan.yaml").plan()
    assert plan.would_stop is True
    assert plan.checks[0].detail == "its dry run ended with exit 1 (see: recordstream run sets/wlan.yaml --dry-run)"


@pytest.mark.parametrize(
    ("output", "said"),
    [
        (
            "INFO loading\nError: Cannot resolve class: a.B at wlan.yaml:3:5\n",
            "Cannot resolve class: a.B at wlan.yaml:3:5",
        ),
        (
            'Traceback (most recent call last):\n  File "x.py", line 1\nyaml.parser.ParserError: while parsing a '
            'flow mapping\n  in "wlan.yaml", line 4\n',
            "yaml.parser.ParserError: while parsing a flow mapping",
        ),
    ],
)
def test_a_process_that_crashed_while_planning_says_why_and_how_to_see_it(
    processes: _FakeProcesses, output: str, said: str
) -> None:
    processes.plan, processes.output_without_plan = None, output
    plan = RunConfig(config="wlan.yaml").plan()
    assert plan.checks[0].detail == (
        f"its dry run ended with exit 1: {said} (see: recordstream run wlan.yaml --dry-run)"
    )


def test_a_refusal_carried_up_with_its_hint_is_printed_once(processes: _FakeProcesses) -> None:
    processes.plan = Plan(title="DatasetProcessor", checks=[Check("the source can be read", False, "no files in /r")])
    lines = plan_lines(Sequence(steps=[RunConfig(config="wlan.yaml", name="wlan", hint="make it")]).plan())
    failed = [line for line in lines if "FAILED" in line]
    assert failed == ["      FAILED: wlan: its inputs — no files in /r — run: make it"]


def test_a_run_config_reads_a_relative_config_from_the_folder_the_workflow_runs_in(
    processes: _FakeProcesses, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``config`` is handed to ``recordstream run`` as written: a relative path is the run folder's, ``pkg://`` a
    package's — the step adds no folder of its own."""
    monkeypatch.chdir(tmp_path)
    RunConfig(config="pkg://some-project/config/a.yaml").run()
    assert _config_of(processes.commands[0]) == "pkg://some-project/config/a.yaml"


def test_a_run_config_without_an_environment_hands_its_process_the_workflows_own(processes: _FakeProcesses) -> None:
    RunConfig(config="a.yaml").plan()
    RunConfig(config="a.yaml").run()
    assert processes.environments == [None, None]


def test_a_run_config_sets_its_environment_where_the_machine_leaves_it_unset(
    processes: _FakeProcesses, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine's own value wins: the step sets a variable only for its process, and only where the environment the
    workflow runs in leaves it unset."""
    monkeypatch.delenv("WORKFLOW_ALLOC", raising=False)
    monkeypatch.setenv("WORKFLOW_RATIO", "0.7")
    step = RunConfig(
        config="train.yaml", environment={"WORKFLOW_ALLOC": "expandable_segments:True", "WORKFLOW_RATIO": 0.5}
    )
    step.run()
    (env,) = processes.environments
    assert env is not None
    assert env["WORKFLOW_ALLOC"] == "expandable_segments:True" and env["WORKFLOW_RATIO"] == "0.7"
    assert env["PATH"] == os.environ["PATH"]
    assert "WORKFLOW_ALLOC" not in os.environ  # the workflow's own environment is untouched


def test_a_run_config_plans_in_the_environment_it_runs_in_and_says_what_it_sets(
    processes: _FakeProcesses, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WORKFLOW_ALLOC", raising=False)
    monkeypatch.setenv("WORKFLOW_RATIO", "0.7")
    plan = RunConfig(
        config="train.yaml", environment={"WORKFLOW_ALLOC": "expandable_segments:True", "WORKFLOW_RATIO": 0.5}
    ).plan()
    (env,) = processes.environments
    assert env is not None and env["WORKFLOW_ALLOC"] == "expandable_segments:True"
    assert plan.details == {"environment": "WORKFLOW_ALLOC=expandable_segments:True"}
    assert "      environment: WORKFLOW_ALLOC=expandable_segments:True" in plan_lines(Plan(title="s", steps=[plan]))


# --------------------------------------------------------------------------- #
# Sequence: checks first, only, plan
# --------------------------------------------------------------------------- #
def test_a_failed_check_stops_the_sequence_before_its_first_step_runs(tmp_path: Path) -> None:
    sequence = Sequence(
        steps=[
            _Step("first"),
            Require(condition=PathExists(path=str(tmp_path / "x")), message="make x"),
            _Step("never"),
        ]
    )
    with pytest.raises(RequirementNotMet, match=r"\[2/3\] .*make x"):
        sequence.run()
    assert _RUN_LOG == []


def test_a_step_checked_at_its_turn_does_not_stop_the_sequence_before_it(
    processes: _FakeProcesses, tmp_path: Path
) -> None:
    """A step whose inputs an earlier step writes (a check after a conversion) says ``checked_first: false``."""
    processes.plan = Plan(title="DatasetProcessor", checks=[Check("the source can be read", False, "nothing yet")])
    Sequence(steps=[_Step("convert"), RunConfig(config="verify.yaml", checked_first=False)]).run()
    assert _RUN_LOG == ["convert"]
    assert _config_of(processes.commands[-1]) == "verify.yaml" and "--dry-run" not in processes.commands[-1]


def test_only_runs_the_named_steps_and_every_step_without_a_name(tmp_path: Path) -> None:
    Sequence(steps=[_Step("check"), _Step("wlan", name="wlan"), _Step("nr", name="nr")], only=["nr"]).run()
    assert _RUN_LOG == ["check", "nr"]


def test_an_empty_only_runs_every_step() -> None:
    Sequence(steps=[_Step("a", name="a"), _Step("b", name="b")], only=[]).run()
    assert _RUN_LOG == ["a", "b"]


def test_a_named_sequence_is_one_step_the_sequence_around_it_picks_by_name() -> None:
    """A workflow's option made of several runs (its train and val runs, then a check) is one named inner sequence."""
    Sequence(
        steps=[
            Sequence(name="a", steps=[_Step("a train"), _Step("a val")]),
            Sequence(name="c", steps=[_Step("c1 train"), _Step("c2 train"), _Step("c check")]),
        ],
        only=["c"],
    ).run()
    assert _RUN_LOG == ["c1 train", "c2 train", "c check"]


def test_a_failed_check_inside_a_named_sequence_stops_the_workflow_before_its_first_step(tmp_path: Path) -> None:
    sequence = Sequence(
        steps=[
            _Step("first"),
            Sequence(name="c", steps=[Require(condition=PathExists(path=str(tmp_path / "x")), message="make x")]),
        ]
    )
    with pytest.raises(RequirementNotMet, match=r"\[2/2\] c: make x"):
        sequence.run()
    assert _RUN_LOG == []


@pytest.mark.parametrize(
    ("name", "title", "shown"),
    [("", "", "Sequence"), ("c", "", "c"), ("c", "option C: c1, c2, c3", "option C: c1, c2, c3")],
)
def test_a_sequences_plan_is_titled_by_its_title_else_its_name(name: str, title: str, shown: str) -> None:
    plan = Sequence(name=name, title=title, steps=[_Step("x")]).plan()
    assert plan.title == shown
    assert [step.title for step in plan.steps] == ["_Step"]


def test_the_plan_lines_show_an_inner_sequences_steps_beneath_its_title() -> None:
    lines = plan_lines(Sequence(steps=[Sequence(name="a", steps=[_Step("x"), _Step("y")])]).plan())
    assert lines == ["[1/1] a", "      _Step:", "        run _Step", "      _Step:", "        run _Step"]


def test_only_naming_no_step_is_refused_with_the_names_there_are() -> None:
    with pytest.raises(WorkflowError, match="wlam .*wlan, nr"):
        Sequence(steps=[_Step("a", name="wlan"), _Step("b", name="nr")], only=["wlam"]).run()
    assert _RUN_LOG == []


def test_the_plan_of_a_sequence_holds_each_selected_steps_plan_and_runs_nothing(
    processes: _FakeProcesses, tmp_path: Path
) -> None:
    sequence = Sequence(
        steps=[
            Require(condition=PathExists(path=str(tmp_path)), message=""),
            RunConfig(config="wlan.yaml", name="wlan"),
            RunConfig(config="nr.yaml", name="nr"),
            _Step("plain"),
        ],
        only=["nr"],
    )
    plan = sequence.plan()
    assert [step.title for step in plan.steps] == [f"{tmp_path} exists", "nr", "_Step"]
    assert plan.steps[2].actions == ["run _Step"]
    assert _RUN_LOG == [] and all("--dry-run" in command for command in processes.commands)


def test_a_dry_run_reports_every_step_that_would_stop_not_only_the_first(tmp_path: Path) -> None:
    plan = Sequence(
        steps=[
            Require(condition=PathExists(path=str(tmp_path / "a")), message="make a"),
            Require(condition=PathExists(path=str(tmp_path / "b")), message="make b"),
        ]
    ).plan()
    assert plan.would_stop is True
    text = "\n".join(plan_lines(plan))
    assert "make a" in text and "make b" in text


def test_a_step_checked_at_its_turn_does_not_make_the_dry_run_stop(processes: _FakeProcesses) -> None:
    processes.plan = Plan(title="DatasetProcessor", checks=[Check("the source can be read", False, "nothing yet")])
    plan = Sequence(steps=[RunConfig(config="verify.yaml", checked_first=False)]).plan()
    assert plan.would_stop is False


def test_plan_of_a_runnable_without_a_plan_says_it_would_run_it() -> None:
    assert plan_of(_Step("x")) == Plan(title="_Step", actions=["run _Step"])


# --------------------------------------------------------------------------- #
# Flags the workflow does not use: handed to the runs it starts
# --------------------------------------------------------------------------- #
def test_a_forwarded_flag_comes_after_the_settings_so_the_typed_one_wins(processes: _FakeProcesses) -> None:
    step = RunConfig(config="a.yaml", scopes=["val"], settings={"workers": "auto", "folder": "/x"})
    step.forward_flags({"show_progress": True, "workers": 3, "tags": ["a", "b"]})
    step.run()
    assert processes.commands[0][processes.commands[0].index("a.yaml") + 1 :] == [
        "--scope",
        "val",
        "workers=auto",
        "folder=/x",
        "show_progress=true",
        "workers=3",
        "tags=[a, b]",
    ]


def test_a_run_with_forward_off_gets_its_settings_only(processes: _FakeProcesses) -> None:
    step = RunConfig(config="a.yaml", settings={"workers": "auto"}, forward=False)
    step.forward_flags({"show_progress": True})
    step.run()
    assert processes.commands[0][processes.commands[0].index("a.yaml") + 1 :] == ["workers=auto"]


def test_a_run_forwarded_nothing_starts_the_command_it_always_did(processes: _FakeProcesses) -> None:
    step = RunConfig(config="a.yaml", settings={"workers": "auto"})
    before = step.command
    step.forward_flags({})
    assert step.command == before


def test_a_sequence_hands_the_flags_to_every_run_in_it_and_the_plan_shows_what_runs(
    processes: _FakeProcesses,
) -> None:
    inner = Sequence(steps=[RunConfig(config="inner.yaml", name="inner")], name="group")
    flowing = Sequence(steps=[_Step("x"), RunConfig(config="first.yaml", name="first"), inner])
    flowing.forward_flags({"show_progress": True})
    flowing.plan()
    planned = [command[:-3] for command in processes.commands if command[-3:] == ["--dry-run", "--plan-format", "json"]]
    processes.commands.clear()
    flowing.run()
    assert [command[-2:] for command in planned] == [
        ["first.yaml", "show_progress=true"],
        ["inner.yaml", "show_progress=true"],
    ]
    assert [command for command in processes.commands if "--dry-run" not in command] == planned


def test_a_conditional_and_a_switch_hand_the_flags_to_all_their_branches(processes: _FakeProcesses) -> None:
    if_ = Conditional(condition=True, if_true=RunConfig(config="yes.yaml"), if_false=RunConfig(config="no.yaml"))
    case = Switch(
        select="k", cases={"k": RunConfig(config="k.yaml"), "m": RunConfig(config="m.yaml")}, default=_Step("d")
    )
    Sequence(steps=[if_, case]).forward_flags({"workers": 2})
    commands = [if_.if_true.command, if_.if_false.command, case.cases["k"].command, case.cases["m"].command]
    assert [command[-1] for command in commands] == ["workers=2"] * 4


def test_a_step_that_starts_no_run_is_left_alone() -> None:
    step = _Step("x")
    Sequence(steps=[step]).forward_flags({"workers": 2})
    assert not hasattr(step, "forwarded")


# --------------------------------------------------------------------------- #
# The plan, as lines and as JSON
# --------------------------------------------------------------------------- #
def test_the_plan_lines_number_the_steps_and_show_checks_actions_and_commands(processes: _FakeProcesses) -> None:
    processes.plan = Plan(title="DatasetProcessor", details={"records": 1000, "start": 312, "to_build": 688})
    plan = Sequence(steps=[_Step("x"), RunConfig(config="wlan.yaml", scopes=["val"], name="wlan")]).plan()
    lines = plan_lines(plan)
    assert lines[0] == "[1/2] _Step"
    assert "      run _Step" in lines
    assert "[2/2] wlan" in lines
    assert "      recordstream run wlan.yaml --scope val" in lines
    assert any(line.strip() == "records: 1000, start: 312, to_build: 688" for line in lines)


def test_a_check_reads_as_one_line_and_a_failed_one_says_what_to_do() -> None:
    plan = Plan(
        title="s",
        steps=[
            Plan(title="env", checks=[Check("DATA_ROOT is set", False, "source project.bashrc")]),
            Plan(title="disk", checks=[Check("170 GB free at /data (412 GB free)", True, "")]),
        ],
    )
    lines = plan_lines(plan)
    assert "      FAILED: DATA_ROOT is set — source project.bashrc" in lines
    assert "      ok: 170 GB free at /data (412 GB free)" in lines


def test_a_plan_round_trips_through_json() -> None:
    plan = Plan(
        title="s",
        checks=[Check("c", True, "")],
        actions=["delete /x (1.0K)"],
        commands=[["recordstream", "run", "a.yaml"]],
        details={"records": 2, "sink": None},
        steps=[Plan(title="child", checked_first=False)],
    )
    assert Plan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan


def test_the_steps_build_with_no_arguments() -> None:
    for cls in (EnvSet, FreeSpace, HasFiles, ReplaceFolder, Require, RunConfig, Sequence):
        cls()


def test_a_step_that_reads_the_environment_reads_it_when_it_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    step = Require(condition=EnvSet(name="WORKFLOW_PROBE"))
    monkeypatch.setenv("WORKFLOW_PROBE", "x")
    step.run()
    monkeypatch.delenv("WORKFLOW_PROBE")
    with pytest.raises(RequirementNotMet):
        step.run()
