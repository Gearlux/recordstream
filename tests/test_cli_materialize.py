"""``materialize_runnable`` — the flat config's top-level keys must reach the runnable.

The regression this pins cost a debugging session and would have cost worse: a bare
``flow()`` on the bound ``runnable:`` node drops every broadcast sibling SILENTLY. A
dropped ``train_set:`` eventually surfaces as an empty dataset; a dropped
``max_epochs: 3`` never surfaces at all — the run proceeds on the constructor default
and looks configured.

Why it happened: liquifai's DI materializes a command parameter *against the loaded
document* only when the parameter is annotated with a configurable class. A generic
runner annotates ``runnable: Any`` precisely because the runnable is polymorphic, so DI
hands over the raw Fluid and deep-flows it with no document — the very genericity that
makes one runner serve every task is what disabled broadcasting.
"""

from typing import Any, Optional

import confluid
import pytest
from confluid import configurable
from liquifai.context import LiquifyContext, get_context, set_context

from recordstream.cli import materialize_runnable


@configurable
class _Runner:
    """A stand-in runnable with the ergonomic-knob shape a flat config drives."""

    def __init__(self, dataset: Optional[Any] = None, max_epochs: int = 10, name: str = "default") -> None:
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.name = name

    def run(self) -> str:
        return self.name


FLAT_CONFIG = """
runnable: !class:tests.test_cli_materialize._Runner
max_epochs: 3
name: from_the_flat_config
dataset: !class:recordstream.Stream
"""


@pytest.fixture
def liquifai_context() -> Any:
    """Install/remove a liquifai context the way the app does around a command."""
    previous = get_context()

    def _install(config_data: Any) -> None:
        ctx = LiquifyContext(name="test")
        ctx.config_data = config_data
        set_context(ctx)

    yield _install
    set_context(previous)


def _node(text: str = FLAT_CONFIG) -> Any:
    """The document + its ``runnable:`` node, loaded the way liquifai loads it (until="document")."""
    document = confluid.load(text, until="document")
    return document, document["runnable"]


def test_top_level_keys_broadcast_into_the_runnable(liquifai_context: Any) -> None:
    """THE regression: a bare flow() would leave every one of these at its default."""
    document, node = _node()
    liquifai_context(document)

    runner = materialize_runnable(node)

    assert runner.max_epochs == 3, "a bare flow() leaves this at the ctor default of 10 — silently"
    assert runner.name == "from_the_flat_config"
    assert runner.dataset is not None


def test_a_bare_flow_would_have_dropped_them(liquifai_context: Any) -> None:
    """The counterfactual, executed — so the pin above cannot silently stop meaning anything."""
    from confluid import flow

    _, node = _node()
    dropped = flow(node)

    assert dropped.max_epochs == 10 and dropped.name == "default" and dropped.dataset is None


def test_a_live_object_passes_through_untouched(liquifai_context: Any) -> None:
    """Only a Fluid needs building — an already-built runnable must not be rebuilt."""
    liquifai_context({"max_epochs": 3})
    live = _Runner(max_epochs=99)

    assert materialize_runnable(live) is live
    assert live.max_epochs == 99


def test_no_liquifai_context_falls_back_to_flow(liquifai_context: Any) -> None:
    """Called outside an app (a library user, a test) — build it, don't raise."""
    set_context(None)
    _, node = _node()

    runner = materialize_runnable(node)

    assert isinstance(runner, _Runner)
    assert runner.max_epochs == 10  # nothing to broadcast FROM


def test_a_root_fluid_document_falls_back_to_flow(liquifai_context: Any) -> None:
    """A YAML whose root is a single `!class:` has no siblings — nothing is lost."""
    document = confluid.load("!class:tests.test_cli_materialize._Runner\nmax_epochs: 5\n", until="document")
    liquifai_context(document)  # not a dict

    runner = materialize_runnable(document)

    assert isinstance(runner, _Runner) and runner.max_epochs == 5


def test_the_built_runnable_still_runs(liquifai_context: Any) -> None:
    """End of the line: what the CLI does after building."""
    document, node = _node()
    liquifai_context(document)

    assert materialize_runnable(node).run() == "from_the_flat_config"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
