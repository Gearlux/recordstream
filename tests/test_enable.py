"""``Enable`` — the one-flag op-list toggle.

Pins the toggle contract: the flag is the DECLARED ``enabled`` parameter (a settable
property), several wrappers are told apart by ``name`` (which scopes the CLI flag to
``--<name>.enabled``), and the retired dynamic-toggle form (any boolean attribute name
becoming the flag) is rejected loudly instead of being silently ignored.
"""

from typing import List

import pytest
from confluid import accepts_broadcast, accepts_key, to_pydantic

from recordstream import Record
from recordstream.ops.enable import Enable


def _tag(record: Record) -> Record:
    return {**record, "seen": True}


class TestEnableToggle:
    def test_default_is_enabled(self) -> None:
        # Zero-config wrapper runs its ops — the toggle only ever has to be written to turn it OFF.
        assert Enable(ops=[_tag])({"x": 1}) == {"x": 1, "seen": True}

    def test_disabled_passes_through(self) -> None:
        op = Enable(ops=[_tag], enabled=False)
        assert op({"x": 1}) == {"x": 1}
        assert op.enabled is False

    def test_constructible_from_python_in_one_call(self) -> None:
        # The whole wrapper — ops, toggle, name — comes from the constructor, so a
        # generated tool/form/canvas call can build it without post-construction setattr.
        op = Enable(ops=[_tag], enabled=False, name="visualize")
        assert (op.name, op.enabled, op.ops) == ("visualize", False, [_tag])

    def test_post_construction_setattr_is_the_yaml_path(self) -> None:
        # What `enabled: false` in YAML does when confluid setattr's it after __init__.
        op = Enable(ops=[_tag])
        op.enabled = False
        assert op({"x": 1}) == {"x": 1}
        op.enabled = True
        assert op({"x": 1}) == {"x": 1, "seen": True}

    def test_named_wrappers_toggle_independently(self) -> None:
        # `name` scopes the CLI flag (--overlay.enabled vs --labelstudio.enabled); here we
        # simulate the post-override state the two addressed writes produce.
        overlay = Enable(ops=[_tag], name="overlay", enabled=True)
        labelstudio = Enable(ops=[_tag], name="labelstudio", enabled=False)
        assert overlay({"x": 1}) == {"x": 1, "seen": True}
        assert labelstudio({"x": 1}) == {"x": 1}

    def test_non_bool_toggle_raises(self) -> None:
        # No silent truthiness: a quoted YAML bool / typo'd value fails at set time.
        with pytest.raises(TypeError, match="must be a bool"):
            Enable(ops=[_tag], enabled="true")  # type: ignore[arg-type]
        op = Enable(ops=[_tag])
        with pytest.raises(TypeError, match="must be a bool"):
            op.enabled = 1  # type: ignore[assignment]


class TestIntrospectionContract:
    """The toggle must be reachable from every front-end, not just YAML."""

    def test_declared_parameters_are_the_schema(self) -> None:
        # to_pydantic drives navigaitor's MCP/form schemas and StreamStudio's widgets —
        # a toggle absent here is a toggle no GUI or tool call can set.
        assert set(to_pydantic(Enable).model_fields) == {"ops", "enabled", "name"}

    @pytest.mark.parametrize("key", ["ops", "enabled", "name"])
    def test_keys_are_cli_settable(self, key: str) -> None:
        # accepts_key gates liquifai's addressed `--<name>.<key>` writes;
        # accepts_broadcast gates the bare `--<key>` form.
        assert accepts_key(Enable, key) is True
        assert accepts_broadcast(Enable, key) is True

    def test_retired_dynamic_toggle_name_is_not_settable(self) -> None:
        assert accepts_key(Enable, "visualize") is False
        assert accepts_broadcast(Enable, "visualize") is False


class TestLazyValidation:
    def test_empty_ops_raises_on_first_call(self) -> None:
        op = Enable()  # zero-arg construction stays valid
        with pytest.raises(ValueError, match="non-empty 'ops'"):
            op({"x": 1})

    def test_stray_boolean_attribute_raises_with_migration_hint(self) -> None:
        # The retired form (`visualize: false` as a bare YAML kwarg) lands as a
        # post-construction attribute nothing reads — reject it instead of silently
        # running the ops the user meant to gate.
        op = Enable(ops=[_tag])
        op.visualize = False  # type: ignore[attr-defined]  # what the old YAML form produced
        with pytest.raises(ValueError, match=r"unexpected boolean attribute\(s\) \['visualize'\]"):
            op({"x": 1})

    def test_migration_hint_names_the_replacement_spelling(self) -> None:
        op = Enable(ops=[_tag])
        op.visualize = False  # type: ignore[attr-defined]
        with pytest.raises(ValueError) as excinfo:
            op({"x": 1})
        message = str(excinfo.value)
        assert "name: visualize" in message and "enabled:" in message
        assert "--visualize.enabled" in message

    def test_validation_runs_once_then_stays_out_of_the_hot_path(self) -> None:
        op = Enable(ops=[_tag])
        assert op({"x": 1}) == {"x": 1, "seen": True}
        # A stray attribute set AFTER the first record is not re-scanned per record.
        op.visualize = False  # type: ignore[attr-defined]
        assert op({"x": 2}) == {"x": 2, "seen": True}


class TestOpsChain:
    def test_ops_run_in_sequence(self) -> None:
        def _bump(record: Record) -> Record:
            return {**record, "x": record["x"] + 1}

        op = Enable(ops=[_bump, _bump, _tag])
        assert op({"x": 1}) == {"x": 3, "seen": True}

    def test_close_propagates_to_inner_ops(self) -> None:
        closed: List[str] = []

        class _Sink:
            def __call__(self, record: Record) -> Record:
                return record

            def close(self) -> None:
                closed.append("sink")

        Enable(ops=[_Sink()]).close()
        assert closed == ["sink"]
