"""``Enable`` — the one-flag op-list toggle.

Pins the toggle-attribute contract: ANY boolean attribute set post-construction is the
toggle and its NAME is the CLI flag; ``enable`` is the documented generic name for the
named-wrapper pattern (``--overlay.enable``); the class's own members (``ops`` /
``enabled`` / ``flag_name``) are RESERVED — read-only properties reject a same-named
YAML kwarg loudly at configure time.
"""

import pytest

from sampleflux.ops.enable import Enable


def _tag(record):
    return {**record, "seen": True}


class TestEnableToggle:
    def test_enable_named_toggle_off_passes_through(self) -> None:
        op = Enable(ops=[_tag])
        op.enable = False  # what `enable: false` in YAML does (post-construction setattr)
        assert op({"x": 1}) == {"x": 1}
        assert op.enabled is False
        assert op.flag_name == "enable"

    def test_enable_named_toggle_on_fires_ops(self) -> None:
        op = Enable(ops=[_tag])
        op.enable = True
        assert op({"x": 1}) == {"x": 1, "seen": True}
        assert op.enabled is True

    def test_named_wrappers_toggle_independently(self) -> None:
        # Two wrappers, same generic `enable` attr — the instance `name` scopes the CLI flag
        # (--overlay.enable vs --labelstudio.enable); here we simulate the post-config state.
        overlay, labelstudio = Enable(ops=[_tag]), Enable(ops=[_tag])
        overlay.name, labelstudio.name = "overlay", "labelstudio"
        overlay.enable, labelstudio.enable = True, False
        assert overlay({"x": 1}) == {"x": 1, "seen": True}
        assert labelstudio({"x": 1}) == {"x": 1}

    def test_semantic_toggle_name_still_works(self) -> None:
        op = Enable(ops=[_tag])
        op.visualize = True
        assert op.flag_name == "visualize"
        assert op({"x": 1}) == {"x": 1, "seen": True}

    def test_reserved_names_raise_on_set(self) -> None:
        # `enabled` / `flag_name` are read-only introspection properties — a YAML kwarg
        # with one of those names fails loudly instead of silently shadowing the API.
        for reserved in ("enabled", "flag_name"):
            with pytest.raises(AttributeError):
                setattr(Enable(ops=[_tag]), reserved, False)

    def test_zero_toggles_raises(self) -> None:
        with pytest.raises(RuntimeError, match="exactly one boolean toggle"):
            Enable(ops=[_tag])({"x": 1})

    def test_multiple_toggles_raises(self) -> None:
        op = Enable(ops=[_tag])
        op.enable, op.visualize = True, False
        with pytest.raises(RuntimeError, match="exactly one boolean toggle"):
            op({"x": 1})

    def test_empty_ops_raises(self) -> None:
        op = Enable()
        op.enable = True
        with pytest.raises(ValueError, match="non-empty 'ops'"):
            op({"x": 1})
