"""Pins the "Partial Initialization & Zero-Arg Construction" convention for ALL recordstream configurables.

Every ``@configurable`` class in recordstream MUST be constructible with no arguments and do no
functional work in ``__init__`` (no I/O, no network, no eager materialization). This walks the
whole package, discovers every ``@configurable`` class, and asserts ``Cls()`` succeeds — so a
newly-added class that violates the convention (a required ctor arg, or a constructor that opens a
file / loads a dataset) fails here. See confluid ``AGENTS.md`` → "Partial Initialization & Zero-Arg
Construction" and recordstream ``AGENTS.md`` → "Partial Evaluation".
"""

import importlib
import pkgutil
from typing import List

import pytest

import recordstream


def _all_recordstream_configurables() -> List[type]:
    """Import every recordstream submodule and collect the ``@configurable`` classes defined in recordstream."""
    seen: dict = {}
    for modinfo in pkgutil.walk_packages(recordstream.__path__, prefix="recordstream."):
        try:
            module = importlib.import_module(modinfo.name)
        except Exception:  # pragma: no cover - optional/heavy deps absent in some envs
            continue
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and getattr(obj, "__confluid_configurable__", False)
                and getattr(obj, "__module__", "").startswith("recordstream")
            ):
                seen[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return list(seen.values())


_CONFIGURABLES = _all_recordstream_configurables()


def test_discovery_found_the_configurables() -> None:
    # Guard against the walker silently finding nothing (which would make the parametrized
    # test below vacuously pass). recordstream has well over a dozen @configurable classes.
    assert len(_CONFIGURABLES) >= 20


@pytest.mark.parametrize("cls", _CONFIGURABLES, ids=lambda c: c.__name__)
def test_zero_arg_construction(cls: type) -> None:
    # The whole point of the convention: building any configurable must succeed with no arguments
    # and do no functional work (so it can be configured post-construction).
    instance = cls()
    assert instance is not None


def test_sources_do_not_materialize_on_construction() -> None:
    # The lazy caches stay empty until first use — no dataset load / partition / offset compute
    # happens in __init__.
    from recordstream.sources import ConcatSource, DatasetSplit, HuggingFaceSource, RangeSource

    assert HuggingFaceSource()._dataset is None
    assert DatasetSplit()._views == {}
    assert RangeSource()._indices is None
    assert ConcatSource()._offsets is None
