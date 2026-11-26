"""Every exception recordstream defines derives from :class:`recordstream.RecordstreamError` and keeps the builtin it
stood for, so ``except RecordstreamError`` catches all of them and an older ``except ValueError`` still works.

The classes are found by walking the package's modules, never listed by hand: a new exception that forgets the root
fails here. A module that needs an extra this host lacks is skipped, naming it.
"""

import importlib
import inspect
import pkgutil
from typing import Dict, List, Type

import pytest

import recordstream
from recordstream import RecordstreamError
from recordstream.draws import DrawRefused, DrawSpecError
from recordstream.flow.steps import StepReferenceError
from recordstream.ops.contract import ChainContractError, ContractError
from recordstream.resources import OutOfMemory
from recordstream.storage.cache import CacheBudgetExceeded
from recordstream.workflow import RequirementNotMet, StepFailed, WorkflowError

#: What each exception was before the root, the builtin an older ``except`` names.
BUILTIN_OF: Dict[Type[BaseException], Type[BaseException]] = {
    DrawSpecError: ValueError,
    DrawRefused: ValueError,
    StepReferenceError: ValueError,
    ContractError: ValueError,
    ChainContractError: ValueError,
    OutOfMemory: MemoryError,
    CacheBudgetExceeded: Exception,
    WorkflowError: RuntimeError,
    RequirementNotMet: RuntimeError,
    StepFailed: RuntimeError,
}


def _defined_exceptions() -> List[Type[BaseException]]:
    """Every public exception class a recordstream module defines (its own, not one it imports)."""
    found: List[Type[BaseException]] = []
    for module_info in pkgutil.walk_packages(recordstream.__path__, prefix="recordstream."):
        try:
            module = importlib.import_module(module_info.name)
        except ImportError:  # an extra this host lacks (keras, torch, …): it is checked where it is installed
            continue
        for name, member in vars(module).items():
            if (
                inspect.isclass(member)
                and issubclass(member, BaseException)
                and member.__module__ == module.__name__
                and not name.startswith("_")
            ):
                found.append(member)
    return found


def test_every_exception_the_package_defines_derives_from_the_root() -> None:
    defined = _defined_exceptions()
    assert defined, "the walk found no exception class at all"
    outside = [f"{cls.__module__}.{cls.__qualname__}" for cls in defined if not issubclass(cls, RecordstreamError)]
    assert outside == []


@pytest.mark.parametrize("error", list(BUILTIN_OF), ids=lambda cls: cls.__name__)
def test_an_error_is_caught_by_the_root_and_by_the_builtin_it_was(error: Type[BaseException]) -> None:
    if error is StepFailed:
        raised: BaseException = error("refused", 3)
    elif error is StepReferenceError:
        raised = error("refused", key="from", ref="zz", target="zz")
    else:
        raised = error("refused")
    with pytest.raises(RecordstreamError):
        raise raised
    with pytest.raises(BUILTIN_OF[error]):
        raise raised


def test_the_root_is_an_exception_and_exported_from_the_package() -> None:
    assert issubclass(RecordstreamError, Exception)
    assert "RecordstreamError" in recordstream.__all__
