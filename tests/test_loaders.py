"""The deferred torch DataLoader triple (``recordstream.loaders.loader_slots``).

Two claims: the slots ARE the construction every torch training runnable used to bake inline
(train shuffled, eval not, one shared kwarg set, ``persistent_workers`` derived), and the
module is torch-ONLY by design — deliberately not reachable from the package root, so
``import recordstream`` keeps pulling no ML framework.
"""

import ast
from pathlib import Path

import pytest
from confluid.fluid import Fluid

import recordstream
from recordstream import Stream, collate_list, collate_records

torch = pytest.importorskip("torch")


def _kwargs(marker: object) -> dict:
    """A slot holds a LazyClass MARKER pre-flow; its stored kwargs are what these tests pin."""
    assert isinstance(marker, Fluid)
    return marker.kwargs


from confluid import flow  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from recordstream.loaders import LoaderSlots, loader_slots  # noqa: E402


def test_the_triple_is_split_addressed_with_the_shuffle_split() -> None:
    slots = loader_slots(batch_size=4, num_workers=0)
    assert isinstance(slots, LoaderSlots)
    assert _kwargs(slots.train)["shuffle"] is True
    assert _kwargs(slots.val)["shuffle"] is False
    assert _kwargs(slots.test)["shuffle"] is False


def test_the_shared_kwargs_are_baked_and_persistent_workers_derives() -> None:
    for split, marker in zip(("train", "val", "test"), loader_slots(batch_size=8, num_workers=0)):
        assert _kwargs(marker)["batch_size"] == 8, split
        assert _kwargs(marker)["collate_fn"] is collate_records, split
        assert _kwargs(marker)["persistent_workers"] is False, split
    assert _kwargs(loader_slots(batch_size=8, num_workers=2).train)["persistent_workers"] is True


def test_the_collate_is_the_batch_shape_choice() -> None:
    """A detection consumer passes collate_list; the slot carries it verbatim."""
    assert (
        _kwargs(loader_slots(batch_size=2, num_workers=0, collate_fn=collate_list).train)["collate_fn"] is collate_list
    )


def test_further_loader_kwargs_pass_through_to_all_three() -> None:
    """The code channel for the rest of the DataLoader surface (pin_memory, drop_last, ...)."""
    slots = loader_slots(batch_size=2, num_workers=0, drop_last=True)
    assert all(_kwargs(marker)["drop_last"] is True for marker in slots)


def test_shuffle_is_refused_as_a_shared_kwarg() -> None:
    """It is the one PER-SPLIT kwarg the helper owns; the error names the way out."""
    with pytest.raises(ValueError, match="per-split"):
        loader_slots(batch_size=2, num_workers=0, shuffle=False)


def test_a_slot_flows_into_a_working_loader() -> None:
    """The whole point of the deferral: `flow(slot, dataset=...)` at run time yields batches."""
    stream = Stream(source=[{"x": float(i)} for i in range(4)])
    loader = flow(loader_slots(batch_size=2, num_workers=0).val, dataset=stream)
    assert isinstance(loader, DataLoader)
    batches = list(loader)
    assert len(batches) == 2
    assert [float(v) for v in batches[0]["x"]] == [0.0, 1.0]


def test_the_module_is_not_reachable_from_the_package_root() -> None:
    """It imports torch at module level, so the root must neither import nor advertise it —
    that is what keeps `import recordstream` framework-free (the ops.torch pattern)."""
    assert "loader_slots" not in recordstream.__all__
    root_source = (Path(recordstream.__file__)).read_text()
    for node in ast.walk(ast.parse(root_source)):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "recordstream.loaders", "the package root must not import recordstream.loaders"
