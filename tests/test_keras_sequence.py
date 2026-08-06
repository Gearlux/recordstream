"""`RecordSequence` — the DataLoader half Keras leaves to the caller.

These pin the plumbing and NOTHING about a task: row order, batch slicing, the short final
batch, the per-epoch reshuffle, and that `transform` is what decides the model's input shape.
A test here that mentioned classes, `(x, y)` tuples or multi-hot targets would mean the task
had leaked back into the engine.

Skipped as a module when keras is absent — importing `recordstream.keras` is also what sets
`KERAS_BACKEND`, so this must be the first keras-touching import in the process.
"""

import importlib.util
import os
from typing import Any

import numpy as np
import pytest

keras_module = pytest.importorskip("recordstream.keras")
RecordSequence = keras_module.RecordSequence

from recordstream import Image, Label, Stream  # noqa: E402 - after the keras availability gate


def _records(n: int = 10) -> list:
    return [
        {"image": Image(np.full((4, 4, 3), i, dtype="uint8"), layout="HWC"), "class": Label(i), "idx": i}
        for i in range(n)
    ]


class _CountingSource:
    """A map-style source that records how often it is measured and indexed."""

    def __init__(self, records: list) -> None:
        self.records = records
        self.len_calls = 0
        self.reads: list = []

    def __len__(self) -> int:
        self.len_calls += 1
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        self.reads.append(index)
        return self.records[index]


# --------------------------------------------------------------------------- #
# The default: collated record batches, no task shape at all
# --------------------------------------------------------------------------- #


def test_without_a_transform_a_batch_is_the_collated_record() -> None:
    """`transform=None` is the identity, so the class is usable — and testable — task-free."""
    seq = RecordSequence(Stream(source=_records(8)), batch_size=4)
    batch = seq[0]

    assert set(batch) == {"image", "class", "idx"}
    assert np.asarray(batch["image"]).shape == (4, 4, 4, 3)
    assert batch["idx"] == [0, 1, 2, 3]


def test_the_transform_decides_what_keras_receives() -> None:
    """The `collate_fn` equivalent: the batch shape is the caller's decision, not this class's."""
    seq = RecordSequence(Stream(source=_records(4)), batch_size=2, transform=lambda b: ("shaped", b["idx"]))

    assert seq[0] == ("shaped", [0, 1])


def test_the_transform_sees_the_same_batch_as_batch() -> None:
    seen: list = []
    seq = RecordSequence(Stream(source=_records(4)), batch_size=2, transform=lambda b: seen.append(b))
    seq[1]

    assert seen[0]["idx"] == seq.batch(1)["idx"]


# --------------------------------------------------------------------------- #
# Batching — what a DataLoader would have done
# --------------------------------------------------------------------------- #


def test_the_last_batch_is_short_not_padded() -> None:
    seq = RecordSequence(Stream(source=_records(10)), batch_size=4)

    assert len(seq) == 3
    assert len(seq[2]["idx"]) == 2


def test_batch_size_one_yields_one_batch_per_record() -> None:
    seq = RecordSequence(Stream(source=_records(5)), batch_size=1)

    assert len(seq) == 5
    assert [b["idx"] for b in seq.batches()] == [[0], [1], [2], [3], [4]]


def test_batches_walks_every_batch_in_epoch_order() -> None:
    """The pairing half of prediction: the caller needs the batch its output came from."""
    seq = RecordSequence(Stream(source=_records(7)), batch_size=3)

    assert [b["idx"] for b in seq.batches()] == [[0, 1, 2], [3, 4, 5], [6]]


# --------------------------------------------------------------------------- #
# Shuffling — the other thing a DataLoader owns
# --------------------------------------------------------------------------- #


def test_unshuffled_order_is_the_sources_order() -> None:
    seq = RecordSequence(Stream(source=_records(6)), batch_size=6)

    assert seq[0]["idx"] == [0, 1, 2, 3, 4, 5]


def test_shuffling_changes_the_order_but_not_the_content() -> None:
    records = _records(12)
    shuffled = RecordSequence(Stream(source=records), batch_size=12, shuffle=True, seed=1)

    order = shuffled[0]["idx"]
    assert order != sorted(order)
    assert sorted(order) == list(range(12))


def test_the_seed_makes_a_shuffled_run_reproducible() -> None:
    records = _records(12)
    a = RecordSequence(Stream(source=records), batch_size=12, shuffle=True, seed=7)
    b = RecordSequence(Stream(source=records), batch_size=12, shuffle=True, seed=7)
    c = RecordSequence(Stream(source=records), batch_size=12, shuffle=True, seed=8)

    assert a[0]["idx"] == b[0]["idx"]
    assert a[0]["idx"] != c[0]["idx"]


def test_on_epoch_end_reshuffles_when_shuffling() -> None:
    seq = RecordSequence(Stream(source=_records(12)), batch_size=12, shuffle=True, seed=1)
    first = seq[0]["idx"]
    seq.on_epoch_end()

    assert seq[0]["idx"] != first


def test_on_epoch_end_is_a_no_op_when_not_shuffling() -> None:
    """Evaluation and prediction depend on this: their row order must survive every epoch."""
    seq = RecordSequence(Stream(source=_records(6)), batch_size=6)
    seq.on_epoch_end()

    assert seq[0]["idx"] == [0, 1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- #
# Lazy construction — the recordstream constructor rule
# --------------------------------------------------------------------------- #


def test_the_constructor_never_touches_the_source() -> None:
    """`len(source)` is real work for a deferred source (a HuggingFaceSource LOADS to answer it),
    so the row order is built on first use, not in `__init__`."""
    source = _CountingSource(_records(4))
    seq = RecordSequence(source, batch_size=2)

    assert source.len_calls == 0 and source.reads == []

    len(seq)
    assert source.len_calls == 1


def test_zero_arg_construction_works_and_a_missing_source_is_reported_lazily() -> None:
    seq = RecordSequence()

    with pytest.raises(RuntimeError, match="source"):
        len(seq)


def test_indices_are_computed_once() -> None:
    source = _CountingSource(_records(8))
    seq = RecordSequence(source, batch_size=4)
    len(seq), seq[0], seq[1]

    assert source.len_calls == 1


# --------------------------------------------------------------------------- #
# PyDataset's prefetch knobs — DECLARED, not smuggled through **kwargs
# --------------------------------------------------------------------------- #


def test_the_prefetch_knobs_reach_pydataset() -> None:
    """They are Keras's, but reachable only if we forward them — and a form/schema generator
    reads the SIGNATURE, so `**kwargs` would have made them invisible as well as unreachable."""
    seq = RecordSequence(Stream(source=_records(4)), workers=3, use_multiprocessing=True, max_queue_size=7)

    assert (seq.workers, seq.use_multiprocessing, seq.max_queue_size) == (3, True, 7)


def test_the_knob_defaults_are_keras_own() -> None:
    """Restated in our signature, so declaring them changes no behaviour for anyone."""
    seq = RecordSequence(Stream(source=_records(4)))

    assert (seq.workers, seq.use_multiprocessing, seq.max_queue_size) == (1, False, 10)


def test_every_knob_is_a_declared_parameter() -> None:
    """The regression guard for the rule itself: no `**kwargs` escape hatch may appear here."""
    import inspect

    params = inspect.signature(RecordSequence.__init__).parameters

    assert not [p for p in params.values() if p.kind is inspect.Parameter.VAR_KEYWORD]
    assert {"source", "batch_size", "shuffle", "seed", "transform"} <= set(params)
    assert {"workers", "use_multiprocessing", "max_queue_size"} <= set(params)


def test_threaded_prefetch_produces_the_same_batches() -> None:
    """Executed rather than asserted about: Keras drives a worker pool over this object, so the
    row order and the collate must survive being read off the calling thread."""
    ordered = RecordSequence(Stream(source=_records(9)), batch_size=3)
    threaded = RecordSequence(Stream(source=_records(9)), batch_size=3, workers=2)

    assert [b["idx"] for b in threaded.batches()] == [b["idx"] for b in ordered.batches()]


# --------------------------------------------------------------------------- #
# The KERAS_BACKEND ordering — the reason this is a module and not a loose class
# --------------------------------------------------------------------------- #
# Keras 3 reads KERAS_BACKEND at IMPORT time and defaults to `tensorflow`, which the extra does
# not install: a bare `import keras` dies inside keras.src.tree with ModuleNotFoundError. The
# setdefault has to run in the LOWEST layer that imports keras, because import sorters put a
# library import above a first-party one — a consumer's own shim loses the race.


def test_the_backend_default_is_in_effect_before_keras_was_imported() -> None:
    """The variable and the LOADED backend must agree — Keras consults it once and never again."""
    assert os.environ["KERAS_BACKEND"] == keras_module.keras_backend()


def test_the_default_is_the_first_INSTALLED_backend_not_a_hardcoded_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-coding `torch` would fail on a TensorFlow-only install exactly as Keras's own
    `tensorflow` default fails on the torch-only one this exists to fix."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "jax" else None)

    assert keras_module._first_installed_backend() == "jax"


def test_with_no_engine_installed_it_defers_to_keras_own_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let KERAS raise then — its error names the package to install; ours would guess."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    assert keras_module._first_installed_backend() == "tensorflow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --------------------------------------------------------------------------- #
# The collate is selectable here too — the torch half's `collate_fn`, on the Keras side
# --------------------------------------------------------------------------- #
def test_the_collate_defaults_to_the_record_key() -> None:
    from recordstream.keras import RecordSequence

    assert RecordSequence().collate == "record"


def test_a_registered_key_selects_the_batch_shape() -> None:
    """A `PyDataset` had no way to say "don't stack" — that made the collate a torch-only
    choice, which is the gap `register_collate` existed for and nothing used."""
    import numpy as np

    from recordstream import Image
    from recordstream.keras import RecordSequence

    records = [{"image": Image(np.zeros((3, 8, 8), dtype="float32"), layout="CHW")} for _ in range(2)]

    stacked = RecordSequence(records, batch_size=2, collate="record").batch(0)["image"]
    listed = RecordSequence(records, batch_size=2, collate="list").batch(0)["image"]
    assert getattr(stacked, "shape", None) == (2, 3, 8, 8)
    assert isinstance(listed, list) and len(listed) == 2


def test_a_collate_FUNCTION_is_accepted_too() -> None:
    """Keys serve JSON-carrying tool surfaces; a function stays the normal Python path."""
    from recordstream import collate_list
    from recordstream.keras import RecordSequence

    seq = RecordSequence([{"x": 1}, {"x": 2}], batch_size=2, collate=collate_list)
    assert seq.batch(0) == {"x": [1, 2]}


def test_an_unknown_key_names_the_registered_ones() -> None:
    import pytest

    from recordstream.keras import RecordSequence

    with pytest.raises(KeyError, match="known:"):
        RecordSequence([{"x": 1}], batch_size=1, collate="nope").batch(0)
