"""The Keras boundary: the backend default, the ``keras`` handle, and the batching adapter.

Three things live here, in the order they have to happen.

**1. The backend default.** Keras 3 reads ``KERAS_BACKEND`` **at import time** and defaults to
``tensorflow``, which ``recordstream[keras]`` does not install (Keras 3 is an API, not a runtime —
it needs exactly one of torch / TensorFlow / JAX underneath, and which one is the operator's
choice). A bare ``import keras`` on a torch-backed install therefore dies with
``ModuleNotFoundError: No module named 'tensorflow'`` raised from inside
``keras.src.tree.optree_impl`` — a traceback that says nothing about what to do. So this module
sets a default the install can actually honour, the first backend that is PRESENT, and only when
the operator has not chosen: ``setdefault`` never overrides an explicit ``KERAS_BACKEND=jax``.

**2. The ``keras`` handle.** Consumers import keras THROUGH here (``from recordstream.keras import
keras``) so the ordering above cannot be got wrong by an import that happens to land first — which
is a real hazard, because import sorters group a library import ahead of a first-party one.

**3. The batching adapter.** For torch, recordstream supplies one function and the framework does
the rest: hand :func:`~recordstream.collate.collate_records` to a ``DataLoader`` as its
``collate_fn`` and torch owns the row order, the batch slicing and the per-epoch reshuffle (a
``Stream`` is map-style, which is all a ``DataLoader`` needs — see
:class:`~recordstream.core.mapstyle.MapStyle`). Keras 3 has no ``DataLoader``:
``keras.utils.PyDataset.__getitem__`` must return a whole BATCH, so somebody has to write that
loop. :class:`RecordSequence` is that loop and nothing else — row order, slicing, reshuffle,
``collate_records`` — the DataLoader half, kept beside the collate half it calls instead of
re-appearing in every training project.

What a batch BECOMES stays the caller's, exactly as ``collate_fn`` is the caller's on the torch
side: ``transform`` maps one collated record to what the model consumes, so task shapes (a
classifier's ``(x, y)`` tuple, a multi-input model's dict) never enter this module.

Import it by PATH — ``RecordSequence`` is deliberately absent from the package root, because
``inspect.getmembers`` (what :func:`recordstream.discovery.scan_module` and the GUI bridges use)
getattrs every name a module advertises, so a lazy root export would import keras on every
discovery scan of a torch-only install::

    from recordstream.keras import RecordSequence

    seq = RecordSequence(stream, batch_size=32, shuffle=True, transform=to_xy)
    model.fit(seq, epochs=3)
"""

import importlib.util
import os
from typing import Any, Callable, Iterator, Optional, cast

import numpy as np

from recordstream.collate import collate_records
from recordstream.core import MapStyle
from recordstream.items import Record

#: Compute backends Keras 3 can run on, in the order this package prefers them. torch first
#: because it is the one every other recordstream extra already implies; the rest are honoured
#: whenever an operator has them installed.
_BACKENDS = ("torch", "tensorflow", "jax")


def _first_installed_backend() -> str:
    """The first of :data:`_BACKENDS` actually importable, else Keras's own default.

    Picking a backend that is PRESENT rather than one we assume: hard-coding ``torch`` would fail
    on a TensorFlow-only install exactly as Keras's own ``tensorflow`` default fails on the
    torch-only install this exists to fix. Uses ``find_spec`` so nothing is imported just to look,
    which matters because it runs at import time of this module.

    Example::

        os.environ.setdefault("KERAS_BACKEND", _first_installed_backend())  # "torch" here
    """
    for name in _BACKENDS:
        if importlib.util.find_spec(name) is not None:
            return name
    return "tensorflow"  # Keras's own default — let IT raise, naming the package to install


#: Set BEFORE keras is imported, and only if unset — an explicit choice always wins.
os.environ.setdefault("KERAS_BACKEND", _first_installed_backend())

import keras  # noqa: E402 - deliberately after the environment default above

__all__ = ["RecordSequence", "keras", "keras_backend"]


def keras_backend() -> str:
    """The Keras backend actually in effect (``torch`` / ``tensorflow`` / ``jax``).

    Reads Keras rather than the environment variable, so it reports what is LOADED — Keras
    consults ``KERAS_BACKEND`` once at import, and changing it afterwards has no effect. Worth
    logging at the start of a run whose engine is selectable.
    """
    return str(keras.backend.backend())


class RecordSequence(keras.utils.PyDataset):
    """A map-style record source as a ``keras.utils.PyDataset`` of collated batches.

    The DataLoader half of Keras batching: row order, batch slicing, per-epoch reshuffle, and
    :func:`~recordstream.collate.collate_records`. It is deliberately task-blind — ``transform``
    is the caller's ``collate_fn``-equivalent and decides what the model actually receives.

    Args:
        source: Any map-style record source (a ``Stream`` is one).
        batch_size: Rows per batch. A short final batch is yielded as-is, never padded.
        shuffle: Reshuffle the row order at the end of every epoch (training).
        seed: Shuffle seed, so a shuffled run is reproducible.
        transform: Maps one collated record batch to what the model consumes. ``None`` hands
            over the batched record itself.
        workers: ``PyDataset`` prefetch workers. ``1`` (Keras's own default) loads batches on
            the calling thread; higher values overlap the record walk with the training step,
            which is what a slow source (decode, resize, remote read) needs.
        use_multiprocessing: Run those workers as PROCESSES instead of threads. Each one
            re-pickles this object and its source, so it is the wrong default for a source
            holding an open handle — reach for it only when the per-record work is
            GIL-bound and does not parallelize with threads.
        max_queue_size: How many prefetched batches may wait. The memory ceiling of
            prefetching: batches are held whole, so a large value on large batches is a
            real footprint.
    """

    def __init__(
        self,
        source: Optional[MapStyle] = None,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int = 0,
        transform: Optional[Callable[[Record], Any]] = None,
        workers: int = 1,
        use_multiprocessing: bool = False,
        max_queue_size: int = 10,
    ) -> None:
        # Keras's own defaults, restated so they are DECLARED parameters rather than reachable
        # only through `**kwargs` — the every-knob-is-a-declared-parameter rule (architecture
        # §6): a form/schema generator enumerates the signature, and an undeclared knob is
        # invisible to it.
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.source = source
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.transform = transform
        self._rng = np.random.default_rng(seed)
        self._indices: Optional[np.ndarray] = None

    @property
    def indices(self) -> np.ndarray:
        """Row order for the current epoch — built on first use, reshuffled by ``on_epoch_end``.

        Built lazily rather than in the constructor because ``len(source)`` is real work for a
        deferred source (a ``HuggingFaceSource`` loads its dataset to answer it), and recordstream
        constructors do none. It is also where a missing ``source`` is reported, so zero-arg
        construction stays possible.
        """
        if self._indices is None:
            if self.source is None:
                raise RuntimeError("RecordSequence: no 'source' to batch — wire the dataset first.")
            order = np.arange(len(self.source))
            if self.shuffle:
                self._rng.shuffle(order)
            self._indices = order
        return self._indices

    def __len__(self) -> int:
        """Number of batches — Keras asks once per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def batch(self, index: int) -> Record:
        """The collated record batch at ``index``, BEFORE ``transform``."""
        rows = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        source = cast(MapStyle, self.source)  # the `indices` read above validated it
        # Declared local, not a bare return: `@register_collate` types every registered collate
        # as `CollateFn = Callable[[Sequence[Any]], Any]` — deliberately loose, because the
        # registry holds task collates with divergent conventions — which erases
        # `collate_records`' own `-> Record` at the call site.
        collated: Record = collate_records([source[int(i)] for i in rows])
        return collated

    def batches(self) -> Iterator[Record]:
        """Every collated batch, in the current epoch's order.

        The pairing half of prediction: a model emits ``[N, ...]`` while a
        :class:`~recordstream.predictions.PredictionsSink` writes per record, so the caller needs
        the batch its output came from to read that batch's metadata back
        (:func:`~recordstream.batch.batch_metadata`).
        """
        for index in range(len(self)):
            yield self.batch(index)

    def __getitem__(self, index: int) -> Any:
        """What Keras feeds the model: the collated batch, mapped by ``transform``."""
        batch = self.batch(index)
        return batch if self.transform is None else self.transform(batch)

    def on_epoch_end(self) -> None:
        """Reshuffle between epochs when training. Keras calls this itself."""
        if self.shuffle:
            self._rng.shuffle(self.indices)
