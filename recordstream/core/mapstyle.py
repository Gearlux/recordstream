"""What recordstream MEANS by "a dataset", said structurally — no framework import.

Pure types: the :class:`MapStyle` Protocol and the :data:`RecordSource` union built on it.
Deliberately depends on nothing but ``typing`` + the record alias, so every other core
module can import it without ordering constraints. The function that NORMALIZES a wired slot
into one of these (``ensure_record_dataset``) lives beside ``Stream`` instead, because
building a ``Stream`` is its whole body.
"""

from typing import Any, Iterable, Protocol, Union, runtime_checkable

from recordstream.items import Record


@runtime_checkable
class MapStyle(Protocol):
    """A map-style dataset: ``len(ds)`` and ``ds[i]``.

    What recordstream MEANS by "a dataset", said structurally so the engine never imports a
    framework to express it. ``Stream`` used to inherit ``torch.utils.data.Dataset``, which made
    torch a hard dependency of a package whose own work is numpy — for nothing: that base is not
    load-bearing. ``DataLoader`` duck-types its argument (a plain object with these two methods
    works), nothing in the workspace does ``isinstance(x, Dataset)``, and the annotation is the
    only thing the inheritance ever bought.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


#: What a wired dataset slot may hold — the contract ``ensure_record_dataset`` enforces,
#: named ONCE here rather than restated by every consumer: anything MAP-STYLE (``__len__`` +
#: ``__getitem__`` — which a ``Stream`` is), or any iterable of records (a recordstream
#: source, a plain list of record dicts). Consumers annotate their slots
#: ``Optional[Partial[RecordSource]]`` — ``Partial`` because they flow the slot at run time.
#:
#: Expressed with the structural :class:`MapStyle` rather than ``torch.utils.data.Dataset`` so the
#: engine can say "a dataset" without importing a framework; torch's ``DataLoader`` is itself
#: duck-typed and consumes either.
RecordSource = Union[MapStyle, Iterable[Record]]
