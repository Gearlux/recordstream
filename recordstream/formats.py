"""The file-format registry — extension-decided decoding for dropped files.

A *file format* answers three questions about a file PATH, cheaply and by name: does
this file belong to me (``matches``), is it a paired format's companion half whose
content rides in with a sibling (``consumes``), and — only when actually asked — what
record does it decode to (``read``). The file-listing surfaces dispatch through the
registry: :class:`~recordstream.sources.files.FilesSource` keeps consumed companions
out of the LISTING (so ``len()``/ids stay honest without reading anything) and
:class:`~recordstream.ops.formats.ReadFile` turns each listed path into a record.

The registry itself is format-blind, like the item and op-family registries beside it:
a domain package registers a module under the ``recordstream.formats`` entry-point
group exposing ``formats() -> Iterable[FileFormat]``, and its formats decode wherever
this engine lists files. A failing entry is a warning and a skip, never fatal — one
half-installed format package must not blank every consumer's drop path.

Whether a LONE half of a paired format is readable is each format's own decision:
``consumes`` gates only the companion of a PRESENT pair, so a format whose data half
is self-describing simply matches and reads it standalone.
"""

import re
from importlib import metadata
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from loggair import get_logger

from recordstream.items import Record

logger = get_logger(__name__)

#: The entry-point group a domain package registers its formats module under.
FORMAT_GROUP = "recordstream.formats"

#: A drop store's ordering prefix (``0001_name.ext``) — stripped for sibling pairing.
_COPY_PREFIX = re.compile(r"^\d+_")


@runtime_checkable
class FileFormat(Protocol):
    """The structural contract a file format implements.

    ``matches``/``consumes`` are NAME-level (a suffix test plus at most a sibling
    ``stat``) because the listing calls them per dropped file; only ``read`` may open a
    file — or import a heavy backend, which is why a format keeps its backend imports
    inside ``read`` (the registry scan constructs every format object, so a
    module-level backend import would break every consumer's discovery bootstrap where
    that backend is absent).
    """

    name: str

    def matches(self, path: Path) -> bool:
        """Whether ``path`` is a file this format reads as a PRIMARY (name-only, no I/O)."""
        ...

    def consumes(self, path: Path) -> bool:
        """Whether ``path`` is a companion half consumed via its sibling (name + stat)."""
        ...

    def read(self, path: Path, *, mmap: bool = True) -> Record:
        """Decode the file at ``path`` into a record; raise a LOCATED error."""
        ...


_FORMATS: Optional[Tuple[FileFormat, ...]] = None


def _iter_entries() -> Iterable[metadata.EntryPoint]:
    """The ``recordstream.formats`` entry points — a seam tests stand fake entries into."""
    return metadata.entry_points(group=FORMAT_GROUP)


def _formats_from(source_name: str, module: object) -> List[FileFormat]:
    hook = getattr(module, "formats", None)
    if not callable(hook):
        logger.warning("Format module {} exposes no formats() hook — skipped.", source_name)
        return []
    return list(hook())


def file_formats() -> Tuple[FileFormat, ...]:
    """Every installed file format, in entry-point order.

    The order is the dispatch tie-break within one registration — formats whose claims
    overlap ACROSS packages have no defined order, so overlapping claims are a
    format-design error, not something to resolve by installation accident. The scan
    runs once per process and is cached; registration is import-cheap because a
    format's ``matches``/``consumes`` must be name-level and its backend imports live
    inside ``read``.
    """
    global _FORMATS
    if _FORMATS is not None:
        return _FORMATS
    found: List[FileFormat] = []
    for entry in _iter_entries():
        try:
            found.extend(_formats_from(entry.name, entry.load()))
        except Exception as exc:  # one broken package must not blank every drop path
            logger.warning("File-format entry point {!r} failed to load: {}", entry.name, exc)
    _FORMATS = tuple(found)
    return _FORMATS


def bare_name(name: str) -> str:
    """``name`` with a drop store's ``NNNN_`` ordering prefix stripped.

    For a format's error messages: the stripped name is the file the user actually
    dropped, which is what a "drop X beside it" instruction must name.
    """
    return _COPY_PREFIX.sub("", name)


def _counter(path: Path) -> Optional[int]:
    match = _COPY_PREFIX.match(path.name)
    return int(match.group()[:-1]) if match else None


def sibling(path: Path, own_tail: str, want_tail: str) -> Optional[Path]:
    """The counterpart file for ``path`` — its other half under a paired format.

    ``own_tail``/``want_tail`` are name TAILS stripped from and appended to the shared
    base by explicit concatenation (never ``with_suffix`` — a stem containing a dot
    loses its tail under that), so a bare tail like ``p0.bin`` pairs as readily as a
    dot-suffix. Pairing tolerates a drop store's ``NNNN_`` ordering prefix, and several
    prefix-stripped matches are resolved by RANK in drop order: the store's counter
    prefix numbers files as they were dropped, so sorting each kind by counter, the k-th
    own-tail copy of a name pairs the k-th want-tail copy. Re-dropping a pair is the
    normal case that produces this — refusing it 400'd a whole listing, and
    nearest-counter alone still tied on a back-to-back re-drop. A copy ranked past the
    last twin shares the final one — every candidate is a copy of the same dropped file.
    A copy with no counter cannot be ranked and refuses, naming the candidates.

    Args:
        path: The file whose counterpart is wanted; its name must end in ``own_tail``.
        own_tail: The tail ``path``'s name carries (e.g. ``".meta"``, ``"p0.bin"``).
        want_tail: The counterpart's tail (e.g. ``".data"``, ``".scp"``).

    Returns:
        The counterpart path, or ``None`` when no candidate exists.

    Raises:
        ValueError: Several candidates and not every copy carries a counter to rank by.
    """
    base = path.name[: -len(own_tail)]
    exact = path.parent / (base + want_tail)
    if exact.exists():
        return exact
    wanted = bare_name(base) + want_tail
    matches = [p for p in path.parent.glob(f"*{want_tail}") if bare_name(p.name) == wanted]
    if len(matches) <= 1:
        return matches[0] if matches else None
    kin = [p for p in path.parent.glob(f"*{own_tail}") if bare_name(p.name) == bare_name(path.name)]
    counters = {p: _counter(p) for p in [path, *matches, *kin]}
    if any(c is None for c in counters.values()):
        raise ValueError(
            f"{path.name} pairs ambiguously — {len(matches)} files match {wanted!r} after "
            f"stripping the copy prefix and not every copy carries a drop counter to rank "
            f"them by: {', '.join(sorted(p.name for p in matches))}"
        )
    rank = sorted(kin, key=lambda p: counters[p] or 0).index(path.parent / path.name)
    twins = sorted(matches, key=lambda p: counters[p] or 0)
    return twins[min(rank, len(twins) - 1)]


__all__ = ["FORMAT_GROUP", "FileFormat", "bare_name", "file_formats", "sibling"]
