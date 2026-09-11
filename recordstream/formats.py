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
from typing import Collection, Iterable, List, Optional, Protocol, Sequence, Tuple, cast, runtime_checkable

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

    ``matches``/``consumes`` are NAME-level and cheap (a suffix test, a sibling
    ``stat``, at most a small sidecar-metadata peek for a format whose sidecars name
    their data file) because the listing calls them per dropped file; only ``read`` may
    decode the data — or import a heavy backend, which is why a format keeps its
    backend imports inside ``read`` (the registry scan constructs every format object,
    so a module-level backend import would break every consumer's discovery bootstrap
    where that backend is absent).
    """

    name: str

    def matches(self, path: Path) -> bool:
        """Whether ``path`` is a file this format reads as a PRIMARY (name-only, no I/O)."""
        ...

    def consumes(self, path: Path) -> bool:
        """Whether ``path`` is a companion half consumed via its sibling (name + stat,
        at most a sidecar peek — never a data decode)."""
        ...

    def read(self, path: Path, *, mmap: bool = True) -> Record:
        """Decode the file at ``path`` into a record; raise a LOCATED error."""
        ...

    # OPTIONAL capabilities a format MAY implement — probed structurally by whoever
    # dispatches them, never required. `scan` is the one this protocol documents:
    #
    #     def scan(self, path: Path) -> Record: ...
    #
    # "What does this file's METADATA say?" — its sidecar or its header, never its
    # payload — answered as an ordinary record with the payload left out, so every op
    # and graph downstream reads it exactly like a decoded one. It exists because for
    # the formats that pair a small metadata file with a large data file the two
    # questions differ in cost by orders of magnitude, and a consumer surveying a whole
    # listing can only afford the cheap one. See :func:`scan_file`.
    #
    #     scan_stands_in: Tuple[str, ...]
    #
    # Which entries `scan` fills with a STAND-IN rather than the real thing — in practice
    # the payload's own key. A survey wants the physics that lives ON that item (the rate,
    # the centre, the position), so a format may keep the entry and empty its array rather
    # than drop it; measured on two formats, `scan` reports 0 samples where `read` reports
    # 1 000 001 and 100 000 000. That makes the entry PRESENT but not ANSWERED, which is a
    # difference a survey may ignore and a caller answering a QUESTION about it may not.
    # Absent (the default) means the cheap answer stands in for nothing. See
    # :func:`scan_answered`.


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


def scan_file(path: Path, formats: Optional[Sequence[FileFormat]] = None) -> Optional[Record]:
    """The record ``path``'s METADATA alone describes, or ``None`` when nothing can say.

    The cheap half of the registry: a format MAY implement ``scan(path) -> Record``
    (sidecar or header, never the payload) and this dispatches to the first one claiming
    the path, in registry order — the same order and the same companion rule
    :class:`~recordstream.ops.formats.ReadFile` follows, so a survey and a decode never
    disagree about which file is a primary.

    ``None`` is returned — never a full ``read`` — for a companion half, a file no format
    claims, and a claiming format that does not implement ``scan``. **The fallback is
    deliberately absent:** the caller asked the cheap question precisely because the
    expensive one was unaffordable at this scale, so silently answering it instead would
    turn a survey of ten thousand files into a decode of ten thousand files. A caller that
    wants the payload after all asks the format for it by name.

    A ``scan`` that RAISES is left to the caller: a malformed sidecar is a located error
    like a malformed file, and whether one bad file skips or fails the survey is the
    survey's decision, not this layer's.

    Args:
        path: The file to describe.
        formats: Formats to dispatch over, in order; ``None`` = the installed registry.
    """
    for fmt in file_formats() if formats is None else formats:
        if fmt.consumes(path):
            return None  # its metadata rides in with the sibling's record
        if fmt.matches(path):
            scan = getattr(fmt, "scan", None)
            return cast(Record, scan(path)) if callable(scan) else None
    return None


class StandIn:
    """The value a cheap answer puts where the real thing would be. A singleton: :data:`STAND_IN`."""

    def __repr__(self) -> str:  # pragma: no cover - a leak into real data is what this makes readable
        return "<recordstream STAND_IN: this entry was never read>"


#: Marks an entry a cheap answer could not really supply — see :func:`scan_answered`.
STAND_IN = StandIn()


def scan_answered(path: Path, formats: Optional[Sequence[FileFormat]] = None) -> Optional[Record]:
    """:func:`scan_file`'s record with every STAND-IN entry's value replaced by :data:`STAND_IN`.

    The difference matters to exactly one kind of caller. A SURVEY wants the stand-in as it
    is: the rate and the centre live on the payload item, so a format keeps that entry and
    empties its array. A caller answering a QUESTION about that entry must not be handed it —
    a filter given the placeholder tests an empty array and answers confidently wrong
    (measured: ``scan`` reports 0 samples where ``read`` reports 1 000 001).

    It is REPLACED rather than removed, and that is the load-bearing choice. Removing it broke
    the next op in a real chain outright (``RenameField: unknown key 'signal'``), because a
    chain is written against the record a full read produces. And a key is not a stable handle
    anyway — the same chain renames ``signal`` to ``input`` two steps later, so a caller
    checking a NAME would be checking the wrong one. The marker rides the VALUE, so it
    survives every move a chain makes, and :func:`answers` is the check.

    ``None`` for the same three cases :func:`scan_file` returns it: a companion half, a file
    no format claims, a claiming format that cannot answer cheaply.
    """
    for fmt in file_formats() if formats is None else formats:
        if fmt.consumes(path):
            return None
        if fmt.matches(path):
            scan = getattr(fmt, "scan", None)
            if not callable(scan):
                return None
            record = dict(cast(Record, scan(path)))
            for key in getattr(fmt, "scan_stands_in", ()) or ():
                if key in record:
                    record[key] = STAND_IN
            return record
    return None


def answers(record: Record, keys: Collection[str]) -> bool:
    """Whether ``record`` really carries every one of ``keys`` — no missing, no placeholders.

    The one test behind every cheap-then-verify walk, so a source and a chain cannot disagree
    about what counts as an answer.
    """
    return all(key in record and record[key] is not STAND_IN for key in keys)


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


__all__ = [
    "FORMAT_GROUP",
    "STAND_IN",
    "FileFormat",
    "StandIn",
    "answers",
    "bare_name",
    "file_formats",
    "scan_answered",
    "scan_file",
    "sibling",
]
