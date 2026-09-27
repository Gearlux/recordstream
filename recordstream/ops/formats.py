"""ReadFile — turn a dropped ``{file}`` record into the record its file decodes to.

The file-listing decoder: which formats exist is the registry's business
(:func:`recordstream.formats.file_formats` — domain packages register via the
``recordstream.formats`` entry-point group), so this op stays format-blind. Per format,
in registry order: a path the format ``consumes`` is DROPPED (``None`` — the paired
format's companion half, its content rides in with the sibling's record), a path it
``matches`` is decoded (a decode error rides out to the caller — a drop surface reports
it as this file's refusal), and a file NO format knows is REFUSED naming the installed
formats — the alternative, passing it through, was measured to produce a broken
``{"file": ...}`` row with no decoded content and no error anywhere.
"""

from pathlib import Path
from typing import Optional, Sequence

from confluid import configurable

from recordstream.formats import FORMAT_GROUP, FileFormat, file_formats
from recordstream.items import Record


@configurable(category="op", group="formats")
class ReadFile:
    """Decode a ``{file}`` record through the file-format registry.

    Args:
        field: The record key holding the dropped file's path. Defaults to ``file``.
        mmap: Map payloads instead of reading them where the format supports it
            (default) — pages load only when a consumer slices them, so listing huge
            files costs nothing.
        formats: Formats to dispatch over, in order. ``None`` (default) = every
            installed format from the registry, resolved lazily on first call.
    """

    def __init__(
        self,
        field: str = "file",
        mmap: bool = True,
        formats: Optional[Sequence[FileFormat]] = None,
    ) -> None:
        self.field = field
        self.mmap = bool(mmap)
        self.formats = formats

    @property
    def _formats(self) -> Sequence[FileFormat]:
        return file_formats() if self.formats is None else self.formats

    def __call__(self, record: Record) -> Optional[Record]:
        value = record.get(self.field)
        if value is None:
            return record
        path = Path(str(value))
        for fmt in self._formats:
            if fmt.consumes(path):
                return None  # its sibling's record carries the content — this half is consumed
            if fmt.matches(path):
                return {**record, **fmt.read(path, mmap=self.mmap)}
        names = ", ".join(fmt.name for fmt in self._formats) or "none"
        raise ValueError(
            f"ReadFile: no file format matches {path.name!r} — installed formats: "
            f"{names}. A format arrives with the package that implements it "
            f"(registered under the {FORMAT_GROUP!r} entry-point group)."
        )

    def for_projection(self) -> "_ScanningReadFile":
        """This op's CHEAP variant, for a key-restricted walk — see :class:`_ScanningReadFile`.

        Implements :class:`recordstream.projection.SupportsCheapProjection`.
        """
        return _ScanningReadFile(field=self.field, mmap=self.mmap, formats=self.formats)


class _ScanningReadFile(ReadFile):
    """``ReadFile``'s cheap twin: the record a file's METADATA describes, payload left out.

    Never built in a config (deliberately NOT ``@configurable`` — the ``_SplitView``
    precedent): it is what :meth:`ReadFile.for_projection` hands a key-restricted walk, not a
    knob. Reading a whole capture to answer a question about its annotations is the cost this
    exists to remove — measured over a 1430-recording library, 143.5 ms per record decoded
    against 0.37 ms scanned.

    Three outcomes, and each one has to be its own, or the caller cannot tell them apart:

    * a COMPANION half is dropped (``None``) exactly where a full read drops it — a projected
      walk must yield the same records as a full one or the Nth record is no longer the Nth id;
    * a format that cannot answer cheaply (no ``scan``, or none claims the file) returns the
      record UNCHANGED, so the caller sees the keys it wanted are missing and re-runs the real
      chain — which is also where an unknown file raises its located refusal;
    * otherwise the metadata's record, its declared STAND-IN entries carrying
      :data:`~recordstream.formats.STAND_IN` (``scan_answered``): the payload key is present
      in a scan with an EMPTY array so a survey can read the physics off it, and a filter
      handed that would test an empty array and answer wrong. Marking rather than removing is
      what lets the REST of the chain run — deleting the entry broke the next op outright
      (``RenameField: unknown key 'signal'``), because a chain is written against the record a
      full read produces.
    """

    def __call__(self, record: Record) -> Optional[Record]:
        from recordstream.formats import scan_answered

        value = record.get(self.field)
        if value is None:
            return record
        path = Path(str(value))
        for fmt in self._formats:
            if fmt.consumes(path):
                return None
            if fmt.matches(path):
                scanned = scan_answered(path, formats=[fmt])
                return record if scanned is None else {**record, **scanned}
        return record


__all__ = ["ReadFile"]
