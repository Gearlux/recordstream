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


__all__ = ["ReadFile"]
