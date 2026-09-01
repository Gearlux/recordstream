"""``FilesSource`` — a plain list of file paths, each one record."""

from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from confluid import configurable
from loggair import get_logger

from recordstream.formats import FileFormat, file_formats
from recordstream.items import Record

logger = get_logger(__name__)


@configurable(category="source")
class FilesSource:
    """A list of FILES as records — ``{"file": "<path>"}`` each, nothing more.

    The dataset counterpart of "someone handed me files": no split, no metadata layout, no
    labels — and deliberately NO decoding. The source knows nothing about what a file MEANS;
    turning the path into content is an OP's job (``ReadImage`` for pictures, the
    format-registry ``ReadFile`` for anything a registered format decodes), so the same
    source serves every file kind and the graph shows how a file becomes a record. A
    consuming workspace passes the list (e.g. files a user dropped) and the chain takes it
    from there.

    One listing rule IS format-aware: a PAIRED format's companion half (the file whose
    content rides in with a sibling — decided by the registry's ``consumes()``, a cheap
    name-level test that never decodes the data) is kept out of the listing, so
    ``len()``/ids count one record per pair. With no format packages installed the
    listing is the plain file list.

    Args:
        files: The file paths to serve, in the order to serve them.
        name: Id prefix a consuming viewer derives record ids from.
        exclude: Optional filename glob whose matches are NOT served as records — a
            manual filter on top of the registry rule (name-level, no file reads).
        formats: Formats whose ``consumes()`` decides the companion exclusion. ``None``
            (default) = every installed format from the registry, resolved lazily; an
            empty list restores the plain listing.
    """

    def __init__(
        self,
        files: Optional[List[str]] = None,
        name: str = "files",
        exclude: str = "",
        formats: Optional[Sequence[FileFormat]] = None,
    ) -> None:
        self.files = [str(f) for f in (files or [])]
        self.name = name
        self.exclude = exclude
        self.formats = formats

    @property
    def _format_list(self) -> Sequence[FileFormat]:
        return file_formats() if self.formats is None else self.formats

    @property
    def _served(self) -> List[str]:
        served = self.files
        if self.exclude:
            from fnmatch import fnmatch

            served = [f for f in served if not fnmatch(Path(f).name, self.exclude)]
        formats = self._format_list
        if formats:
            served = [f for f in served if not any(fmt.consumes(Path(f)) for fmt in formats)]
        return served

    def __len__(self) -> int:
        return len(self._served)

    def __getitem__(self, index: int) -> Record:
        return {"file": str(Path(self._served[index]))}

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"FilesSource(files={len(self.files)})"
