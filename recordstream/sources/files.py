"""``FilesSource`` — a plain list of file paths, each one record."""

from pathlib import Path
from typing import Iterator, List, Optional

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record

logger = get_logger(__name__)


@configurable(category="source")
class FilesSource:
    """A list of FILES as records — ``{"file": "<path>"}`` each, nothing more.

    The dataset counterpart of "someone handed me files": no split, no metadata layout, no
    labels — and deliberately NO decoding. The source knows nothing about what a file MEANS;
    turning the path into content is an OP's job (``ReadImage`` for pictures, a domain's own
    reader for anything else), so the same source serves every file kind and the graph shows
    how a file becomes a record. A consuming workspace passes the list (e.g. files a user
    dropped) and the chain takes it from there.

    Args:
        files: The file paths to serve, in the order to serve them.
        name: Id prefix a consuming viewer derives record ids from.
        exclude: Optional filename glob whose matches are NOT served as records — for a
            PAIRED format whose companion file is consumed via its sibling (name-level,
            no file reads).
    """

    def __init__(self, files: Optional[List[str]] = None, name: str = "files", exclude: str = "") -> None:
        self.files = [str(f) for f in (files or [])]
        self.name = name
        self.exclude = exclude

    @property
    def _served(self) -> List[str]:
        if not self.exclude:
            return self.files
        from fnmatch import fnmatch

        return [f for f in self.files if not fnmatch(Path(f).name, self.exclude)]

    def __len__(self) -> int:
        return len(self._served)

    def __getitem__(self, index: int) -> Record:
        return {"file": str(Path(self._served[index]))}

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"FilesSource(files={len(self.files)})"
