"""``GroupedFilesSink`` — copy each record's file into a folder per group."""

import os
import shutil
from pathlib import Path
from typing import Any, Union

import confluid
from loggair import get_logger

from recordstream.items import Record, item_value
from recordstream.storage.base import DataSink

logger = get_logger(__name__)


# category="sink": surfaced by visual editors as a sink node docking into a DatasetProcessor's sink slot.
@confluid.configurable(category="sink")
class GroupedFilesSink(DataSink):
    """Copy every record's FILE under ``path``, the files of one group together in a folder of their own.

    A record names its file under :attr:`file` and, when it is in a group, the group under :attr:`group`. A grouped
    file lands in ``<prefix><group>/`` (a number padded to four digits: ``group_0007``); a record without the group
    entry — or with ``None`` there — lands outside any such folder. The files are COPIED: the dataset read from
    is left as it is. :class:`~recordstream.sources.grouped.GroupedFilesSource` reads the folder back, one record
    per group.

    ``root`` keeps each file's place under it, and the group folder goes INSIDE that place, so a dataset of class
    folders stays one — ``cat/1.png`` of group 3 lands in ``cat/group_0003/1.png``, and a group whose files sit
    in two class folders is split over them. Without ``root`` only the file's name is kept.

    A file that would land on an existing one is refused, never overwritten: two files of one name need ``root``,
    and a second run needs an empty ``path``.

    Args:
        path: Folder the files are copied under (created when the first record is written).
        root: Folder each file's place is kept relative to; empty (default) = the file's name only.
        file: Record entry naming the file to copy.
        group: Record entry naming the file's group; a record without it is in no group.
        prefix: What a group's folder name starts with.
    """

    def __init__(
        self,
        path: Union[str, Path] = "",
        root: Union[str, Path] = "",
        file: str = "file",
        group: str = "group",
        prefix: str = "group_",
    ) -> None:
        # Partial / zero-arg: store config only; folders are created when a record is written.
        self.path = str(path)
        self.root = str(root)
        self.file = file
        self.group = group
        self.prefix = prefix
        self._written = 0

    @staticmethod
    def _expanded(folder: str) -> Path:
        return Path(os.path.expanduser(os.path.expandvars(folder)))

    def _folder_of(self, group: Any) -> str:
        if hasattr(group, "item"):
            group = group.item()  # a numpy number
        return f"{self.prefix}{group:04d}" if isinstance(group, int) else f"{self.prefix}{group}"

    def write(self, record: Record) -> None:
        """Copy the record's file to its place under ``path``."""
        if self.file not in record:
            raise KeyError(f"GroupedFilesSink: the record has no {self.file!r} entry (it carries {sorted(record)})")
        if not self.path:
            raise ValueError("GroupedFilesSink: `path` is empty — name the folder the files are copied under")
        source = Path(str(item_value(record[self.file])))
        place = Path()
        if self.root:
            root = self._expanded(self.root)
            try:
                place = source.relative_to(root).parent
            except ValueError:
                raise ValueError(f"GroupedFilesSink: {str(source)!r} is not under root {str(root)!r}") from None
        group = item_value(record.get(self.group))
        if group is not None:
            place = place / self._folder_of(group)
        target = self._expanded(self.path) / place / source.name
        if target.exists():
            raise FileExistsError(
                f"GroupedFilesSink: {str(source)!r} would land on {str(target)!r}, which exists — set `root` to "
                "keep two files of one name apart, or write into an empty folder"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self._written += 1

    def flush(self) -> None:
        """Nothing is pending: every file is copied as its record is written."""
        logger.debug(f"GroupedFilesSink: {self._written} file(s) copied under {self.path!r}")
