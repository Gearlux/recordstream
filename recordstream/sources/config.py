"""``ConfigStreamSource`` — the records of a stream another config holds (a training config's ``train_set``)."""

from typing import Any, Dict, Iterator, List, Optional

from confluid import configurable
from loggair import get_logger

from recordstream.items import Record

logger = get_logger(__name__)


@configurable(category="source")
class ConfigStreamSource:
    """The records of the stream config ``config`` holds under ``key`` — a dataset defined ONCE, read anywhere.

    A training config already says what its data is: ``train_set: !class:recordstream.Stream`` with a source and its
    ops. This source reads that stream from the config file, so another graph (an analysis, a comparison) walks the
    same records without copying the dataset's definition. The config is loaded on first use, never at construction.

    A training stream usually ends in preprocessing a model needs — resizing, normalising, ``ToTensor`` — which a
    reader of the pictures does not want: by default only the stream's source is read (the records as stored);
    ``apply_ops: true`` reads the stream as the config defines it, every op applied.

    Args:
        config: The config file: a path, or ``pkg://<distribution>/<path>`` for a config a package ships.
        key: The key the stream sits under in the config (``train_set``, ``val_set``, ``test_set``, ``stream``).
        scopes: Scopes to activate while the config is loaded (``framework=lightning``) — as on its command line.
        apply_ops: Read the stream with its ops applied; ``False`` reads its source alone.
    """

    def __init__(
        self,
        config: str = "",
        key: str = "train_set",
        scopes: Optional[List[str]] = None,
        apply_ops: bool = False,
    ) -> None:
        self.config = config
        self.key = key
        self.scopes = list(scopes) if scopes else []
        self.apply_ops = bool(apply_ops)
        self._records: Any = None

    @property
    def records(self) -> Any:
        """What the config holds under :attr:`key` — its source, or the whole stream (``apply_ops``). Loaded once.

        Raises:
            ValueError: no config named, a key the config does not hold, or a key holding no stream.
        """
        if self._records is None:
            import confluid

            if not self.config:
                raise ValueError("ConfigStreamSource: name the config file the stream is read from (config)")
            document = confluid.load(self.config, scopes=self.scopes or None)
            if not isinstance(document, dict) or self.key not in document:
                held = ", ".join(str(k) for k in document) if isinstance(document, dict) else "nothing"
                raise ValueError(f"ConfigStreamSource: {self.config} holds no {self.key!r} — it holds: {held}")
            stream = document[self.key]
            if not self.apply_ops:
                source = getattr(stream, "source", None)
                if source is None:
                    raise ValueError(
                        f"ConfigStreamSource: {self.config}'s {self.key!r} is not a stream with a source "
                        f"({type(stream).__name__})"
                    )
                self._records = source
            else:
                if not hasattr(stream, "__iter__"):
                    raise ValueError(f"ConfigStreamSource: {self.config}'s {self.key!r} yields no records")
                self._records = stream
            read = "with its ops" if self.apply_ops else "its source"
            logger.info(f"ConfigStreamSource: {self.config} → {self.key}, {read}")
        return self._records

    @property
    def source(self) -> Any:
        """What the records come from — :attr:`records`, so the free functions that follow ``.source``
        (``dataset_uri``) name the config's dataset."""
        return self.records

    @property
    def produces(self) -> Dict[str, str]:
        """The entries the config's stream declares (its own ``produces``), ``{}`` when it declares none or the config
        cannot be read — loading the config builds its classes and reads no record."""
        try:
            declared = getattr(self.records, "produces", None)
        except Exception:  # noqa: BLE001 - a config that cannot be read declares nothing; iterating says why
            return {}
        return dict(declared) if isinstance(declared, dict) else {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Record:
        record: Record = self.records[index]
        return record

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)


__all__ = ["ConfigStreamSource"]
