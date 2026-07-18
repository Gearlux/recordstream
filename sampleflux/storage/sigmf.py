"""SigMF storage — one recording (``.sigmf-data`` + ``.sigmf-meta``) per sample.

`SigMF <https://sigmf.org>`_ is the open Signal Metadata Format for raw recordings: a
binary sample file plus a JSON metadata file with ``global`` / ``captures`` /
``annotations`` sections. ``SigMFSink``/``SigMFSource`` are the sampleflux carrier pair
(siblings of the HDF5/Zarr pairs, additive — no migration of existing datasets):

- the sink writes ``Sample.input`` as the raw ``.sigmf-data`` payload (``core:datatype``
  derived from the numpy dtype) and the sample metadata into the ``.sigmf-meta`` JSON;
- the source reads a directory of recordings back into Sample triplets.

sampleflux stays domain-neutral: metadata keys are carried VERBATIM — recognised
``core:``-prefixed keys land in their SigMF section, everything else rides the
namespaced ``sampleflux:<key>`` extension in ``global`` (SigMF explicitly supports
namespaced extensions). The waveform VOCABULARY (mapping ``samplerate`` →
``core:sample_rate``, regions → annotations, the torchsig collisions) lives in
``waivefront.vocab`` and plugs in via the ``meta_encoder``/``meta_decoder`` hooks
(dotted callable paths, lazily resolved like ``WrappedOp.f``).

JSON is hand-rolled deliberately (the format is a stable, simple spec; no dependency to
churn). ``core:sha512`` is optional (``checksum=True``).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from confluid import configurable
from loggair import get_logger

from sampleflux.sample import Sample
from sampleflux.storage.base import DataSink, DataSource, Storage, to_numpy

logger = get_logger("sampleflux.storage.sigmf")

SIGMF_VERSION = "1.0.0"
_EXTENSION_PREFIX = "sampleflux:"

# numpy dtype <-> SigMF core:datatype (little-endian; the practical interchange subset).
_DTYPE_TO_SIGMF: Dict[str, str] = {
    "complex64": "cf32_le",
    "complex128": "cf64_le",
    "float32": "rf32_le",
    "float64": "rf64_le",
    "int16": "ri16_le",
    "int32": "ri32_le",
    "uint8": "ru8",
    "int8": "ri8",
}
_SIGMF_TO_DTYPE: Dict[str, str] = {v: k for k, v in _DTYPE_TO_SIGMF.items()}

MetaEncoder = Callable[[Dict[str, Any]], Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]
MetaDecoder = Callable[[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]]


def _json_safe(value: Any) -> Optional[Any]:
    """``value`` if JSON-serializable (numpy scalars unwrapped), else None."""
    if isinstance(value, np.generic):
        value = value.item()
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None
    return value


def _passthrough_encode(meta: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The domain-neutral default encoder: core:* keys verbatim, the rest namespaced into global."""
    global_section: Dict[str, Any] = {}
    for key, value in meta.items():
        safe = _json_safe(value)
        if safe is None:
            logger.debug(f"SigMFSink: metadata key {key!r} is not JSON-serializable — skipped")
            continue
        if str(key).startswith("core:"):
            global_section[str(key)] = safe
        else:
            global_section[f"{_EXTENSION_PREFIX}{key}"] = safe
    return global_section, [], []


def _passthrough_decode(
    global_section: Dict[str, Any], captures: List[Dict[str, Any]], annotations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Inverse of :func:`_passthrough_encode` — unwrap the namespaced keys, keep core:* verbatim."""
    meta: Dict[str, Any] = {}
    for key, value in global_section.items():
        if key.startswith(_EXTENSION_PREFIX):
            meta[key[len(_EXTENSION_PREFIX) :]] = value
        elif key.startswith("core:") and key not in ("core:datatype", "core:version"):
            meta[key] = value
    if captures:
        meta["core:captures"] = captures
    if annotations:
        meta["core:annotations"] = annotations
    return meta


def _resolve_hook(hook: Union[str, Callable[..., Any], None], default: Callable[..., Any]) -> Callable[..., Any]:
    """A dotted-path / callable / empty hook resolved lazily (the ``WrappedOp.f`` pattern).

    Accepts BOTH ``module:function`` (the discovery-native form) and the friendlier
    dotted ``module.function`` (last dot promoted to the separator).
    """
    if hook is None or hook == "":
        return default
    if callable(hook):
        return hook
    from sampleflux.discovery import resolve_callable

    path = str(hook)
    if ":" not in path and "." in path:
        module, _, attr = path.rpartition(".")
        path = f"{module}:{attr}"
    return resolve_callable(path)


# category="sink": surfaced as a FluxStudio sink node (SAMPLEFLUX_OBJECT:sink → DatasetProcessor.sink).
@configurable(category="sink")
class SigMFSink(Storage, DataSink):
    """Write each Sample as a SigMF recording pair in a directory.

    ``Sample.input`` becomes the raw ``.sigmf-data`` payload; metadata is encoded into
    the ``.sigmf-meta`` JSON via ``meta_encoder`` (default: the domain-neutral
    passthrough — ``core:*`` keys verbatim, others under ``sampleflux:<key>``; wire
    ``waivefront.vocab.to_sigmf`` for the waveform vocabulary). A JSON-serializable
    ``Sample.target`` rides ``sampleflux:target`` (SigMF is an input-centric recording
    format; array targets are skipped with a debug note).

    Args:
        path: Directory the recordings are written into; required at write time, validated lazily.
        prefix: Recording filename prefix; files are ``<prefix><NNNNN>.sigmf-{data,meta}``.
        meta_encoder: Dotted path or callable, metadata -> (global, captures, annotations). Blank = passthrough.
        checksum: When True, write the ``core:sha512`` of the data payload into the metadata.
    """

    def __init__(
        self,
        path: Union[str, Path] = "",
        prefix: str = "rec_",
        meta_encoder: Union[str, MetaEncoder] = "",
        checksum: bool = False,
    ) -> None:
        # Lazy / zero-arg: store config only; the directory is created lazily in open().
        self.path = Path(path)
        self.prefix = str(prefix)
        self.meta_encoder = meta_encoder
        self.checksum = bool(checksum)
        self._counter = 0
        self._opened = False

    def open(self) -> "SigMFSink":
        if not self._opened:
            if str(self.path) in ("", "."):
                raise ValueError("SigMFSink: 'path' (the output directory) is required")
            self.path.mkdir(parents=True, exist_ok=True)
            self._opened = True
        return self

    def close(self) -> None:
        self._opened = False

    def write(self, sample: Sample) -> None:
        self.open()
        data = np.ascontiguousarray(to_numpy(sample.input))
        datatype = _DTYPE_TO_SIGMF.get(str(data.dtype))
        if datatype is None:
            raise TypeError(
                f"SigMFSink: dtype {data.dtype!s} has no SigMF core:datatype mapping "
                f"(supported: {sorted(_DTYPE_TO_SIGMF)})"
            )
        stem = self.path / f"{self.prefix}{self._counter:05d}"
        data.tofile(stem.with_suffix(".sigmf-data"))

        encoder = _resolve_hook(self.meta_encoder, _passthrough_encode)
        global_section, captures, annotations = encoder(dict(sample.meta))
        global_section = {
            "core:datatype": datatype,
            "core:version": SIGMF_VERSION,
            **global_section,
        }
        if self.checksum:
            global_section["core:sha512"] = hashlib.sha512(data.tobytes()).hexdigest()
        target = _json_safe(sample.target)
        if sample.target is not None:
            if target is None:
                logger.debug("SigMFSink: non-JSON-serializable target skipped (SigMF is input-centric)")
            else:
                global_section[f"{_EXTENSION_PREFIX}target"] = target
        if not captures:
            captures = [{"core:sample_start": 0}]

        meta_doc = {"global": global_section, "captures": captures, "annotations": annotations}
        stem.with_suffix(".sigmf-meta").write_text(json.dumps(meta_doc, indent=2, sort_keys=True))
        self._counter += 1

    def flush(self) -> None:
        return None


@configurable
class SigMFSource(Storage, DataSource):
    """Read a directory of SigMF recordings back into Sample triplets.

    The inverse of :class:`SigMFSink`: each ``.sigmf-meta``/``.sigmf-data`` pair yields
    one Sample — the payload as ``input`` (dtype from ``core:datatype``), metadata
    decoded via ``meta_decoder`` (default: the passthrough inverse; wire
    ``waivefront.vocab.from_sigmf`` for the waveform vocabulary), and a stored
    ``sampleflux:target`` restored to ``Sample.target``.

    Args:
        path: Directory holding the recordings; required at read time, validated lazily.
        meta_decoder: Dotted path or callable, (global, captures, annotations) -> metadata. Blank = passthrough.
    """

    def __init__(self, path: Union[str, Path] = "", meta_decoder: Union[str, MetaDecoder] = "") -> None:
        # Lazy / zero-arg: store config only; the directory is validated on first access.
        self.path = Path(path)
        self.meta_decoder = meta_decoder

    def open(self) -> "SigMFSource":
        return self

    def close(self) -> None:
        return None

    def _meta_files(self) -> List[Path]:
        if str(self.path) in ("", ".") or not self.path.is_dir():
            raise ValueError(f"SigMFSource: 'path' {str(self.path)!r} is not a directory of SigMF recordings")
        return sorted(self.path.glob("*.sigmf-meta"))

    def _read(self, meta_path: Path) -> Sample:
        doc = json.loads(meta_path.read_text())
        global_section: Dict[str, Any] = doc.get("global", {})
        captures: List[Dict[str, Any]] = doc.get("captures", [])
        annotations: List[Dict[str, Any]] = doc.get("annotations", [])

        datatype = str(global_section.get("core:datatype", ""))
        dtype = _SIGMF_TO_DTYPE.get(datatype)
        if dtype is None:
            raise ValueError(f"SigMFSource: {meta_path.name}: unsupported core:datatype {datatype!r}")
        data = np.fromfile(meta_path.with_suffix(".sigmf-data"), dtype=np.dtype(dtype))

        decoder = _resolve_hook(self.meta_decoder, _passthrough_decode)
        target_key = f"{_EXTENSION_PREFIX}target"
        target = global_section.get(target_key)
        decodable = {k: v for k, v in global_section.items() if k != target_key}
        metadata = decoder(decodable, captures, annotations)
        return Sample(input=data, target=target, metadata=metadata)

    def __iter__(self) -> Iterator[Sample]:
        for meta_path in self._meta_files():
            yield self._read(meta_path)

    def __len__(self) -> int:
        return len(self._meta_files())

    def __getitem__(self, index: int) -> Sample:
        return self._read(self._meta_files()[index])

    def iter_metadata(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """(recording stem, decoded metadata) WITHOUT loading any data payload (SupportsMetadataScan)."""
        decoder = _resolve_hook(self.meta_decoder, _passthrough_decode)
        target_key = f"{_EXTENSION_PREFIX}target"
        for meta_path in self._meta_files():
            doc = json.loads(meta_path.read_text())
            global_section = {k: v for k, v in doc.get("global", {}).items() if k != target_key}
            yield meta_path.stem, decoder(global_section, doc.get("captures", []), doc.get("annotations", []))
