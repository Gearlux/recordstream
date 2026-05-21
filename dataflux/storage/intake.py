"""Generic adapter that exposes any intake DataSource as a DataFlux DataSource.

This bridge lets DataFlux pipelines consume any data exposed via the
`intake <https://intake.readthedocs.io>`_ catalog system. It is deliberately
container-agnostic: ``xarray.DataArray`` / ``xarray.Dataset``, ``numpy.ndarray``,
``pandas.DataFrame``, and arbitrary Python objects all map cleanly onto the
DataFlux ``Sample(input, target, metadata)`` triplet.

The ``intake`` import is lazy so DataFlux's hard dependency surface is unchanged
— users only pay the cost when they actually construct an ``IntakeSource``.
"""

from pathlib import Path
from typing import Any, Iterator, Optional, Union

import confluid
import numpy as np
import torch
from logflow import get_logger

from dataflux.sample import Sample
from dataflux.storage.base import DataSource, Storage

logger = get_logger(__name__)

_INTAKE_INSTALL_HINT = "IntakeSource requires the 'intake' package. " "Install it with: pip install 'intake>=2.0'"


@confluid.configurable
class IntakeSource(Storage, DataSource):
    """Wrap an intake DataSource so it satisfies the DataFlux ``DataSource`` protocol.

    Each intake partition becomes one ``Sample``. Mapping rules:

      - ``xarray.DataArray`` / ``xarray.Dataset``: ``Sample.input`` is the
        underlying numpy array (wrapped as a torch tensor when possible),
        ``Sample.metadata`` is ``dict(da.attrs)``, and if ``target_attr`` is
        set, ``Sample.target`` is read from ``attrs[target_attr]``.
      - ``numpy.ndarray``: ``Sample.input`` is the array as a torch tensor;
        no target, empty metadata.
      - Everything else: routed through :py:meth:`dataflux.sample.Sample.from_any`.

    Construct with either a (catalog_path, source_name) pair (preferred for
    Confluid serialization symmetry) or a pre-instantiated intake source via
    ``source`` (handy for tests and ad-hoc use).

    Args:
        catalog_path: Path to an intake catalog YAML.
        source_name: Key inside the catalog identifying the source.
        source: Pre-instantiated intake DataSource. Mutually exclusive with
                the (catalog_path, source_name) pair.
        target_attr: Name of an attribute key whose value should be lifted to
                     ``Sample.target`` for xarray containers. ``None`` means
                     no target.
    """

    def __init__(
        self,
        catalog_path: Optional[Union[str, Path]] = None,
        source_name: Optional[str] = None,
        source: Any = None,
        target_attr: Optional[str] = None,
    ) -> None:
        if source is None and (catalog_path is None or source_name is None):
            raise ValueError(
                "IntakeSource requires either a (catalog_path, source_name) pair "
                "or a pre-instantiated `source` object."
            )
        if source is not None and (catalog_path is not None or source_name is not None):
            raise ValueError("IntakeSource: pass either (catalog_path, source_name) or `source`, not both.")

        try:
            import intake  # noqa: F401
        except ImportError as e:  # pragma: no cover - exercised via test_missing_intake_dep
            raise ImportError(_INTAKE_INSTALL_HINT) from e

        self.catalog_path: Optional[str] = str(catalog_path) if catalog_path is not None else None
        self.source_name = source_name
        self.target_attr = target_attr
        self._source = source
        self._catalog: Any = None

    def _resolve_source(self) -> Any:
        if self._source is not None:
            return self._source
        import intake

        logger.info(f"IntakeSource: opening catalog {self.catalog_path!r}, source {self.source_name!r}")
        self._catalog = intake.open_catalog(self.catalog_path)
        self._source = self._catalog[self.source_name]
        return self._source

    def open(self) -> "IntakeSource":
        self._resolve_source()
        return self

    def close(self) -> None:
        src = self._source
        if src is not None and hasattr(src, "_close"):
            try:
                src._close()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"IntakeSource: source close raised {e!r}")
        # Only drop catalog-derived sources; user-supplied ones stay alive.
        if self._catalog is not None:
            self._source = None
        self._catalog = None

    def __iter__(self) -> Iterator[Sample]:
        src = self._resolve_source()
        npart = self._npartitions(src)
        for i in range(npart):
            yield self._to_sample(src.read_partition(i))

    def __len__(self) -> int:
        return self._npartitions(self._resolve_source())

    @staticmethod
    def _npartitions(src: Any) -> int:
        # intake 2.x lazily populates the schema; npartitions reads 0 until discover() runs.
        if hasattr(src, "discover"):
            try:
                src.discover()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"IntakeSource: discover() raised {e!r}")
        n = getattr(src, "npartitions", None)
        if n is None or n == 0:
            schema = src._get_schema()
            n = getattr(schema, "npartitions", 1) or 1
        return int(n)

    def _to_sample(self, value: Any) -> Sample:
        # xarray: DataArray and Dataset both expose .values + .attrs.
        if hasattr(value, "values") and hasattr(value, "attrs"):
            attrs = dict(value.attrs)
            data = value.values
            tensor = self._array_to_tensor(data)
            target = attrs.get(self.target_attr) if self.target_attr else None
            return Sample(input=tensor, target=target, metadata=attrs)
        if isinstance(value, np.ndarray):
            return Sample(input=self._array_to_tensor(value), target=None, metadata={})
        return Sample.from_any(value)

    @staticmethod
    def _array_to_tensor(array: np.ndarray) -> Any:
        # Complex dtypes are preserved; non-complex go through torch.from_numpy directly.
        try:
            return torch.from_numpy(array)
        except TypeError:
            # Older torch versions can't ingest some dtypes (e.g. complex on CPU < 1.8).
            # Fall back to returning the numpy array; downstream ops can handle it.
            return array
