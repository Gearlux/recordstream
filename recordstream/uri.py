"""Dataset identity — the URI/URL a source carries for the data it reads.

A run record that says *which* data produced it needs a stable, comparable handle on
that data. A source already knows one: a Hub repo id, a directory on disk, a query
against a store. This module is the opt-in protocol for exposing it plus the walk
helpers a consumer uses, with the same shape as :mod:`recordstream.projection` —
a ``Protocol`` (never a base class), free functions that materialize a deferred
source first, and ``None`` for "this source has no such handle", which is not an error.

Two handles, not one, because they answer different questions:

* **``dataset_uri``** — the CANONICAL identifier: machine-parseable, stable across
  machines, and the thing two runs are compared on. ``hf://datasets/ylecun/mnist?split=train``.
* **``dataset_url``** — a link a HUMAN can open, or ``None`` when the data has no web
  page (a local directory has none, and inventing one would be a lie).

Design notes
------------
* **A wrapper propagates the URI VERBATIM.** The handle identifies the *dataset*; how
  much of it a run consumed (a split, a slice, a filter) is a different fact, carried
  by the wrapper's own configuration. Decorating the string would mean the same dataset
  reached through two wrappers no longer compares equal, which is the one property the
  handle exists to have.
* **Following happens HERE, not in every wrapper.** :func:`dataset_uri` follows a
  ``.source`` attribute when the object holds no handle of its own, so every view
  source in this package — and any third-party wrapper using the same attribute name —
  works with no code of its own.
* **A concatenation declines.** Several datasets end to end are not one dataset, so the
  singular functions answer ``None`` for a source holding ``.sources``; use
  :func:`dataset_uris`, which fans out over the members.
"""

from typing import Any, List, Optional, Protocol, runtime_checkable

#: How many ``.source`` hops :func:`dataset_uri` will follow before giving up. Wrapping
#: is shallow in practice (a split over a range over a source is already unusual); the
#: cap is what keeps a cyclic ``source`` reference from hanging a tracking call.
MAX_WRAPPER_DEPTH = 16


@runtime_checkable
class SupportsDatasetIdentity(Protocol):
    """A source that can name the dataset it reads.

    Both members are properties and both may answer ``None`` — a source configured
    with no dataset yet (the zero-arg construction convention) has no identity, and a
    dataset with no web page has no URL. Implementations MUST be cheap and side-effect
    free: this is asked of a source that may never be iterated, so it reads stored
    configuration and never loads, downloads or opens anything.
    """

    @property
    def dataset_uri(self) -> Optional[str]:
        """Canonical, machine-parseable identifier for the data this source reads."""
        ...

    @property
    def dataset_url(self) -> Optional[str]:
        """Browsable link to the data, or ``None`` when it has no web page."""
        ...


def _identity(source: Any, attribute: str) -> Optional[str]:
    """Read ``attribute`` off ``source``, following ``.source`` wrappers.

    The shared body of :func:`dataset_uri` and :func:`dataset_url`: materialize a
    deferred source, read the handle if it has one, otherwise take one hop into the
    wrapped source and ask again.
    """
    current = _materialize(source)
    for _ in range(MAX_WRAPPER_DEPTH):
        if current is None:
            return None
        value = getattr(current, attribute, None)
        if value:
            return str(value)
        wrapped = getattr(current, "source", None)
        if wrapped is None or wrapped is current:
            return None
        current = _materialize(wrapped)
    return None


def _materialize(node: Any) -> Any:
    """Build ``node`` only if it is a DEFERRED config marker; hand a live object back untouched.

    Narrower than a bare ``flow()``, and the difference is the "asking never loads" property.
    ``flow()`` on a LIVE object still runs confluid's post-construction ``solidify()`` hook, so a
    source that grows one would be materialized merely by being asked its name. A marker has
    nothing to read until it is built, and building one is cheap by the lazy-construction rule.
    """
    from confluid import Fluid, flow

    return flow(node) if isinstance(node, Fluid) else node


def dataset_uri(source: Any) -> Optional[str]:
    """The canonical URI identifying the dataset ``source`` reads, or ``None``.

    A DEFERRED source (a ``!class:`` marker straight out of a config) is materialized
    first, exactly as :func:`~recordstream.projection.project` does, so a caller never
    has to remember which entry point flows and which does not.

    A wrapper that holds no URI of its own but wraps another source via ``.source`` —
    every view source in this package does — reports the wrapped source's URI
    unchanged. A wrapper holding SEVERAL sources answers ``None``; ask
    :func:`dataset_uris` instead.

    Args:
        source: Any source, view, stream, or deferred config marker. ``None`` is accepted
            and answers ``None``, so a caller needs no guard for an unwired slot.

    Example::

        dataset_uri(HuggingFaceSource(path="ylecun/mnist"))     # hf://datasets/ylecun/mnist?split=train
        dataset_uri(Stream(source=split.train))                 # the same string — the split is not part of it
    """
    return _identity(source, "dataset_uri")


def dataset_url(source: Any) -> Optional[str]:
    """The browsable URL for the dataset ``source`` reads, or ``None``.

    The human-facing twin of :func:`dataset_uri`, with identical flowing and wrapper
    rules. ``None`` is the ordinary answer for data with no web page — a directory on
    disk, a store on a mounted volume — and never an error.

    Args:
        source: Any source, view, stream, or deferred config marker.
    """
    return _identity(source, "dataset_url")


def dataset_uris(source: Any) -> List[str]:
    """Every dataset URI reachable from ``source``, in order, deduplicated.

    The plural form, for a source that concatenates several datasets: it fans out over
    a ``.sources`` list (recursively, so a concatenation of concatenations works) while
    a single source yields its one URI. Sources with no URI contribute nothing rather
    than a ``None`` entry, so the result is directly usable.

    Args:
        source: Any source, view, stream, or deferred config marker.

    Example::

        dataset_uris(ConcatSource(sources=[a, b]))   # ['hf://datasets/…', 'file:///…']
    """
    found: List[str] = []

    def visit(node: Any, depth: int) -> None:
        if node is None or depth > MAX_WRAPPER_DEPTH:
            return
        node = _materialize(node)
        uri = dataset_uri(node)
        if uri:
            if uri not in found:
                found.append(uri)
            return
        # No single identity: a concatenation is the case worth descending into. `.source`
        # is already covered by `dataset_uri` above, so only the plural attribute is left.
        members = getattr(node, "sources", None)
        if isinstance(members, (list, tuple)):
            for member in members:
                visit(member, depth + 1)

    visit(source, 0)
    return found


__all__ = [
    "MAX_WRAPPER_DEPTH",
    "SupportsDatasetIdentity",
    "dataset_uri",
    "dataset_uris",
    "dataset_url",
]
