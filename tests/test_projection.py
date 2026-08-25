"""Tests for :mod:`recordstream.projection` — the key-addressed walk helpers.

Focused on :func:`recordstream.first_value`, the one-peek primitive: what it unwraps,
what it skips, what it costs, and that it inherits ``iter_key``'s three properties
(projection-aware sources, deferred sources, laziness).
"""

from typing import Any, Collection, Dict, Iterator, List

from confluid import PartialClass

from recordstream import Label, MultiLabel, first_value, is_class_id
from recordstream.items import Record


class _Source:
    """A plain iterable source — no projection protocol, no laziness tricks."""

    def __init__(self, records: List[Record]) -> None:
        self.records = records

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)


class _CountingProjectionSource:
    """A projection-aware source that records what it was asked for, and counts reads."""

    def __init__(self, records: List[Record]) -> None:
        self.records = records
        self.requested: List[Collection[str]] = []
        self.reads = 0

    def project(self, keys: Collection[str]) -> Iterator[Record]:
        self.requested.append(set(keys))
        for record in self.records:
            self.reads += 1
            yield {k: v for k, v in record.items() if k in keys}

    def __iter__(self) -> Iterator[Record]:  # pragma: no cover - project() is what runs
        return iter(self.records)


def test_first_value_returns_the_first_present_value() -> None:
    source = _Source([{"class": "cat"}, {"class": "dog"}])
    assert first_value(source, "class") == "cat"


def test_first_value_skips_leading_nones_and_missing_keys() -> None:
    """``None`` means "no value here", which is not an answer about the column's kind."""
    source = _Source([{"class": None}, {"other": 1}, {"class": "dog"}])
    assert first_value(source, "dog_key_absent_everywhere") is None
    assert first_value(source, "class") == "dog"


def test_an_all_none_column_answers_none_rather_than_raising() -> None:
    source = _Source([{"class": None}, {"class": None}])
    assert first_value(source, "class") is None


def test_an_empty_source_answers_none() -> None:
    assert first_value(_Source([]), "class") is None


def test_a_label_item_unwraps_to_its_value() -> None:
    source = _Source([{"class": Label(value="cat", classes=["cat", "dog"])}])
    assert first_value(source, "class") == "cat"


def test_a_multilabel_item_unwraps_to_its_values_LIST() -> None:
    """The peek is how a consumer learns the column is multi-label — by the ITEM type.

    ``iter_key`` unwraps a ``MultiLabel`` to its list, so a sequence here IS multi-label
    rather than a guess about what a list might mean.
    """
    source = _Source([{"class": MultiLabel(values=["cat", "dog"])}])
    peeked = first_value(source, "class")
    assert peeked == ["cat", "dog"]
    assert isinstance(peeked, (list, tuple, set))


def test_the_peek_answers_the_names_versus_ids_question() -> None:
    """The canonical call site: one peek + ``is_class_id`` decides whether a LabelMap is needed."""
    assert not is_class_id(first_value(_Source([{"class": "cat"}]), "class"))
    assert is_class_id(first_value(_Source([{"class": 3}]), "class"))


def test_first_value_stops_at_the_first_hit() -> None:
    """One peek, not a walk — the whole reason to call this instead of ``list(iter_key(...))``."""
    source = _CountingProjectionSource([{"class": i} for i in range(100)])
    assert first_value(source, "class") == 0
    assert source.reads == 1


def test_first_value_asks_only_for_the_requested_key() -> None:
    """A projection-aware source never builds the values this does not ask for."""
    source = _CountingProjectionSource([{"class": 0, "image": "expensive"}])
    assert first_value(source, "class") == 0
    assert source.requested == [{"class"}]


def test_first_value_materializes_a_deferred_source() -> None:
    """``project()`` flows a ``!class:`` marker, so a caller writes no ``flow()`` here."""
    marker = PartialClass(_Source, records=[{"class": "cat"}])
    assert first_value(marker, "class") == "cat"


def test_plain_values_pass_through_verbatim() -> None:
    payload: Dict[str, Any] = {"samplerate": 30.72e6}
    assert first_value(_Source([payload]), "samplerate") == 30.72e6
