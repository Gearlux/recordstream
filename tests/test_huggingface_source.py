"""HuggingFaceSource unit pins that need no Hub access (the dataset cache is stubbed)."""

from recordstream.sources.huggingface import HuggingFaceSource


def _with_rows(n: int, count: int = 0) -> HuggingFaceSource:
    source = HuggingFaceSource(path="stub/dataset", count=count)
    source._dataset = list(range(n))  # the lazy cache — a list satisfies len()/iteration
    return source


def test_len_reports_the_dataset_when_count_is_unset() -> None:
    """``count`` of 0 (or None) means "all records" — a bare ``self.count`` would report 0 and
    make the source look empty to a len()-based consumer."""
    assert len(_with_rows(29)) == 29


def test_len_reports_count_when_it_caps() -> None:
    assert len(_with_rows(29, count=10)) == 10


def test_len_is_clamped_when_count_exceeds_the_split() -> None:
    """A ``count`` larger than the split must never be reported verbatim: a map-style consumer
    trusts ``len()`` for its index space, so a lying length surfaces as an ``IndexError`` deep
    inside a DataLoader worker, one epoch in (found the hard way — ``count: 32`` over cppe-5's
    29-row test split killed the first validation pass)."""
    assert len(_with_rows(29, count=32)) == 29
