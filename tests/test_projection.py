"""Tests for :mod:`recordstream.projection` — the key-addressed walk helpers.

Focused on :func:`recordstream.first_value`, the one-peek primitive: what it unwraps,
what it skips, what it costs, and that it inherits ``iter_key``'s three properties
(projection-aware sources, deferred sources, laziness).
"""

from typing import Any, Collection, Dict, Iterator, List, cast

from confluid import PartialClass

from recordstream import Label, MultiLabel, first_value, is_class_id
from recordstream.items import Record


class _Source:
    """A plain iterable source — no projection protocol, no laziness tricks."""

    def __init__(self, records: List[Record]) -> None:
        self.records = records

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)


class _Lazy:
    """A source backed by a GENERATOR — walking it is observable, so laziness is testable."""

    def __init__(self, records: Iterator[Record]) -> None:
        self.records = records

    def __iter__(self) -> Iterator[Record]:
        return self.records


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


class TestStreamForwardsProjection:
    """A ``Stream`` with NO ops forwards ``project`` to its source — with ops it must not.

    The generic iterate-then-filter form exists because an op may CONSUME the entry another
    one produces (the image becomes the label), so a stream with ops has to run its chain.
    A stream with an EMPTY chain adds nothing to the records, so filtering through it used to
    throw away the source's efficient path — a label-only walk decoded every image anyway.
    """

    class _ProjectingSource:
        def __init__(self) -> None:
            self.asked: list = []

        def project(self, keys: Any) -> Iterator[Record]:
            self.asked.append(set(keys))
            yield {"class": 7}

        def __iter__(self) -> Iterator[Record]:
            yield {"class": 7, "image": "DECODED"}

    def test_an_ops_free_stream_forwards_to_the_source(self) -> None:
        from recordstream import Stream

        source = self._ProjectingSource()
        rows = list(Stream(source=source).project({"class"}))
        assert source.asked == [{"class"}], "the source's efficient path must be the one that runs"
        assert rows == [{"class": 7}]

    def test_a_stream_WITH_ops_still_runs_its_chain(self) -> None:
        """The CON case: the op may produce the requested entry, so the chain must run."""
        from recordstream import Stream

        source = self._ProjectingSource()
        stamp = lambda record: {**record, "stamped": True}  # noqa: E731
        rows = list(Stream(source=source, ops=[stamp]).project({"class", "stamped"}))
        assert source.asked == [], "forwarding would skip the op that makes the entry"
        assert rows == [{"class": 7, "stamped": True}]

    def test_an_ops_free_stream_over_a_plain_source_keeps_the_generic_path(self) -> None:
        rows = list(__import__("recordstream").Stream(source=[{"class": 1, "image": "X"}]).project({"class"}))
        assert rows == [{"class": 1}]


class TestTheViewSourcesForwardProjection:
    """A wrapper that only SLICES must not throw away its source's efficient path.

    ``project()`` takes a source's own cheap walk only when THAT object implements the
    protocol, so a wrapper without one falls back to reading every record whole — even when
    the thing it wraps projects perfectly. Measured on a signal corpus of 15 MB records:
    0.006 s per record straight from the source, 0.423 s through a ``RangeSource`` around it,
    for the same records and the same keys. These four wrappers slice, concatenate or reorder;
    none of them looks inside a record, so each forwards.
    """

    class _Projecting:
        """An indexable source whose projected walk is DISTINGUISHABLE from a full read."""

        def __init__(self, count: int = 6) -> None:
            self.count = count
            self.asked: List[Any] = []
            self.projected = 0
            self.decoded = 0

        def _record(self, index: int) -> Record:
            return {"class": index, "image": "DECODED"}

        def project(self, keys: Collection[str]) -> Iterator[Record]:
            self.asked.append(set(keys))
            for index in range(self.count):
                self.projected += 1
                yield {k: v for k, v in self._record(index).items() if k in keys}

        def __getitem__(self, index: int) -> Record:
            if not 0 <= index < self.count:
                raise IndexError(index)
            self.decoded += 1
            return self._record(index)

        def __len__(self) -> int:
            return self.count

    def test_range_source_slices_the_projection_instead_of_reading_records(self) -> None:
        from recordstream.projection import project
        from recordstream.sources.range import RangeSource

        inner = self._Projecting()
        rows = list(project(RangeSource(source=inner, start=1, stop=4), ("class",)))
        assert rows == [{"class": 1}, {"class": 2}, {"class": 3}]
        assert inner.asked == [{"class"}]
        assert inner.decoded == 0, "the window was read whole instead of projected"

    def test_range_source_stops_once_its_window_is_delivered(self) -> None:
        """A window at the front must not walk the tail — the generator is abandoned."""
        from recordstream.projection import project
        from recordstream.sources.range import RangeSource

        inner = self._Projecting(count=1000)
        rows = list(project(RangeSource(source=inner, start=0, stop=3), ("class",)))
        assert [r["class"] for r in rows] == [0, 1, 2]
        assert inner.projected == 3, f"walked {inner.projected} records to deliver 3"

    def test_concat_source_chains_each_sub_sources_own_projection(self) -> None:
        from recordstream.projection import project
        from recordstream.sources.concat import ConcatSource

        first, second = self._Projecting(count=2), self._Projecting(count=2)
        rows = list(project(ConcatSource(sources=[first, second]), ("class",)))
        assert [r["class"] for r in rows] == [0, 1, 0, 1]
        assert first.asked == [{"class"}] and second.asked == [{"class"}]
        assert first.decoded == 0 and second.decoded == 0

    def test_a_split_view_yields_its_own_records_in_its_own_order(self) -> None:
        """A split's indices are SHUFFLED, so order is the part that has to be reconstructed."""
        from recordstream.projection import project
        from recordstream.sources.split import DatasetSplit

        inner = self._Projecting(count=10)
        split = DatasetSplit(source=inner, val_fraction=0.2, seed=42)
        expected = [record["class"] for record in split.val]
        inner.decoded = 0
        assert [r["class"] for r in project(split.val, ("class",))] == expected
        assert inner.decoded == 0, "the view was read whole instead of projected"

    def test_a_split_selected_by_name_projects_that_view(self) -> None:
        from recordstream.projection import project
        from recordstream.sources.split import DatasetSplit

        inner = self._Projecting(count=10)
        selected = DatasetSplit(source=inner, val_fraction=0.2, seed=42, split="val")
        view = DatasetSplit(source=self._Projecting(count=10), val_fraction=0.2, seed=42).val
        assert [r["class"] for r in project(selected, ("class",))] == [r["class"] for r in project(view, ("class",))]

    def test_a_joint_stream_chains_its_sub_streams_projections(self) -> None:
        from recordstream import Stream
        from recordstream.core.stream import JointStream
        from recordstream.projection import project

        first, second = self._Projecting(count=2), self._Projecting(count=2)
        joint = JointStream(streams=[Stream(source=cast(Any, first)), Stream(source=cast(Any, second))])
        assert [r["class"] for r in project(joint, ("class",))] == [0, 1, 0, 1]
        assert first.decoded == 0 and second.decoded == 0

    def test_a_wrapper_over_a_plain_source_still_works(self) -> None:
        """Forwarding must not REQUIRE the protocol — a source without one keeps the fallback."""
        from recordstream.projection import project
        from recordstream.sources.range import RangeSource

        class _Indexable:
            """What RangeSource asks of a source, and nothing more — no ``project``."""

            def __init__(self) -> None:
                self.records = [{"class": i, "image": "DECODED"} for i in range(5)]

            def __getitem__(self, index: int) -> Record:
                return self.records[index]

            def __len__(self) -> int:
                return len(self.records)

        rows = list(project(RangeSource(source=_Indexable(), start=1, stop=3), ("class",)))
        assert rows == [{"class": 1}, {"class": 2}]


class TestProjectIndices:
    """The shared ordering primitive the three index wrappers use."""

    def test_it_yields_the_named_indices_in_the_named_order(self) -> None:
        from recordstream.projection import project_indices

        source = _Source([{"n": i} for i in range(5)])
        assert list(project_indices(source, ("n",), [3, 0, 4])) == [{"n": 3}, {"n": 0}, {"n": 4}]

    def test_an_increasing_order_streams_without_holding_records(self) -> None:
        """The common case (a slice, an unshuffled split) must stay lazy."""
        from recordstream.projection import project_indices

        walked = []

        def records() -> Iterator[Record]:
            for i in range(100):
                walked.append(i)
                yield {"n": i}

        first = next(iter(project_indices(_Lazy(records()), ("n",), [0, 1, 2])))
        assert first == {"n": 0}
        assert walked == [0], "the whole source was walked before the first record came out"

    def test_it_stops_at_the_last_index_it_needs(self) -> None:
        from recordstream.projection import project_indices

        walked = []

        def records() -> Iterator[Record]:
            for i in range(100):
                walked.append(i)
                yield {"n": i}

        assert list(project_indices(_Lazy(records()), ("n",), [1, 0])) == [{"n": 1}, {"n": 0}]
        assert walked == [0, 1], f"walked {walked[-1] + 1} records to deliver 2"

    def test_no_indices_is_an_empty_walk_that_reads_nothing(self) -> None:
        from recordstream.projection import project_indices

        source = _CountingProjectionSource([{"n": 1}])
        assert list(project_indices(source, ("n",), [])) == []
        assert source.reads == 0


class TestAChainWithOpsCanAnswerCheaply:
    """A `Stream` WITH ops must run its chain — but an op may know a cheaper way to answer.

    The measured case is a file-reading chain: decoding a whole capture to test its
    annotations cost 143.5 ms per record over a 1430-recording library, where the sidecar
    beside it answers in 0.37 ms. The op offers a cheap variant of ITSELF (it cannot know
    which keys are wanted); the stream runs that chain, checks whether the record carries
    what was asked for, and re-runs the real chain when it does not.
    """

    class _Expensive:
        """An op that decodes a payload — and can answer from a sidecar instead."""

        def __init__(self, sidecar: bool = False) -> None:
            self.sidecar = sidecar
            self.decodes = 0

        def __call__(self, record: Record) -> Any:
            if record["file"] == "companion":
                return None  # a companion half — dropped by BOTH variants
            if self.sidecar:
                # the metadata's answer: everything but the payload
                return {**record, "label": record["file"], "seen": True}
            self.decodes += 1
            return {**record, "label": record["file"], "payload": "DECODED", "seen": True}

        def for_projection(self) -> "TestAChainWithOpsCanAnswerCheaply._Expensive":
            return type(self)(sidecar=True)  # type(self): a subclass must get ITS cheap twin

    def _stream(self, files: List[str]) -> Any:
        from recordstream import Stream

        return Stream(source=[{"file": name} for name in files], ops=[self._Expensive()])

    def test_a_key_the_cheap_chain_answers_never_decodes(self) -> None:
        stream = self._stream(["a", "b", "c"])
        assert [r["label"] for r in stream.project(("label",))] == ["a", "b", "c"]
        assert stream.ops[0].decodes == 0, "the payload was decoded to answer a question about labels"

    def test_a_key_only_the_real_chain_has_falls_back(self) -> None:
        """The cheap answer has no `payload`, so asking for one must re-run the real chain."""
        stream = self._stream(["a", "b"])
        assert [r["payload"] for r in stream.project(("payload",))] == ["DECODED", "DECODED"]
        assert stream.ops[0].decodes == 2

    def test_the_fallback_is_PER_RECORD_not_per_walk(self) -> None:
        """A sidecar carrying the answer for one file and not the next is the normal case."""
        from recordstream import Stream

        class _Patchy(TestAChainWithOpsCanAnswerCheaply._Expensive):
            def __call__(self, record: Record) -> Any:
                out = super().__call__(record)
                if out is not None and self.sidecar and record["file"] == "b":
                    out.pop("label")  # this one's metadata does not say
                return out

        op = _Patchy()
        stream = Stream(source=[{"file": n} for n in ("a", "b", "c")], ops=[op])
        assert [r["label"] for r in stream.project(("label",))] == ["a", "b", "c"]
        assert op.decodes == 1, "only the record whose cheap answer fell short is re-read"

    def test_a_record_the_cheap_chain_DROPS_is_not_re_read(self) -> None:
        """Dropping is a decision about the FILE, not about the keys — and the projected walk
        must yield the same records a full one does, or the Nth record is no longer the Nth id."""
        from recordstream import Stream

        stream = self._stream(["a", "companion", "c"])
        assert [r["label"] for r in stream.project(("label",))] == ["a", "c"]
        assert stream.ops[0].decodes == 0
        assert [r["label"] for r in Stream(source=stream.source, ops=[self._Expensive()])] == ["a", "c"]

    def test_an_ops_chain_offering_nothing_cheap_is_unchanged(self) -> None:
        from recordstream import Stream

        def plain(record: Record) -> Record:
            return {**record, "n": record["file"] * 2}

        stream = Stream(source=[{"file": "a"}], ops=[plain])
        assert list(stream.project(("n",))) == [{"n": "aa"}]

    def test_the_cheap_path_is_refused_where_the_stream_would_not_run_sequentially(self) -> None:
        """Workers and a chunk size change WHAT iteration yields; the projection must not
        quietly take a different route than the stream it belongs to."""
        from recordstream import Stream

        for chunk_size in (2, 0):
            stream = Stream(source=[{"file": "a"}], ops=[self._Expensive()], chunk_size=chunk_size)
            assert (stream._cheap_ops() is None) is bool(chunk_size)


class TestACheapChainThatCannotFinish:
    """A chain is written against the record a full READ produces, so an op reaching for
    something the cheap step could not supply is expected, not exceptional.

    Measured on a real chain: marking the payload entry rather than removing it is what keeps
    the rest of the chain running (`RenameField: unknown key 'signal'` when it was removed) —
    and a chain that fails anyway simply gets the real chain's answer.
    """

    class _Cheapable:
        def __init__(self, cheap: bool = False) -> None:
            self.cheap = cheap
            self.reads = 0

        def __call__(self, record: Record) -> Record:
            if self.cheap:
                return {**record, "meta": "from the sidecar"}
            self.reads += 1
            return {**record, "meta": "from the file", "payload": "DECODED"}

        def for_projection(self) -> "TestACheapChainThatCannotFinish._Cheapable":
            return type(self)(cheap=True)

    def test_an_op_that_raises_on_the_cheap_record_falls_back(self) -> None:
        from recordstream import Stream
        from recordstream.ops.structure import RenameField

        op = self._Cheapable()
        stream = Stream(source=[{"file": "a"}], ops=[op, RenameField(src="payload", dst="input")])
        assert [r["meta"] for r in stream.project(("meta",))] == ["from the file"]
        assert op.reads == 1, "the raise must be a verdict, not an escape"

    def test_a_marked_entry_counts_as_an_ABSENCE_however_it_is_renamed(self) -> None:
        """The marker rides the value, so the chain can move it anywhere and it still says
        'this was never read' — which a key-name check could not survive."""
        from recordstream import Stream
        from recordstream.formats import STAND_IN
        from recordstream.ops.structure import RenameField

        class _Marking(TestACheapChainThatCannotFinish._Cheapable):
            def __call__(self, record: Record) -> Record:
                out = super().__call__(record)
                if self.cheap:
                    out["payload"] = STAND_IN
                return out

        op = _Marking()
        stream = Stream(source=[{"file": "a"}], ops=[op, RenameField(src="payload", dst="input")])
        assert [r["meta"] for r in stream.project(("meta",))] == ["from the sidecar"]
        assert op.reads == 0, "a question the sidecar answers must not read the file"
        assert [r["input"] for r in stream.project(("input",))] == ["DECODED"]
        assert op.reads == 1, "a question about the marked entry must read the file"
