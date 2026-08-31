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


class _Decoded:
    """A decoded PIL-style image: ``datasets`` gives it a ``filename``."""

    def __init__(self, filename: str) -> None:
        self.filename = filename


class TestTheRecordRemembersWhereItCameFrom:
    """A file-backed dataset stamps ``hf_file``, so a consumer can rewrite or MOVE the original.

    It can be learned nowhere else: the decoded item is an array with no memory of its origin,
    and a folder dataset's whole point is that the file on disk is the thing being curated.
    """

    def _record(self, cell: object) -> dict:
        source = HuggingFaceSource(path="stub/dataset")
        return source._to_record({"image": cell, "label": 0}, [])

    def test_a_decoded_image_reports_its_filename(self) -> None:
        record = self._record(_Decoded("/data/mnist_png/test/0/10.png"))
        assert record["hf_file"].value == "/data/mnist_png/test/0/10.png"

    def test_an_undecoded_cell_reports_its_path(self) -> None:
        """``datasets`` hands back ``{"bytes": …, "path": …}`` when the column is not decoded."""
        record = self._record({"bytes": b"", "path": "/data/mnist_png/test/3/7.png"})
        assert record["hf_file"].value == "/data/mnist_png/test/3/7.png"

    def test_a_dataset_with_no_file_gets_NO_entry(self) -> None:
        """A Hub dataset is parquet-backed — an entry saying "" would claim a file exists."""
        import numpy as np

        assert "hf_file" not in self._record(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_the_provenance_entries_stay_beside_it(self) -> None:
        record = self._record(_Decoded("/data/x.png"))
        assert record["hf_path"].value == "stub/dataset" and "hf_split" in record


class TestTheClassVocabulary:
    """Read from the dataset's own schema — metadata, never a walk over the records."""

    def _source(self, features: object, loaded: object = None) -> HuggingFaceSource:
        source = HuggingFaceSource(path="stub/dataset")
        if loaded is not None:
            source._dataset = loaded
        return source

    def test_a_loaded_dataset_answers_from_its_features(self) -> None:
        class _ClassLabel:
            names = ["cat", "dog"]

        class _Loaded:
            features = {"image": object(), "label": _ClassLabel()}

        source = HuggingFaceSource(path="stub/dataset")
        source._dataset = _Loaded()
        assert source.class_names == ["cat", "dog"]

    def test_a_column_that_is_not_a_ClassLabel_yields_nothing(self) -> None:
        """A detection set nests its classes in an object field — there is no vocabulary here."""

        class _Loaded:
            features = {"image": object(), "label": object()}

        source = HuggingFaceSource(path="stub/dataset")
        source._dataset = _Loaded()
        assert source.class_names == []

    def test_the_CONFIGURED_column_is_read_not_the_first_ClassLabel(self) -> None:
        """A dataset may carry several; only the caller knows which one it targets."""

        class _Names:
            def __init__(self, names: list) -> None:
                self.names = names

        class _Loaded:
            features = {"label": _Names(["a", "b"]), "species": _Names(["cat", "dog"])}

        source = HuggingFaceSource(path="stub/dataset", target_feature="species")
        source._dataset = _Loaded()
        assert source.class_names == ["cat", "dog"]


class _NarrowableDataset:
    """A stand-in for ``datasets.Dataset`` that RECORDS how it was narrowed.

    Rows carry a sentinel image value, so a test can tell whether the image column was ever
    part of what iteration touched — the whole point of the narrowing is that it never is.
    """

    def __init__(self, rows: list, columns: list) -> None:
        self._rows = rows
        self.column_names = list(columns)
        self.selected: list = []
        self.cast: list = []

    def select_columns(self, columns: list) -> "_NarrowableDataset":
        self.selected.append(sorted(columns))
        narrowed = _NarrowableDataset(
            [{k: v for k, v in row.items() if k in columns} for row in self._rows], sorted(columns)
        )
        narrowed.selected = self.selected  # share the log so the test reads ONE place
        narrowed.cast = self.cast
        return narrowed

    def cast_column(self, column: str, feature: object) -> "_NarrowableDataset":
        self.cast.append((column, type(feature).__name__))
        undecoded = [{**row, column: {"bytes": b"", "path": f"/files/{i}.png"}} for i, row in enumerate(self._rows)]
        replaced = _NarrowableDataset(undecoded, self.column_names)
        replaced.selected = self.selected
        replaced.cast = self.cast
        return replaced

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._rows)


def _folder_like(n: int = 3) -> HuggingFaceSource:
    source = HuggingFaceSource(path="stub/dataset")
    source._dataset = _NarrowableDataset(
        [{"image": "DECODED-IMAGE", "label": i % 2} for i in range(n)], ["image", "label"]
    )
    return source


class TestProjectionNarrowsTheDataset:
    """``project`` selects the columns BEFORE iterating, so Arrow never decodes the rest.

    Measured before this existed (oxford_iiit_pet, 2000 rows, warm): the projected walk took
    5.6111s against 5.6556s for a FULL walk — no saving at all, while selecting the label
    column first costs 0.0092s (~570x). The docstring promised the saving; this makes it true.
    """

    def test_a_class_only_walk_selects_only_the_target_column(self) -> None:
        source = _folder_like()
        rows = list(source.project({"class"}))
        assert source._dataset.selected == [["label"]]
        assert [r["class"].value for r in rows] == [0, 1, 0]

    def test_the_image_never_reaches_iteration(self) -> None:
        source = _folder_like()
        for record in source.project({"class"}):
            assert "image" not in record and "DECODED-IMAGE" not in map(str, record.values())

    def test_hf_file_rides_the_UNDECODED_image_column(self) -> None:
        """The path is wanted, the pixels are not — ``decode=False`` delivers exactly that."""
        source = _folder_like()
        rows = list(source.project({"class", "hf_file"}))
        assert source._dataset.cast == [("image", "Image")]
        assert rows[0]["hf_file"].value == "/files/0.png"

    def test_asking_for_the_image_keeps_it_decoded(self) -> None:
        """The CON case: a projection that WANTS the image must not un-decode it."""
        source = _folder_like()
        rows = list(source.project({"image", "class"}))
        assert source._dataset.cast == []
        assert source._dataset.selected == [["image", "label"]]
        import numpy as np

        assert isinstance(rows[0]["image"], np.ndarray)

    def test_provenance_keys_read_no_row_data_at_all(self) -> None:
        """`select_columns([])` yields ZERO rows (measured) — so the no-column case must not
        select at all: one record per row, none of them touching the dataset's columns."""
        source = _folder_like()
        rows = list(source.project({"hf_path", "hf_split"}))
        assert source._dataset.selected == [], "no column is needed, so none may be selected"
        assert len(rows) == 3
        assert rows[0]["hf_path"].value == "stub/dataset" and rows[0]["hf_split"].value == "train"

    def test_a_dataset_without_select_columns_falls_back_to_the_full_walk(self) -> None:
        """Older/streaming datasets have no ``select_columns`` — the answer must not change."""
        source = HuggingFaceSource(path="stub/dataset")
        source._dataset = [{"image": "X", "label": 1}]  # a plain list: no narrowing API at all
        rows = list(source.project({"class"}))
        assert [r["class"].value for r in rows] == [1]

    def test_count_still_caps_the_projected_walk(self) -> None:
        source = _folder_like(n=5)
        source.count = 2
        assert len(list(source.project({"class"}))) == 2

    def test_metadata_columns_are_selected_when_asked(self) -> None:
        source = HuggingFaceSource(path="stub/dataset", metadata_features=["extra"])
        source._dataset = _NarrowableDataset(
            [{"image": "DECODED", "label": 1, "extra": "m"}], ["image", "label", "extra"]
        )
        rows = list(source.project({"class", "extra"}))
        assert source._dataset.selected == [["extra", "label"]]
        assert rows[0]["extra"].value == "m"
