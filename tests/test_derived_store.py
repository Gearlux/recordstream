"""The derived store — ``recordstream.storage.derived`` and the ``Derive`` op."""

import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytest
from confluid import configurable, load

from recordstream import RECORD_ID, IdentifiedSource, Stream
from recordstream.items import Boxes, Image, Label
from recordstream.ops.derive import Derive
from recordstream.storage import derived
from recordstream.storage.derived import DerivedStore, DerivedStoreError, producer_hash

CALLS: List[str] = []


@configurable
class _Embed:
    """Writes ``embedding``: three numbers of ``scale``; counts its calls."""

    def __init__(self, scale: float = 1.0, weights: str = "") -> None:
        self.scale = scale
        self.weights = weights

    def __call__(self, record: Any) -> Any:
        CALLS.append(record[RECORD_ID])
        return {**record, "embedding": np.full(3, self.scale), "score": float(self.scale)}


@configurable
class _DropOdd:
    def __call__(self, record: Any) -> Any:
        return None if record["value"] % 2 else record


class _Opaque:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, record: Any) -> Any:
        return record


class _Source:
    dataset_uri = "d"

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> Any:
        if not 0 <= index < 4:
            raise IndexError(index)
        return {"value": index}


@pytest.fixture(autouse=True)
def _no_calls() -> None:
    CALLS.clear()


def _op(root: Path, **settings: Any) -> Derive:
    outputs: Optional[List[str]] = settings.pop("outputs", ["embedding"])
    files = settings.pop("files", None)
    return Derive(ops=[_Embed(**settings)], outputs=outputs, store=DerivedStore(root=str(root)), files=files)


class TestATableKeepsAValuePerRecord:
    def test_a_value_is_read_back_by_another_store(self, tmp_path: Path) -> None:
        with DerivedStore(root=str(tmp_path)) as store:
            table = store.table("p", "embedding")
            table.put("d#1", np.arange(4.0))
            table.put("d#2", np.ones(4))
            assert table.get("d#1").tolist() == [0.0, 1.0, 2.0, 3.0]  # readable before any flush
        table = DerivedStore(root=str(tmp_path)).table("p", "embedding")
        assert table.ids() == ["d#1", "d#2"] and len(table) == 2 and "d#2" in table
        assert table.array(["d#2", "d#1"]).tolist() == [[1.0] * 4, [0.0, 1.0, 2.0, 3.0]]
        assert table.missing(["d#1", "d#9"]) == ["d#9"]

    def test_a_typed_item_comes_back_as_its_type(self, tmp_path: Path) -> None:
        values = {
            "boxes": Boxes(boxes=np.zeros((1, 4)), labels=np.array([1]), scores=None, canvas=(4, 5)),
            "label": Label(2, classes=["a", "b", "c"]),
            "image": Image(np.zeros((2, 2, 3), dtype=np.uint8), layout="HWC"),
            "plain": {"pair": (1, 2), "nothing": None, "text": "x"},
            "number": 0.25,
        }
        with DerivedStore(root=str(tmp_path)) as store:
            for name, value in values.items():
                store.table("p", "values").put(name, value)
        table = DerivedStore(root=str(tmp_path)).table("p", "values")
        boxes = table.get("boxes")
        assert isinstance(boxes, Boxes) and boxes.canvas == (4, 5) and boxes.scores is None
        assert table.get("label") == values["label"]
        image = table.get("image")
        assert isinstance(image, Image) and image.layout == "HWC" and image.dtype == np.uint8
        assert table.get("plain") == values["plain"] and table.get("number") == 0.25

    def test_a_later_value_replaces_an_earlier_one(self, tmp_path: Path) -> None:
        store = DerivedStore(root=str(tmp_path))
        table = store.table("p", "score")
        table.put("d#1", 1.0)
        table.flush()
        table.put("d#1", 2.0)
        store.close()
        again = DerivedStore(root=str(tmp_path)).table("p", "score")
        assert again.get("d#1") == 2.0 and len(again) == 1

    def test_a_missing_value_is_a_key_error_or_the_default(self, tmp_path: Path) -> None:
        table = DerivedStore(root=str(tmp_path)).table("p", "score")
        assert table.get("d#1", None) is None
        with pytest.raises(KeyError):
            table.get("d#1")

    def test_it_writes_a_shard_on_its_own_when_enough_is_held(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(derived, "FLUSH_EVERY", 2)
        table = DerivedStore(root=str(tmp_path)).table("p", "score")
        for index in range(5):
            table.put(f"d#{index}", float(index))
        assert sorted(path.name for path in (tmp_path / "p" / "score").glob("*.npz")) == ["000000.npz", "000001.npz"]
        assert DerivedStore(root=str(tmp_path)).table("p", "score").ids() == ["d#0", "d#1", "d#2", "d#3"]

    def test_what_cannot_be_kept_is_refused_where_it_is_put(self, tmp_path: Path) -> None:
        table = DerivedStore(root=str(tmp_path)).table("p", "values")
        with pytest.raises(DerivedStoreError, match="output of its own"):
            table.put("d#1", {"array": np.zeros(2)})
        with pytest.raises(DerivedStoreError, match="record id"):
            table.put("", 1.0)
        assert len(table) == 0

    def test_a_name_that_is_a_path_is_refused(self, tmp_path: Path) -> None:
        store = DerivedStore(root=str(tmp_path))
        for producer, name in (("../p", "x"), ("p", "a/b"), ("p", ".hidden"), ("", "x")):
            with pytest.raises(DerivedStoreError, match="plain name"):
                store.table(producer, name)

    def test_an_empty_root_and_another_layout_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DerivedStoreError, match="`root` is empty"):
            DerivedStore().table("p", "x")
        folder = tmp_path / "p" / "x"
        folder.mkdir(parents=True)
        (folder / "index.json").write_text(json.dumps({"format": "older", "rows": {}}))
        with pytest.raises(DerivedStoreError, match="laid out as 'older'"):
            len(DerivedStore(root=str(tmp_path)).table("p", "x"))


class TestTheStoreSaysWhatItHoldsAndLetsItGo:
    def test_producers_are_listed_with_their_outputs_and_size(self, tmp_path: Path) -> None:
        with DerivedStore(root=str(tmp_path)) as store:
            store.table("p1", "embedding").put("d#1", np.zeros(2))
            store.table("p1", "score").put("d#1", 1.0)
            store.table("p2", "score").put("d#1", 1.0)
            store.describe("p1", "what p1 is")
        listed = DerivedStore(root=str(tmp_path)).producers()
        assert [entry["producer"] for entry in listed] == ["p1", "p2"]
        assert listed[0]["outputs"] == {"embedding": 1, "score": 1} and listed[0]["description"] == "what p1 is"
        assert listed[0]["bytes"] > 0 and listed[1]["description"] == ""
        assert DerivedStore(root=str(tmp_path / "nowhere")).producers() == []

    def test_a_removed_producer_is_gone_and_the_others_stay(self, tmp_path: Path) -> None:
        store = DerivedStore(root=str(tmp_path))
        store.table("p1", "score").put("d#1", 1.0)
        store.table("p2", "score").put("d#1", 2.0)
        store.flush()
        store.remove("p1")
        assert not (tmp_path / "p1").exists() and "d#1" not in store.table("p1", "score")
        assert store.table("p2", "score").get("d#1") == 2.0


class TestAProducerIsNamedByWhatItIs:
    def test_the_same_settings_give_the_same_name_written_out_or_not(self) -> None:
        assert producer_hash([_Embed()]) == producer_hash([_Embed(scale=1.0)])
        assert producer_hash([_Embed()]) != producer_hash([_Embed(scale=2.0)])
        assert producer_hash([_Embed()]) != producer_hash([_Embed(), _DropOdd()])
        assert producer_hash("my model v3") == producer_hash("my model v3")

    def test_a_files_content_is_part_of_the_name(self, tmp_path: Path) -> None:
        weights = tmp_path / "weights.bin"
        weights.write_bytes(b"one")
        first = producer_hash([_Embed()], files=[weights])
        assert first == producer_hash([_Embed()], files=[str(weights)]) != producer_hash([_Embed()])
        weights.write_bytes(b"two")
        assert producer_hash([_Embed()], files=[weights]) != first

    def test_an_op_whose_settings_cannot_be_written_is_refused(self) -> None:
        with pytest.raises(DerivedStoreError, match="cannot be written as a document"):
            producer_hash([_Opaque(scale=2.0)])


class TestDeriveRunsItsOpsOncePerRecord:
    def test_the_second_read_comes_from_the_store(self, tmp_path: Path) -> None:
        op = _op(tmp_path)
        first = op({RECORD_ID: "d#1", "value": 1})
        second = op({RECORD_ID: "d#1", "value": 1})
        assert CALLS == ["d#1"]
        assert first is not None and second is not None
        assert second["embedding"].tolist() == first["embedding"].tolist() == [1.0, 1.0, 1.0]
        assert second["value"] == 1 and "score" not in second  # only the named outputs are kept

    def test_another_process_finds_what_was_kept(self, tmp_path: Path) -> None:
        op = _op(tmp_path)
        op({RECORD_ID: "d#1"})
        op.close()
        again = _op(tmp_path)({RECORD_ID: "d#1"})
        assert CALLS == ["d#1"] and again is not None and again["embedding"].tolist() == [1.0, 1.0, 1.0]

    def test_another_setting_or_weights_file_is_computed_anew_beside_the_old(self, tmp_path: Path) -> None:
        weights = tmp_path / "w.bin"
        weights.write_bytes(b"one")
        root = tmp_path / "store"
        for op in (_op(root), _op(root, scale=2.0), _op(root, files=[str(weights)])):
            op({RECORD_ID: "d#1"})
            op.close()
        weights.write_bytes(b"two")
        changed = _op(root, files=[str(weights)])
        changed({RECORD_ID: "d#1"})
        changed.close()
        assert CALLS == ["d#1"] * 4
        assert len(DerivedStore(root=str(root)).producers()) == 4

    def test_an_output_missing_for_a_record_runs_the_ops_for_it(self, tmp_path: Path) -> None:
        _op(tmp_path)({RECORD_ID: "d#1"})
        both = _op(tmp_path, outputs=["embedding", "score"])
        assert both({RECORD_ID: "d#2"})["score"] == 1.0  # type: ignore[index]
        assert CALLS == ["d#1", "d#2"]

    def test_the_folder_says_what_the_producer_is(self, tmp_path: Path) -> None:
        op = _op(tmp_path, scale=3.0)
        op({RECORD_ID: "d#1"})
        op.close()
        description = (tmp_path / op.producer / "producer.yaml").read_text()
        assert "_Embed" in description and "scale: 3.0" in description

    def test_a_dropped_record_keeps_nothing(self, tmp_path: Path) -> None:
        store = DerivedStore(root=str(tmp_path))
        op = Derive(ops=[_DropOdd(), _Embed()], outputs=["embedding"], store=store)
        stream = Stream(source=IdentifiedSource(_Source()), ops=[op])
        assert [record[RECORD_ID] for record in stream] == ["d#0", "d#2"]
        assert store.table(op.producer, "embedding").ids() == ["d#0", "d#2"]

    def test_what_is_wrong_is_said_on_the_first_record(self, tmp_path: Path) -> None:
        store = DerivedStore(root=str(tmp_path))
        with pytest.raises(KeyError, match="IdentifiedSource"):
            _op(tmp_path)({"value": 1})
        with pytest.raises(ValueError, match="non-empty 'ops'"):
            Derive(outputs=["x"], store=store)({RECORD_ID: "d#1"})
        with pytest.raises(ValueError, match="requires `outputs`"):
            Derive(ops=[_Embed()], store=store)({RECORD_ID: "d#1"})
        with pytest.raises(ValueError, match="requires a `store`"):
            Derive(ops=[_Embed()], outputs=["embedding"])({RECORD_ID: "d#1"})
        with pytest.raises(KeyError, match="wrote no"):
            Derive(ops=[_Embed()], outputs=["absent"], store=store)({RECORD_ID: "d#1"})

    def test_it_builds_from_yaml(self, tmp_path: Path) -> None:
        document = f"""
        op: !class:recordstream.ops.derive.Derive
          store: !class:recordstream.storage.derived.DerivedStore {{root: {tmp_path}}}
          outputs: [embedding]
          ops:
            - !class:tests.test_derived_store._Embed {{scale: 4.0}}
        """
        op = load(document)["op"]
        record = op({RECORD_ID: "d#1"})
        assert record["embedding"].tolist() == [4.0, 4.0, 4.0]
        assert op.producer == producer_hash([_Embed(scale=4.0)])
