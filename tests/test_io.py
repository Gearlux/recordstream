"""The item codec registry (``recordstream.io``) — default structural codec, overrides, records,
the ``"plain"`` codec path."""

from dataclasses import dataclass

import numpy as np
import pytest

from recordstream import (
    EncodedItem,
    Image,
    Label,
    Regions,
    decode_item,
    decode_record,
    encode_item,
    encode_record,
    register_io,
    register_item,
)
from recordstream.io import PLAIN_TYPE


@register_item
@dataclass
class _IoBlob:
    """A data-bearing wrapper (the shape a domain signal item takes)."""

    data: object = None
    rate: float = 1.0


class TestDefaultCodec:
    def test_array_item_round_trip(self) -> None:
        img = Image(np.arange(12, dtype=np.float32).reshape(2, 2, 3), layout="CHW")
        enc = encode_item(img)
        assert enc.type_name == "Image" and enc.attrs == {"layout": "CHW"}
        back = decode_item(enc)
        assert isinstance(back, Image) and back.layout == "CHW"
        assert np.array_equal(np.asarray(back), np.asarray(img))

    def test_wrapper_item_round_trip(self) -> None:
        blob = _IoBlob(np.ones(4), rate=48000.0)
        enc = encode_item(blob)
        assert enc.type_name == "_IoBlob" and enc.attrs == {"rate": 48000.0}
        assert np.array_equal(enc.payload, np.ones(4))
        back = decode_item(enc)
        assert isinstance(back, _IoBlob) and back.rate == 48000.0

    def test_payloadless_item_round_trip(self) -> None:
        lab = Label("drone", classes=["a", "drone"])
        enc = encode_item(lab)
        assert enc.payload is None and enc.attrs == {"value": "drone", "classes": ["a", "drone"]}
        back = decode_item(enc)
        assert isinstance(back, Label) and back.value == "drone" and back.classes == ["a", "drone"]

    def test_unknown_type_name_raises(self) -> None:
        with pytest.raises(KeyError, match="no item type registered"):
            decode_item(EncodedItem(type_name="Nope", payload=None, attrs={}))


class TestPlainCodec:
    def test_scalar_round_trips_verbatim(self) -> None:
        for value in (-3.0, 7, "text", True):
            enc = encode_item(value)
            assert enc.type_name == PLAIN_TYPE and enc.payload == value and enc.attrs == {}
            assert decode_item(enc) == value

    def test_bare_array_round_trips_verbatim(self) -> None:
        arr = np.arange(6).reshape(2, 3)  # a bare ndarray is NOT a registered item -> "plain"
        enc = encode_item(arr)
        assert enc.type_name == PLAIN_TYPE
        back = decode_item(enc)
        assert type(back) is np.ndarray and np.array_equal(back, arr)

    def test_none_is_plain(self) -> None:
        enc = encode_item(None)
        assert enc.type_name == PLAIN_TYPE and decode_item(enc) is None


class TestRegisteredCodec:
    def test_override_wins_and_round_trips(self) -> None:
        @register_item
        class Compact:  # a type the default codec cannot capture
            def __init__(self, values: list) -> None:
                self.values = values

        register_io(
            Compact,
            encode=lambda item: (np.asarray(item.values), {}),
            decode=lambda payload, attrs: Compact(list(np.asarray(payload))),
        )
        enc = encode_item(Compact([1, 2, 3]))
        assert enc.type_name == "Compact" and np.array_equal(enc.payload, [1, 2, 3])
        back = decode_item(enc)
        assert isinstance(back, Compact) and back.values == [1, 2, 3]


class TestRecordCodec:
    def test_record_round_trip_keys_order_and_plain_entries(self) -> None:
        record = {
            "image": Image(np.zeros((2, 2, 3), dtype=np.float32)),
            "regions": Regions(boxes=[[0, 0, 1, 1]], labels=["a"], canvas=(2, 2)),
            "class": Label("x"),
            "gain_db": -3.0,  # a plain scalar rides the same layout under the "plain" tag
        }
        fields = encode_record(record)
        assert [f.key for f in fields] == ["image", "regions", "class", "gain_db"]
        assert fields[3].item.type_name == PLAIN_TYPE
        back = decode_record(fields)
        assert list(back.keys()) == list(record.keys())
        assert np.array_equal(np.asarray(back["image"]), np.asarray(record["image"]))
        assert back["regions"] == record["regions"]
        assert back["class"] == record["class"]
        assert back["gain_db"] == -3.0
