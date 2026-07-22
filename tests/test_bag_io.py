"""The item codec registry (``sampleflux.bag.io``) — default structural codec, overrides, samples."""

from dataclasses import dataclass

import numpy as np
import pytest

from sampleflux import (
    EncodedItem,
    Image,
    Label,
    Regions,
    TypedSample,
    decode_item,
    decode_sample,
    encode_item,
    encode_sample,
    register_io,
    register_item,
)


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


class TestSampleCodec:
    def test_sample_round_trip_fields_roles_order(self) -> None:
        s = TypedSample(
            {
                "image": Image(np.zeros((2, 2, 3), dtype=np.float32)),
                "regions": Regions(boxes=[[0, 0, 1, 1]], labels=["a"], canvas=(2, 2)),
                "class": Label("x"),
            },
            roles={"regions": "target", "class": "target"},
        )
        fields = encode_sample(s)
        assert [f.key for f in fields] == ["image", "regions", "class"]
        assert [f.role for f in fields] == ["input", "target", "target"]
        back = decode_sample(fields)
        assert back == s
