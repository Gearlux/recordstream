"""Legacy ``Sample`` <-> ``TypedSample`` bridge — lossless round-trip, builder path, errors.

Uses the modality-neutral core items plus a small test-local data-bearing wrapper (the shape a
signal item takes) so the wrapper round-trip is covered without importing a domain package.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from sampleflux.bag.interop import ENCODE_KEY, to_legacy, to_typed
from sampleflux.bag.items import Image, Label, Regions, register_item
from sampleflux.bag.sample import TypedSample
from sampleflux.sample import Sample


@register_item
@dataclass
class _WrapBlob:
    data: object = None
    tag: str = "x"


def _sample() -> TypedSample:
    return TypedSample(
        {
            "image": Image(np.arange(48, dtype=np.float32).reshape(4, 4, 3), layout="HWC"),
            "blob": _WrapBlob(np.arange(16, dtype=np.float32), tag="sig"),
            "regions": Regions(boxes=[[0, 0, 1, 1]], labels=["a"], canvas=(4, 8)),
            "class": Label("drone_x", classes=["noise", "drone_x"]),
        },
        roles={"regions": "target", "class": "target"},
    )


class TestRoundTrip:
    def test_lossless(self) -> None:
        s = _sample()
        assert to_typed(to_legacy(s)) == s

    def test_legacy_exposes_input_target(self) -> None:
        legacy = to_legacy(_sample())
        assert isinstance(legacy, Sample)
        assert np.asarray(legacy.input).shape == (4, 4, 3)  # first input field payload (the image)
        assert ENCODE_KEY in legacy.meta

    def test_reconstructs_item_types_and_meta(self) -> None:
        back = to_typed(to_legacy(_sample()))
        assert isinstance(back["image"], Image) and back["image"].layout == "HWC"
        assert isinstance(back["blob"], _WrapBlob) and back["blob"].tag == "sig"
        assert isinstance(back["regions"], Regions) and back["regions"].canvas == (4, 8)
        assert back.role_of("class") == "target"

    def test_no_input_or_target(self) -> None:
        s = TypedSample({"aux": _WrapBlob(np.ones(4))}, roles={"aux": "aux"})
        legacy = to_legacy(s)
        assert legacy.input is None and legacy.target is None
        assert to_typed(legacy) == s


class TestBuilderPath:
    def test_builder_used_when_no_encoding(self) -> None:
        raw = Sample(input=np.zeros((4, 4, 3)), target="cat", metadata={})

        def builder(sample: Sample) -> TypedSample:
            return TypedSample(
                {"image": Image(sample.input), "class": Label(sample.target)},
                roles={"class": "target"},
            )

        typed = to_typed(raw, builder=builder)
        assert isinstance(typed["image"], Image) and typed["class"].value == "cat"

    def test_no_encoding_no_builder_raises(self) -> None:
        with pytest.raises(ValueError, match="no embedded typed encoding"):
            to_typed(Sample(input=np.zeros(3), target=None, metadata={}))
