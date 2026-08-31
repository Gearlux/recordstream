"""``Normalize`` — the per-channel standardization node between a conversion and a model."""

from typing import Any, Dict, Optional

import numpy as np
import pytest

from recordstream.items import Image, Record
from recordstream.ops.image import Normalize


def _apply(op: Normalize, record: Dict[str, Any]) -> Record:
    out: Optional[Record] = op(record)
    assert out is not None  # a kernel Transform never drops a record
    return out


class TestNormalize:
    def test_the_math_matches_the_albumentations_convention(self) -> None:
        """``(x - mean*max) / (std*max)`` — an all-black uint8 image becomes -mean/std."""
        record = {"image": Image(np.zeros((4, 5, 3), dtype=np.uint8))}
        out = _apply(Normalize(), record)
        expected = -np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
        assert np.allclose(out["image"][0, 0], expected, atol=1e-5)

    def test_the_item_type_survives_in_float32(self) -> None:
        out = _apply(Normalize(), {"image": Image(np.full((2, 2, 3), 128, dtype=np.uint8))})
        assert isinstance(out["image"], Image) and out["image"].dtype == np.float32

    def test_unhandled_values_pass_through(self) -> None:
        record = {"image": Image(np.zeros((2, 2, 3), dtype=np.uint8)), "note": "keep", "n": 3}
        out = _apply(Normalize(), record)
        assert out["note"] == "keep" and out["n"] == 3

    def test_field_pins_the_op_to_one_key(self) -> None:
        record = {"a": Image(np.zeros((2, 2, 3), dtype=np.uint8)), "b": Image(np.zeros((2, 2, 3), dtype=np.uint8))}
        out = _apply(Normalize(field="a"), record)
        assert out["a"].dtype == np.float32 and out["b"].dtype == np.uint8

    def test_custom_statistics_and_max_value(self) -> None:
        record = {"image": Image(np.ones((2, 2, 3), dtype=np.float32))}
        out = _apply(Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_value=1.0), record)
        assert np.allclose(out["image"], 1.0)

    def test_a_2d_map_with_per_channel_stats_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="ConvertToImage"):
            Normalize()({"image": Image(np.zeros((4, 5), dtype=np.uint8))})

    def test_zero_arg_construction_works(self) -> None:
        assert Normalize().max_value == 255.0
