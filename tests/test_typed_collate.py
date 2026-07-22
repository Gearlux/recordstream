"""The typed collate — batched TypedSample convention (golden shapes consumers rely on)."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from sampleflux import Image, Label, Mask, TypedSample, collate, get_collate, register_item


@register_item
@dataclass
class _CollateBlob:
    data: object = None
    rate: float = 1.0


def _sample(i: int) -> TypedSample:
    return TypedSample(
        {
            "image": Image(np.full((4, 5, 3), float(i), dtype=np.float32)),
            "mask": Mask(np.full((4, 5), i, dtype=np.int64)),
            "class": Label(i, classes=["a", "b", "c"]),
        },
        roles={"mask": "target", "class": "target"},
    )


class TestTypedCollate:
    def test_golden_shapes(self) -> None:
        # THE batch convention consumers rely on: batched TypedSample, payloads stacked
        # per field, per-item attrs as lists, roles preserved.
        batch = collate([_sample(0), _sample(1), _sample(2)])
        assert isinstance(batch, TypedSample)
        assert np.asarray(batch["image"]).shape == (3, 4, 5, 3)  # stacked payload
        assert np.asarray(batch["mask"]).shape == (3, 4, 5)
        assert batch["class"].value == [0, 1, 2]  # per-item attrs become lists
        assert batch["class"].classes == [["a", "b", "c"]] * 3
        assert batch.roles == {"image": "input", "mask": "target", "class": "target"}

    def test_auto_dispatch_and_explicit_key(self) -> None:
        samples = [_sample(0), _sample(1)]
        auto = collate(samples)  # TypedSample batch routes to "typed" automatically
        explicit = get_collate("typed")(samples)
        assert isinstance(auto, TypedSample) and isinstance(explicit, TypedSample)
        assert np.array_equal(np.asarray(auto["image"]), np.asarray(explicit["image"]))

    def test_torch_payloads_stack_to_tensor(self) -> None:
        samples = [
            TypedSample({"sig": _CollateBlob(torch.ones(8) * i, rate=float(i))}, roles={"sig": "input"})
            for i in range(2)
        ]
        batch = collate(samples)
        assert isinstance(batch["sig"].data, torch.Tensor) and batch["sig"].data.shape == (2, 8)
        assert batch["sig"].rate == [0.0, 1.0]

    def test_heterogeneous_batch_raises(self) -> None:
        odd = TypedSample({"other": Label("x")})
        with pytest.raises(ValueError, match="do not match the batch fields"):
            collate([_sample(0), odd])

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="empty batch"):
            get_collate("typed")([])

    def test_non_typed_items_raise(self) -> None:
        with pytest.raises(TypeError, match="expected TypedSample"):
            get_collate("typed")([1, 2, 3])
