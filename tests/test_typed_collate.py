"""The record collate — batched record convention (golden shapes consumers rely on)."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from recordstream import Image, Label, Mask, Record, Regions, collate, collate_records, get_collate, register_item


@register_item
@dataclass
class _CollateBlob:
    data: object = None
    rate: float = 1.0


def _record(i: int) -> Record:
    return {
        "image": Image(np.full((4, 5, 3), float(i), dtype=np.float32)),
        "mask": Mask(np.full((4, 5), i, dtype=np.int64)),
        "class": Label(i, classes=["a", "b", "c"]),
        "gain_db": float(i) - 3.0,  # a plain scalar entry
    }


class TestRecordCollate:
    def test_golden_shapes(self) -> None:
        # THE batch convention consumers rely on: ONE batched record dict, payloads stacked
        # per key, per-record item attrs as lists, plain values as plain lists.
        batch = collate([_record(0), _record(1), _record(2)])
        assert isinstance(batch, dict)
        assert np.asarray(batch["image"]).shape == (3, 4, 5, 3)  # stacked payload
        assert isinstance(batch["image"], Image)
        assert np.asarray(batch["mask"]).shape == (3, 4, 5)
        assert batch["class"].value == [0, 1, 2]  # per-record attrs become lists
        assert batch["class"].classes == [["a", "b", "c"]] * 3
        assert batch["gain_db"] == [-3.0, -2.0, -1.0]  # plain values -> a plain list

    def test_default_key_and_explicit_key(self) -> None:
        records = [_record(0), _record(1)]
        default = collate(records)  # the default registry key is "record"
        explicit = get_collate("record")(records)
        assert isinstance(default, dict) and isinstance(explicit, dict)
        assert np.array_equal(np.asarray(default["image"]), np.asarray(explicit["image"]))
        assert get_collate("record") is collate_records

    def test_item_attr_lists_decode_back_into_one_item(self) -> None:
        batch = collate_records([_record(0), _record(1)])
        # The batched Image is ONE Image whose layout attr is the per-record list.
        assert isinstance(batch["image"], Image) and batch["image"].layout == ["HWC", "HWC"]

    def test_torch_payloads_stack_to_tensor(self) -> None:
        records = [{"sig": _CollateBlob(torch.ones(8) * i, rate=float(i))} for i in range(2)]
        batch = collate(records)
        assert isinstance(batch["sig"].data, torch.Tensor) and batch["sig"].data.shape == (2, 8)
        assert batch["sig"].rate == [0.0, 1.0]

    def test_plain_string_values_batch_as_list(self) -> None:
        batch = collate_records([{"f": "a.iq"}, {"f": "b.iq"}])
        assert batch["f"] == ["a.iq", "b.iq"]

    def test_heterogeneous_batch_raises(self) -> None:
        odd = {"other": Label("x")}
        with pytest.raises(ValueError, match="do not match the batch keys"):
            collate([_record(0), odd])

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="empty batch"):
            get_collate("record")([])
        with pytest.raises(ValueError, match="empty batch"):
            collate([])

    def test_non_dict_items_raise(self) -> None:
        with pytest.raises(TypeError, match="expected record dicts"):
            get_collate("record")([1, 2, 3])

    def test_unknown_key_raises_with_known_keys(self) -> None:
        with pytest.raises(KeyError, match="no collate registered"):
            get_collate("nope")


# --------------------------------------------------------------------------- #
# The collate REGISTRY: a task collate opts out of the generic folding rules
# (the docs/record-model.md detection example — variable-N boxes cannot stack).
# --------------------------------------------------------------------------- #
def _ragged_detection_records():
    return [
        {
            "image": Image(np.zeros((4, 4, 3), dtype=np.float32)),
            "target": Regions(boxes=[[0, 0, 2, 2]], labels=[1]),
            "pack": "a",
        },
        {
            "image": Image(np.zeros((4, 4, 3), dtype=np.float32)),
            "target": Regions(boxes=[[0, 0, 1, 1], [1, 1, 3, 3], [0, 2, 2, 4]], labels=[0, 1, 0]),
            "pack": "b",
        },
    ]


def test_generic_collate_leaves_ragged_regions_as_lists() -> None:
    # Rule 2: the generic fold can only give per-record lists for a wrapper item —
    # exactly why detection registers its own collate.
    batch = collate_records(_ragged_detection_records())
    assert isinstance(batch["target"], Regions)
    assert [len(b) for b in batch["target"].boxes] == [1, 3]


def test_registered_task_collate_produces_its_own_batch_contract() -> None:
    import torch

    from recordstream import collate, get_collate, register_collate

    @register_collate("_test_detection")
    def detection_collate(items):
        images = torch.stack([torch.as_tensor(np.asarray(r["image"])).permute(2, 0, 1) for r in items])
        targets = [
            {
                "boxes": torch.as_tensor(r["target"].boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.as_tensor(r["target"].labels, dtype=torch.int64),
            }
            for r in items
        ]
        metadata = [{k: v for k, v in r.items() if k not in ("image", "target")} for r in items]
        return {"images": images, "targets": targets, "metadata": metadata}

    assert get_collate("_test_detection") is detection_collate
    batch = collate(_ragged_detection_records(), key="_test_detection")
    assert tuple(batch["images"].shape) == (2, 3, 4, 4)
    assert [tuple(t["boxes"].shape) for t in batch["targets"]] == [(1, 4), (3, 4)]  # raggedness preserved
    assert batch["metadata"] == [{"pack": "a"}, {"pack": "b"}]
