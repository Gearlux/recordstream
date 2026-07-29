"""The tensorization + target-shaping ops over dict records.

Pins the native transforms that build a classification pipeline's model INPUT array and its
encoded TARGET ``Label`` on plain record dicts:

* :class:`recordstream.ops.torch.ToTensor` — array-bearing key → a LIVE CHW-float ``torch.Tensor``
  (a plain record value);
* :class:`recordstream.ops.target.EncodeTarget` / ``DecodeTarget`` — class-name ↔ class-id ``Label``.

Each op REUSES its shared conversion helper, so the op output is pinned identical to the
helper (parity). recordstream-only — no domain-package import.
"""

import numpy as np
import pytest
import torch
from confluid.registry import get_registry, resolve_class

from recordstream import Image, Label, Mask, collate_records, item_data
from recordstream.ops.image import ConvertToImage
from recordstream.ops.target import DecodeTarget, EncodeTarget
from recordstream.ops.torch import ToTensor, to_tensor

_MAP = {"cat": 0, "dog": 1, "fox": 2}
_INV = {0: "cat", 1: "dog", 2: "fox"}


def _hwc_uint8() -> np.ndarray:
    return (np.arange(4 * 5 * 3).reshape(4, 5, 3) % 256).astype(np.uint8)


# --------------------------------------------------------------------------- #
# ToTensor
# --------------------------------------------------------------------------- #
class TestToTensor:
    def test_produces_chw_float_tensor(self) -> None:
        arr = _hwc_uint8()
        out = ToTensor()({"image": Image(arr)})
        tensor = out["image"]
        assert isinstance(tensor, torch.Tensor)
        assert tuple(tensor.shape) == (3, 4, 5)  # HWC -> CHW
        assert tensor.dtype == torch.float32
        assert float(tensor.max()) <= 1.0  # normalized

    def test_parity_with_to_tensor_helper(self) -> None:
        arr = _hwc_uint8()
        out = ToTensor()({"image": Image(arr)})
        expected = to_tensor(arr).numpy()
        assert np.array_equal(np.asarray(out["image"]), expected)

    def test_parity_no_normalize(self) -> None:
        arr = _hwc_uint8()
        out = ToTensor(normalize=False)({"image": Image(arr)})
        expected = to_tensor(arr, normalize=False).numpy()
        assert np.array_equal(np.asarray(out["image"]), expected)

    def test_output_is_a_plain_live_tensor(self) -> None:
        # The record model holds arbitrary values: the tensor rides AS-IS (no Image wrap — an
        # NDArrayItem coerces via np.asarray and cannot hold a live tensor). item_data passes
        # a plain value through unchanged.
        out = ToTensor()({"image": Image(_hwc_uint8())})
        assert isinstance(out["image"], torch.Tensor)
        assert not isinstance(out["image"], Image)
        assert item_data(out["image"]) is out["image"]

    def test_new_output_key_keeps_source(self) -> None:
        arr = _hwc_uint8()
        out = ToTensor(output="tensor")({"image": Image(arr)})
        assert np.asarray(out["tensor"]).shape == (3, 4, 5)
        # original entry left as-is (HWC uint8)
        assert np.asarray(out["image"]).shape == (4, 5, 3)

    def test_explicit_field(self) -> None:
        rec = {"a": Mask(np.zeros((2, 2), dtype=np.uint8)), "b": Image(_hwc_uint8())}
        out = ToTensor(field="b")(rec)
        assert np.asarray(out["b"]).shape == (3, 4, 5)

    def test_default_picks_first_array_key(self) -> None:
        rec = {"lbl": Label("cat"), "image": Image(_hwc_uint8())}
        out = ToTensor()(rec)
        assert np.asarray(out["image"]).shape == (3, 4, 5)

    def test_missing_explicit_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in record"):
            ToTensor(field="nope")({"image": Image(_hwc_uint8())})

    def test_no_array_field_raises(self) -> None:
        with pytest.raises(ValueError, match="no array-bearing field"):
            ToTensor()({"lbl": Label("cat")})

    def test_record_collate_stacks_payloads(self) -> None:
        # The record collate stacks the CHW-float Image payloads into a batched array.
        a = ToTensor()({"image": Image(_hwc_uint8())})
        b = ToTensor()({"image": Image(_hwc_uint8())})
        batch = collate_records([a, b])
        assert np.asarray(batch["image"]).shape == (2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# EncodeTarget / DecodeTarget
# --------------------------------------------------------------------------- #
class TestEncodeDecodeTarget:
    def test_encode_name_to_id(self) -> None:
        out = EncodeTarget(mapping=_MAP)({"y": Label("cat")})
        assert isinstance(out["y"], Label)
        assert out["y"].value == 0

    def test_encode_maps_every_name(self) -> None:
        for name in _MAP:
            out = EncodeTarget(mapping=_MAP)({"y": Label(name)})
            assert out["y"].value == _MAP[name]

    def test_encode_preserves_classes_vocab(self) -> None:
        out = EncodeTarget(mapping=_MAP)({"y": Label("dog", classes=list(_MAP))})
        assert out["y"].value == 1
        assert out["y"].classes == list(_MAP)

    def test_encode_new_output_key(self) -> None:
        out = EncodeTarget(mapping=_MAP, output="target_id")({"y": Label("fox")})
        assert out["target_id"].value == 2
        assert out["y"].value == "fox"  # source left intact

    def test_encode_ignore_unknown(self) -> None:
        out = EncodeTarget(mapping=_MAP, ignore_unknown=True, default=-1)({"y": Label("bird")})
        assert out["y"].value == -1

    def test_encode_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            EncodeTarget(mapping=_MAP)({"y": Label("bird")})

    def test_encode_empty_mapping_raises_lazily(self) -> None:
        op = EncodeTarget()  # constructible with no mapping (lazy)
        with pytest.raises(ValueError, match="at least one entry"):
            op({"y": Label("cat")})

    def test_decode_id_to_name(self) -> None:
        for cid in _INV:
            out = DecodeTarget(mapping=_INV)({"y": Label(cid)})
            assert out["y"].value == _INV[cid]

    def test_encode_then_decode_round_trip(self) -> None:
        encoded = EncodeTarget(mapping=_MAP)({"y": Label("dog")})
        assert encoded["y"].value == 1
        decoded = DecodeTarget(mapping=_INV)(encoded)
        assert decoded["y"].value == "dog"

    def test_decode_empty_mapping_raises_lazily(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            DecodeTarget()({"y": Label(0)})

    def test_encode_non_label_field_raises(self) -> None:
        with pytest.raises(TypeError, match="expected a Label"):
            EncodeTarget(mapping=_MAP, field="image")({"image": Image(_hwc_uint8())})

    def test_encode_no_label_field_raises(self) -> None:
        with pytest.raises(ValueError, match="no Label/MultiLabel field"):
            EncodeTarget(mapping=_MAP)({"image": Image(_hwc_uint8())})


# --------------------------------------------------------------------------- #
# End-to-end classification input/target path (recordstream-only).
# --------------------------------------------------------------------------- #
def test_classification_input_and_target_chain() -> None:
    # Source-shaped record: an HWC image + a class-NAME label — key names carry meaning.
    record = {"image": Image(_hwc_uint8()), "class": Label("cat", classes=list(_MAP))}
    # Build the model INPUT (CHW float) and the encoded TARGET id.
    out = EncodeTarget(mapping=_MAP, field="class")(ToTensor(field="image")(record))

    # Input entry: a LIVE CHW-float tensor (a plain record value).
    assert isinstance(out["image"], torch.Tensor)
    assert tuple(out["image"].shape) == (3, 4, 5)
    assert out["image"].dtype == torch.float32

    # Target entry: an int-id Label carrying the vocabulary.
    assert isinstance(out["class"], Label)
    assert out["class"].value == 0
    assert out["class"].classes == list(_MAP)


def test_convert_then_tensor_chain() -> None:
    # A raw 2-D array entry runs ConvertToImage -> ToTensor into a CHW-float input.
    arr = np.arange(6 * 4).reshape(6, 4).astype(np.float32)
    out = ToTensor(field="image")(ConvertToImage(colormap="gray")({"spec": Mask(arr)}))
    assert isinstance(out["image"], torch.Tensor)
    assert tuple(out["image"].shape) == (3, 6, 4)


# --------------------------------------------------------------------------- #
# Discovery + zero-arg construction.
# --------------------------------------------------------------------------- #
def test_zero_arg_constructible() -> None:
    assert ToTensor().output == ""
    assert EncodeTarget().mapping == {}
    assert DecodeTarget().mapping == {}


@pytest.mark.parametrize(
    ("name", "cls", "group"),
    [
        ("ToTensor", ToTensor, "torch"),
        ("EncodeTarget", EncodeTarget, "structure"),
        ("DecodeTarget", DecodeTarget, "structure"),
    ],
)
def test_discovery_tags(name: str, cls: type, group: str) -> None:
    assert cls.__confluid_category__ == "op"  # type: ignore[attr-defined]
    assert cls.__confluid_group__ == group  # type: ignore[attr-defined]
    assert resolve_class(name) is cls
    registry = get_registry()
    assert name in registry.list_classes(category="op")
    assert name in registry.list_classes(group=group)
