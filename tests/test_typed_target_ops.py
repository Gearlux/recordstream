"""Typed-bag TWINS of the tensorization + target-shaping ops.

Pins the native typed transforms that let a ``TypedSample`` classification pipeline build its
model INPUT tensor and its encoded TARGET ``Label`` without the legacy ``Sample`` path:

* :class:`sampleflux.ops.torch.ToTensor` — array-bearing field → CHW-float ``Image`` item;
* :class:`sampleflux.ops.target.MetadataToTarget` — a field / attr value → a target ``Label``;
* :class:`sampleflux.ops.target.EncodeTarget` / ``DecodeTarget`` — class-name ↔ class-id ``Label``.

Each twin REUSES its legacy op's math, so the twin's output is pinned byte-identical to a legacy
run on the equivalent ``Sample`` (parity). sampleflux-only — no waivefront import.
"""

import numpy as np
import pytest
import torch
from confluid.registry import get_registry, resolve_class

from sampleflux import Image, Label, Mask, TypedSample
from sampleflux.collate import typed_collate
from sampleflux.ops.image import ConvertToImage
from sampleflux.ops.target import DecodeTarget, DecodeTargetOp, EncodeTarget, EncodeTargetOp, MetadataToTarget
from sampleflux.ops.torch import ToTensor, ToTensorOp
from sampleflux.sample import Sample

_MAP = {"cat": 0, "dog": 1, "fox": 2}
_INV = {0: "cat", 1: "dog", 2: "fox"}


def _hwc_uint8() -> np.ndarray:
    return (np.arange(4 * 5 * 3).reshape(4, 5, 3) % 256).astype(np.uint8)


# --------------------------------------------------------------------------- #
# ToTensor
# --------------------------------------------------------------------------- #
class TestToTensor:
    def test_produces_chw_float_image_role_preserved(self) -> None:
        arr = _hwc_uint8()
        out = ToTensor()(TypedSample({"image": Image(arr)}, roles={"image": "input"}))
        img = out["image"]
        assert isinstance(img, Image)
        assert img.layout == "CHW"
        payload = np.asarray(img)
        assert payload.shape == (3, 4, 5)  # HWC -> CHW
        assert payload.dtype == np.float32
        assert payload.max() <= 1.0  # normalized
        assert out.role_of("image") == "input"  # replaced in place -> role preserved

    def test_parity_with_legacy_tensor(self) -> None:
        arr = _hwc_uint8()
        typed = ToTensor()(TypedSample({"image": Image(arr)}))
        legacy = ToTensorOp()(Sample(input=arr, target=None, metadata={})).input
        assert isinstance(legacy, torch.Tensor)
        assert np.array_equal(np.asarray(typed["image"]), legacy.numpy())

    def test_parity_no_normalize(self) -> None:
        arr = _hwc_uint8()
        typed = ToTensor(normalize=False)(TypedSample({"image": Image(arr)}))
        legacy = ToTensorOp(normalize=False)(Sample(input=arr, target=None, metadata={})).input
        assert np.array_equal(np.asarray(typed["image"]), legacy.numpy())

    def test_payload_is_numpy_not_live_tensor(self) -> None:
        # NDArrayItem coerces its payload via np.asarray, so an Image CANNOT hold a live tensor;
        # the stored CHW-float payload is a numpy array (values identical to the legacy tensor).
        from sampleflux.bag.items import item_data

        out = ToTensor()(TypedSample({"image": Image(_hwc_uint8())}))
        assert isinstance(item_data(out["image"]), np.ndarray)

    def test_new_output_field_tagged_input(self) -> None:
        arr = _hwc_uint8()
        out = ToTensor(output="tensor")(TypedSample({"image": Image(arr)}, roles={"image": "input"}))
        assert np.asarray(out["tensor"]).shape == (3, 4, 5)
        assert out.role_of("tensor") == "input"
        # original field left as-is (HWC uint8)
        assert np.asarray(out["image"]).shape == (4, 5, 3)

    def test_explicit_field(self) -> None:
        s = TypedSample({"a": Mask(np.zeros((2, 2), dtype=np.uint8)), "b": Image(_hwc_uint8())})
        out = ToTensor(field="b")(s)
        assert np.asarray(out["b"]).shape == (3, 4, 5)

    def test_default_picks_first_array_field(self) -> None:
        s = TypedSample({"lbl": Label("cat"), "image": Image(_hwc_uint8())})
        out = ToTensor()(s)
        assert np.asarray(out["image"]).shape == (3, 4, 5)

    def test_missing_explicit_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in sample"):
            ToTensor(field="nope")(TypedSample({"image": Image(_hwc_uint8())}))

    def test_no_array_field_raises(self) -> None:
        with pytest.raises(ValueError, match="no array-bearing field"):
            ToTensor()(TypedSample({"lbl": Label("cat")}))

    def test_typed_collate_stacks_payloads(self) -> None:
        # The typed collate stacks the CHW-float Image payloads into a batched array.
        a = ToTensor()(TypedSample({"image": Image(_hwc_uint8())}))
        b = ToTensor()(TypedSample({"image": Image(_hwc_uint8())}))
        batch = typed_collate([a, b])
        assert np.asarray(batch["image"]).shape == (2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# MetadataToTarget
# --------------------------------------------------------------------------- #
class TestMetadataToTarget:
    def test_promotes_label_value_to_target(self) -> None:
        s = TypedSample({"class": Label("cat")}, roles={"class": "aux"})
        out = MetadataToTarget(field="class", output="target")(s)
        assert isinstance(out["target"], Label)
        assert out["target"].value == "cat"
        assert out.role_of("target") == "target"

    def test_default_picks_first_label(self) -> None:
        s = TypedSample({"image": Image(_hwc_uint8()), "y": Label("dog")})
        out = MetadataToTarget()(s)
        assert out["target"].value == "dog"
        assert out.role_of("target") == "target"

    def test_read_named_attribute(self) -> None:
        # Read a carried attribute off a field (a value that rode as item-scoped metadata).
        s = TypedSample({"y": Label("cat", classes=["cat", "dog"])})
        out = MetadataToTarget(field="y", key="classes", output="vocab")(s)
        assert out["vocab"].value == ["cat", "dog"]

    def test_missing_attribute_raises(self) -> None:
        with pytest.raises(AttributeError, match="no attribute 'nope'"):
            MetadataToTarget(field="y", key="nope")(TypedSample({"y": Label("cat")}))

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field 'nope' not in sample"):
            MetadataToTarget(field="nope")(TypedSample({"y": Label("cat")}))

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="sample is empty"):
            MetadataToTarget()(TypedSample({}))


# --------------------------------------------------------------------------- #
# EncodeTarget / DecodeTarget
# --------------------------------------------------------------------------- #
class TestEncodeDecodeTarget:
    def test_encode_name_to_id_role_target(self) -> None:
        out = EncodeTarget(mapping=_MAP)(TypedSample({"y": Label("cat")}, roles={"y": "target"}))
        assert isinstance(out["y"], Label)
        assert out["y"].value == 0
        assert out.role_of("y") == "target"

    def test_encode_parity_with_legacy(self) -> None:
        for name in _MAP:
            typed = EncodeTarget(mapping=_MAP)(TypedSample({"y": Label(name)}))
            legacy = EncodeTargetOp(mapping=_MAP)(Sample(input=None, target=name, metadata={})).target
            assert typed["y"].value == legacy

    def test_encode_preserves_classes_vocab(self) -> None:
        out = EncodeTarget(mapping=_MAP)(TypedSample({"y": Label("dog", classes=list(_MAP))}))
        assert out["y"].value == 1
        assert out["y"].classes == list(_MAP)

    def test_encode_new_output_field(self) -> None:
        out = EncodeTarget(mapping=_MAP, output="target_id")(TypedSample({"y": Label("fox")}))
        assert out["target_id"].value == 2
        assert out.role_of("target_id") == "target"
        assert out["y"].value == "fox"  # source left intact

    def test_encode_ignore_unknown(self) -> None:
        out = EncodeTarget(mapping=_MAP, ignore_unknown=True, default=-1)(TypedSample({"y": Label("bird")}))
        assert out["y"].value == -1

    def test_encode_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            EncodeTarget(mapping=_MAP)(TypedSample({"y": Label("bird")}))

    def test_encode_empty_mapping_raises_lazily(self) -> None:
        op = EncodeTarget()  # constructible with no mapping (lazy)
        with pytest.raises(ValueError, match="at least one entry"):
            op(TypedSample({"y": Label("cat")}))

    def test_decode_id_to_name_parity(self) -> None:
        for cid in _INV:
            typed = DecodeTarget(mapping=_INV)(TypedSample({"y": Label(cid)}))
            legacy = DecodeTargetOp(mapping=_INV)(Sample(input=None, target=cid, metadata={})).target
            assert typed["y"].value == legacy

    def test_encode_then_decode_round_trip(self) -> None:
        s = TypedSample({"y": Label("dog")})
        encoded = EncodeTarget(mapping=_MAP)(s)
        assert encoded["y"].value == 1
        decoded = DecodeTarget(mapping=_INV)(encoded)
        assert decoded["y"].value == "dog"

    def test_decode_empty_mapping_raises_lazily(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            DecodeTarget()(TypedSample({"y": Label(0)}))

    def test_encode_non_label_field_raises(self) -> None:
        with pytest.raises(TypeError, match="expected a Label"):
            EncodeTarget(mapping=_MAP, field="image")(TypedSample({"image": Image(_hwc_uint8())}))

    def test_encode_no_label_field_raises(self) -> None:
        with pytest.raises(ValueError, match="no Label field"):
            EncodeTarget(mapping=_MAP)(TypedSample({"image": Image(_hwc_uint8())}))


# --------------------------------------------------------------------------- #
# End-to-end typed classification input/target path (sampleflux-only).
# --------------------------------------------------------------------------- #
def test_typed_classification_input_and_target_chain() -> None:
    # Source-shaped bag: an HWC image (role input) + a class-NAME label (role target).
    sample = TypedSample(
        {"image": Image(_hwc_uint8()), "class": Label("cat", classes=list(_MAP))},
        roles={"image": "input", "class": "target"},
    )
    # Build the model INPUT tensor (CHW float) and the encoded TARGET id — no legacy Sample.
    out = EncodeTarget(mapping=_MAP, field="class")(ToTensor(field="image")(sample))

    # Input field: a CHW-float Image tagged input.
    assert isinstance(out["image"], Image)
    assert out["image"].layout == "CHW"
    assert np.asarray(out["image"]).shape == (3, 4, 5)
    assert np.asarray(out["image"]).dtype == np.float32
    assert out.role_of("image") == "input"
    assert out.inputs().keys() == {"image"}

    # Target field: an int-id Label tagged target.
    assert isinstance(out["class"], Label)
    assert out["class"].value == 0
    assert out.role_of("class") == "target"
    assert out.targets().keys() == {"class"}


def test_convert_then_tensor_chain() -> None:
    # A raw 2-D array field runs ConvertToImage -> ToTensor into a CHW-float input.
    arr = np.arange(6 * 4).reshape(6, 4).astype(np.float32)
    out = ToTensor(field="image")(ConvertToImage(colormap="gray")(TypedSample({"spec": Mask(arr)})))
    assert out["image"].layout == "CHW"
    assert np.asarray(out["image"]).shape == (3, 6, 4)


# --------------------------------------------------------------------------- #
# Discovery + zero-arg construction.
# --------------------------------------------------------------------------- #
def test_zero_arg_constructible() -> None:
    assert ToTensor().output == ""
    assert MetadataToTarget().output == "target"
    assert EncodeTarget().mapping == {}
    assert DecodeTarget().mapping == {}


@pytest.mark.parametrize(
    ("name", "cls", "group"),
    [
        ("ToTensor", ToTensor, "torch"),
        ("MetadataToTarget", MetadataToTarget, "structure"),
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
