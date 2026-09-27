"""``ConvertMode`` — the channel-layout coercion that used to be ``ToTensor(mode=...)``.

A mixed-mode dataset (cppe-5 ships RGB, RGBA and grayscale rows) reaches a model with ragged
channel counts unless something forces one layout — found the hard way, as a channel-mismatch
crash MID-EPOCH after the 3-channel rows had trained fine. The coercion goes through PIL, which
represents ``uint8`` pixels only, so a non-``uint8`` image is REFUSED here (inside ``ToTensor``
it was silently left alone, which is how a float RGBA image would have slipped through).
"""

from typing import get_args

import numpy as np
import pytest
import torch

from recordstream import Image, Mask, Stream, batch_tensor, collate_records
from recordstream.ops.image import IMAGE_MODES, ConvertMode, ImageMode
from recordstream.ops.numpy import Scale
from recordstream.ops.torch import ToTensor


def _uint8(*shape: int) -> np.ndarray:
    return (np.arange(int(np.prod(shape))).reshape(shape) % 256).astype(np.uint8)


class TestConvertMode:
    def test_rgba_becomes_rgb(self) -> None:
        out = ConvertMode()({"image": Image(_uint8(4, 5, 4))})["image"]
        assert np.asarray(out).shape == (4, 5, 3)

    def test_a_2d_grayscale_image_becomes_rgb(self) -> None:
        out = ConvertMode()({"image": Image(_uint8(4, 5))})["image"]
        assert np.asarray(out).shape == (4, 5, 3)

    def test_a_singleton_channel_becomes_rgb(self) -> None:
        out = ConvertMode()({"image": Image(_uint8(4, 5, 1))})["image"]
        assert np.asarray(out).shape == (4, 5, 3)

    def test_rgb_stays_byte_identical(self) -> None:
        arr = _uint8(4, 5, 3)
        assert np.array_equal(np.asarray(ConvertMode()({"image": Image(arr)})["image"]), arr)

    def test_rgb_becomes_grayscale(self) -> None:
        out = ConvertMode(mode="L")({"image": Image(_uint8(4, 5, 3))})["image"]
        assert np.asarray(out).shape == (4, 5)

    def test_the_item_type_survives(self) -> None:
        out = ConvertMode()({"image": Image(_uint8(4, 5, 4))})["image"]
        assert isinstance(out, Image) and out.layout == "HWC"

    def test_a_float_image_is_refused_with_the_fix(self) -> None:
        with pytest.raises(ValueError, match=r"ConvertMode: 'image' is float32 — PIL modes hold uint8 pixels"):
            ConvertMode()({"image": Image(np.zeros((4, 5, 4), dtype=np.float32))})

    def test_a_channels_first_image_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"ConvertMode: 'image' is declared CHW"):
            ConvertMode()({"image": Image(_uint8(3, 4, 5), layout="CHW")})

    def test_blank_field_converts_every_image_and_no_mask(self) -> None:
        record = {"a": Image(_uint8(4, 5, 4)), "b": Image(_uint8(4, 5)), "target": Mask(_uint8(4, 5))}
        out = ConvertMode()(record)
        assert np.asarray(out["a"]).shape == (4, 5, 3) and np.asarray(out["b"]).shape == (4, 5, 3)
        assert out["target"] is record["target"]

    def test_a_named_field_converts_only_that_entry(self) -> None:
        record = {"a": Image(_uint8(4, 5, 4)), "b": Image(_uint8(4, 5, 4))}
        out = ConvertMode(field="b")(record)
        assert np.asarray(out["a"]).shape == (4, 5, 4) and np.asarray(out["b"]).shape == (4, 5, 3)

    def test_a_named_field_may_hold_a_pil_image(self) -> None:
        from PIL import Image as PILImage

        out = ConvertMode(field="image")({"image": PILImage.fromarray(_uint8(4, 5, 4))})["image"]
        assert np.asarray(out).shape == (4, 5, 3)

    def test_a_missing_field_is_named(self) -> None:
        with pytest.raises(ValueError, match=r"ConvertMode: field 'nope' not in record"):
            ConvertMode(field="nope")({"image": Image(_uint8(4, 5))})

    def test_the_mode_is_a_closed_choice(self) -> None:
        assert IMAGE_MODES == get_args(ImageMode) == ("RGB", "RGBA", "L")


def test_a_mixed_mode_dataset_reaches_the_model_as_one_batch_of_3_channel_floats() -> None:
    """The cppe-5 story end to end: mode, then range, then tensor — and the rows stack."""
    rows = [{"image": Image(_uint8(4, 5, 3))}, {"image": Image(_uint8(4, 5, 4))}, {"image": Image(_uint8(4, 5))}]
    stream = Stream(source=rows, ops=[ConvertMode(mode="RGB"), Scale(), ToTensor()])
    batch = batch_tensor(collate_records(list(stream)), "image")
    assert tuple(batch.shape) == (3, 3, 4, 5) and batch.dtype == torch.float32
    assert 0.0 <= float(batch.min()) and float(batch.max()) <= 1.0
