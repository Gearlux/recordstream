"""A ``Mask`` may name its own values: ``classes[v]`` is the class of every cell holding ``v``.

The same convention as ``Boxes.classes`` and ``Label.classes`` — the vocabulary travels WITH the encoded data, so a
viewer can name and colour a mask whose source declares no class list of its own.
"""

import pickle

import numpy as np

from recordstream.io import decode_item, encode_item
from recordstream.items import Mask


def test_a_mask_names_nothing_unless_told() -> None:
    assert Mask(np.zeros((2, 2), dtype=np.int16)).classes is None


def test_a_mask_carries_the_names_of_its_values() -> None:
    mask = Mask(np.array([[-1, 0], [0, -1]], dtype=np.int16), classes=["5GNR"])
    assert mask.classes == ["5GNR"]


def test_the_names_survive_numpy_operations() -> None:
    mask = Mask(np.array([[-1, 0], [0, -1]], dtype=np.int16), classes=["5GNR"])
    for changed in (mask[:1], np.flip(mask, axis=0)):
        assert isinstance(changed, Mask) and changed.classes == ["5GNR"]


def test_the_names_survive_a_pickle() -> None:
    back = pickle.loads(pickle.dumps(Mask(np.zeros((2, 2), dtype=np.int16), classes=["LTE", "WLAN"])))
    assert isinstance(back, Mask) and back.classes == ["LTE", "WLAN"]


def test_the_names_survive_the_item_codec() -> None:
    back = decode_item(encode_item(Mask(np.zeros((2, 2), dtype=np.int16), classes=["BTE"])))
    assert isinstance(back, Mask) and back.classes == ["BTE"]
