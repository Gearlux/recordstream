"""The engine's op-FAMILY dispatch (``core._apply_op``) — libraries run AS-IS, end to end.

Pins the record-model headline: native type-dispatched ops, BARE albumentations transforms
(kwarg-vocabulary call, one joint draw, item re-wrap), and BARE torchvision ``transforms.v2``
transforms (dict call) all sit in ONE ``Flux.ops`` list with no wrapper/adapter classes —
plus the family classifiers, YAML mapping-form ops docs, spawn-parallel with a bare library
op, ``field=`` targeting, and the ``WrappedOp``/``FilterOp`` raw-callable routes.
"""

from pathlib import Path
from typing import Dict, List, Optional

import albumentations as A
import numpy as np
import pytest
import torch
from confluid import configurable
from torchvision.transforms import v2

from sampleflux import FilterOp, Image, Label, Mask, Pipeline, Record, Transform, WrappedOp
from sampleflux.core import Flux, _apply_op, _is_albumentations, _is_torchvision_v2


# --------------------------------------------------------------------------- #
# Module-level fixtures (spawn workers pickle records, ops, and callables).
# --------------------------------------------------------------------------- #
def _base_record(i: int = 0) -> Record:
    rng = np.random.default_rng(i)
    return {
        "image": Image(rng.random((16, 20, 3)).astype(np.float32)),
        "mask": Mask((rng.random((16, 20)) > 0.5).astype(np.uint8)),
        "class": Label("drone_x", classes=["noise", "drone_x"]),
        "gain_db": -3.0,
    }


def spawn_records() -> List[Record]:
    """Module-level source fixture so the spawn-parallel test's records pickle."""
    return [_base_record(i) for i in range(4)]


def keep_even_gain(record: Record) -> bool:
    """Module-level FilterOp predicate (pickles across spawn workers)."""
    return int(record["idx"]) % 2 == 0


@configurable(category="op")
class AddOffset(Transform):
    """Adds a fixed offset to every Image value (Awgn-style configurable native op).

    Args:
        offset: The value added to every Image payload.
        field: Apply only to this record key. None (default) = every Image value.
    """

    handles = (Image,)

    def __init__(self, offset: float = 0.0, field: Optional[str] = None) -> None:
        super().__init__(field=field)
        self.offset = float(offset)

    def get_params(self, record: Record) -> Dict[str, float]:
        return {"offset": self.offset}


@AddOffset.kernel(Image)
def _add_offset_image(value: Image, params: Dict[str, float]) -> Image:
    return Image(np.asarray(value) + params["offset"], layout=value.layout)


# --------------------------------------------------------------------------- #
# Family classifiers.
# --------------------------------------------------------------------------- #
class TestFamilyClassifiers:
    def test_is_albumentations_positive(self) -> None:
        assert _is_albumentations(A.HorizontalFlip(p=1.0))
        assert _is_albumentations(A.Compose([A.HorizontalFlip(p=1.0)]))

    def test_is_albumentations_negative(self) -> None:
        assert not _is_albumentations(v2.RandomCrop(4))
        assert not _is_albumentations(AddOffset())
        assert not _is_albumentations({"image": None})
        assert not _is_albumentations(lambda r: r)

    def test_is_torchvision_v2_positive(self) -> None:
        assert _is_torchvision_v2(v2.RandomCrop(4))
        assert _is_torchvision_v2(v2.ToImage())

    def test_is_torchvision_v2_negative(self) -> None:
        assert not _is_torchvision_v2(A.HorizontalFlip(p=1.0))
        assert not _is_torchvision_v2(AddOffset())
        assert not _is_torchvision_v2(object())


# --------------------------------------------------------------------------- #
# Albumentations family.
# --------------------------------------------------------------------------- #
class TestAlbumentationsFamily:
    def test_joint_move_under_one_bare_compose_with_bboxes(self) -> None:
        # Box-carrying augmentation is albumentations' own Compose(bbox_params=...) dropped in
        # BARE — ONE joint draw moves image + mask + bboxes together; item types survive.
        record = {**_base_record(), "bboxes": [[2, 3, 6, 7]], "labels": ["drone"]}
        flip = A.Compose(
            [A.HorizontalFlip(p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
        out = _apply_op(record, flip)
        assert out is not None
        assert np.array_equal(np.asarray(out["image"]), np.asarray(record["image"])[:, ::-1])
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(record["mask"])[:, ::-1])
        assert [round(v) for v in out["bboxes"][0]] == [14, 3, 18, 7]  # W=20: x -> W-x
        assert isinstance(out["image"], Image) and isinstance(out["mask"], Mask)  # re-wrapped
        assert out["class"].value == "drone_x" and out["gain_db"] == -3.0  # non-alb keys untouched

    def test_zero_known_keys_is_passthrough(self) -> None:
        # A record with NO albumentations-vocabulary keys passes through unchanged.
        record = {"spec": Mask(np.zeros((4, 4))), "gain_db": 1.0}
        out = _apply_op(record, A.HorizontalFlip(p=1.0))
        assert out is record

    def test_extra_record_entries_never_reach_the_library(self) -> None:
        # Scalars / Labels are not in _ALB_KEYS — the op receives only image/mask and the
        # extras ride through verbatim (same objects).
        record = _base_record()
        out = _apply_op(record, A.HorizontalFlip(p=1.0))
        assert out is not None
        assert out["class"] is record["class"] and out["gain_db"] == -3.0


# --------------------------------------------------------------------------- #
# Mixed ops list end-to-end through Flux.
# --------------------------------------------------------------------------- #
class TestMixedOpsList:
    _OPS = [
        AddOffset(offset=0.5),  # native type-dispatched op
        A.HorizontalFlip(p=1.0),  # bare albumentations
        v2.ToImage(),  # bare torchvision v2: explicit numpy HWC -> CHW tv_tensor conversion
        v2.RandomCrop(4),  # bare torchvision v2
    ]

    def test_iteration(self) -> None:
        flux = Flux(source=[_base_record()], ops=list(self._OPS))
        (out,) = list(flux)
        assert isinstance(out["image"], torch.Tensor)
        assert tuple(out["image"].shape) == (3, 4, 4)
        assert out["class"].value == "drone_x"  # rode through every family untouched

    def test_random_access(self) -> None:
        flux = Flux(source=[_base_record(0), _base_record(1)], ops=list(self._OPS))
        out = flux[1]
        assert isinstance(out["image"], torch.Tensor) and tuple(out["image"].shape) == (3, 4, 4)

    def test_pipeline_nests_the_same_families(self) -> None:
        # The same mixed list nested inside Pipeline (which routes through _apply_op).
        out = Pipeline(list(self._OPS))(_base_record())
        assert out is not None and tuple(out["image"].shape) == (3, 4, 4)


# --------------------------------------------------------------------------- #
# YAML ops docs — mapping-form bare library entries + configurable ctor binding.
# --------------------------------------------------------------------------- #
class TestOpsYaml:
    def test_mapping_form_bare_albumentations_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "ops.yaml"
        path.write_text("ops:\n  - !class:albumentations.HorizontalFlip {p: 1.0}\n")
        record = _base_record()
        flux = Flux.from_ops_yaml(str(path), source=[record])
        (out,) = list(flux)
        assert np.array_equal(np.asarray(out["image"]), np.asarray(record["image"])[:, ::-1])
        assert isinstance(out["image"], Image)

    def test_configurable_ctor_param_bound_from_yaml(self, tmp_path: Path) -> None:
        # An Awgn-style native op with a ctor param set in the YAML doc: the value reaches
        # the constructor and the op output reflects it.
        path = tmp_path / "ops.yaml"
        path.write_text("ops:\n  - !class:tests.test_op_families.AddOffset {offset: 3.0}\n")
        record = {"image": Image(np.zeros((2, 3, 3), dtype=np.float32))}
        flux = Flux.from_ops_yaml(str(path), source=[record])
        (out,) = list(flux)  # a @configurable entry stays a deferred marker until route entry
        assert np.allclose(np.asarray(out["image"]), 3.0)
        (op,) = flux.ops  # _check_ops_materialized flowed + cached the live op in place
        assert isinstance(op, AddOffset) and op.offset == 3.0


# --------------------------------------------------------------------------- #
# Spawn-parallel with a bare albumentations op (+ FilterOp drop on the parallel route).
# --------------------------------------------------------------------------- #
def test_spawn_parallel_with_bare_albumentations_op() -> None:
    records = [{**r, "idx": i} for i, r in enumerate(spawn_records())]
    flux = Flux(source=records, ops=[A.HorizontalFlip(p=1.0), FilterOp(keep_even_gain)]).parallel(2)
    results = flux.collect()
    assert [int(r["idx"]) for r in results] == [0, 2]  # FilterOp dropped odd records in workers
    for out, want in zip(results, [records[0], records[2]]):
        assert isinstance(out["image"], Image)  # item type survived pickle + re-wrap
        assert np.array_equal(np.asarray(out["image"]), np.asarray(want["image"])[:, ::-1])


# --------------------------------------------------------------------------- #
# field= targeting.
# --------------------------------------------------------------------------- #
def test_field_targets_one_of_two_image_keys() -> None:
    record = {
        "a": Image(np.zeros((2, 2, 3), dtype=np.float32)),
        "b": Image(np.zeros((2, 2, 3), dtype=np.float32)),
    }
    out = AddOffset(offset=1.0, field="a")(record)
    assert out is not None
    assert np.allclose(np.asarray(out["a"]), 1.0)  # pinned key moved
    assert np.allclose(np.asarray(out["b"]), 0.0)  # sibling untouched


# --------------------------------------------------------------------------- #
# Raw-callable routes: WrappedOp / Flux.map / FilterOp drops everywhere.
# --------------------------------------------------------------------------- #
def double(x: np.ndarray) -> np.ndarray:
    """Module-level payload function for WrappedOp (stored as an importable path)."""
    return x * 2


def bump_gain(record: Record) -> Record:
    """Module-level whole-record function for WrappedOp(key=None)."""
    return {**record, "gain_db": record["gain_db"] + 1.0}


class TestWrappedOpAndMap:
    def test_key_targets_payload_and_preserves_item(self) -> None:
        record = {"m": Mask(np.ones((2, 2))), "gain_db": 0.0}
        out = WrappedOp(double, key="m")(record)
        assert out is not None
        assert isinstance(out["m"], Mask) and np.allclose(np.asarray(out["m"]), 2.0)
        assert record["m"] is not out["m"]  # copy-on-write

    def test_key_on_plain_value_replaces_verbatim(self) -> None:
        out = WrappedOp(double, key="g")({"g": 3})
        assert out is not None and out["g"] == 6  # plain value: no item to re-wrap

    def test_key_none_receives_whole_record(self) -> None:
        out = WrappedOp(bump_gain)({"gain_db": -3.0})
        assert out is not None and out["gain_db"] == -2.0

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError, match="nope"):
            WrappedOp(double, key="nope")({"m": Mask(np.ones(2))})

    def test_flux_map_key(self) -> None:
        flux = Flux(source=[{"m": Mask(np.ones((2, 2)))}]).map(double, key="m")
        (out,) = list(flux)
        assert isinstance(out["m"], Mask) and np.allclose(np.asarray(out["m"]), 2.0)


class TestFilterDropRoutes:
    def test_sequential_iteration_drops(self) -> None:
        records = [{"i": 0}, {"i": 1}, {"i": 2}]
        flux = Flux(source=records, ops=[FilterOp(lambda r: r["i"] != 1)])
        assert [r["i"] for r in flux] == [0, 2]

    def test_getitem_on_filtered_record_raises_index_error(self) -> None:
        flux = Flux(source=[{"i": 0}], ops=[FilterOp(lambda r: False)])
        with pytest.raises(IndexError, match="filtered out"):
            flux[0]

    def test_pipeline_propagates_drop(self) -> None:
        assert Pipeline([FilterOp(lambda r: False)])({"i": 0}) is None

    def test_flux_filter_helper(self) -> None:
        flux = Flux(source=[{"i": 0}, {"i": 1}]).filter(lambda r: r["i"] > 0)
        assert [r["i"] for r in flux] == [1]

    def test_unset_predicate_raises_lazily(self) -> None:
        with pytest.raises(ValueError, match="predicate"):
            FilterOp()({"i": 0})
