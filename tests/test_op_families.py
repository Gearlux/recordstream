"""The engine's op-FAMILY dispatch (``core._apply_op``) — libraries run AS-IS, end to end.

Pins the record-model headline: native type-dispatched ops, BARE albumentations transforms
(kwarg-vocabulary call, one joint draw, item re-wrap), and BARE torchvision ``transforms.v2``
transforms (dict call) all sit in ONE ``Stream.ops`` list with no wrapper/adapter classes —
plus the family classifiers, YAML mapping-form ops docs, spawn-parallel with a bare library
op, ``field=`` targeting, and the ``WrappedOp``/``FilterOp`` raw-callable routes.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import albumentations as A
import numpy as np
import pytest
import torch
from confluid import configurable
from torchvision.transforms import v2

from recordstream import Boxes, FilterOp, Image, Label, Mask, Pipeline, Record, Transform, WrappedOp
from recordstream.core import Stream, _apply_op, _is_albumentations, _is_torchvision_v2


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
# Mixed ops list end-to-end through Stream.
# --------------------------------------------------------------------------- #
class TestMixedOpsList:
    _OPS = [
        AddOffset(offset=0.5),  # native type-dispatched op
        A.HorizontalFlip(p=1.0),  # bare albumentations
        v2.ToImage(),  # bare torchvision v2: explicit numpy HWC -> CHW tv_tensor conversion
        v2.RandomCrop(4),  # bare torchvision v2
    ]

    def test_iteration(self) -> None:
        stream = Stream(source=[_base_record()], ops=list(self._OPS))
        (out,) = list(stream)
        assert isinstance(out["image"], torch.Tensor)
        assert tuple(out["image"].shape) == (3, 4, 4)
        assert out["class"].value == "drone_x"  # rode through every family untouched

    def test_random_access(self) -> None:
        stream = Stream(source=[_base_record(0), _base_record(1)], ops=list(self._OPS))
        out = stream[1]
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
        stream = Stream.from_ops_yaml(str(path), source=[record])
        (out,) = list(stream)
        assert np.array_equal(np.asarray(out["image"]), np.asarray(record["image"])[:, ::-1])
        assert isinstance(out["image"], Image)

    def test_configurable_ctor_param_bound_from_yaml(self, tmp_path: Path) -> None:
        # An Awgn-style native op with a ctor param set in the YAML doc: the value reaches
        # the constructor and the op output reflects it.
        path = tmp_path / "ops.yaml"
        path.write_text("ops:\n  - !class:tests.test_op_families.AddOffset {offset: 3.0}\n")
        record = {"image": Image(np.zeros((2, 3, 3), dtype=np.float32))}
        stream = Stream.from_ops_yaml(str(path), source=[record])
        (out,) = list(stream)  # a @configurable entry stays a deferred marker until route entry
        assert np.allclose(np.asarray(out["image"]), 3.0)
        (op,) = stream.ops  # _check_ops_materialized flowed + cached the live op in place
        assert isinstance(op, AddOffset) and op.offset == 3.0


# --------------------------------------------------------------------------- #
# Spawn-parallel with a bare albumentations op (+ FilterOp drop on the parallel route).
# --------------------------------------------------------------------------- #
def test_spawn_parallel_with_bare_albumentations_op() -> None:
    records = [{**r, "idx": i} for i, r in enumerate(spawn_records())]
    stream = Stream(source=records, ops=[A.HorizontalFlip(p=1.0), FilterOp(keep_even_gain)]).parallel(2)
    results = stream.collect()
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
# Raw-callable routes: WrappedOp / Stream.map / FilterOp drops everywhere.
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

    def test_stream_map_key(self) -> None:
        stream = Stream(source=[{"m": Mask(np.ones((2, 2)))}]).map(double, key="m")
        (out,) = list(stream)
        assert isinstance(out["m"], Mask) and np.allclose(np.asarray(out["m"]), 2.0)


class TestFilterDropRoutes:
    def test_sequential_iteration_drops(self) -> None:
        records = [{"i": 0}, {"i": 1}, {"i": 2}]
        stream = Stream(source=records, ops=[FilterOp(lambda r: r["i"] != 1)])
        assert [r["i"] for r in stream] == [0, 2]

    def test_getitem_on_filtered_record_raises_index_error(self) -> None:
        stream = Stream(source=[{"i": 0}], ops=[FilterOp(lambda r: False)])
        with pytest.raises(IndexError, match="filtered out"):
            stream[0]

    def test_pipeline_propagates_drop(self) -> None:
        assert Pipeline([FilterOp(lambda r: False)])({"i": 0}) is None

    def test_stream_filter_helper(self) -> None:
        stream = Stream(source=[{"i": 0}, {"i": 1}]).filter(lambda r: r["i"] > 0)
        assert [r["i"] for r in stream] == [1]

    def test_unset_predicate_raises_lazily(self) -> None:
        with pytest.raises(ValueError, match="predicate"):
            FilterOp()({"i": 0})


# --------------------------------------------------------------------------- #
# The open op-family registry (register_op_family) — third-party libraries
# --------------------------------------------------------------------------- #
class FakeLibScale:
    """Stands in for a foreign library's op type — deliberately NOT record-callable,
    so a test passing proves dispatch went through the registered invoker."""

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor


def is_fakelib(op: object) -> bool:
    """Module-level matcher (pickles by reference for the spawn test)."""
    return isinstance(op, FakeLibScale)


def invoke_fakelib(record: Record, op: FakeLibScale) -> Record:
    """Module-level invoker — the fake library's calling convention."""
    return {**record, "gain_db": record["gain_db"] * op.factor}


def invoke_fakelib_override(record: Record, op: FakeLibScale) -> Record:
    """A second invoker for the shadowing / replacement tests."""
    return {**record, "gain_db": -999.0}


@pytest.fixture()
def family_registry() -> Iterator[None]:
    """Snapshot/restore the global registry so registrations never leak between tests."""
    from recordstream import core

    snapshot = list(core._OP_FAMILIES)
    yield
    core._OP_FAMILIES[:] = snapshot


class TestOpFamilyRegistry:
    def test_builtins_are_registered_through_the_same_registry(self) -> None:
        from recordstream import registered_op_families

        assert registered_op_families()[:2] == ("albumentations", "torchvision_v2")

    def test_registered_family_dispatches_via_invoker(self, family_registry: None) -> None:
        from recordstream import register_op_family

        register_op_family("fakelib", is_fakelib, invoke_fakelib)
        out = _apply_op(_base_record(), FakeLibScale(factor=3.0))
        assert out is not None and out["gain_db"] == -9.0  # -3.0 * 3 — via the invoker, op never called
        assert isinstance(out["image"], Image)  # rest of the record untouched

    def test_registered_family_runs_in_stream_ops_list(self, family_registry: None) -> None:
        from recordstream import register_op_family

        register_op_family("fakelib", is_fakelib, invoke_fakelib)
        out = list(Stream(source=[_base_record()], ops=[FakeLibScale(factor=2.0), lambda r: {**r, "tag": 1}]))
        assert out[0]["gain_db"] == -6.0 and out[0]["tag"] == 1  # mixes with native ops in ONE list

    def test_last_registered_family_wins_overlap(self, family_registry: None) -> None:
        from recordstream import register_op_family

        register_op_family("fakelib", is_fakelib, invoke_fakelib)
        register_op_family("fakelib_specific", is_fakelib, invoke_fakelib_override)  # same matcher, later
        out = _apply_op(_base_record(), FakeLibScale())
        assert out is not None and out["gain_db"] == -999.0

    def test_reregistering_name_replaces_in_place(self, family_registry: None) -> None:
        from recordstream import register_op_family, registered_op_families

        register_op_family("fakelib", is_fakelib, invoke_fakelib)
        n = len(registered_op_families())
        register_op_family("fakelib", is_fakelib, invoke_fakelib_override)
        assert len(registered_op_families()) == n  # replaced, not duplicated
        out = _apply_op(_base_record(), FakeLibScale())
        assert out is not None and out["gain_db"] == -999.0

    def test_unmatched_op_falls_back_to_native_call(self, family_registry: None) -> None:
        out = _apply_op(_base_record(), lambda r: {**r, "native": True})
        assert out is not None and out["native"] is True

    def test_spawn_parallel_ships_family_to_workers(self, family_registry: None) -> None:
        from recordstream import register_op_family

        register_op_family("fakelib", is_fakelib, invoke_fakelib)
        stream = Stream(source=spawn_records(), ops=[FakeLibScale(factor=2.0)]).parallel(2)
        results = list(stream)
        assert len(results) == 4
        assert all(r["gain_db"] == -6.0 for r in results)  # invoker ran INSIDE the workers


class TestFormulaReducers:
    def test_array_reducers_are_function_style(self) -> None:
        # amax/amin/mean/std/median are pre-bound numpy callables in the sandbox namespace.
        from recordstream.ops.formula import FormulaOp

        rec = {"image": Image(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0)}
        out = FormulaOp(formula="amax(a) * 0.5", field="image")(rec)
        assert float(np.asarray(out["image"])) == pytest.approx(0.5)

    def test_attribute_reduction_is_not_part_of_the_contract(self) -> None:
        # a.max() depends on numpy's lazy-import cache (KeyError '__import__' in a cold
        # process): the FUNCTION form is the sanctioned spelling. We only pin that the
        # function form never regresses; the attribute form is deliberately unpinned.
        from recordstream.ops.formula import _FORMULA_NAMESPACE

        assert {"amax", "amin", "mean", "std", "median"} <= set(_FORMULA_NAMESPACE)


@contextmanager
def _captured_warnings() -> Iterator[List[str]]:
    """Collect loggair WARNING records emitted inside the block.

    `caplog` cannot see these — loggair is loguru, which does not propagate to stdlib logging —
    and its sink is ENQUEUED, so reading a captured stream races the writer. `logger.complete()`
    is the deterministic flush (the workspace forbids sleeping for one).
    """
    from loguru import logger

    collected: List[str] = []
    sink_id = logger.add(lambda message: collected.append(str(message)), level="WARNING")
    try:
        yield collected
        logger.complete()
    finally:
        logger.remove(sink_id)


class TestGeometryLeavingBoxesBehind:
    """A `Boxes` is not in albumentations' key vocabulary, so it never reaches the library.

    That is correct for the dispatch — passing a foreign item would break the call — but it
    means a geometry-changing transform moves the pixels while the boxes stay put, with no
    error of its own. Measured: `A.Resize` takes a 200x200 image to 64x64 and leaves the boxes
    on `[10, 10, 100, 100]`; `A.HorizontalFlip` mirrors the pixels while changing NO shape at
    all, which is why the condition is the library's spatial/photometric taxonomy rather than
    "did the raster change".
    """

    @staticmethod
    def _record() -> Record:
        return {
            "image": Image(np.zeros((200, 200, 3), dtype="uint8")),
            "target": Boxes(boxes=np.array([[10.0, 10.0, 100.0, 100.0]]), labels=np.array([1])),
        }

    @pytest.fixture(autouse=True)
    def _forget_previous_warnings(self) -> Iterator[None]:
        """The once-per-type memo is module state — clear it so tests do not shadow each other."""
        from recordstream.core.families import _WARNED_SPATIAL

        snapshot = set(_WARNED_SPATIAL)
        _WARNED_SPATIAL.clear()
        yield
        _WARNED_SPATIAL.clear()
        _WARNED_SPATIAL.update(snapshot)

    def test_the_desync_is_real_and_silent_without_the_guard(self) -> None:
        """The premise, asserted rather than assumed: pixels move, boxes do not."""
        out = _apply_op(self._record(), A.Resize(height=64, width=64))
        assert out is not None
        assert out["image"].shape[:2] == (64, 64)
        assert out["target"].boxes.tolist() == [[10.0, 10.0, 100.0, 100.0]], "boxes stayed behind"

    def test_a_resize_warns_naming_both_ways_out(self) -> None:
        with _captured_warnings() as warnings:
            _apply_op(self._record(), A.Resize(height=64, width=64))
        assert len(warnings) == 1
        assert "bbox_params" in warnings[0] and "ResizeDetection" in warnings[0]

    def test_a_flip_warns_though_NO_shape_changes(self) -> None:
        with _captured_warnings() as warnings:
            _apply_op(self._record(), A.HorizontalFlip(p=1.0))
        assert len(warnings) == 1, "a raster-change test would miss this one entirely"

    def test_an_image_only_transform_stays_silent(self) -> None:
        """`Normalize` is an `ImageOnlyTransform` — it cannot touch geometry, so there is
        nothing to warn about. Reading the library's own taxonomy is what makes this exact."""
        with _captured_warnings() as warnings:
            _apply_op(self._record(), A.Normalize())
        assert warnings == []

    def test_a_compose_is_recursed(self) -> None:
        composed = A.Compose([A.Normalize(), A.RandomCrop(height=8, width=8)])
        with _captured_warnings() as warnings:
            _apply_op(self._record(), composed)
        assert len(warnings) == 1, "the spatial transform is nested one level down"

    def test_it_warns_once_per_transform_type(self) -> None:
        with _captured_warnings() as warnings:
            _apply_op(self._record(), A.Resize(height=64, width=64))
            _apply_op(self._record(), A.Resize(height=32, width=32))
        assert len(warnings) == 1, "the message is about the configuration, not the record"

    def test_the_DOCUMENTED_yaml_way_out_actually_runs(self) -> None:
        """The warning names a fix, so the fix has to work — this is that exact YAML.

        Note it must go through a `Stream`: deferred `!class:` markers are flowed at route
        entry, so applying them straight out of `confluid.load` hands `_apply_op` a marker.
        """
        import confluid

        document = """
ops:
  - !class:recordstream.ops.structure.RenameField { src: my_boxes, dst: bboxes }
  - !class:albumentations.Compose
    transforms: [!class:albumentations.HorizontalFlip { p: 1.0 }]
    bbox_params: !class:albumentations.BboxParams { format: pascal_voc, label_fields: [labels] }
"""
        record = {
            "image": np.zeros((100, 100, 3), dtype="uint8"),
            "my_boxes": [[10.0, 10.0, 40.0, 40.0]],
            "labels": [1],
        }
        with _captured_warnings() as warnings:
            out = list(Stream(source=[record], ops=confluid.load(document)["ops"]))[0]
        assert warnings == []
        assert [round(v, 1) for v in out["bboxes"][0]] == [60.0, 10.0, 90.0, 40.0], "mirrored across x"

    def test_the_correct_spelling_is_NOT_warned_about_and_moves_the_boxes(self) -> None:
        """Boxes in the library's own vocabulary: it moves them in the same joint draw."""
        composed = A.Compose(
            [A.Resize(height=64, width=64)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
        record = {
            "image": np.zeros((200, 200, 3), dtype="uint8"),
            "bboxes": [[10.0, 10.0, 100.0, 100.0]],
            "labels": [1],
        }
        with _captured_warnings() as warnings:
            out = _apply_op(record, composed)
        assert warnings == []
        assert out is not None
        assert [round(v, 1) for v in out["bboxes"][0]] == [3.2, 3.2, 32.0, 32.0], "boxes scaled with the image"


class TestV2GeometryLeavingBoxesBehind:
    """The same gap in the OTHER family, reached by a different route.

    albumentations misses a `Boxes` because it is not in the KEY vocabulary; torchvision v2
    misses it because it is not one of v2's tv_tensor TYPES. Measured: `v2.Resize((64, 64))`
    takes a 200x200 image to 64x64 with the boxes still on `[10, 10, 100, 100]`, while the same
    transform over a `tv_tensors.BoundingBoxes` rescales them to `[3.2, 3.2, 32, 32]`.
    """

    @staticmethod
    def _record() -> Record:
        return {
            "image": Image(np.zeros((200, 200, 3), dtype="uint8")),
            "target": Boxes(boxes=torch.tensor([[10.0, 10.0, 100.0, 100.0]]), labels=torch.tensor([1])),
        }

    @pytest.fixture(autouse=True)
    def _forget_previous_warnings(self) -> Iterator[None]:
        from recordstream.core.families import _WARNED_SPATIAL

        snapshot = set(_WARNED_SPATIAL)
        _WARNED_SPATIAL.clear()
        yield
        _WARNED_SPATIAL.clear()
        _WARNED_SPATIAL.update(snapshot)

    def test_the_desync_is_real(self) -> None:
        out = _apply_op(self._record(), v2.Compose([v2.ToImage(), v2.Resize((64, 64))]))
        assert out is not None
        assert tuple(out["image"].shape[-2:]) == (64, 64)
        assert out["target"].boxes.tolist() == [[10.0, 10.0, 100.0, 100.0]], "boxes stayed behind"

    def test_a_geometric_transform_warns_naming_the_way_out(self) -> None:
        with _captured_warnings() as warnings:
            _apply_op(self._record(), v2.RandomHorizontalFlip(p=1.0))
        assert len(warnings) == 1
        assert "BoundingBoxes" in warnings[0] and "ResizeDetection" in warnings[0]

    def test_a_non_geometric_transform_stays_silent(self) -> None:
        with _captured_warnings() as warnings:
            _apply_op(self._record(), v2.ColorJitter(brightness=0.5))
        assert warnings == []

    def test_a_compose_is_recursed(self) -> None:
        with _captured_warnings() as warnings:
            _apply_op(self._record(), v2.Compose([v2.ToImage(), v2.Resize((32, 32))]))
        assert len(warnings) == 1

    def test_v2s_OWN_box_type_is_transformed_and_not_warned_about(self) -> None:
        from torchvision import tv_tensors

        record = {
            "image": tv_tensors.Image(torch.zeros(3, 200, 200, dtype=torch.uint8)),
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor([[10.0, 10.0, 100.0, 100.0]]), format="XYXY", canvas_size=(200, 200)
            ),
        }
        with _captured_warnings() as warnings:
            out = _apply_op(record, v2.Resize((64, 64)))
        assert warnings == []
        assert out is not None
        assert [round(v, 1) for v in out["boxes"].tolist()[0]] == [3.2, 3.2, 32.0, 32.0]

    def test_the_geometry_signal_still_matches_this_torchvision(self) -> None:
        """The signal is a PRIVATE module path, so it can go stale on a torchvision upgrade.

        It fails OPEN (no warning, nothing else changes), which is the right direction for a
        diagnostic but also the direction that rots unnoticed — so assert the classification
        directly rather than only through a warning that would silently stop appearing.
        """
        from recordstream.core.families import _is_v2_geometry

        assert _is_v2_geometry(v2.Resize((8, 8)))
        assert _is_v2_geometry(v2.RandomHorizontalFlip())
        assert _is_v2_geometry(v2.RandomCrop(8))
        assert not _is_v2_geometry(v2.ColorJitter())
        assert not _is_v2_geometry(v2.Normalize(mean=[0.0], std=[1.0]))
