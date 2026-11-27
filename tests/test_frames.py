"""Frames — the mapping from the record as stored into the frames later steps leave it in."""

from pathlib import Path

import numpy as np
import pytest

from recordstream.frames import (
    FRAMES,
    IDENTITY,
    FrameError,
    Frames,
    add_step,
    compose,
    frames_of,
    inverse,
    map_boxes,
    map_points,
    map_record_boxes,
)
from recordstream.items import Boxes
from recordstream.ops.target import ResizeDetection
from recordstream.storage.derived import DerivedStore


class TestAMappingIsAScaleAndAnOffsetPerAxis:
    def test_a_point_and_a_box_move_through_it(self) -> None:
        half_then_shift = (0.5, 0.25, 10.0, 20.0)
        assert map_points([[100, 200]], half_then_shift).tolist() == [[60.0, 70.0]]
        assert map_boxes([[0, 0, 100, 200]], half_then_shift).tolist() == [[10.0, 20.0, 60.0, 70.0]]

    def test_two_steps_are_one_mapping_and_it_can_be_undone(self) -> None:
        resize, pad = (0.5, 0.5, 0.0, 0.0), (1.0, 1.0, 0.0, 16.0)
        both = compose(resize, pad)
        boxes = [[10.0, 20.0, 30.0, 40.0]]
        assert map_boxes(boxes, both).tolist() == map_boxes(map_boxes(boxes, resize), pad).tolist()
        assert np.allclose(map_boxes(map_boxes(boxes, both), inverse(both)), boxes)
        assert compose(both, inverse(both)) == pytest.approx(IDENTITY)

    def test_a_mirrored_axis_keeps_a_box_the_right_way_round(self) -> None:
        assert map_boxes([[10, 0, 30, 5]], (-1.0, 1.0, 100.0, 0.0)).tolist() == [[70.0, 0.0, 90.0, 5.0]]

    def test_a_zero_scale_cannot_be_undone(self) -> None:
        with pytest.raises(ValueError, match="cannot be undone"):
            inverse((0.0, 1.0, 0.0, 0.0))


class TestARecordCarriesItsFrames:
    def test_a_record_no_step_has_moved_has_only_its_own_frame(self) -> None:
        assert frames_of({}).into("record") == IDENTITY
        with pytest.raises(FrameError, match=r"no mapping into the frame 'model' \(the frames are: \['record'\]\)"):
            frames_of({}).into("model")

    def test_a_step_writes_its_mapping_and_a_later_one_continues_it(self) -> None:
        record = add_step({"image": 1}, "model", (0.5, 0.5, 0.0, 0.0))
        record = add_step(record, "model", (1.0, 1.0, 0.0, 16.0))
        assert record["image"] == 1 and frames_of(record).into("model") == (0.5, 0.5, 0.0, 16.0)

    def test_a_frame_can_start_from_another(self) -> None:
        record = add_step({}, "model", (0.5, 0.5, 0.0, 0.0))
        record = add_step(record, "feature_map", (1 / 16, 1 / 16, 0.0, 0.0), after="model")
        assert map_record_boxes(record, [[0, 0, 64, 64]], to="feature_map").tolist() == [[0.0, 0.0, 2.0, 2.0]]
        assert map_record_boxes(record, [[0, 0, 2, 2]], from_="feature_map", to="model").tolist() == [[0, 0, 32, 32]]
        assert map_record_boxes(record, [[0, 0, 32, 32]], from_="model").tolist() == [[0.0, 0.0, 64.0, 64.0]]

    def test_the_input_record_is_not_changed_and_the_records_frame_is_not_a_step(self) -> None:
        first = add_step({}, "model", (0.5, 0.5, 0.0, 0.0))
        add_step(first, "model", (2.0, 2.0, 0.0, 0.0))
        assert frames_of(first).into("model") == (0.5, 0.5, 0.0, 0.0)
        with pytest.raises(ValueError, match="names the record as stored"):
            add_step({}, "record", IDENTITY)

    def test_another_value_under_the_entry_is_refused(self) -> None:
        with pytest.raises(TypeError, match="not a Frames"):
            frames_of({FRAMES: {"model": IDENTITY}})

    def test_it_is_kept_and_read_back_as_itself(self, tmp_path: Path) -> None:
        with DerivedStore(root=str(tmp_path)) as store:
            store.table("p", FRAMES).put("d#1", Frames(maps={"model": (0.5, 0.25, 0.0, 16.0)}))
        back = DerivedStore(root=str(tmp_path)).table("p", FRAMES).get("d#1")
        assert isinstance(back, Frames) and back.into("model") == (0.5, 0.25, 0.0, 16.0)


class TestAResizeWritesItsFrame:
    def _record(self) -> dict:
        return {"image": np.zeros((100, 200, 3), dtype=np.uint8), "target": Boxes(boxes=[[20.0, 10.0, 60.0, 50.0]])}

    def test_a_box_of_the_resized_picture_goes_back_to_the_stored_one(self) -> None:
        out = ResizeDetection(width=100, height=25, frame="model")(self._record())
        assert frames_of(out).into("model") == (0.5, 0.25, 0.0, 0.0)
        resized = np.asarray(out["target"].boxes)
        assert resized.tolist() == [[10.0, 2.5, 30.0, 12.5]]
        assert map_record_boxes(out, resized, from_="model").tolist() == [[20.0, 10.0, 60.0, 50.0]]

    def test_a_second_resize_continues_the_same_frame(self) -> None:
        out = ResizeDetection(width=50, height=50, frame="model")(
            ResizeDetection(width=100, height=25, frame="model")(self._record())
        )
        assert frames_of(out).into("model") == (0.25, 0.5, 0.0, 0.0)

    def test_without_a_frame_name_nothing_is_written(self) -> None:
        assert FRAMES not in ResizeDetection(width=100, height=25)(self._record())
