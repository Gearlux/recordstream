"""Frames — where a pixel of the record as stored lands after the steps that moved it.

A box is stated in a raster's pixels. Once a step resizes, crops or pads that raster, the box and the raster
are in a NEW frame, and whoever needs the old one back — a prediction drawn on the stored picture, a labelled
box looked up in a network's feature map — needs the mapping between the two. A step that changes the frame
writes it down, in the record, under :data:`FRAMES`:

* :class:`Frames` — a record entry holding, for each named frame, the mapping from the frame of the record as
  stored (:data:`RECORD_FRAME`) into it: per axis a scale and an offset, ``x' = x * sx + ox``. That covers a
  resize, a crop, a pad and a letterbox, and any chain of them;
* :func:`add_step` — what a frame-changing op calls to write its own step after the ones before it;
* :func:`map_boxes` / :func:`map_points` — boxes or points from one frame to another.

A step that cannot be undone point for point (a rotation by an arbitrary angle, a warp) is not an axis-aligned
mapping and writes none: a frame with no mapping is a refusal for whoever asks, never a guess.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from recordstream.exceptions import RecordstreamError
from recordstream.items import Record, register_item

#: The entry a record's frame mappings are carried under.
FRAMES = "frames"
#: The frame every mapping starts from: the record as its source gave it.
RECORD_FRAME = "record"

#: ``(sx, sy, ox, oy)``: a point ``(x, y)`` of the record's frame is ``(x * sx + ox, y * sy + oy)`` in the frame.
FrameMap = Tuple[float, float, float, float]

IDENTITY: FrameMap = (1.0, 1.0, 0.0, 0.0)


class FrameError(RecordstreamError, KeyError):
    """A frame was asked for that the record holds no mapping to."""


@register_item
@dataclass
class Frames:
    """The mapping from the record's stored frame into each named frame.

    Attributes:
        maps: Frame name -> ``(sx, sy, ox, oy)``. The record's own frame (:data:`RECORD_FRAME`) needs no
            entry: it is the identity.
    """

    maps: Dict[str, FrameMap] = field(default_factory=dict)

    def into(self, name: str) -> FrameMap:
        """The mapping from the record's frame into frame ``name``."""
        if name == RECORD_FRAME:
            return IDENTITY
        if name not in self.maps:
            raise FrameError(f"no mapping into the frame {name!r} (the frames are: {[RECORD_FRAME, *self.maps]})")
        sx, sy, ox, oy = self.maps[name]
        return (float(sx), float(sy), float(ox), float(oy))

    def between(self, from_: str, to: str) -> FrameMap:
        """The mapping from frame ``from_`` into frame ``to``."""
        return compose(inverse(self.into(from_)), self.into(to))


def compose(first: FrameMap, then: FrameMap) -> FrameMap:
    """The one mapping that applies ``first`` and then ``then``."""
    sx1, sy1, ox1, oy1 = first
    sx2, sy2, ox2, oy2 = then
    return (sx1 * sx2, sy1 * sy2, ox1 * sx2 + ox2, oy1 * sy2 + oy2)


def inverse(mapping: FrameMap) -> FrameMap:
    """The mapping that undoes ``mapping``."""
    sx, sy, ox, oy = mapping
    if sx == 0 or sy == 0:
        raise ValueError(f"a frame mapping with a zero scale {mapping} cannot be undone")
    return (1.0 / sx, 1.0 / sy, -ox / sx, -oy / sy)


def frames_of(record: Record) -> Frames:
    """The record's frame mappings; an empty :class:`Frames` for a record no step has re-framed."""
    held = record.get(FRAMES)
    if held is None:
        return Frames()
    if not isinstance(held, Frames):
        raise TypeError(f"the record's {FRAMES!r} entry is a {type(held).__name__}, not a Frames")
    return held


def add_step(record: Record, frame: str, step: FrameMap, *, after: str = "") -> Record:
    """``record`` with frame ``frame`` moved by ``step``: what a frame-changing op returns.

    ``step`` maps the frame the op RECEIVED into the frame it leaves. That received frame is ``after`` when
    given, else ``frame`` itself when the record already has it (the op moves an existing frame further),
    else the record's own frame (the op is the first to leave it).

    Args:
        record: The record the op received.
        frame: The name of the frame the op leaves its raster in.
        step: The op's own mapping, ``(sx, sy, ox, oy)``.
        after: The frame the op received its raster in, when it is neither ``frame`` nor the record's.
    """
    if frame == RECORD_FRAME:
        raise ValueError(f"{RECORD_FRAME!r} names the record as stored; a step leaves it for another frame")
    held = frames_of(record)
    start = after or (frame if frame in held.maps else RECORD_FRAME)
    maps = dict(held.maps)
    maps[frame] = compose(held.into(start), step)
    return {**record, FRAMES: Frames(maps=maps)}


def map_points(points: Any, mapping: FrameMap) -> np.ndarray:
    """``[N, 2]`` ``(x, y)`` points through ``mapping``."""
    sx, sy, ox, oy = mapping
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.asarray(array * np.array([sx, sy]) + np.array([ox, oy]), dtype=np.float64)


def map_boxes(boxes: Any, mapping: FrameMap) -> np.ndarray:
    """``[N, 4]`` ``[x0, y0, x1, y1]`` boxes through ``mapping``; a mirrored axis keeps ``x0 <= x1``."""
    sx, sy, ox, oy = mapping
    array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    moved = array * np.array([sx, sy, sx, sy]) + np.array([ox, oy, ox, oy])
    x = np.sort(moved[:, [0, 2]], axis=1)
    y = np.sort(moved[:, [1, 3]], axis=1)
    return np.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], axis=1)


def map_record_boxes(record: Record, boxes: Any, from_: str = RECORD_FRAME, to: str = RECORD_FRAME) -> np.ndarray:
    """``boxes`` of frame ``from_`` in frame ``to``, by the mappings ``record`` carries.

    Example::

        in_model = map_record_boxes(record, record["target"].boxes, to="model")
    """
    return map_boxes(boxes, frames_of(record).between(from_, to))


__all__ = [
    "FRAMES",
    "IDENTITY",
    "RECORD_FRAME",
    "FrameError",
    "FrameMap",
    "Frames",
    "add_step",
    "compose",
    "frames_of",
    "inverse",
    "map_boxes",
    "map_points",
    "map_record_boxes",
]
