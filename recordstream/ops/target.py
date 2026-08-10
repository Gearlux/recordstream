"""Target-shaping transforms over plain-dict records.

* :class:`EncodeTarget` / :class:`DecodeTarget` map a class-name ``Label`` to a class-id
  ``Label`` and back through an explicit lookup ``mapping`` — the declarative analogue of
  scikit-learn's ``LabelEncoder``. The mapping is pinned in config, NOT fitted, so
  train / eval / predict share one identical ordering.
* :class:`CocoToTorchVisionDetection` turns a HuggingFace / COCO ``objects`` annotation
  (``{bbox, category}``) into a torchvision detection target rendered as a
  :class:`~recordstream.Boxes` item.
* :class:`MasksToDetectionBoxes` derives detection boxes from a segmentation ``Mask``.

The detection conversions are the modality-neutral, image-detection counterparts of
a signal package's domain-specific region ops. The encoded target value is written verbatim; wrap
it into a framework tensor downstream (e.g. a collate function) when a loss needs one.
"""

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from confluid import configurable

from recordstream.items import Boxes, Label, Mask, MultiLabel, Record, item_data
from recordstream.transform import Transform

#: COCO / HuggingFace bounding-box layouts (all in absolute pixels). Closed set so a typo
#: fails at the call site and UIs / form-specs enumerate the choices.
BBoxFormat = Literal["xywh", "xyxy", "cxcywh"]


def _source_frame(record: Record) -> Optional[Tuple[int, int]]:
    """The ``(H, W)`` raster a record's boxes are stated in, or ``None`` when it cannot be read.

    An annotation gives coordinates in the pixel space of the image it annotates, so the frame
    is the IMAGE's — which the record carries but the annotation does not. It is looked up under
    the ``"image"`` key first (this engine's declared key vocabulary — what the op-family
    dispatch hands to a library, what the coupled resize defaults to, what the dataset sources
    yield), then as the first :class:`~recordstream.Image` item.

    It is deliberately narrow: only a declared image is trusted. A generic "first array with two
    dimensions" search would happily read a ``Boxes``' own ``[N, 4]`` box array as an ``N x 4``
    raster and record a confident lie. ``None`` is an ordinary answer — it leaves ``canvas``
    exactly as it was before this was recorded at all, so nothing depends on the lookup
    succeeding.
    """
    from recordstream.items import Image as ImageItem
    from recordstream.ops.image import image_frame

    candidate = record.get("image")
    if candidate is not None:
        frame = image_frame(candidate)
        if frame is not None:
            return frame
    for value in record.values():
        if isinstance(value, ImageItem):
            frame = image_frame(value)
            if frame is not None:
                return frame
    return None


def _lookup(value: Any, mapping: Dict[Any, Any], ignore_unknown: bool, default: Any, op_name: str) -> Any:
    """Return ``mapping[value]``, or ``default`` when missing and ``ignore_unknown``.

    A plain module-level function shared by :class:`EncodeTarget` / :class:`DecodeTarget`.
    """
    if value in mapping:
        return mapping[value]
    if ignore_unknown:
        return default
    record_keys = list(mapping)[:8]
    suffix = "..." if len(mapping) > 8 else ""
    raise KeyError(
        f"{op_name}: value {value!r} not in mapping (keys: {record_keys}{suffix}). "
        "Pass ignore_unknown=True to substitute `default` instead."
    )


def coco_to_detection(
    objects: Any,
    bbox_key: str = "bbox",
    category_key: str = "category",
    bbox_format: BBoxFormat = "xywh",
    label_offset: int = 0,
) -> Dict[str, Any]:
    """Convert a COCO / HuggingFace ``objects`` mapping to ``{"boxes": [N,4] xyxy, "labels": [N]}`` tensors.

    Each box is, by COCO convention, ``[x, y, w, h]`` in absolute pixels; ``category`` is an
    integer class id. An empty annotation yields empty ``[0,4]`` / ``[0]`` tensors (the
    negative-example contract torchvision detectors accept).
    """
    import torch

    if not isinstance(objects, dict):
        raise TypeError(
            f"coco_to_detection: expected a COCO/HF objects mapping "
            f"(a dict with {bbox_key!r}/{category_key!r}); got {type(objects).__name__}."
        )
    raw_boxes = objects.get(bbox_key) or []
    raw_labels = objects.get(category_key) or []

    if len(raw_boxes):
        boxes = torch.as_tensor(raw_boxes, dtype=torch.float32).reshape(-1, 4)
        if bbox_format == "xywh":  # COCO: top-left + size
            x, y, w, h = boxes.unbind(-1)
            boxes = torch.stack([x, y, x + w, y + h], dim=-1)
        elif bbox_format == "cxcywh":  # center + size
            cx, cy, w, h = boxes.unbind(-1)
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        # "xyxy": already in the output layout
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)

    if len(raw_labels):
        labels = torch.as_tensor(list(raw_labels), dtype=torch.int64).reshape(-1) + label_offset
    else:
        labels = torch.zeros((0,), dtype=torch.int64)

    return {"boxes": boxes, "labels": labels}


def masks_to_detection(
    mask: Any,
    label: int = 1,
    connected: bool = False,
    min_area: int = 1,
    connectivity: int = 4,
) -> Dict[str, Any]:
    """Derive ``{"boxes": [N,4] xyxy, "labels": [N]}`` tensors from a 2-D integer segmentation mask.

    ``connected=False`` (default) — an instance mask: each distinct non-zero pixel value is
    one object. ``connected=True`` — binarize then split into connected components. Every box
    gets class id ``label``. An empty mask yields empty ``[0,4]`` / ``[0]`` tensors.
    """
    import torch

    if hasattr(mask, "convert"):  # PIL image (e.g. an 'L' instance mask)
        mask = np.array(mask)
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise TypeError(
            f"masks_to_detection: expected a 2-D segmentation mask; got shape {getattr(mask, 'shape', None)}."
        )

    boxes: list = []
    if connected:
        from recordstream.ops.numpy import connected_component_boxes

        for x0, y0, x1, y1 in connected_component_boxes(mask != 0, min_area, connectivity):
            boxes.append((float(x0), float(y0), float(x1), float(y1)))
    else:
        for value in np.unique(mask):
            if int(value) == 0:
                continue
            ys, xs = np.where(mask == value)
            if int(ys.size) < min_area:
                continue
            boxes.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))

    if boxes:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.full((len(boxes),), label, dtype=torch.int64)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.int64)
    return {"boxes": boxes_t, "labels": labels_t}


@configurable(category="op", group="structure")
class EncodeTarget(Transform):
    """A class-NAME ``Label`` → a class-ID ``Label``.

    Reads a :class:`~recordstream.Label` field (``field``; blank picks the first ``Label``)
    whose ``.value`` is a raw class name and maps it to its class id through the config-pinned
    ``mapping`` — the declarative ``LabelEncoder`` analogue. The result is a new
    :class:`~recordstream.Label` (carrying the source label's ``classes`` vocabulary) written
    under ``output`` — blank (default) replaces the source field in place.

    Args:
        mapping: Lookup from raw label name → class id, e.g. ``{"DJI AVATA2": 2, ...}``. Must be non-empty.
        ignore_unknown: When ``False`` (default), raise on a label missing from ``mapping``; when
            ``True``, substitute ``default``.
        default: Value written for an unknown label when ``ignore_unknown=True`` (default ``0``).
        field: ``Label`` field to encode; blank (default) picks the first ``Label`` field.
        output: Key the encoded ``Label`` is written to; blank (default) replaces the source field in place.
    """

    handles = (Label, MultiLabel)
    consumes = (Label, MultiLabel)
    produces = (Label, MultiLabel)

    def __init__(
        self,
        mapping: Optional[Dict[Any, Any]] = None,
        ignore_unknown: bool = False,
        default: Any = 0,
        field: str = "",
        output: str = "",
    ) -> None:
        super().__init__()
        # Lazy / zero-arg: store config only; the non-empty requirement is validated lazily in __call__.
        self.mapping = dict(mapping) if mapping else {}
        self.ignore_unknown = bool(ignore_unknown)
        self.default = default
        self.field = str(field)
        self.output = str(output)

    def _find_label(self, record: Record) -> str:
        """Resolve the KEY of the label field to encode (``self.field`` or the first label item).

        Matches a :class:`~recordstream.Label` OR a :class:`~recordstream.MultiLabel` — both are
        label items, and a multi-label target must be encodeable through the same op.
        """
        if self.field:
            if self.field not in record:
                raise ValueError(f"EncodeTarget: field {self.field!r} not in record (keys: {list(record)})")
            item = record[self.field]
            if not isinstance(item, (Label, MultiLabel)):
                raise TypeError(
                    f"EncodeTarget: field {self.field!r} is {type(item).__name__}, expected a Label or MultiLabel"
                )
            return self.field
        for key, _item in ((k, v) for k, v in record.items() if isinstance(v, (Label, MultiLabel))):
            return key
        raise ValueError(f"EncodeTarget: no Label/MultiLabel field in record (keys: {list(record)})")

    def __call__(self, record: Record) -> Record:
        if not self.mapping:
            raise ValueError("EncodeTarget: mapping must contain at least one entry.")
        key = self._find_label(record)
        label = record[key]
        out_key = self.output or key
        if isinstance(label, MultiLabel):
            values = [_lookup(v, self.mapping, self.ignore_unknown, self.default, "EncodeTarget") for v in label.values]
            return {**record, out_key: MultiLabel(values, classes=label.classes)}
        encoded = _lookup(label.value, self.mapping, self.ignore_unknown, self.default, "EncodeTarget")
        return {**record, out_key: Label(encoded, classes=label.classes)}


@configurable(category="op", group="structure")
class DecodeTarget(Transform):
    """A class-ID ``Label`` → a class-NAME ``Label`` (inverse of :class:`EncodeTarget`).

    Reads a :class:`~recordstream.Label` field (``field``; blank picks the first ``Label``)
    whose ``.value`` is an encoded class id and maps it back to its label name through
    ``mapping`` — the readback half used in prediction / reporting. The result is a new
    :class:`~recordstream.Label` written under ``output`` (blank replaces in place).

    Args:
        mapping: Lookup from class id → label name, e.g. ``{2: "DJI AVATA2", ...}``. Must be non-empty.
        ignore_unknown: When ``False`` (default), raise on an id missing from ``mapping``; when
            ``True``, substitute ``default``.
        default: Value written for an unknown id when ``ignore_unknown=True`` (default ``None``).
        field: ``Label`` field to decode; blank (default) picks the first ``Label`` field.
        output: Key the decoded ``Label`` is written to; blank (default) replaces the source field in place.
    """

    handles = (Label, MultiLabel)
    consumes = (Label, MultiLabel)
    produces = (Label, MultiLabel)

    def __init__(
        self,
        mapping: Optional[Dict[Any, Any]] = None,
        ignore_unknown: bool = False,
        default: Any = None,
        field: str = "",
        output: str = "",
    ) -> None:
        super().__init__()
        # Lazy / zero-arg: store config only; the non-empty requirement is validated lazily in __call__.
        self.mapping = dict(mapping) if mapping else {}
        self.ignore_unknown = bool(ignore_unknown)
        self.default = default
        self.field = str(field)
        self.output = str(output)

    def _find_label(self, record: Record) -> str:
        """Resolve the KEY of the label field to decode (``self.field`` or the first label item).

        Matches a :class:`~recordstream.Label` OR a :class:`~recordstream.MultiLabel` — both are
        label items, and a multi-label target must be decodeable through the same op.
        """
        if self.field:
            if self.field not in record:
                raise ValueError(f"DecodeTarget: field {self.field!r} not in record (keys: {list(record)})")
            item = record[self.field]
            if not isinstance(item, (Label, MultiLabel)):
                raise TypeError(
                    f"DecodeTarget: field {self.field!r} is {type(item).__name__}, expected a Label or MultiLabel"
                )
            return self.field
        for key, _item in ((k, v) for k, v in record.items() if isinstance(v, (Label, MultiLabel))):
            return key
        raise ValueError(f"DecodeTarget: no Label/MultiLabel field in record (keys: {list(record)})")

    def __call__(self, record: Record) -> Record:
        if not self.mapping:
            raise ValueError("DecodeTarget: mapping must contain at least one entry.")
        key = self._find_label(record)
        label = record[key]
        out_key = self.output or key
        if isinstance(label, MultiLabel):
            values = [_lookup(v, self.mapping, self.ignore_unknown, self.default, "DecodeTarget") for v in label.values]
            return {**record, out_key: MultiLabel(values, classes=label.classes)}
        decoded = _lookup(label.value, self.mapping, self.ignore_unknown, self.default, "DecodeTarget")
        return {**record, out_key: Label(decoded, classes=label.classes)}


@configurable(category="op", group="structure")
class CocoToTorchVisionDetection(Transform):
    """A COCO / HF ``objects`` annotation → a target ``Boxes``.

    Reads a source field (``field``; blank picks the first :class:`~recordstream.Label`, else the
    first field) carrying a HuggingFace / COCO ``objects`` mapping and rewrites it to the
    torchvision detection target, riding as a :class:`~recordstream.Boxes` item under
    ``output`` (``boxes`` = the ``[N, 4]`` float32 xyxy tensor, ``labels`` = the ``[N]`` int64
    class-id tensor). An empty annotation yields empty ``[0,4]`` / ``[0]``
    tensors (the negative-example contract).

    Args:
        bbox_key: Key in the objects mapping holding per-box coordinates (default ``"bbox"``).
        category_key: Key holding the per-box integer class ids (default ``"category"``).
        bbox_format: Box layout in pixels — ``xywh`` (COCO, default), ``xyxy``, or ``cxcywh``; output is xyxy.
        label_offset: Added to each class id (default ``0``). Set ``1`` to reserve class ``0`` for background.
        field: Source field with the objects mapping; blank (default) picks the first ``Label``, else the first field.
        output: Key the target ``Boxes`` is written to (added if new).
    """

    handles = (Label,)
    consumes = (Label,)
    produces = (Boxes,)

    def __init__(
        self,
        bbox_key: str = "bbox",
        category_key: str = "category",
        bbox_format: BBoxFormat = "xywh",
        label_offset: int = 0,
        field: str = "",
        output: str = "target",
    ) -> None:
        super().__init__()
        self.bbox_key = str(bbox_key)
        self.category_key = str(category_key)
        self.bbox_format = bbox_format
        self.label_offset = int(label_offset)
        self.field = str(field)
        self.output = str(output)

    def _find_source(self, record: Record) -> str:
        """Resolve the KEY of the source field (``self.field``, else the first ``Label``, else the first field)."""
        if self.field:
            if self.field not in record:
                raise ValueError(
                    f"CocoToTorchVisionDetection: field {self.field!r} not in record (keys: {list(record)})"
                )
            return self.field
        for key, _item in ((k, v) for k, v in record.items() if isinstance(v, Label)):
            return key
        for key in record:
            return key
        raise ValueError("CocoToTorchVisionDetection: record is empty — no source field to read")

    def __call__(self, record: Record) -> Record:
        key = self._find_source(record)
        item = record[key]
        objects = item.value if isinstance(item, Label) else item_data(item)
        target = coco_to_detection(objects, self.bbox_key, self.category_key, self.bbox_format, self.label_offset)
        # `canvas` IS the frame the boxes are stated in — a COCO box is in the source image's
        # pixel space, so it is knowable here and recording it costs one lookup. Left None when
        # no image is in the record, which is what every Boxes carried before this.
        return {
            **record,
            self.output: Boxes(boxes=target["boxes"], labels=target["labels"], canvas=_source_frame(record)),
        }


@configurable(category="op", group="structure")
class MasksToDetectionBoxes(Transform):
    """A segmentation ``Mask`` → a target ``Boxes``.

    Reads the :class:`~recordstream.Mask` at ``field`` (blank = the first ``Mask`` in the record,
    else the first array-bearing item) as a 2-D integer mask and derives one tight
    ``[x0,y0,x1,y1]`` box per object. The target rides as a :class:`~recordstream.Boxes` item
    under ``output``. An empty mask yields empty ``[0,4]`` / ``[0]`` tensors.

    Args:
        label: Foreground class id assigned to every derived box (default ``1``; class 0 = background).
        connected: True = connected-components on a binary mask; False (default) = each non-zero value is one instance.
        min_area: Drop objects whose mask area (in pixels) is below this (default ``1``).
        connectivity: Connected-components neighborhood when ``connected=True`` — ``4`` or ``8`` (default ``4``).
        field: Name of the ``Mask`` field to read; blank (default) picks the first ``Mask`` (else the first array).
        output: Key the target ``Boxes`` is written to (added if new).
    """

    handles = (Mask,)
    consumes = (Mask,)
    produces = (Boxes,)

    def __init__(
        self,
        label: int = 1,
        connected: bool = False,
        min_area: int = 1,
        connectivity: int = 4,
        field: str = "",
        output: str = "target",
    ) -> None:
        super().__init__()
        self.label = int(label)
        self.connected = bool(connected)
        self.min_area = int(min_area)
        self.connectivity = int(connectivity)
        self.field = str(field)
        self.output = str(output)

    def _find_mask(self, record: Record) -> np.ndarray:
        """Resolve the mask array (``self.field``, else the first ``Mask``, else the first array-bearing item)."""
        if self.field:
            if self.field not in record:
                raise ValueError(f"MasksToDetectionBoxes: field {self.field!r} not in record (keys: {list(record)})")
            data = item_data(record[self.field])
        else:
            data = None
            for _key, item in ((k, v) for k, v in record.items() if isinstance(v, Mask)):
                data = item_data(item)
                break
            if data is None:
                for _key, item in record.items():
                    payload = item_data(item)
                    if isinstance(payload, np.ndarray):
                        data = payload
                        break
            if data is None:
                raise ValueError(
                    f"MasksToDetectionBoxes: no Mask or array-bearing field in record (keys: {list(record)})"
                )
        if not isinstance(data, np.ndarray):
            raise TypeError(f"MasksToDetectionBoxes: expected an np.ndarray mask, got {type(data).__name__}")
        return data

    def __call__(self, record: Record) -> Record:
        mask = self._find_mask(record)
        target = masks_to_detection(mask, self.label, self.connected, self.min_area, self.connectivity)
        # The boxes were derived FROM this mask, so its shape is the frame exactly — no lookup,
        # no fallback, nothing to be wrong about.
        height, width = int(mask.shape[0]), int(mask.shape[1])
        return {
            **record,
            self.output: Boxes(boxes=target["boxes"], labels=target["labels"], canvas=(height, width)),
        }


@configurable(category="op", group="structure")
class ResizeDetection(Transform):
    """Resize an image AND scale its detection-target boxes in ONE coupled step.

    The detection twin of the joint image+mask draw: a fixed-input-size detector needs the image
    resized, and a resize that moved the pixels without moving the boxes would silently train on
    misplaced targets. Reads the image under ``input_key`` (PIL or a uint8 HWC/2-D array),
    resizes it to ``(height, width)`` (bilinear, PIL), and scales the
    :class:`~recordstream.Boxes` under ``target_key`` by the same factors — torch boxes
    stay torch, numpy stays numpy. The resized ``Boxes`` records the new frame in ``canvas``.

    Ops that need no fixed size (torchvision detectors resize internally) simply omit this op —
    it exists for the detectors that require pre-sized square inputs.

    Args:
        width: Target width in pixels; required at use (validated lazily, ``0`` = unset).
        height: Target height in pixels; required at use (validated lazily, ``0`` = unset).
        input_key: Record key carrying the image (default ``"image"``).
        target_key: Record key carrying the target ``Boxes``; a record without it resizes the image alone.
    """

    consumes = (Boxes,)
    produces = (Boxes,)

    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        input_key: str = "image",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.input_key = str(input_key)
        self.target_key = str(target_key)

    def _resize_image(self, payload: Any) -> Any:
        from PIL import Image as PILImage

        if hasattr(payload, "convert"):  # PIL
            return payload.resize((self.width, self.height), PILImage.Resampling.BILINEAR)
        array = np.asarray(payload)
        if array.dtype != np.uint8:
            raise TypeError(
                f"ResizeDetection: expected a PIL image or a uint8 array under {self.input_key!r}; "
                f"got dtype {array.dtype}. Run it BEFORE any float conversion (e.g. before ToTensor)."
            )
        resized = PILImage.fromarray(array).resize((self.width, self.height), PILImage.Resampling.BILINEAR)
        return np.array(resized)

    @staticmethod
    def _scale_boxes(boxes: Any, sx: float, sy: float) -> Any:
        """Scale ``[N, 4]`` xyxy boxes by per-axis factors, preserving the array framework."""
        from recordstream._compat import is_torch_tensor

        if is_torch_tensor(boxes):
            import torch

            return boxes * torch.tensor([sx, sy, sx, sy], dtype=boxes.dtype)
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * np.array([sx, sy, sx, sy])

    def __call__(self, record: Record) -> Record:
        if self.width < 1 or self.height < 1:
            raise ValueError(f"ResizeDetection needs positive width/height (got {self.width}x{self.height}).")
        if self.input_key not in record:
            raise ValueError(f"ResizeDetection: field {self.input_key!r} not in record (keys: {list(record)})")
        item = record[self.input_key]
        payload = item_data(item)
        frame = _source_frame({"image": item})
        if frame is None:
            raise ValueError(
                f"ResizeDetection: the value under {self.input_key!r} is not an image "
                f"(got {type(payload).__name__}) — it has no raster to resize."
            )
        orig_h, orig_w = frame
        resized = self._resize_image(payload)
        merged = dict(record)
        from recordstream.items import NDArrayItem, with_data

        merged[self.input_key] = with_data(item, resized) if isinstance(item, NDArrayItem) else resized

        target = record.get(self.target_key)
        if isinstance(target, Boxes):
            import dataclasses

            # An EMPTY target is re-framed too. Scaling no boxes is a no-op, but leaving the
            # canvas behind would make a negative example the ONE record in a set whose frame is
            # unknown — and a frame check that silently skips exactly the records with nothing
            # to check is a check that reports a clean bill for the wrong reason.
            sx, sy = float(self.width) / float(orig_w), float(self.height) / float(orig_h)
            boxes = self._scale_boxes(target.boxes, sx, sy) if len(target.boxes) else target.boxes
            merged[self.target_key] = dataclasses.replace(target, boxes=boxes, canvas=(self.height, self.width))
        return merged


__all__ = [
    "EncodeTarget",
    "DecodeTarget",
    "CocoToTorchVisionDetection",
    "MasksToDetectionBoxes",
    "ResizeDetection",
    "coco_to_detection",
    "masks_to_detection",
]
