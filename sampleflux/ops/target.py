"""Move and encode the supervised ``target`` field.

Companions to the input↔metadata movers (:class:`~sampleflux.ops.stash.StashInputOp` /
:class:`~sampleflux.ops.stash.UnstashInputOp`) and
:class:`~sampleflux.ops.swap.SwapInputTargetOp`:

* :class:`MetadataToTargetOp` moves a value from ``metadata`` onto ``sample.target``.
* :class:`EncodeTargetOp` / :class:`DecodeTargetOp` map ``sample.target`` through an
  explicit lookup and back — the declarative analogue of scikit-learn's
  ``LabelEncoder``. The label→id mapping is pinned in config, NOT fitted from
  whatever labels happen to appear, so train / eval / predict share one identical
  ordering.

The first three are deliberately small, value-agnostic plumbing ops (no ``ACCEPTS`` /
``PRODUCES`` contract, like ``copy`` / ``swap`` / ``stash``). The encoded value is
written verbatim (e.g. a plain ``int``); wrap it into a framework tensor downstream
(e.g. a collate function) when a loss needs one.

* :class:`CocoToTorchVisionDetectionOp` is the one structured-target op here: it turns a
  HuggingFace / COCO ``objects`` annotation (``{bbox, category}``) into the torchvision
  detection target ``{"boxes": xyxy, "labels"}`` (torch tensors). It is the generic,
  image-detection counterpart of waivefront's signal-domain ``RegionsToDetectionBoxesOp``.
"""

from typing import Any, Dict, Literal, Optional

from confluid import configurable

from sampleflux.bag.items import Label, item_data
from sampleflux.bag.sample import TypedSample
from sampleflux.bag.transform import Transform
from sampleflux.sample import Sample

#: COCO / HuggingFace bounding-box layouts (all in absolute pixels). Closed set so a typo
#: fails at the call site and UIs / form-specs enumerate the choices.
BBoxFormat = Literal["xywh", "xyxy", "cxcywh"]


def _lookup(value: Any, mapping: Dict[Any, Any], ignore_unknown: bool, default: Any, op_name: str) -> Any:
    """Return ``mapping[value]``, or ``default`` when missing and ``ignore_unknown``.

    Shared by :class:`EncodeTargetOp` / :class:`DecodeTargetOp`. A plain
    module-level function (NOT a base class) so the ops stay independent
    callables — SampleFlux Functional Purity.
    """
    if value in mapping:
        return mapping[value]
    if ignore_unknown:
        return default
    sample_keys = list(mapping)[:8]
    suffix = "..." if len(mapping) > 8 else ""
    raise KeyError(
        f"{op_name}: value {value!r} not in mapping (keys: {sample_keys}{suffix}). "
        "Pass ignore_unknown=True to substitute `default` instead."
    )


@configurable(category="op", group="structure")
class MetadataToTargetOp:
    """Set ``sample.target := metadata[key]``; optionally copy it to ``metadata[target_key]``.

    The metadata→target counterpart of :class:`~sampleflux.ops.stash.StashInputOp` /
    :class:`~sampleflux.ops.stash.UnstashInputOp` (which move input↔metadata). Typical
    use: a raw label rides in ``metadata`` and must become the supervised ``target``
    before :class:`EncodeTargetOp` overwrites it with a class id.

    Args:
        key: Metadata key to read the value from into ``sample.target`` (defaults to ``""``; a missing
            or empty key surfaces as a ``KeyError`` when the op runs, per the lazy-init convention).
        target_key: When set, the value is also written to ``metadata[target_key]``
            (so the raw label survives a later ``EncodeTargetOp`` and can be decoded
            back). ``None`` (default) leaves ``metadata`` untouched.
    """

    def __init__(self, key: str = "", target_key: Optional[str] = None) -> None:
        # Lazy / zero-arg: store config only; a missing key surfaces lazily as a KeyError in __call__.
        self.key = str(key)
        self.target_key = str(target_key) if target_key is not None else None

    def __call__(self, sample: Sample) -> Sample:
        if self.key not in sample.meta:
            raise KeyError(
                f"MetadataToTargetOp: sample.meta has no key {self.key!r}. " f"Available keys: {sorted(sample.meta)}"
            )
        value = sample.meta[self.key]
        if self.target_key is not None:
            sample.meta[self.target_key] = value
        return sample._replace(target=value)


@configurable(category="op", group="structure")
class EncodeTargetOp:
    """Encode ``sample.target`` through an explicit lookup ``mapping``.

    The declarative analogue of scikit-learn's ``LabelEncoder``: maps a raw target
    (typically a string label) to its class id via a config-pinned ``mapping``.
    Pinning the mapping — rather than fitting it from whatever labels appear — keeps
    train / eval / predict on one identical label→id ordering. The plain mapping value
    is written (framework-agnostic); tensorize the target downstream when a loss needs it.

    Args:
        mapping: Lookup from raw target → encoded value, e.g. ``{"DJI AVATA2": 2, ...}``.
            Must be non-empty.
        ignore_unknown: When ``False`` (default), raise on a target missing from
            ``mapping``; when ``True``, substitute ``default``.
        default: Value written for an unknown target when ``ignore_unknown=True``.
            Defaults to ``0``.
    """

    def __init__(
        self, mapping: Optional[Dict[Any, Any]] = None, ignore_unknown: bool = False, default: Any = 0
    ) -> None:
        # Lazy / zero-arg: store config only; the non-empty requirement is validated lazily in __call__.
        self.mapping = dict(mapping) if mapping else {}
        self.ignore_unknown = bool(ignore_unknown)
        self.default = default

    def __call__(self, sample: Sample) -> Sample:
        if not self.mapping:
            raise ValueError("EncodeTargetOp: mapping must contain at least one entry.")
        encoded = _lookup(sample.target, self.mapping, self.ignore_unknown, self.default, "EncodeTargetOp")
        return sample._replace(target=encoded)


@configurable(category="op", group="structure")
class DecodeTargetOp:
    """Decode ``sample.target`` through a lookup ``mapping`` (inverse of :class:`EncodeTargetOp`).

    Maps an encoded target (e.g. an integer class id) back to its label (e.g. a class
    name) — the readback half used in prediction / reporting.

    Args:
        mapping: Lookup from encoded value → decoded value, e.g. ``{2: "DJI AVATA2", ...}``.
            Must be non-empty.
        ignore_unknown: When ``False`` (default), raise on a target missing from
            ``mapping``; when ``True``, substitute ``default``.
        default: Value written for an unknown target when ``ignore_unknown=True``.
            Defaults to ``None``.
    """

    def __init__(
        self, mapping: Optional[Dict[Any, Any]] = None, ignore_unknown: bool = False, default: Any = None
    ) -> None:
        # Lazy / zero-arg: store config only; the non-empty requirement is validated lazily in __call__.
        self.mapping = dict(mapping) if mapping else {}
        self.ignore_unknown = bool(ignore_unknown)
        self.default = default

    def __call__(self, sample: Sample) -> Sample:
        if not self.mapping:
            raise ValueError("DecodeTargetOp: mapping must contain at least one entry.")
        decoded = _lookup(sample.target, self.mapping, self.ignore_unknown, self.default, "DecodeTargetOp")
        return sample._replace(target=decoded)


@configurable(category="op", group="structure")
class CocoToTorchVisionDetectionOp:
    """Convert a HuggingFace / COCO ``objects`` annotation to a torchvision detection target.

    HuggingFace object-detection datasets (e.g. ``cppe-5``) carry per-image annotations as an
    ``objects`` mapping — ``{"bbox": [[...], ...], "category": [...], ...}`` — where each box is,
    by COCO convention, ``[x, y, w, h]`` in absolute pixels and ``category`` is an integer class
    id. ``HuggingFaceSource(target_feature="objects")`` lands that mapping verbatim on
    ``sample.target``; this op rewrites it to the shape ``raidar.detection.detection_collate_fn``
    and the detection trainer consume::

        sample.target = {"boxes": [N, 4] float32 xyxy-pixel, "labels": [N] int64}

    The modality-neutral, image-detection counterpart of waivefront's signal-domain
    :class:`~waivefront.targets.RegionsToDetectionBoxesOp` (which projects time/frequency
    regions) — it lives in core sampleflux because the COCO→xyxy conversion is fully generic.
    The input image is left untouched (tensorize it with :class:`~sampleflux.ops.torch.ToTensorOp`).
    An empty annotation yields empty ``[0,4]`` / ``[0]`` tensors (the negative-example contract
    torchvision detectors accept).

    Args:
        bbox_key: Key in the objects mapping holding per-box coordinates (default ``"bbox"``).
        category_key: Key holding the per-box integer class ids (default ``"category"``).
        bbox_format: Box layout in pixels — ``xywh`` (COCO, default), ``xyxy``, or ``cxcywh``; output is xyxy.
        label_offset: Added to each class id (default ``0``). Set ``1`` to reserve class ``0`` for background.
    """

    def __init__(
        self,
        bbox_key: str = "bbox",
        category_key: str = "category",
        bbox_format: BBoxFormat = "xywh",
        label_offset: int = 0,
    ) -> None:
        # Lazy / zero-arg: store config only. The objects-shaped target is validated in __call__.
        self.bbox_key = str(bbox_key)
        self.category_key = str(category_key)
        self.bbox_format = bbox_format
        self.label_offset = int(label_offset)

    def __call__(self, sample: Sample) -> Sample:
        # torch is imported lazily so this module stays import-light for the value-agnostic
        # plumbing ops above (which need no framework).
        import torch

        objects = sample.target
        if not isinstance(objects, dict):
            raise TypeError(
                f"CocoToTorchVisionDetectionOp: sample.target must be a COCO/HF objects mapping "
                f"(a dict with {self.bbox_key!r}/{self.category_key!r}); got {type(objects).__name__}. "
                "Wire HuggingFaceSource(target_feature='objects') upstream."
            )
        raw_boxes = objects.get(self.bbox_key) or []
        raw_labels = objects.get(self.category_key) or []

        if len(raw_boxes):
            boxes = torch.as_tensor(raw_boxes, dtype=torch.float32).reshape(-1, 4)
            if self.bbox_format == "xywh":  # COCO: top-left + size
                x, y, w, h = boxes.unbind(-1)
                boxes = torch.stack([x, y, x + w, y + h], dim=-1)
            elif self.bbox_format == "cxcywh":  # center + size
                cx, cy, w, h = boxes.unbind(-1)
                boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
            # "xyxy": already in the output layout
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        if len(raw_labels):
            labels = torch.as_tensor(list(raw_labels), dtype=torch.int64).reshape(-1) + self.label_offset
        else:
            labels = torch.zeros((0,), dtype=torch.int64)

        return sample._replace(target={"boxes": boxes, "labels": labels})


@configurable(category="op", group="structure")
class MasksToDetectionBoxesOp:
    """Convert a segmentation MASK on ``sample.target`` to a torchvision detection target.

    Reads a 2-D integer mask (PIL ``L`` image or ndarray) and rewrites ``sample.target`` to
    ``{"boxes": [N,4] float32 xyxy-pixel, "labels": [N] int64}`` — the tight per-object box. This is
    the derivation the official torchvision **Penn-Fudan** object-detection tutorial performs (the
    dataset ships masks, not boxes). Two object-separation modes:

    * ``connected=False`` (default) — an **instance mask**: each distinct non-zero pixel value is one
      object (box = the tight extent of ``mask == value``). Penn-Fudan's ``instance_id`` mask (pixels
      ``1..N``, one per pedestrian) is exactly this — exact even when objects touch.
    * ``connected=True`` — a **binary / semantic mask**: binarize (non-zero), then split into connected
      components via :func:`sampleflux.ops.numpy.connected_component_bboxes` (one box per blob). Use for
      a semantic mask (all objects share one value) or a model's predicted foreground mask.

    Every box gets class id ``label`` (one foreground class; class 0 stays background — so a 1-class
    dataset like Penn-Fudan derives ``num_classes = 2``). The input image is left untouched (tensorize
    with :class:`~sampleflux.ops.torch.ToTensorOp` ``mode="RGB"``). An empty mask yields empty ``[0,4]`` /
    ``[0]`` tensors (the negative-example contract torchvision detectors accept).

    Args:
        label: Foreground class id assigned to every derived box (default ``1``; class 0 = background).
        connected: True = connected-components on a binary mask; False (default) = each non-zero value is one instance.
        min_area: Drop objects whose mask area (in pixels) is below this (default ``1``).
        connectivity: Connected-components neighborhood when ``connected=True`` — ``4`` or ``8`` (default ``4``).
    """

    def __init__(self, label: int = 1, connected: bool = False, min_area: int = 1, connectivity: int = 4) -> None:
        # Lazy / zero-arg: store config only; the mask shape is validated in __call__.
        self.label = int(label)
        self.connected = bool(connected)
        self.min_area = int(min_area)
        self.connectivity = int(connectivity)

    def __call__(self, sample: Sample) -> Sample:
        import numpy as np
        import torch

        mask = sample.target
        if hasattr(mask, "convert"):  # PIL image (e.g. an 'L' instance mask)
            mask = np.array(mask)
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise TypeError(
                f"MasksToDetectionBoxesOp: sample.target must be a 2-D segmentation mask "
                f"(PIL 'L' image or 2-D array); got shape {getattr(mask, 'shape', None)}. "
                "Wire HuggingFaceSource(target_feature='<mask column>') upstream."
            )

        boxes: list = []
        if self.connected:
            from sampleflux.ops.numpy import connected_component_bboxes

            # row/col-inclusive (r0,r1,c0,c1) → xyxy-pixel (x0,y0,x1,y1) with exclusive far edge.
            for r0, r1, c0, c1 in connected_component_bboxes(mask != 0, self.min_area, self.connectivity):
                boxes.append((float(c0), float(r0), float(c1 + 1), float(r1 + 1)))
        else:
            for value in np.unique(mask):
                if int(value) == 0:
                    continue
                ys, xs = np.where(mask == value)
                if int(ys.size) < self.min_area:
                    continue
                boxes.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.full((len(boxes),), self.label, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        return sample._replace(target={"boxes": boxes_t, "labels": labels_t})


@configurable(category="op", group="structure")
class MetadataToTarget(Transform):
    """Typed twin of :class:`MetadataToTargetOp` — promote a field / attr value into a target ``Label``.

    The typed-bag counterpart of :class:`MetadataToTargetOp`. The legacy op copies
    ``metadata[key]`` onto ``sample.target``, but the typed model has NO shared metadata dict — every
    item OWNS its metadata, and the supervised label already rides a :class:`~sampleflux.Label` field.
    So this twin reads a value from a SOURCE field (``field``; blank picks the first ``Label``, else
    the first field) — either the field's natural value (a ``Label``'s ``.value``, otherwise the
    item's array payload) or, when ``key`` is set, the named ATTRIBUTE of the source item — and writes
    a fresh :class:`~sampleflux.Label` under ``output`` tagged ``target``.

    REDUNDANCY. In a typical typed classification pipeline the source emits the label directly as a
    ``Label`` field already tagged ``target``, so this op is usually a NO-OP-ish re-home and is NOT
    needed. It is provided for parity / config-compat with the legacy ``metadata → target`` step and
    for the case where a label rode as another item's attribute (``key=``) and must become a
    dedicated target ``Label``.

    Args:
        field: Source field to read; blank (default) picks the first ``Label`` field, else the first field.
        key: Optional attribute name to read off the source item (e.g. a carried label attr); blank
            (default) reads the item's natural value (a ``Label``'s ``.value``, else its array payload).
        output: Field the target ``Label`` is written to (added if new); its role is set to ``target``.
    """

    handles = (Label,)
    consumes = (Label,)
    produces = (Label,)

    def __init__(self, field: str = "", key: str = "", output: str = "target") -> None:
        super().__init__()
        self.field = str(field)
        self.key = str(key)
        self.output = str(output)

    def _find_source(self, sample: TypedSample) -> str:
        """Resolve the KEY of the source field (``self.field``, else first ``Label``, else first field)."""
        if self.field:
            if self.field not in sample.keys():
                raise ValueError(
                    f"MetadataToTarget: field {self.field!r} not in sample (fields: {list(sample.keys())})"
                )
            return self.field
        for key, _item in sample.items_of_type(Label):
            return key
        for key in sample.keys():
            return key
        raise ValueError("MetadataToTarget: sample is empty — no source field to read")

    def __call__(self, sample: TypedSample) -> TypedSample:
        key = self._find_source(sample)
        item = sample[key]
        if self.key:
            if not hasattr(item, self.key):
                raise AttributeError(
                    f"MetadataToTarget: field {key!r} ({type(item).__name__}) has no attribute {self.key!r}"
                )
            value = getattr(item, self.key)
        elif isinstance(item, Label):
            value = item.value
        else:
            value = item_data(item)
        out = sample.replace_field(self.output, Label(value))
        return out.set_role(self.output, "target")


@configurable(category="op", group="structure")
class EncodeTarget(Transform):
    """Typed twin of :class:`EncodeTargetOp` — a class-NAME ``Label`` → a class-ID ``Label`` (role ``target``).

    The typed-bag counterpart of :class:`EncodeTargetOp`: it reads a :class:`~sampleflux.Label` field
    (``field``; blank picks the first ``Label``) whose ``.value`` is a raw class name and maps it to
    its class id through the config-pinned ``mapping`` — the declarative ``LabelEncoder`` analogue.
    This twin REUSES the legacy ``EncodeTargetOp`` verbatim (its non-empty-mapping validation AND its
    shared ``_lookup`` logic), so the encoded value is byte-identical. The result is a new
    :class:`~sampleflux.Label` (carrying the source label's ``classes`` vocabulary) written under
    ``output`` — blank (default) replaces the source field in place — tagged ``target``.

    Pinning the mapping (rather than fitting it) keeps train / eval / predict on one identical
    label→id ordering. The non-empty-mapping requirement is validated LAZILY when the op runs (the
    zero-arg default stays constructible per the lazy-init convention).

    Args:
        mapping: Lookup from raw label name → class id, e.g. ``{"DJI AVATA2": 2, ...}``. Must be non-empty.
        ignore_unknown: When ``False`` (default), raise on a label missing from ``mapping``; when
            ``True``, substitute ``default``.
        default: Value written for an unknown label when ``ignore_unknown=True`` (default ``0``).
        field: ``Label`` field to encode; blank (default) picks the first ``Label`` field.
        output: Field the encoded ``Label`` is written to; blank (default) replaces the source field
            in place. Its role is set to ``target``.
    """

    handles = (Label,)
    consumes = (Label,)
    produces = (Label,)

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

    def _find_label(self, sample: TypedSample) -> str:
        """Resolve the KEY of the ``Label`` field to encode (``self.field`` or the first ``Label``)."""
        if self.field:
            if self.field not in sample.keys():
                raise ValueError(f"EncodeTarget: field {self.field!r} not in sample (fields: {list(sample.keys())})")
            item = sample[self.field]
            if not isinstance(item, Label):
                raise TypeError(f"EncodeTarget: field {self.field!r} is {type(item).__name__}, expected a Label")
            return self.field
        for key, _item in sample.items_of_type(Label):
            return key
        raise ValueError(f"EncodeTarget: no Label field in sample (fields: {list(sample.keys())})")

    def __call__(self, sample: TypedSample) -> TypedSample:
        key = self._find_label(sample)
        label = sample[key]
        # Reuse the legacy op VERBATIM (non-empty validation + shared _lookup) for byte-parity.
        encoded = EncodeTargetOp(self.mapping, self.ignore_unknown, self.default)(
            Sample(input=None, target=label.value, metadata={})
        ).target
        out_key = self.output or key
        out = sample.replace_field(out_key, Label(encoded, classes=label.classes))
        return out.set_role(out_key, "target")


@configurable(category="op", group="structure")
class DecodeTarget(Transform):
    """Typed twin of :class:`DecodeTargetOp` — a class-ID ``Label`` → a class-NAME ``Label`` (inverse of encode).

    The typed-bag counterpart of :class:`DecodeTargetOp`: it reads a :class:`~sampleflux.Label` field
    (``field``; blank picks the first ``Label``) whose ``.value`` is an encoded class id and maps it
    back to its label name through ``mapping`` — the readback half used in prediction / reporting.
    This twin REUSES the legacy ``DecodeTargetOp`` verbatim, so the decoded value is byte-identical.
    The result is a new :class:`~sampleflux.Label` (carrying the source label's ``classes``) written
    under ``output`` — blank (default) replaces the source field in place — tagged ``target``.

    Args:
        mapping: Lookup from class id → label name, e.g. ``{2: "DJI AVATA2", ...}``. Must be non-empty.
        ignore_unknown: When ``False`` (default), raise on an id missing from ``mapping``; when
            ``True``, substitute ``default``.
        default: Value written for an unknown id when ``ignore_unknown=True`` (default ``None``).
        field: ``Label`` field to decode; blank (default) picks the first ``Label`` field.
        output: Field the decoded ``Label`` is written to; blank (default) replaces the source field
            in place. Its role is set to ``target``.
    """

    handles = (Label,)
    consumes = (Label,)
    produces = (Label,)

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

    def _find_label(self, sample: TypedSample) -> str:
        """Resolve the KEY of the ``Label`` field to decode (``self.field`` or the first ``Label``)."""
        if self.field:
            if self.field not in sample.keys():
                raise ValueError(f"DecodeTarget: field {self.field!r} not in sample (fields: {list(sample.keys())})")
            item = sample[self.field]
            if not isinstance(item, Label):
                raise TypeError(f"DecodeTarget: field {self.field!r} is {type(item).__name__}, expected a Label")
            return self.field
        for key, _item in sample.items_of_type(Label):
            return key
        raise ValueError(f"DecodeTarget: no Label field in sample (fields: {list(sample.keys())})")

    def __call__(self, sample: TypedSample) -> TypedSample:
        key = self._find_label(sample)
        label = sample[key]
        # Reuse the legacy op VERBATIM (non-empty validation + shared _lookup) for byte-parity.
        decoded = DecodeTargetOp(self.mapping, self.ignore_unknown, self.default)(
            Sample(input=None, target=label.value, metadata={})
        ).target
        out_key = self.output or key
        out = sample.replace_field(out_key, Label(decoded, classes=label.classes))
        return out.set_role(out_key, "target")


__all__ = [
    "MetadataToTargetOp",
    "EncodeTargetOp",
    "DecodeTargetOp",
    "CocoToTorchVisionDetectionOp",
    "MasksToDetectionBoxesOp",
    "MetadataToTarget",
    "EncodeTarget",
    "DecodeTarget",
]
