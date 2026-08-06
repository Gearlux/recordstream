"""Applying ONE op to ONE record — the op-family registry and the dispatch chokepoint.

The bottom of the op-facing layer: every composing op in :mod:`recordstream.ops` reaches
:func:`_apply_op` from here, and nothing here imports back out (see ``docs/architecture.md``
§5 on the import direction this protects). Also home to the ``EXPANDS`` protocol
(:func:`_op_expands` / :func:`_expand`), because "does this op expand, and how do I run it?"
is the same question as "how do I apply this op" — the graph kernel asks all three together.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from loggair import get_logger

from recordstream.items import NDArrayItem, Record, Regions, with_data

logger = get_logger(__name__)


def _op_expands(op: Any) -> bool:
    """True when an op is a 1→N expanding op (explicit ``EXPANDS = True`` class attribute)."""
    return bool(getattr(op, "EXPANDS", False))


def _expand(op: Any, record: Any) -> List[Any]:
    """Run a 1→N EXPANDING op and return its flattened children."""
    raw = op(record)
    if raw is None:
        return []
    return [child for child in raw if child is not None]


#: An op-family MATCHER recognises a library's op objects. Keep it IMPORT-FREE — inspect
#: ``type(op).__mro__`` module names rather than importing the library.
OpMatcher = Callable[[Any], bool]
#: An op-family INVOKER applies one foreign op with its library's native calling
#: convention: ``(record, op) -> Optional[Record]`` (``None`` = drop the record).
OpInvoker = Callable[[Record, Any], Optional[Record]]

#: The registered op families, in registration order. Dispatch checks LAST-registered
#: first, so a later (more specific) family can shadow an earlier one.
_OP_FAMILIES: List[Tuple[str, OpMatcher, OpInvoker]] = []


def register_op_family(name: str, matcher: OpMatcher, invoker: OpInvoker) -> None:
    """Teach the engine to invoke a NEW library's ops natively — the open extension point.

    ``matcher(op) -> bool`` recognises the family's op objects (keep it import-free —
    inspect ``type(op).__mro__`` module names); ``invoker(record, op)`` applies one op with
    the library's own calling convention and returns the new record (``None`` drops it).
    Re-registering a ``name`` REPLACES that family in place; otherwise the family is
    appended, and dispatch checks last-registered first (a more specific family shadows an
    earlier one — register yours after the built-ins to win an overlap).

    Both callables MUST be module-level functions (picklable by reference): the engine's
    spawn-parallel routes ship non-builtin families to worker processes by pickling them.

    Example — kornia augmentations (``nn.Module``s over batched BCHW tensors)::

        def is_kornia(op) -> bool:
            return any(c.__module__.startswith("kornia.augmentation") for c in type(op).__mro__)

        def invoke_kornia(record, op):
            img = record["image"]                    # a CHW torch.Tensor (e.g. after ToTensor)
            out = op(img.unsqueeze(0)).squeeze(0)    # kornia draws once per batch call
            return {**record, "image": out}

        register_op_family("kornia", is_kornia, invoke_kornia)
    """
    entry = (str(name), matcher, invoker)
    for i, (existing, _, _) in enumerate(_OP_FAMILIES):
        if existing == name:
            _OP_FAMILIES[i] = entry
            return
    _OP_FAMILIES.append(entry)


def registered_op_families() -> Tuple[str, ...]:
    """The registered op-family names, in registration/dispatch-precedence order."""
    return tuple(name for name, _, _ in _OP_FAMILIES)


def _sync_op_families(families: Optional[List[Tuple[str, OpMatcher, OpInvoker]]]) -> None:
    """Merge families shipped from the parent process into this process's registry.

    Spawn workers import this module (built-ins present) but never re-run the user's
    registration side effects — the parallel routes therefore pass the parent's
    non-builtin entries along and merge them here (idempotent by name).
    """
    for name, matcher, invoker in families or []:
        register_op_family(name, matcher, invoker)


def _extra_op_families() -> List[Tuple[str, OpMatcher, OpInvoker]]:
    """The non-builtin registry entries — what a spawn worker cannot rebuild by import alone."""
    return [entry for entry in _OP_FAMILIES if entry[0] not in _BUILTIN_FAMILIES]


#: The record keys albumentations understands — its OWN target vocabulary. An albumentations
#: op receives exactly these keys (the ones present) and nothing else, so extra record
#: entries (scalars, domain items) never reach a library that would reject them.
_ALB_KEYS: Tuple[str, ...] = ("image", "mask", "masks", "bboxes", "keypoints", "labels")


def _is_albumentations(op: Any) -> bool:
    """True for an albumentations transform / ``Compose`` — by MRO module name (no import here)."""
    return any(getattr(cls, "__module__", "").startswith("albumentations") for cls in type(op).__mro__)


#: Whether :func:`_disable_cv2_threading` has already run. Module-level, so the cost of the
#: guarantee is one bool check per op application.
_CV2_THREADING_DISABLED = False


def _disable_cv2_threading() -> None:
    """Turn OpenCV's internal thread pool off, ONCE, the first time albumentations is used.

    **This prevents a SIGSEGV, not a slowdown.** albumentations runs on OpenCV, whose thread pool
    is not fork-safe: once a parent process has executed a cv2 op, a FORKED child inheriting that
    pool crashes with *"DataLoader worker ... is killed by signal: Segmentation fault: 11"* — no
    Python traceback, because the child never gets to raise. Measured on macOS, 2026-08-02, with a
    ``DataLoader(num_workers=2)`` over a ``Stream`` whose ops list contained ``A.Resize``: it
    segfaults reliably with the pool on and passes reliably with it off.

    It is the SECOND fork hazard on this path and is independent of the first
    (:func:`~recordstream.ensure_materialized`, which warms a lazy source so the child does not
    run a download through the non-fork-safe ``_scproxy``). Neither fixes the other: with the
    source warmed and the pool ON the worker still dies, and with the pool off a cold source is
    still built in the child. A consumer that forks needs both.

    Turning the pool off costs nothing where it matters. Inside a ``DataLoader`` worker the
    WORKER is the parallelism — cv2's own threads oversubscribe the machine rather than help —
    which is why albumentations' own documentation recommends exactly this for multiprocessing
    loaders.

    Done HERE, at the one place this package invokes albumentations, rather than at import: a
    process that never uses albumentations must not have its OpenCV settings changed by importing
    a data library, and cv2 must not become an import-time dependency of the engine.
    """
    global _CV2_THREADING_DISABLED
    if _CV2_THREADING_DISABLED:
        return
    _CV2_THREADING_DISABLED = True  # set FIRST: a cv2-less install must not retry on every record
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception as exc:  # pragma: no cover - albumentations without cv2 is not a real install
        logger.debug(f"could not disable OpenCV threading ({exc}); a forked DataLoader worker may crash.")


#: Transform classes already warned about (see :func:`_warn_if_regions_are_left_behind`). Keyed
#: by CLASS, not instance: the message describes a configuration pattern, and two `Resize`s in
#: one chain have the same thing wrong with them.
_WARNED_SPATIAL: Set[type] = set()


def _has_spatial_transform(op: Any, depth: int = 0) -> bool:
    """True when ``op`` contains a transform that would MOVE boxes, per albumentations' own taxonomy.

    The library already draws this line: a ``DualTransform`` is defined as one that applies to
    boxes and masks as well as the image (``Resize``, ``HorizontalFlip``, ``RandomCrop``), while
    an ``ImageOnlyTransform`` cannot touch geometry (``Normalize``, ``ColorJitter``). Reading THAT
    distinction is why this needs no list of transform names to drift out of date, and no guess
    about what a given transform does.

    Matched by MRO class NAME rather than `isinstance`, for the same reason
    :func:`_is_albumentations` matches by module name: recognising a library must never import
    one. A ``Compose`` is recursed through its ``transforms``, depth-capped against a cycle.
    """
    if any(cls.__name__ == "DualTransform" for cls in type(op).__mro__):
        return True
    if depth >= 4:
        return False
    children = getattr(op, "transforms", None)
    if not children:
        return False
    return any(_has_spatial_transform(child, depth + 1) for child in children)


def _warn_if_regions_are_left_behind(record: Record, op: Any, passed: Dict[str, Any]) -> None:
    """Warn once when a geometry-changing transform ran while a ``Regions`` sat out the call.

    This family passes the op EXACTLY the keys of albumentations' own vocabulary, which is what
    lets a bare library transform work unmodified — but a detection target rides as a ``Regions``
    item under a key of the pipeline's choosing, so it is not in that vocabulary and does not get
    passed. Measured: a bare ``A.Resize`` moves a 200x200 image to 64x64 and leaves the boxes on
    ``[10, 10, 100, 100]``; a bare ``A.HorizontalFlip`` mirrors the pixels and leaves the boxes
    where they were WITHOUT changing the raster at all — which is why the condition here is the
    library's spatial/photometric taxonomy and not "did the image size change".

    Nothing errors either way: the shapes stay valid and only the coordinates become wrong, so a
    model trains against misplaced targets and reports nothing. It stays a WARNING rather than an
    error because a record may legitimately carry regions describing something other than the
    image being augmented — this family cannot know, and refusing the call would break a pipeline
    that is right.

    The fix is to speak the library's vocabulary: put boxes under ``bboxes`` with their
    ``labels`` and declare ``bbox_params`` on the ``Compose``, and the library moves them in the
    same joint draw. For a plain deterministic resize, ``ops.target.ResizeDetection`` does the
    coupled step over the ``Regions`` form directly.
    """
    if "bboxes" in passed or type(op) in _WARNED_SPATIAL:
        return
    keys = [key for key, value in record.items() if isinstance(value, Regions)]
    if not keys or not _has_spatial_transform(op):
        return
    _WARNED_SPATIAL.add(type(op))
    logger.warning(
        f"albumentations {type(op).__name__} changes GEOMETRY, but this record's detection "
        f"boxes ({keys}) ride as a Regions item, which is not in the library's key vocabulary "
        f"({', '.join(_ALB_KEYS)}) — so the pixels moved and the boxes did not. Put boxes under "
        f"'bboxes' + 'labels' with A.Compose(..., bbox_params=A.BboxParams(...)) so the library "
        f"moves them in the same draw, or use recordstream.ops.target.ResizeDetection for a "
        f"plain coupled resize. (Warned once per transform type.)"
    )


def _invoke_albumentations(record: Record, op: Any) -> Optional[Record]:
    """albumentations dispatches by KWARG NAME: hand the op exactly its own target keys
    present in the record (one call = one joint draw across them); array outputs are
    re-wrapped in the incoming value's item type (``with_data``) so ``Image``/``Mask``
    keep their type and metadata. Box-carrying augmentation belongs in albumentations' own
    ``A.Compose(..., bbox_params=...)`` — format handling is Compose's job in that library.
    """
    _disable_cv2_threading()
    kwargs = {k: record[k] for k in _ALB_KEYS if k in record}
    if not kwargs:
        logger.debug(
            f"albumentations op {type(op).__name__} received no known keys "
            f"({', '.join(_ALB_KEYS)}) — record keys: {list(record)}; passing through."
        )
        return record
    _warn_if_regions_are_left_behind(record, op, kwargs)
    out = op(**kwargs)
    merged = dict(record)
    for key, value in out.items():
        original = record.get(key)
        if isinstance(original, NDArrayItem) and not isinstance(value, NDArrayItem):
            value = with_data(original, value)
        merged[key] = value
    return merged


def _is_torchvision_v2(op: Any) -> bool:
    """True for a torchvision ``transforms.v2`` transform — by MRO module name (no import here)."""
    return any(getattr(cls, "__module__", "").startswith("torchvision.transforms.v2") for cls in type(op).__mro__)


#: torchvision v2's geometric transforms all live in ONE private module — the closest thing the
#: library has to albumentations' `DualTransform` marker. Private, so this can go stale across a
#: torchvision release; it FAILS OPEN (no warning, nothing else changes), which is the right
#: direction for a diagnostic.
_V2_GEOMETRY_MODULE = "torchvision.transforms.v2._geometry"


def _is_v2_geometry(op: Any, depth: int = 0) -> bool:
    """True when a v2 transform (or one nested in a ``Compose``) changes GEOMETRY.

    The behavioural test — apply it to a throwaway ``BoundingBoxes`` and see whether they move —
    would be authoritative and is deliberately NOT used: running a transform speculatively draws
    from the RNG, which would change the augmentation stream of the run being diagnosed. A
    diagnostic must not alter what it observes.
    """
    if any(getattr(cls, "__module__", "") == _V2_GEOMETRY_MODULE for cls in type(op).__mro__):
        return True
    if depth >= 4:
        return False
    children = getattr(op, "transforms", None)
    if not children:
        return False
    return any(_is_v2_geometry(child, depth + 1) for child in children)


def _invoke_torchvision_v2(record: Record, op: Any) -> Optional[Record]:
    """torchvision v2 natively walks a dict: params sampled once, tensor/tv_tensor/PIL
    leaves transformed, everything else passed through — called as-is.

    "Everything else passed through" is where detection boxes fall: v2 recognises its OWN
    ``tv_tensors`` types, and a :class:`~recordstream.Regions` is not one, so a geometric
    transform moves the pixels and leaves the boxes — the same silent desync the albumentations
    family has, reached by a different route (there the boxes are not in the key vocabulary; here
    they are not in the TYPE vocabulary). Measured: ``v2.Resize((64, 64))`` takes a 200x200 image
    to 64x64 with the boxes still on ``[10, 10, 100, 100]``, while the same transform over a
    ``tv_tensors.BoundingBoxes`` correctly rescales them to ``[3.2, 3.2, 32, 32]``.
    """
    _warn_if_v2_leaves_regions_behind(record, op)
    return cast(Record, op(record))


def _warn_if_v2_leaves_regions_behind(record: Record, op: Any) -> None:
    """The v2 twin of :func:`_warn_if_regions_are_left_behind` — once per transform type."""
    if type(op) in _WARNED_SPATIAL:
        return
    keys = [key for key, value in record.items() if isinstance(value, Regions)]
    if not keys or not _is_v2_geometry(op):
        return
    _WARNED_SPATIAL.add(type(op))
    logger.warning(
        f"torchvision v2 {type(op).__name__} changes GEOMETRY, but this record's detection boxes "
        f"({keys}) ride as a Regions item, which is not one of v2's tv_tensors types — so v2 "
        f"passes them through untouched while the pixels move. Carry boxes as "
        f"torchvision.tv_tensors.BoundingBoxes(..., format=…, canvas_size=…) so v2 transforms "
        f"them in the same call, or use recordstream.ops.target.ResizeDetection for a plain "
        f"coupled resize. (Warned once per transform type.)"
    )


# The built-in families register through the SAME open registry third parties use —
# one mechanism, no privileged code path. Registered at import, so spawn workers
# rebuild them by importing this module.
register_op_family("albumentations", _is_albumentations, _invoke_albumentations)
register_op_family("torchvision_v2", _is_torchvision_v2, _invoke_torchvision_v2)
_BUILTIN_FAMILIES: Tuple[str, ...] = ("albumentations", "torchvision_v2")


def _apply_op(record: Record, op: Any) -> Optional[Record]:
    """Apply one op to the record dict — the engine's op-FAMILY dispatch.

    The single op-application chokepoint shared by the sequential, parallel (via
    :func:`~recordstream.core.stream._worker_task`), streamed, and random-access
    (``__getitem__``) paths; composing ops (``Pipeline`` / ``Parallel`` / ``Enable`` /
    ``RandomApply`` / ``ConfigureOp``) route their inner ops through here so every op is
    applied identically. Each op family is invoked the way its library expects — no
    wrapper/adapter classes: the registered families (:func:`register_op_family`; built-ins
    ``albumentations`` / ``torchvision_v2``) are checked LAST-registered first, and an op
    matching none of them is a native/wiring op called ``op(record) -> Optional[Record]``
    (``None`` drops the record — filter semantics).
    """
    for _name, matcher, invoker in reversed(_OP_FAMILIES):
        try:
            matched = matcher(op)
        except Exception:  # pragma: no cover - a defensive matcher never breaks dispatch
            matched = False
        if matched:
            return invoker(record, op)
    return cast(Optional[Record], op(record))
