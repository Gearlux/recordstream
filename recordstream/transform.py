"""``Transform`` — type-dispatched record ops with once-per-record parameters, plus ``Pipeline``.

A transform records its random / configured parameters ONCE per record (:meth:`Transform.get_params`),
then walks the dict and, for each value whose type it handles, applies the registered
kernel (:mod:`recordstream.dispatch`). Values it does not handle pass through untouched.

Two properties fall out of this shape for free:

* **Cross-field consistency.** Because params are sampled once and shared, one op moves
  every handled value with the SAME decision (two ``Signal`` values in one record get the
  same drawn SNR) — the torchvision-v2 model.
* **Open extension.** A new value type is taught to an existing op with one
  ``@MyOp.kernel(NewType)`` registration and no core edit.

Targeting is by TYPE; the ``field`` parameter pins an op to one named key when a record
holds several values of a handled type.

recordstream ships NO native augmentation ops — geometric/photometric augmentation comes from
the libraries (torchvision ``transforms.v2`` / albumentations) dropped into an ops list
AS-IS; the engine invokes each op family natively (see ``recordstream.core.families._apply_op``).
There are no wrapper/adapter classes.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from confluid import configurable

from recordstream.dispatch import Kernel, dispatch, register_kernel
from recordstream.items import Record, item_data, with_data

__all__ = [
    "Transform",
    "Pipeline",
    "FunctionTransform",
    "as_transform",
]


class Transform:
    """Base class for type-dispatched record ops (see the module docstring).

    Subclasses declare ``handles`` (the value types they process) and register a kernel per
    type via ``@MyOp.kernel(ItemType)``. Override :meth:`get_params` to record shared
    parameters once per record.

    **The type-interface attributes** (``handles`` / ``consumes`` / ``optional`` /
    ``produces``) describe the op to readers and machines (a visual editor's typed sockets,
    a pipeline linter) WITHOUT executing it. They are DECLARATIVE: nothing validates them
    against the op's behavior, and the base ``__call__`` dispatches on the KERNEL REGISTRY,
    never on ``handles`` — the one exception is :class:`FunctionTransform`, whose
    ``handles`` IS its ``isinstance`` application gate. Declare them truthfully or not at
    all; a kernel op keeps ``handles`` mirroring its registered kernels, and a
    type-CHANGING op (``__call__`` override) states its ``consumes``/``produces`` type flow
    explicitly because no kernel registration reveals it. Full guidance:
    ``docs/record-model.md`` → "Declaring an op's type interface".

    Args:
        field: Apply only to this record key (still type-gated). None (default) = every value of a handled type.
    """

    #: Value types this op processes (everything else passes through). DECLARATIVE for
    #: kernel ops (the kernel registry decides dispatch — keep this mirroring it);
    #: ENFORCED only by FunctionTransform, where it is the isinstance application gate.
    handles: Tuple[type, ...] = ()
    #: Type interface — input types the op NEEDS to do useful work. Empty = same as
    #: ``handles``. Never gates execution; validate a hard requirement lazily in __call__.
    consumes: Tuple[type, ...] = ()
    #: Type interface — input types used when present but not required (e.g. a geometric
    #: op that also moves a Mask if the record has one). Never gates execution.
    optional: Tuple[type, ...] = ()
    #: Type interface — value types this op ADDS or CHANGES (its output contract; what a
    #: downstream op can rely on finding). Never gates execution.
    produces: Tuple[type, ...] = ()

    def __init__(self, field: Optional[str] = None) -> None:
        self.field = field

    @classmethod
    def kernel(cls, item_type: type) -> Callable[[Kernel], Kernel]:
        """Register a kernel for ``item_type`` on this op (decorator over :func:`register_kernel`)."""
        return register_kernel(cls, item_type)

    def get_params(self, record: Record) -> Dict[str, Any]:
        """Record the shared parameters for one call. Default: no params."""
        return {}

    def __call__(self, record: Record) -> Optional[Record]:
        params = self.get_params(record)
        out = dict(record)
        for key, value in record.items():
            if self.field is not None and key != self.field:
                continue
            kernel = dispatch(type(self), type(value))
            if kernel is None:
                continue
            out[key] = kernel(value, params)
        return out


@configurable(category="op", group="compose")
class Pipeline:
    """Sequential application of ops — ``Pipeline([a, b, c])(record)`` is ``c(b(a(record)))``.

    THE compose-group unit: wrap an ordered list of ops so they appear as one named block in
    a config and one node on a visual canvas. Entries may be native ops, BARE library
    transforms (torchvision ``transforms.v2`` / albumentations — invoked natively by the
    engine's op-family dispatch), or config-deferred markers (built on first use). If any op
    returns ``None`` the chain stops and propagates ``None`` (filter-drop semantics).

    Args:
        transforms: Ordered ops applied in sequence (bare library transforms allowed). Defaults to ``[]`` (identity).
    """

    def __init__(self, transforms: Optional[Sequence[Any]] = None) -> None:
        # Partial / zero-arg: store config only; marker flow happens on first call.
        self.transforms: List[Any] = list(transforms) if transforms else []

    def __call__(self, record: Record) -> Optional[Record]:
        from confluid import flow
        from confluid.fluid import Fluid

        # _apply_op = the engine's op-family dispatch, so a bare albumentations /
        # torchvision-v2 transform nests here exactly as in a bare ops list.
        from recordstream.core import _apply_op

        current: Optional[Record] = record
        for i, op in enumerate(self.transforms):
            if current is None:
                return None
            if isinstance(op, Fluid):
                op = flow(op)
                self.transforms[i] = op
            if op is None:
                continue
            current = _apply_op(current, op)
        return current

    def close(self) -> None:
        """Propagate close() to inner ops that own resources (e.g. a sink op)."""
        for op in self.transforms:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()

    def __repr__(self) -> str:
        return f"Pipeline([{', '.join(type(t).__name__ for t in self.transforms)}])"


class FunctionTransform(Transform):
    """An op that applies one plain function ``fn(data) -> data`` to every handled value.

    The escape hatch for custom ops: no kernel registration, no subclass — wrap a
    function and say which value types it applies to (via :func:`as_transform`).
    """

    def __init__(self, fn: Callable[[Any], Any], handles: Sequence[type], field: Optional[str] = None) -> None:
        super().__init__(field=field)
        self._fn = fn
        self.handles = tuple(handles)

    def __call__(self, record: Record) -> Optional[Record]:
        out = dict(record)
        for key, value in record.items():
            if self.field is not None and key != self.field:
                continue
            if isinstance(value, self.handles):
                out[key] = with_data(value, self._fn(item_data(value)))
        return out


def as_transform(fn: Callable[[Any], Any], handles: Sequence[type], field: Optional[str] = None) -> FunctionTransform:
    """Wrap a plain ``fn(data) -> data`` as a :class:`FunctionTransform` over ``handles``."""
    return FunctionTransform(fn, handles, field=field)
