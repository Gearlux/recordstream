"""Shared generator for the per-transform augmentation op families (``Alb*`` / ``Tv*``).

Mirrors the waivefront-helios transform auto-bridge: walk a library's public transform
classes and generate ONE ``@configurable`` SampleFlux op per transform — a subclass of the
library's adapter op (:class:`~sampleflux.ops.albumentations.AlbumentationsOp` /
:class:`~sampleflux.ops.torchvision.TorchvisionTransformOp`) whose constructor mirrors the
transform's own parameters (plus the adapter's ``target`` / ``seed`` knobs). Each
generated op:

* is a normal sample-scoped op (``__call__(sample)``) — it chains in a ``Flux`` ops list,
  a ``TransformChain`` / ``RandomApply``, a Confluid YAML (``!class:AlbHorizontalFlip``),
  or a visual canvas exactly like any hand-written op;
* exposes ``raw_transform`` — the configured library transform instance — so an adapter
  op's ``transforms`` list unwraps a wired generated op back to the library object;
* carries a synthesized ``__signature__`` / ``__annotations__`` / ``Args:`` docstring so
  static introspection (``to_pydantic`` → form-specs / MCP schemas, ``parse_param_docs``
  → widget tooltips) sees the transform's real parameters.

The mandatory name prefix (``Alb`` / ``Tv``) keeps confluid's flat, name-keyed registry
collision-free — albumentations and torchvision share many bare names (``ColorJitter``,
``Normalize``, ``Resize``, …).
"""

import inspect
# Callable/Dict/Literal/Sequence/Union are referenced only inside the wrapped transforms'
# STRING annotations, which get_type_hints() evaluates against THIS module's globals when
# to_pydantic introspects a synthesized __init__ — so they must be importable here (flake8
# can't see string-annotation usage).
from typing import Any, Iterable, List, Optional, Tuple, Type, get_type_hints

from confluid import configurable
from loggair import get_logger

from sampleflux.ops.albumentations import TargetMode

logger = get_logger(__name__)

#: Adapter-owned constructor names — a library transform whose ctor collides is skipped.
_RESERVED = frozenset({"target", "seed", "transform", "transforms"})

_TARGET_DOC = "    target: Joint-augmentation mode — ``none`` (input-only, default), ``mask``, or ``boxes``."
_SEED_DOC = "    seed: Compose seed for deterministic draws. ``None`` = non-deterministic (default)."


def _param_specs(transform_cls: type) -> List[inspect.Parameter]:
    """The transform constructor's named parameters (``self`` and variadics dropped)."""
    sig = inspect.signature(transform_cls.__init__)  # type: ignore[misc]
    return [
        p
        for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def _synth_doc(name: str, transform_cls: type, seed_param: bool) -> str:
    """The generated op's docstring: the library docstring with the adapter params spliced in.

    ``parse_param_docs`` reads the (first) Google-style ``Args:`` block, so the adapter's
    ``target`` / ``seed`` lines are inserted right after the library's own ``Args:``
    heading — or a fresh block is appended when the library docstring has none.
    """
    extra = [_TARGET_DOC] + ([_SEED_DOC] if seed_param else [])
    lib_doc = inspect.getdoc(transform_cls) or f"{name} transform (see the library documentation)."
    lines = lib_doc.splitlines()
    if any(ln.strip() == "Args:" for ln in lines):
        out: List[str] = []
        for ln in lines:
            out.append(ln)
            if ln.strip() == "Args:":
                out.extend(extra)
        return "\n".join(out)
    return lib_doc + "\n\nArgs:\n" + "\n".join(extra)


def _make_op(
    name: str,
    transform_cls: type,
    *,
    base: type,
    prefix: str,
    group: str,
    module_name: str,
    seed_param: bool,
) -> type:
    """One generated op class wrapping ``transform_cls`` (see the module docstring)."""
    specs = _param_specs(transform_cls)
    pnames = [p.name for p in specs]
    clash = _RESERVED & set(pnames)
    if clash:
        raise ValueError(f"constructor params clash with adapter params: {sorted(clash)}")
    defaults = {p.name: (None if p.default is inspect.Parameter.empty else p.default) for p in specs}
    required = {p.name for p in specs if p.default is inspect.Parameter.empty}
    op_name = f"{prefix}{name}"
    allowed = set(pnames) | {"target"} | ({"seed"} if seed_param else set())

    def __init__(self: Any, **kwargs: Any) -> None:
        # Lazy / zero-arg: store config only (required transform params default to None and
        # surface lazily via the library's own missing-argument error on first call).
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"{op_name}: unexpected parameters {sorted(unknown)}")
        if seed_param:
            base.__init__(self, target=kwargs.get("target", "none"), seed=kwargs.get("seed"))  # type: ignore[misc]
        else:
            base.__init__(self, target=kwargs.get("target", "none"))  # type: ignore[misc]
        for pname in pnames:
            setattr(self, pname, kwargs.get(pname, defaults[pname]))
        self._params_key = None

    # Synthesized signature/annotations: static introspection (to_pydantic / FluxStudio
    # widgets / parse_param_docs) sees the transform's real parameters, keyword-only, with
    # the adapter's target/seed appended LAST. Required params are defaulted to None so
    # zero-arg construction always works (the workspace lazy-init mandate).
    # Resolve annotations to REAL objects against the transform's own module. A library's
    # param annotations are often strings (PEP 563) referencing names local to that module
    # (cv2, Literal, the library's own aliases); left as strings they'd blow up later when
    # to_pydantic's get_type_hints evals them against THIS module. Anything that still won't
    # resolve degrades to Any so introspection never chokes on a stray name.
    try:
        _hints = get_type_hints(transform_cls.__init__)
    except Exception:
        _hints = {}

    def _resolve(p: inspect.Parameter) -> Any:
        ann = _hints.get(p.name, p.annotation)
        return Any if isinstance(ann, str) else ann

    sig_params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for p in specs:
        sig_params.append(
            p.replace(kind=inspect.Parameter.KEYWORD_ONLY, default=defaults[p.name], annotation=_resolve(p))
        )
    sig_params.append(
        inspect.Parameter("target", inspect.Parameter.KEYWORD_ONLY, default="none", annotation=TargetMode)
    )
    if seed_param:
        sig_params.append(
            inspect.Parameter("seed", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Optional[int])
        )
    __init__.__signature__ = inspect.Signature(sig_params)  # type: ignore[attr-defined]
    annotations = {p.name: _resolve(p) for p in specs if p.annotation is not inspect.Parameter.empty}
    annotations["target"] = TargetMode
    if seed_param:
        annotations["seed"] = Optional[int]
    __init__.__annotations__ = annotations

    def raw_transform(self: Any) -> Any:
        kwargs = {}
        for pname in pnames:
            value = getattr(self, pname, None)
            if value is None and pname in required:
                continue  # omitted → the library raises its own clear missing-argument error
            kwargs[pname] = value
        return transform_cls(**kwargs)

    base_pipeline = base.pipeline.fget  # type: ignore[attr-defined]

    def pipeline(self: Any) -> Any:
        # Rebuild the wrapped transform when any mirrored param changed (post-construction
        # configuration), then reuse the adapter's compose/cache machinery verbatim.
        key: tuple = tuple(repr(getattr(self, pname, None)) for pname in pnames)
        key += (self.target, getattr(self, "seed", None))
        if key != getattr(self, "_params_key", None):
            self._params_key = key
            self.transform = self.raw_transform
        return base_pipeline(self)

    namespace = {
        "__init__": __init__,
        # The base __call__ re-stated in the class dict: canvas op-classification checks
        # vars(cls) for __call__, and inherited-only methods are invisible to it.
        "__call__": base.__call__,
        "__doc__": _synth_doc(name, transform_cls, seed_param),
        "__module__": module_name,
        "raw_transform": property(
            raw_transform, doc="The configured library transform instance (built fresh per access)."
        ),
        "pipeline": property(pipeline, doc="The live library pipeline for the current parameter values."),
        "LIBRARY_CLS": transform_cls,
    }
    cls = type(op_name, (base,), namespace)
    return configurable(category="op", group=group, random=True)(cls)


def generate_transform_ops(
    *,
    classes: Iterable[Tuple[str, type]],
    base: Type[Any],
    prefix: str,
    group: str,
    module_globals: dict,
    seed_param: bool,
) -> List[str]:
    """Generate one op per ``(name, transform_cls)`` into ``module_globals``; returns the sorted names.

    Per-class failures (uninspectable constructor, adapter-param clash) skip that
    transform with a DEBUG note and never break the module import — the helios-bridge
    warn-and-continue contract.
    """
    module_name = module_globals.get("__name__", base.__module__)
    names: List[str] = []
    for name, transform_cls in classes:
        try:
            op_cls = _make_op(
                name,
                transform_cls,
                base=base,
                prefix=prefix,
                group=group,
                module_name=module_name,
                seed_param=seed_param,
            )
        except Exception as exc:
            logger.debug(f"augment bridge: skipping {name}: {exc}")
            continue
        module_globals[op_cls.__name__] = op_cls
        names.append(op_cls.__name__)
    logger.debug(f"augment bridge: generated {len(names)} {prefix}* ops in group {group!r}")
    return sorted(names)


__all__ = ["generate_transform_ops"]
