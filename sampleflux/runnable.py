"""The runnable protocol: gradient + progress marker mixins for long-running objects.

A *runnable* is any object exposing a no-arg ``run(self)`` — a trainer, an
evaluator, a dataset processor, a workflow. This module carries the two stateless
mixins a runnable inherits so a GUI executor (one that runs the object as a graph
node) can cooperate with it WITHOUT this package importing the GUI framework:

* :class:`TorchRunner` — marks a runnable whose ``run()`` needs autograd (it
  performs gradient-based optimization). A GUI executor that evaluates nodes under
  ``torch.inference_mode()`` reads the duck-typed ``__torch_runner__`` flag and
  re-enables autograd for the duration of ``run()``.
* :class:`ProgressReporting` — gives a runnable a framework-free progress callback.
  The executor injects a ``(value, total, desc) -> None`` sink via
  :meth:`~ProgressReporting.set_progress_callback`; the runnable drains it from its
  loop via :meth:`~ProgressReporting._report_progress` (a silent no-op when no sink
  was injected, so a plain CLI run is unaffected).

Both are pure marker/utility mixins (no required state) so they compose cleanly
with any base class (a ``torch.nn.Module``, a ``LightningModule``, a plain object).
This package declares only *that* a runnable needs autograd / can report progress;
the framework-specific bridges (e.g. a Lightning ``Callback`` that maps per-batch
hooks onto :meth:`~ProgressReporting._report_progress`) live in the consuming
framework package, not here.

It also carries the :func:`entrypoint` method marker: a runnable that exposes SEVERAL
capabilities from one class — the merged train+eval classes drive ``fit`` /
``evaluate`` / ``test`` / ``predict`` off a single ``task`` knob — annotates each such
method with the ``task`` value it runs and a ``role`` label (``"trainer"`` /
``"evaluator"`` / ``"predictor"``). A discovery consumer (a config generator, a visual
editor) reads these via :func:`runnable_entrypoints` to learn that one class both
trains and evaluates, instead of assuming a separate class per role.
"""

from typing import Callable, Dict, List, Optional

from loggair import get_logger

logger = get_logger(__name__)

#: Attribute stamped on a method by :func:`entrypoint`.
_ENTRYPOINT_ATTR = "__runnable_entrypoint__"

#: A progress sink: ``(value, total, description) -> None``. ``value`` / ``total`` are
#: floats in the same unit (optimizer steps, samples); ``description`` is a short stage label.
ProgressCallback = Callable[[float, float, str], None]


class TorchRunner:
    """Mixin marking a runnable whose ``run()`` performs gradient-based optimization.

    Pure marker — no state, no ``__init__`` — so it composes cleanly with any base
    (``torch.nn.Module``, ``pytorch_lightning.LightningModule``, …). It lets a GUI
    executor distinguish "this run needs autograd" from "this run is inference-only".

    Why it exists: a GUI executor may call ``run()`` from inside a graph evaluation
    wrapped in ``torch.inference_mode()`` for cheap, grad-free node evaluation. Under
    inference mode every tensor created — model parameters, forward activations, the
    loss — is an inference tensor with no autograd graph, so ``loss.backward()`` dies
    with *"element 0 of tensors does not require grad and does not have a grad_fn"*.
    The executor reads the duck-typed ``__torch_runner__`` flag (no hard import on the
    GUI side) and re-enables normal autograd for the duration of ``run()``.

    Inference-only runnables (a pure evaluator, a dataset processor) deliberately do
    NOT inherit this — they run as-is under the executor's inference mode.
    """

    __torch_runner__: bool = True


class ProgressReporting:
    """Mixin giving a runnable a framework-free progress callback.

    Pure mixin — no ``__init__``, no required state. The executor injects a sink via
    :meth:`set_progress_callback`; the runnable drains it from its loop via
    :meth:`_report_progress`. When no sink is injected (a plain CLI run, or any
    non-GUI caller) every call is a silent no-op, so behaviour is unchanged.
    """

    #: Set by the executor via :meth:`set_progress_callback`; ``None`` ⇒ progress reporting is a no-op.
    _progress_callback: Optional[ProgressCallback] = None

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        """Inject (or clear, with ``None``) the progress sink the executor drains. Idempotent."""
        self._progress_callback = callback

    def _report_progress(self, value: float, total: Optional[float], desc: str = "") -> None:
        """Report ``value`` of ``total`` to the injected callback; no-op when unset or invalid.

        Never raises — a broken progress sink must never abort a run (it is cosmetic). A ``None`` or
        non-positive ``total`` (an unbounded / unknown length, e.g. a streaming source) is skipped so
        the executor's bar stays in its indeterminate state rather than dividing by zero.
        """
        callback = self._progress_callback
        if callback is None or not total or total <= 0:
            return
        try:
            callback(float(value), float(total), desc)
        except Exception:  # noqa: BLE001 - progress reporting must never break the run
            logger.debug("progress callback raised; ignoring", exc_info=True)


def entrypoint(task: str, role: str = "runnable", primary: bool = False) -> Callable:
    """Mark a runnable method as a named capability entry point.

    A class that drives several capabilities off one ``task`` knob (the merged train+eval
    runnables: ``fit`` / ``evaluate`` / ``test`` / ``predict``) annotates each entry-point
    method so a discovery consumer (a config generator, a visual editor) can learn — from
    ONE class — which capabilities it exposes, instead of assuming a separate class per role.

    Args:
        task: The ``task`` value the runnable's ``run()`` dispatches to for this method
            (e.g. ``"fit"`` / ``"test"``).
        role: A capability label — conventionally ``"trainer"`` (fits/trains),
            ``"evaluator"`` (computes metrics over a held-out set), or ``"predictor"``
            (streams predictions). Free-form so new capabilities need no change here.
        primary: When several methods share a ``role``, marks the default one (e.g. ``test``
            is the primary ``"evaluator"`` over ``evaluate``/validate).
    """

    def deco(fn: Callable) -> Callable:
        setattr(fn, _ENTRYPOINT_ATTR, {"task": task, "role": role, "primary": primary})
        return fn

    return deco


def runnable_entrypoints(cls: type) -> Dict[str, Dict[str, object]]:
    """Return ``{method_name: {"task", "role", "primary"}}`` for every :func:`entrypoint` method.

    Walks the MRO so an inherited entry point is found, and reads the marker off the raw
    function object (never triggering a property getter).
    """
    out: Dict[str, Dict[str, object]] = {}
    for klass in reversed(cls.__mro__):
        for name, attr in vars(klass).items():
            meta = getattr(attr, _ENTRYPOINT_ATTR, None)
            if meta is not None:
                out[name] = dict(meta)
    return out


def entrypoint_tasks(cls: type, role: str) -> List[str]:
    """Return the ``task`` values of ``cls``'s entry points with the given ``role``, primary first."""
    matches = [(name, meta) for name, meta in runnable_entrypoints(cls).items() if meta.get("role") == role]
    matches.sort(key=lambda nm: not nm[1].get("primary", False))  # primary (True) sorts first
    return [str(meta["task"]) for _, meta in matches]


__all__ = [
    "ProgressCallback",
    "ProgressReporting",
    "TorchRunner",
    "entrypoint",
    "entrypoint_tasks",
    "runnable_entrypoints",
]
