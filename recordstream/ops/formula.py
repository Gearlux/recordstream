"""``FormulaOp`` — evaluate a math formula over one record entry.

The op-form of a visual canvas *Math* node: a restricted Python expression over one
named variable bound to the ``field``-keyed record value (plus the stdlib ``math``
namespace and the scalar helpers ``abs``/``min``/``max``/``round``/``pow`` — no builtins,
so ``__import__``/``open``/``exec`` are unavailable). Its main consumer is an ops-export's
value-chain compilation: an on-canvas ``… → Extract → Math → widget`` wire becomes
``ConfigureOp(ops=[…, FormulaOp(field, formula)], target=…, param=…)``, so the per-record
value survives serialization.
"""

import math as _math
from typing import Any, Dict

import numpy as _np
from confluid import configurable

from recordstream.items import Record, item_data, with_data

# Every public ``math`` symbol + the scalar built-in helpers, mirroring the canvas Math
# node's namespace. The bound variable shadows same-named constants (e.g. ``e``).
_FORMULA_NAMESPACE: Dict[str, Any] = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
_FORMULA_NAMESPACE.update({"abs": abs, "min": min, "max": max, "round": round, "pow": pow})
# Array reducers, FUNCTION style (``amax(a) * 0.5``) — pre-bound numpy callables whose
# internal lazy imports resolve via numpy's own globals. The ATTRIBUTE form (``a.max()``)
# is NOT guaranteed under the sandbox: numpy's C reductions lazy-import through the
# CALLING frame, whose ``__builtins__`` is empty here (KeyError: '__import__') unless some
# earlier code already warmed that import in this process. Teach the function form.
_FORMULA_NAMESPACE.update({"amax": _np.max, "amin": _np.min, "mean": _np.mean, "std": _np.std, "median": _np.median})


@configurable(category="op", group="compose")
class FormulaOp:
    """Replace the ``field``-keyed record value with ``formula`` evaluated over it.

    Args:
        formula: Expression over ``var`` — math.*, abs/min/max/round/pow + reducers amax/amin/mean/std/median.
        field: Record key whose value the formula reads and replaces; required at call time.
        var: Variable name the incoming value binds to. Defaults to ``a``.
    """

    def __init__(self, formula: str = "a", field: str = "", var: str = "a") -> None:
        # Lazy / zero-arg: store config only; formula and field are validated at first call.
        self.formula = str(formula)
        self.field = str(field)
        self.var = str(var)

    def __call__(self, record: Record) -> Record:
        if not self.formula.strip():
            raise ValueError("FormulaOp: 'formula' must be a non-empty expression")
        if not self.field:
            raise ValueError("FormulaOp: 'field' (the record key to evaluate over) is required")
        if self.field not in record:
            raise KeyError(f"FormulaOp: record has no key {self.field!r} (keys: {list(record)})")
        item = record[self.field]
        namespace = {**_FORMULA_NAMESPACE, self.var: item_data(item)}
        try:
            value = eval(self.formula, {"__builtins__": {}}, namespace)  # noqa: S307 - restricted namespace
        except Exception as exc:
            raise ValueError(f"FormulaOp: formula {self.formula!r} failed: {exc}") from exc
        try:
            new_value = with_data(item, value)
        except TypeError:
            new_value = value  # a plain (non-item) value is replaced verbatim
        return {**record, self.field: new_value}
