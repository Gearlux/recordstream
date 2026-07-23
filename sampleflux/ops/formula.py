"""``FormulaOp`` — evaluate a math formula over the primary input item.

The op-form of a visual canvas *Math* node: a restricted Python expression over one
named variable bound to the incoming primary input item (plus the stdlib ``math`` namespace
and the scalar helpers ``abs``/``min``/``max``/``round``/``pow`` — no builtins, so
``__import__``/``open``/``exec`` are unavailable). Its main consumer is the ops-export's
value-chain compilation: an on-canvas ``… → Extract → Math → widget`` wire becomes
``ConfigureOp(ops=[…, FormulaOp(formula)], target=…, param=…)``, so the per-sample value
survives serialization.
"""

import math as _math
from typing import Any, Dict

from confluid import configurable

from sampleflux.bag.items import item_data, with_data
from sampleflux.bag.sample import Sample, primary

# Every public ``math`` symbol + the scalar built-in helpers, mirroring the canvas Math
# node's namespace. The bound variable shadows same-named constants (e.g. ``e``).
_FORMULA_NAMESPACE: Dict[str, Any] = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
_FORMULA_NAMESPACE.update({"abs": abs, "min": min, "max": max, "round": round, "pow": pow})


@configurable(category="op", group="compose")
class FormulaOp:
    """Replace the primary input item with ``formula`` evaluated over it.

    Args:
        formula: Expression over ``var`` (e.g. ``"a * 0.2"``); ``math.*`` + ``abs``/``min``/``max``/``round`` allowed.
        var: Variable name the incoming primary input item binds to. Defaults to ``a``.
    """

    def __init__(self, formula: str = "a", var: str = "a") -> None:
        # Lazy / zero-arg: store config only; the formula is validated at first call.
        self.formula = str(formula)
        self.var = str(var)

    def __call__(self, sample: Sample) -> Sample:
        if not self.formula.strip():
            raise ValueError("FormulaOp: 'formula' must be a non-empty expression")
        key, item = primary(sample, "input")
        namespace = {**_FORMULA_NAMESPACE, self.var: item_data(item)}
        try:
            value = eval(self.formula, {"__builtins__": {}}, namespace)  # noqa: S307 - restricted namespace
        except Exception as exc:
            raise ValueError(f"FormulaOp: formula {self.formula!r} failed: {exc}") from exc
        return sample.replace_field(key, with_data(item, value))
