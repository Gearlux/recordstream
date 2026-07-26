# Per-record op parameters (`ConfigureOp` / `Apply` / `Capture`)

Some op parameters are only known *per record*. Two mechanisms cover this:

- **`ConfigureOp(ops, target, param, source)`** — runs the `ops` compute-chain on the record as a SIDE branch (its transformations are discarded — the original record continues); the `source`-keyed entry of the chain's final record becomes the VALUE (payload-unwrapped via `item_data`), which is set as the `param` attribute of `target` — post-construction configuration, the confluid paradigm — and then `target` is applied to the original record. Use it when the value is *derived from the record itself* (e.g. a threshold from the record's own max) — the whole derivation reads as one node/YAML block.
- **`Capture` + `Apply`** (`sampleflux.ops.context`, see [graph.md](graph.md)) — when the value is an op's runtime **`@output`** (possibly stochastic — a random draw that can't be recomputed): `Capture(op, output, name)` applies the producer and records its live `@output` into a Context cell; a later `Apply(op, param, source)` sets the consumer's `param` from that cell and applies it. This is what graph exporters emit for `@output` → param wires, and the preferred form whenever the value already lives in a cell.

Concretely — a producer that draws a random gain per record and publishes what it ACTUALLY drew
as a confluid `@output` (apply `@output` UNDER `@property`), and a consumer whose `level` gets set
per record. The pair is deliberately a roundtrip: compensating with the captured gain restores the
original image, which proves the LIVE draw — not a recomputation — reached the consumer
(verified: `np.allclose(out["image"], original)` holds for every record):

```python
# mypackage/ops.py
from confluid import configurable, output
from sampleflux import Record, item_data, with_data
import numpy as np

@configurable(category="op", random=True)
class AugmentOp:
    """Scale the image by a random gain drawn per record.

    Args:
        max_gain: Upper bound of the uniform gain draw.
    """

    def __init__(self, max_gain: float = 2.0) -> None:
        self.max_gain = max_gain
        self._applied = 1.0
        self._rng = np.random.default_rng()

    @property
    @output
    def applied_level(self) -> float:
        """The gain the LAST call actually drew — the live @output that Capture records."""
        return self._applied

    def __call__(self, record: Record) -> Record:
        self._applied = float(self._rng.uniform(1.0, self.max_gain))
        img = record["image"]
        return {**record, "image": with_data(img, item_data(img) * self._applied)}

@configurable(category="op")
class CompensateOp:
    """Divide the image by ``level`` — undo a gain applied earlier in the chain.

    Args:
        level: The gain to divide out; set per record by Apply (or ConfigureOp).
    """

    def __init__(self, level: float = 1.0) -> None:
        self.level = level

    def __call__(self, record: Record) -> Record:
        img = record["image"]
        return {**record, "image": with_data(img, item_data(img) / float(self.level))}
```

```yaml
ops:
  # AugmentOp draws a random gain each call; capture the LIVE @output into a cell.
  - !class:sampleflux.ops.context.Capture
    op: !class:mypackage.ops.AugmentOp {}      # or the registered short name: !class:AugmentOp {}
    output: applied_level
    name: __captured_level
  # …then inject the captured value into the consumer's parameter, per record.
  - !class:sampleflux.ops.context.Apply
    op: !class:mypackage.ops.CompensateOp {}
    param: level
    source: __captured_level
```

Reading the pair: `Capture` runs `AugmentOp` once (the record's image is scaled by, say, 1.7×) and
stores `applied_level` = 1.7 in the `__captured_level` cell; `Apply` does
`setattr(compensate_op, "level", 1.7)` and then runs it — post-construction configuration, the
confluid paradigm. Because the value is read off the op AFTER it ran, a stochastic draw is captured
exactly; recomputing it (the naive alternative) would draw a DIFFERENT number.

A self-contained `ConfigureOp` example — derive a per-record threshold from the record's own statistics:

```yaml
ops:
  - !class:sampleflux.ops.configure.ConfigureOp
    ops:
      - !class:sampleflux.ops.formula.FormulaOp {field: image, formula: "amax(a) * 0.5"}
    source: image
    target: !class:sampleflux.ops.numpy.Threshold
      low_op: ">="
    param: low_level
```

Both mechanisms leave the record's own entries untouched: `ConfigureOp`'s compute chain runs on a side-branch copy, and `Capture`/`Apply` move values through the per-record Context. All inner ops (the compute chain, `target`, the wrapped ops of `Capture`/`Apply`) are applied through the engine's op-family dispatch, so a bare library transform works in any of these slots too.
