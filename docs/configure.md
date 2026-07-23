# Per-sample op parameters (`ConfigureOp` / `Apply` / `Capture`)

Some op parameters are only known *per sample*. Two mechanisms cover this:

- **`ConfigureOp(ops, target, param, key)`** — runs `ops` on the sample as a side-branch; the chain's final primary input value (`primary(sample, "input")`) is injected as `target.<param>` and also recorded as an `aux`-role field named `key`, then `target` is applied. Use it when the value is *derived from the sample itself* (e.g. a threshold from the sample's own max) — the whole derivation reads as one node/YAML block.
- **`Capture` + `Apply`** (`sampleflux.ops.context`, see [graph.md](graph.md)) — when the value is an op's runtime **`@output`** (possibly stochastic — a random draw that can't be recomputed): `Capture(op, output, name)` applies the producer and records its live `@output` into a Context cell; a later `Apply(op, param, source)` sets the consumer's `param` from that cell and applies it. This is what graph exporters emit for `@output` → param wires, and the preferred form whenever the value already lives in a cell.

```yaml
ops:
  # AugmentOp draws a random level each call; capture it into a cell.
  - !class:sampleflux.ops.context.Capture
    op: !class:mypackage.ops.AugmentOp {}      # any op exposing a confluid @output
    output: applied_level
    name: __captured_level
  # …then inject the captured value into a later op's parameter per sample.
  - !class:sampleflux.ops.context.Apply
    op: !class:mypackage.ops.CompensateOp {}
    param: level
    source: __captured_level
```

A self-contained `ConfigureOp` example — derive a per-sample threshold from the sample's own statistics:

```yaml
ops:
  - !class:sampleflux.ops.configure.ConfigureOp
    ops:
      - !class:sampleflux.ops.numpy.MaxOp {}
      - !class:sampleflux.ops.formula.FormulaOp { formula: "a * 0.5" }
    target: !class:sampleflux.ops.numpy.ThresholdOp {}
    param: low_level
    key: derived_threshold
```

`ConfigureOp` also records the derived value as an `aux`-role field named `key` (traceability — it persists into a sink); `Capture`/`Apply` move values through the per-sample Context, which never alters the sample's fields.
