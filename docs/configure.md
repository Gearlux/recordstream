# Per-record op parameters (`ConfigureOp` / `Apply` / `Capture`)

Some op parameters are only known *per record*. Two mechanisms cover this:

- **`ConfigureOp(ops, target, param, source)`** — runs the `ops` compute-chain on the record as a SIDE branch (its transformations are discarded — the original record continues); the `source`-keyed entry of the chain's final record becomes the VALUE (payload-unwrapped via `item_data`), which is set as the `param` attribute of `target` — post-construction configuration, the confluid paradigm — and then `target` is applied to the original record. Use it when the value is *derived from the record itself* (e.g. a threshold from the record's own max) — the whole derivation reads as one node/YAML block.
- **`Capture` + `Apply`** (`sampleflux.ops.context`, see [graph.md](graph.md)) — when the value is an op's runtime **`@output`** (possibly stochastic — a random draw that can't be recomputed): `Capture(op, output, name)` applies the producer and records its live `@output` into a Context cell; a later `Apply(op, param, source)` sets the consumer's `param` from that cell and applies it. This is what graph exporters emit for `@output` → param wires, and the preferred form whenever the value already lives in a cell.

```yaml
ops:
  # AugmentOp draws a random level each call; capture it into a cell.
  - !class:sampleflux.ops.context.Capture
    op: !class:mypackage.ops.AugmentOp {}      # any op exposing a confluid @output
    output: applied_level
    name: __captured_level
  # …then inject the captured value into a later op's parameter per record.
  - !class:sampleflux.ops.context.Apply
    op: !class:mypackage.ops.CompensateOp {}
    param: level
    source: __captured_level
```

A self-contained `ConfigureOp` example — derive a per-record threshold from the record's own statistics:

```yaml
ops:
  - !class:sampleflux.ops.configure.ConfigureOp
    ops:
      - !class:sampleflux.ops.formula.FormulaOp {field: image, formula: "a.max() * 0.5"}
    source: image
    target: !class:sampleflux.ops.numpy.Threshold
      low_op: ">="
    param: low_level
```

Both mechanisms leave the record's own entries untouched: `ConfigureOp`'s compute chain runs on a side-branch copy, and `Capture`/`Apply` move values through the per-record Context. All inner ops (the compute chain, `target`, the wrapped ops of `Capture`/`Apply`) are applied through the engine's op-family dispatch, so a bare library transform works in any of these slots too.
