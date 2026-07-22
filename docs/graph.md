# Graph pipelines — flow documents, the FlowGraph engine and Context ops

## Flow documents & the FlowGraph engine (`sampleflux.flow`)

The **readable authoring form** of a graph pipeline is a `flow:` document — named steps where a step's name is how later steps reference its result:

```yaml
flow:
  scaled:  !class:sampleflux.ops.numpy.RescaleOp()          # input: the source sample
  norm:    !class:sampleflux.ops.numpy.StandardizeOp()      # input: previous step
  mask:    !class:sampleflux.ops.numpy.ThresholdOp(low_level=0.5) {from: scaled}   # 2nd reader of `scaled` = fan-out
  thresh:  !class:sampleflux.ops.formula.FormulaOp(formula="a*0.5") {from: norm}
  gated:   !class:sampleflux.ops.numpy.ThresholdOp()
    from: scaled
    bind: {low_level: thresh}          # per-sample param := thresh's result
  out: {from: gated, target_from: mask}                     # pure fan-in (no op)
outputs: out
```

Step grammar (four reserved keys, stripped before the op is built):

- **`from:`** — the input step (omitted = previous step; must name an *earlier* step, so document order is the schedule and cycles are inexpressible).
- **`target_from:` / `metadata_from:`** — fan-in slots (a step result contributes its corresponding field; metadata merges last-write-wins).
- **`bind:`** — `{param: step}` per-sample parameters (a step name = its result's `input`; `step.attr` = the step op's live `@output`, stochastic-correct).

A plain-mapping step with no op (`out: {from: a, target_from: b}`) is a pure fan-in; `{}` is the identity (names the source). Cell lifetimes are **automatic** in both forms.

Two engines, one contract — **bidirectional conversion with execution parity**:

```python
from sampleflux import Flux, FlowGraph, to_ops, from_ops

graph  = FlowGraph.from_yaml("graph.yaml", source=src)   # native named-step engine
flux   = Flux.from_flow_yaml("graph.yaml", source=src)   # same graph, LOWERED to the
                                                         # flat context-ops list (serial)
ops    = to_ops(graph.steps, graph.output_step)          # flow -> flat ops
flow2  = from_ops(ops)                                   # flat ops -> flow (lifting)
```

`FlowGraph` is a `torch.utils.data.Dataset` like `Flux` (`__len__`/`__getitem__`/`.batch`/`.parallel` — parallel runs the lowered form on Flux's spawn pool, one worker implementation). A purely linear flow lowers to the bare op list — zero context ops. See `examples/flow_graph.py` for the full round-trip.

## Graph pipelines on a flat op list (Context ops)

A branchy pipeline — fan-out, fan-in, a value computed on one branch feeding a parameter on another — runs on the **plain sequential `Flux` engine** via six *context ops* (`sampleflux.ops.context`). The engine creates one per-sample **`Context`** (a named-cell store, `sampleflux.context`) around each sample's trip through the op list; the context ops move data between the linear stream and those cells. Graph wiring never touches `sample.metadata` — the metadata bus stays byte-identical to a linear run.

| Op | Semantics |
|---|---|
| `Save(name)` | snapshot the stream sample into a cell (pass-through) — the fork point |
| `Use(name, drop=False)` | stream := the cell's value; deep-copies unless `drop` frees the cell (move) |
| `Drop(names)` | free cells explicitly |
| `Apply(op, param, source, drop=False)` | set `op.<param>` from a cell's value, then apply `op` |
| `Capture(op, output, name)` | apply `op`, record its live `@output` into a cell (stochastic-correct) |
| `Mix(input_from, target_from, metadata_from, drop)` | fan-in: compose a sample from cells + the incoming sample |

```yaml
ops:
  - !class:sampleflux.ops.context.Save(name=fork)              # fork the stream
  - !class:sampleflux.ops.numpy.StandardizeOp()                # branch A rides the stream
  - !class:sampleflux.ops.context.Save(name=branch_a)
  - !class:sampleflux.ops.context.Use(name=fork,drop=true)     # branch B restarts from the fork
  - !class:sampleflux.ops.numpy.ThresholdOp
    low_level: 0.5
  - !class:sampleflux.ops.context.Mix(target_from=branch_a)    # fan-in
    drop: [branch_a]
```

A straight sequence needs none of this — a bare `ops:` list stays exactly as before. Outside an engine (a hand-rolled loop), activate a Context explicitly:

```python
from sampleflux.context import Context, activate

with activate(Context()):
    for op in ops:
        sample = op(sample)
```

Cells hold whole `Sample`s (from `Save`) or raw values (from `Capture`); `Apply` reads a Sample cell's `input`, `Mix` reads each cell's corresponding field. Copy discipline mirrors the stash family: stored by reference, deep-copied on read (`Use` without `drop`), moved on last read (`drop=True`). These ops are what a `flow:` graph document lowers to. Why the wiring plane is an ambient per-sample store instead of `sample.metadata` (and why `FlowGraph` doesn't use it) is recorded in [architecture.md](architecture.md#the-per-sample-context-is-an-ambient-wiring-plane-samplefluxcontext-2026-07-17).

> **What about the stash family?** `sampleflux.ops.stash` (`StashInputOp`/`UnstashInputOp`/`StashTargetOp`/`UnstashTargetOp`) snapshots a field into `sample.metadata` instead of a cell. It is NOT a wiring mechanism — the context ops are — and remains only for the two jobs cells cannot do: carrying a snapshot **across a `Parallel` boundary** (metadata rides the sample through the stream split; cells deliberately raise there) and deliberately **persisting a snapshot into a sink**. Everything else — fan-out, fan-in, cross-branch values — uses the context ops above.

## Reattach an ops-only YAML (`Flux.from_ops_yaml`)

A `{ops: [!class:…()]}` document — e.g. one exported by an external pipeline-authoring tool — can be attached to any source:

```python
from sampleflux import Flux
from sampleflux.sources import HuggingFaceSource

flux = Flux.from_ops_yaml("ops.yaml", source=HuggingFaceSource(path="mnist"))
```

The helper **materializes** the deferred `!class:` markers before attaching (via `confluid.materialize`) — necessary because `confluid.load` leaves markers nested under a mapping key deferred, and a `Flux` rejects deferred markers at iteration by design. The manual equivalent is `Flux(source=src, ops=confluid.materialize(confluid.load("ops.yaml")["ops"]))`.
