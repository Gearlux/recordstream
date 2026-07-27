# Graph pipelines — flow documents, the FlowGraph engine and Context ops

## Flow documents & the FlowGraph engine (`recordstream.flow`)

The **readable authoring form** of a graph pipeline is a `flow:` document — named steps where a step's name is how later steps reference its result:

```yaml
flow:
  spec:    !class:mypkg.MakeSpectrogram {}                  # input: the source record (writes key `image`)
  masked:  !class:recordstream.ops.numpy.Threshold {low_level: 0.5, from: spec}   # 2nd reader of `spec` = fan-out
  thresh:  !class:recordstream.ops.formula.FormulaOp {formula: "amax(a) * 0.6", field: image, from: spec}
  gated:                                                    # a step with bind: uses the plain-mapping form (op: + reserved keys)
    op: !class:recordstream.ops.numpy.Threshold {output: gated_mask}
    from: spec
    bind:
      low_level: thresh[image]     # per-record param := the `image` entry of thresh's result
  out: {from: gated, merge_from: [masked]}                  # fan-in (no op)
outputs: out
```

Two YAML spelling rules (both verified): SCALAR/list reserved keys (`from:`, `merge_from:`) may ride
inside a `!class:` marker's mapping alongside its kwargs — but **`bind:` (a nested mapping) MUST use
the plain-mapping step form** (`op:` + reserved keys, the `gated` step above): a nested mapping under
a `!class:` marker is consumed by Confluid as addressed configuration and never reaches the step
grammar. Write bind refs in block style or quoted — `{low_level: thresh[image]}` inline is a YAML
parse error (`[` opens a flow sequence).

Step grammar (three reserved keys, stripped before the op is built):

- **`from:`** — the input step (omitted = previous step; must name an *earlier* step, so document order is the schedule and cycles are inexpressible).
- **`merge_from:`** — fan-in: UNION the named steps' record ENTRIES into this step's incoming record, in listed order, last-write-wins on a key collision (the `MergeFields` slot semantics).
- **`bind:`** — `{param: ref}` per-record parameters: a bare `step` binds the step's WHOLE result record, `step[key]` the named ENTRY of its record, and `step.attr` the step op's live `@output` (lowered through `Capture` — stochastic-correct).

A plain-mapping step with no op (`out: {from: a, merge_from: [b]}`) is a pure fan-in; `{}` is the identity (names the source). Cell lifetimes are **automatic** in both forms. Steps apply their ops through the engine's op-family dispatch, so bare library transforms sit in flow steps too.

Two engines, one contract — **bidirectional conversion with execution parity**:

```python
from recordstream import Stream, FlowGraph, to_ops, from_ops

graph  = FlowGraph.from_yaml("graph.yaml", source=src)   # native named-step engine
stream   = Stream.from_flow_yaml("graph.yaml", source=src)   # same graph, LOWERED to the
                                                         # flat context-ops list (serial)
ops    = to_ops(graph.steps, graph.output_step)          # flow -> flat ops
flow2  = from_ops(ops)                                   # flat ops -> flow (lifting)
```

`FlowGraph` is a `torch.utils.data.Dataset` like `Stream` (`__len__`/`__getitem__`/`.batch`/`.parallel` — parallel runs the lowered form on Stream's spawn pool, one worker implementation). A purely linear flow lowers to the bare op list — zero context ops.

## Graph pipelines on a flat op list (Context ops)

A branchy pipeline — fan-out, fan-in, a value computed on one branch feeding a parameter on another — runs on the **plain sequential `Stream` engine** via six *context ops* (`recordstream.ops.context`). The engine creates one per-record **`Context`** (a named-cell store, `recordstream.context`) around each record's trip through the op list; the context ops move data between the linear stream and those cells. Graph wiring never mutates the record's entries — a linear run's record stays byte-identical whether or not context threading exists.

| Op | Semantics |
|---|---|
| `Save(name)` | snapshot the stream record into a cell (pass-through) — the fork point |
| `Use(name, drop=False)` | stream := the cell's value; deep-copies unless `drop` frees the cell (move) |
| `Drop(names)` | free cells explicitly |
| `Apply(op, param, source, key="", drop=False)` | set `op.<param>` from a cell (a record cell contributes its `key`-named entry, or the whole record when `key` is blank; a raw cell value verbatim), then apply `op` |
| `Capture(op, output, name)` | apply `op`, record its live `@output` into a cell (stochastic-correct) |
| `MergeFields(sources, keys, drop)` | fan-in: UNION the named cells' entries into the incoming record (listed order, last-write-wins; `keys` restricts the union) |

```yaml
ops:
  - !class:recordstream.ops.context.Save(name=fork)              # fork the stream
  - !class:albumentations.GaussNoise {p: 1.0}                  # branch A rides the stream
  - !class:recordstream.ops.context.Save(name=branch_a)
  - !class:recordstream.ops.context.Use(name=fork,drop=true)     # branch B restarts from the fork
  - !class:recordstream.ops.numpy.Threshold
    low_level: 0.5
  - !class:recordstream.ops.context.MergeFields                  # fan-in
    sources: [branch_a]
    keys: [image]
    drop: [branch_a]
```

A straight sequence needs none of this — a bare `ops:` list stays exactly as before. Outside an engine (a hand-rolled loop), activate a Context explicitly:

```python
from recordstream.context import Context, activate

with activate(Context()):
    for op in ops:
        record = op(record)
```

Cells hold whole records (from `Save`) or raw values (from `Capture`); `Apply` reads a record cell's `key`-named entry (whole record when `key` is blank), `MergeFields` unions each cell's entries. Copy discipline: cells are stored by reference, deep-copied on read (`Use` without `drop`), moved on last read (`drop=True`). On a deliberate key collision at the fan-in, rename on the producing branch first (`RenameField`, `recordstream.ops.structure`). These ops are what a `flow:` graph document lowers to. Why the wiring plane is an ambient per-record store instead of extra record keys (and why `FlowGraph` doesn't use it) is recorded in [architecture.md](architecture.md#3-the-per-record-context-is-an-ambient-wiring-plane-recordstreamcontext-2026-07-17).

> **Carrying a snapshot the context ops cannot?** Context cells are the wiring plane, but they deliberately raise across a `Parallel` boundary and never persist into a sink. For the two jobs cells cannot do — carrying a snapshot **across a `Parallel` boundary** and deliberately **persisting a snapshot into a sink** — copy the value under its own key with `CopyField` (`recordstream.ops.structure`); the snapshot then rides the record as a real entry. Everything else — fan-out, fan-in, cross-branch values — uses the context ops above.

## Reattach an ops-only YAML (`Stream.from_ops_yaml`)

A `{ops: [!class:…()]}` document — e.g. one exported by an external pipeline-authoring tool — can be attached to any source:

```python
from recordstream import Stream
from recordstream.sources import HuggingFaceSource

stream = Stream.from_ops_yaml("ops.yaml", source=HuggingFaceSource(path="mnist"))
```

The helper **materializes** the deferred `!class:` markers eagerly (via `confluid.materialize`) so a broken op fails at load time with the YAML in hand. It is a convenience, not a necessity: `Stream` also flows any still-deferred marker in place at engine-route entry (the same lazy-flow convention the composing ops use), which is what lets a bare mapping-form `!class:albumentations.HorizontalFlip {p: 0.5}` sit directly in an `ops:` list. The manual equivalent is `Stream(source=src, ops=confluid.materialize(confluid.load("ops.yaml")["ops"]))`.
