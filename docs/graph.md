# Graph pipelines — `flow:` documents and the `FlowGraph` engine

A pipeline is a **graph of named steps**. There is ONE engine and ONE execution model; `ops:`
and `flow:` are two spellings of it, and which one you write is purely about whether the
pipeline branches.

`recordstream.core` and `recordstream.flow` are packages with **one module per cohesive unit**.
Import from the package — `from recordstream.core import Stream` / `from recordstream.flow import
FlowGraph` — but spell the **submodule** path in a config, because that is what `cls.__module__`
says and what a generated config emits:

| class | module | `!class:` path |
| --- | --- | --- |
| `Stream`, `JointStream` | `core/stream.py` | `recordstream.core.stream.Stream` |
| `FilterOp`, `WrappedOp` | `core/wrappers.py` | `recordstream.core.wrappers.FilterOp` |
| `FlowGraph` | `flow/graph.py` | `recordstream.flow.graph.FlowGraph` |

The shorter `!class:recordstream.core.Stream` still resolves (Confluid falls back to a
module-path import, and each package re-exports every name), so an older config keeps loading.
Rationale: [docs/architecture.md §12](architecture.md#12-core-and-flow-are-packages-too--layered-by-import-direction-2026-08-01).

## `ops:` — the linear spelling

A straight chain is a graph where every step reads the one before it, so it needs no names:

```yaml
ops:
  - !class:recordstream.ops.image.ConvertToImage {width: 224, height: 224}
  - !class:albumentations.Normalize {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  - !class:recordstream.ops.torch.ToTensor {}
```

The engine compiles that list into positional steps (`s0`, `s1`, `s2`) and runs it on the same
kernel a `flow:` document uses. The names never surface — nothing in an `ops:` document can
reference a step. Repeats stay distinct: the same op twice in a row is two steps.

## `flow:` — the named spelling, for branchy pipelines

When a pipeline forks, merges, or feeds one step's value into another's parameter, the steps
need names — and the name is how a later step refers to an earlier one:

```yaml
flow:
  spec:    !class:mypkg.MakeSpectrogram {}                  # input: the source record
  masked:  !class:recordstream.ops.numpy.Threshold {low_level: 0.5, from: spec}   # 2nd reader of `spec` = fan-out
  thresh:  !class:recordstream.ops.formula.FormulaOp {formula: "amax(a) * 0.6", field: image, from: spec}
  gated:                                                    # a step with bind: uses the plain-mapping form
    op: !class:recordstream.ops.numpy.Threshold {output: gated_mask}
    from: spec
    bind:
      low_level: thresh[image]     # per-record param := the `image` entry of thresh's result
  out: {from: gated, merge_from: [masked]}                  # fan-in (no op)
outputs: out
```

Two YAML spelling rules (both verified): SCALAR/list reserved keys (`from:`, `merge_from:`) may
ride inside a `!class:` marker's mapping alongside its kwargs — but **`bind:` (a nested mapping)
MUST use the plain-mapping step form** (`op:` + reserved keys, the `gated` step above): a nested
mapping under a `!class:` marker is consumed by Confluid as addressed configuration and never
reaches the step grammar. Write bind refs in block style or quoted — `{low_level: thresh[image]}`
inline is a YAML parse error (`[` opens a flow sequence).

Step grammar (three reserved keys, stripped before the op is built):

- **`from:`** — the input step (omitted = previous step; must name an *earlier* step, so document
  order is the schedule and cycles are inexpressible).
- **`merge_from:`** — fan-in: UNION the named steps' record ENTRIES into this step's incoming
  record, in listed order, last-write-wins on a key collision.
- **`bind:`** — `{param: ref}` per-record parameters: a bare `step` binds the step's WHOLE result
  record, `step[key]` the named ENTRY of its record, and `step.attr` the step op's live
  `@output` (read after it ran — stochastic-correct).

A plain-mapping step with no op (`out: {from: a, merge_from: [b]}`) is a pure fan-in; `{}` is the
identity (names the source). `outputs:` picks the yielded step (default: the last). Steps apply
their ops through the engine's op-family dispatch, so bare library transforms sit in flow steps
too.

## Running one

```python
from recordstream import FlowGraph, Stream
from recordstream.sources import HuggingFaceSource

graph = FlowGraph.from_yaml("graph.yaml", source=HuggingFaceSource(path="ylecun/mnist"))
for record in graph:
    ...

graph.parallel(4)          # spawn workers, one future per source record
len(graph); graph[3]       # map-style access (unavailable if a step op is 1→N expanding)
```

`FlowGraph` is a map-style dataset like `Stream` (`__len__`/`__getitem__`/`.batch`/`.parallel`).
Both classes hold a step graph and call the same per-record kernel; a straight chain takes an
env-free fast path in that kernel, so `ops:` costs no more to run than it ever did.

A LINEAR graph converts to a `Stream` (`graph.to_stream()`) because an op list can express a
straight chain. A branchy one does not — and there is no lowering pass that would manufacture a
flat spelling for it. That pass existed until 2026-07-30 (`to_ops`/`from_ops` plus six context
ops that re-encoded dataflow as imperative mutations of a per-record cell store); it was deleted
because it destroyed the very structure every consumer — a compiler, a visual editor, a reader —
wants back. Rationale: [architecture.md](architecture.md).

## Tracing one record (`Tracer`)

`recordstream.flow.Tracer` runs ONE record through a `Stream` or a `FlowGraph` on the same kernel an
ordinary run uses — every node's op sits behind a probe that records what went in and what came out
— so what it shows is what a real run does. Nodes are named `ops[0]`, `ops[1]`, … for a `Stream`
(the position in the ops list; the same op twice is two nodes) and by step key for a flow.

```python
from recordstream.flow import Tracer

tracer = Tracer(graph)                        # nothing is parsed or built until the first use
tracer.names                                  # ['spec', 'mag', 'floor', 'level', 'gated', 'boxes']

tracer.check(seed)                            # one JSON row per node: available entries, consumes, produces, verdict
tracer.run(seed)                              # check, then run; .result is what the graph yielded
tracer.run(seed, until="gated")               # pauses BEFORE 'gated': it is 'paused', later nodes 'not reached'
tracer.step()                                 # runs 'gated', pauses before the next node
tracer.resume()                               # runs to the end

tracer.rerun_from("floor", percentile=75.0)   # 'floor' rebuilt through its constructor; it and everything after recompute
tracer.generations                            # {'spec': 0, 'mag': 0, 'floor': 1, 'level': 1, 'gated': 1, 'boxes': 1}

tracer.value("gated", "mask")                 # the real Mask; side="input" for what the node received
tracer.statuses                               # 'not reached' | 'paused' | 'ok' | 'dropped' | 'error', per node
tracer.to_dict()                              # {where, check, nodes[...], paused_at, result, total_ms}
```

**The check.** Each node is checked with `check_chain` over the nodes whose records reach it (its
`from:` line and its `merge_from:` steps, in schedule order) and itself, so a flag raised earlier is
carried to the gate and a fork's sibling branch does not count; a `bind:` reference hands one value to
a parameter, not a record, so it is not part of the lineage. A refusal is raised before anything runs,
located at the node: `detector.yaml:floor: NoiseFloorEstimate needs the record entry 'spectrogram',
which nothing before it produces — …`. An op that declares the TYPES it consumes and produces rather
than entry names (a `Transform`'s tuple) cannot be checked by name: its verdict is `unverifiable`,
every node after it is checked on incomplete knowledge (`complete: false` — a refusal there is
reported in the row, not raised), and the run shows the truth.

**Reruns.** `rerun_from(node, **params)` rebuilds the node as `type(op)(**{**current, **params})` —
its CURRENT constructor-parameter values (a value the host set after construction survives) with the
new ones written over — so the constructor validates: a refused value raises and the trace is left
exactly as it was. The nodes before keep their recorded outputs and generation; the rebuilt node and
everything after it recompute and carry the next generation. A node that raised at run time is
recorded as `error` with the message (later nodes `not reached`); `rerun_from` that node recovers.

**Snapshots.** By default a snapshot is the object the kernel handed the node — no copy — and an op
that edits its record in place is flagged `in_place: true` on its entry. `Tracer(graph,
copy_snapshots=True)` deep-copies every input and output instead (one 1024x1024 float32 record
through four nodes: 5 MiB by reference, 37 MiB by copy). `to_dict()` never dumps an array: an entry
is summarised as `{type, shape, dtype, min, max}` (a `Mask` as `true_fraction`), with an item's
declared attributes beside it.

Two limits: a 1→N expanding node's trace keeps its last branch only, and a rerun from a node after it
is refused; a `bind:` or `merge_from:` that fails inside the kernel — before the node's op is called —
propagates without being attributed to a node. The full page — a runnable example with its output, the JSON of
one node, every refusal — is [trace.md](trace.md); rationale: [architecture.md](architecture.md) §22.

## Expanding (1→N) steps

A step whose op carries `EXPANDS = True` yields several records from one. The remaining subgraph
runs once per child over its own shallow copy of the step environment, depth-first, so sibling
order matches the nested-loop intuition. Such a pipeline is ITERABLE-ONLY: `__len__`/`__getitem__`
raise, because the expanded index map is unknowable up front.

## Reattach an ops-only YAML (`Stream.from_ops_yaml`)

A `{ops: [!class:…()]}` document — e.g. one exported by a visual editor — can be attached to any
source:

```python
from recordstream import Stream
from recordstream.sources import HuggingFaceSource

stream = Stream.from_ops_yaml("ops.yaml", source=HuggingFaceSource(path="ylecun/mnist"))
```

The helper **materializes** the deferred `!class:` markers eagerly (via `confluid.load`) so
a broken op fails at load time with the YAML in hand. It is a convenience, not a necessity:
`Stream` also flows any still-deferred marker in place at engine-route entry, which is what lets a
bare mapping-form `!class:albumentations.HorizontalFlip {p: 0.5}` sit directly in an `ops:` list.
`FlowGraph.from_ops_yaml` loads the same document as a linear step graph.
