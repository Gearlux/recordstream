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
  - !class:recordstream.ops.torch.ToTensor {normalize: false}
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

The helper **materializes** the deferred `!class:` markers eagerly (via `confluid.materialize`) so
a broken op fails at load time with the YAML in hand. It is a convenience, not a necessity:
`Stream` also flows any still-deferred marker in place at engine-route entry, which is what lets a
bare mapping-form `!class:albumentations.HorizontalFlip {p: 0.5}` sit directly in an `ops:` list.
`FlowGraph.from_ops_yaml` loads the same document as a linear step graph.
