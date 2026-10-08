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
| `Subgraph` | `flow/subgraph.py` | `recordstream.flow.subgraph.Subgraph` |

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

A step name may contain neither `.` (it would read as `step.attr`) nor `/` (it names a node inside
a subgraph, next section).

## Subgraphs: a flow used as one op

`recordstream.flow.subgraph.Subgraph` is an ordinary op whose body is a flow mapping, written inside
the step that uses it. The outer flow sees one step; the inside runs on the same kernel as any flow,
on the record that step receives, and returns the record of its `result:` step:

```yaml
flow:
  prep:
    op: !class:recordstream.flow.subgraph.Subgraph
      steps:
        grey:
          op: !class:recordstream.ops.image.ConvertMode {mode: L}
        scaled:
          op: !class:recordstream.ops.numpy.Scale {source_min: 0.0, source_max: 255.0, target_min: 0.0, target_max: 1.0}
      result: scaled
  mask:
    op: !class:recordstream.ops.numpy.Threshold {low_level: 0.5}
outputs: mask
```

```python
pixels = np.full((120, 160, 3), (20, 20, 30), dtype=np.uint8)   # a dark background
pixels[20:71, 15:61] = 230                                       # two light rectangles
pixels[50:106, 90:146] = 200
graph = FlowGraph.from_yaml("demo.yaml")
float(np.asarray(graph._run({"image": Image(pixels)})["mask"]).mean())    # 0.2855
```

The same answer as the three steps written flat. The rules:

| | |
| --- | --- |
| `steps:` | the flow grammar — an op, a `!class:` marker, or a mapping `{op, from, merge_from, bind}` per step |
| `result:` | the inner step whose record leaves the subgraph; blank = the last inner step. It is `result`, not `outputs`: the document's own top-level `outputs:` would be broadcast into a constructor parameter of that name |
| what the inside reads | only the record the subgraph receives and the inner steps before it — an inner `from:` / `merge_from:` / `bind:` names an inner step |
| what an outer step reads | the subgraph's result record (`from: prep`, `bind: {x: prep[key]}`); never a value of an inner step |
| an expanding op | not inside: a subgraph returns one record per record it receives |
| inner node names | `prep/grey` — in the tracer and in a visual editor. Inner names are scoped: an outer `grey` and an inner `grey` are two steps |
| `consumes` / `produces` / `flags` | derived from the inner ops: by entry name when every inner op declares names, as a tuple of types when one declares types. `produces` and `flags` count only the inner steps on the `from:` / `merge_from:` lineage of `result` — the record the subgraph returns — so a gate after the subgraph sees a flag raised on that lineage, and an entry or flag of a sibling branch is refused exactly as for the same steps written flat |

**Every refusal comes before the first record** — for a subgraph written as a flow step's op or as an
`ops:` list member. `parse_flow` opens a subgraph right after building it, and a `Stream` opens one when
it compiles its ops (every member, with a stream-level op such as `Parallel` in the list too), so
`FlowGraph.steps`, the first `next()` of an iteration and `Tracer.check` all refuse before any op runs.
A subgraph held inside ANOTHER op's slot (`Enable(ops=[…])`, `RandomApply(op=…)`, `Pipeline(transforms=[…])`,
a `Parallel`'s ops) is not opened early: it is refused when that op first calls it, at the first record.
The refusal names the outer step, and the line of the subgraph whose refusal it is when the document has
one — for a nested subgraph that is the INNER marker, not the outermost. Measured (paths shortened):

| you write | you get |
| --- | --- |
| `result: nosuch` | `ValueError: flow step 'prep' (a subgraph): Subgraph: result 'nosuch' does not name an inner step (the steps are: ['grey', 'scaled']) (at bad_result.yaml:3:9)` |
| an inner step reading an OUTER step: `mask: {op: …Threshold {low_level: 100}, from: grey}` | `ValueError: flow step 'prep' (a subgraph): Subgraph: step 'mask' reads 'grey' (from: 'grey'), which is not a step inside this subgraph — a step inside a subgraph reads only the record the subgraph receives and the steps before it inside (the steps are: ['mask']) (at outer_from.yaml:4:9)` |
| an outer `bind: {value: measure.level}` where `level` is a step inside the subgraph `measure` (or `measure/level.level`) | `ValueError: flow step 'stamp': bind value='measure.level' reads 'level' of 'measure', which is a subgraph — a step outside a subgraph cannot read a value of a step inside it; move that step out of the subgraph (at measure.yaml:7:9)` |
| an expanding op inside (here the test op `SgSplit`) | `TypeError: flow step 'fan' (a subgraph): Subgraph: step 'split' (SgSplit) is a 1→N expanding op, and a subgraph returns one record per record it receives — move the step out of the subgraph (at expand.yaml:3:9)` |
| `from:` inside an inner op's marker: `mask: !class:…Threshold {low_level: 100, from: grey}` | `ValueError: flow step 'prep' (a subgraph): Subgraph: step 'mask' carries 'from' inside its op's !class: marker, where Threshold takes it as a plain attribute and the step never reads it — write the step as a mapping: mask: {op: !class:recordstream.ops.numpy.Threshold {...}, from: grey} (at bare_from.yaml:3:9)` |
| a step named `prep/grey` | `ValueError: flow: step name 'prep/grey' may not contain '/' (reserved for the nodes inside a subgraph: 'prep/grey' is the step 'grey' inside the subgraph 'prep')` |
| no steps | `ValueError: flow step 'prep' (a subgraph): Subgraph: steps is empty — give it at least one step` |
| a subgraph as the second member of an `ops:` list, with `result: nosuch` | `ValueError: Stream.ops[1] (a subgraph): Subgraph: result 'nosuch' …` |
| an OUTER step with `from: prep/grey` (or `merge_from:`) | `ValueError: flow step 'm': from: 'prep/grey' reads the inner step 'grey' of 'prep', which is a subgraph — a step outside a subgraph cannot read a step inside it; move that step out of the subgraph` |

Why the marker case is refused: confluid builds a subgraph's inner markers when it builds the
subgraph, and a key the op's constructor does not take is set as a plain attribute — the step never
sees it. Measured before the refusal: the mask came out all `False` (mean `0.0`) where the mapping
spelling gives `0.2855`. One limit: a `bind:` MAPPING inside a marker is dropped by confluid without
a trace (it reads it as addressed configuration), so it cannot be refused — write `bind:` in the
mapping form, as everywhere in a flow.

A subgraph inside a subgraph works the same way; a refusal names the whole path and the line of the
subgraph that refuses (`flow step 'outer' (a subgraph): flow step 'middle' (a subgraph): flow step 'deep'
(a subgraph): Subgraph: result 'nosuch' … (at nested.yaml:15:21)`, where line 15 holds `deep`'s marker).

A reference to no earlier step — `from: zz`, `merge_from: [zz]`, `bind: {x: zz}` — raises
`recordstream.flow.StepReferenceError`, a `ValueError` whose `key`, `ref`, `target`, `step` and `param`
say which reference it was (`flow step 'm': from: 'zz' does not name an EARLIER step …`).

### Using one subgraph twice

1. **Copy the block.** Each `op: !class:…Subgraph` block is an independent copy.
2. **Keep it as a template.** A steps file (point 3) kept aside and inserted as a copy wherever it
   is needed; nothing links the copies back.
3. **Include one steps file.** A steps file holds a bare steps mapping (`grey: {op: …}`, `scaled:
   {op: …}`, comments allowed); each use writes `steps: {include: prep.steps.yaml}`, resolved
   relative to the including file. Each use gets its own op objects (measured: the two uses' `grey`
   ops are two objects), and the run is the same (`0.2855`):

```yaml
flow:
  read: {}                       # names the source record
  prep:
    op: !class:recordstream.flow.subgraph.Subgraph
      steps:
        include: prep.steps.yaml
      result: scaled
  mask:
    op: !class:recordstream.ops.numpy.Threshold {low_level: 0.5}
  prep2:
    from: read
    op: !class:recordstream.flow.subgraph.Subgraph
      steps:
        include: prep.steps.yaml
outputs: mask
```

YAML anchors (`&` / `*` / `<<`) and `!ref:` are not reuse spellings. An anchor merged into `flow:`
runs its steps first, whatever line the merge is written on, and a second merge in the same mapping
vanishes; a `!ref:` makes every use ONE object, so a value one use writes is read by the other, and a
setting written on one use changes both. Rationale: [architecture.md §23](architecture.md#23-a-subgraph-is-an-op-2026-09-29).

## Running one

```python
from recordstream import FlowGraph, Stream
from recordstream.sources import HuggingFaceSource

graph = FlowGraph.from_yaml("graph.yaml", source=HuggingFaceSource(path="ylecun/mnist"))
for record in graph:
    ...

graph.parallel(4)          # spawn workers, at most window × 4 records in flight (window=2 by default)
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
(the position in the ops list; the same op twice is two nodes) and by step key for a flow; a
subgraph step is followed by its inner nodes, `prep/grey` (see [trace.md](trace.md#inside-a-subgraph)).

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
