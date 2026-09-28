# Tracing one record (`Tracer`)

`recordstream.flow.Tracer` runs ONE record through a `Stream` or a `FlowGraph` and keeps a snapshot
per node: what the node received, what it produced, how long it took, the parameter values it ran
with. You can stop the run BEFORE a node, step, resume, and rerun one node with a new parameter value
— only that node and the ones after it recompute.

It is the engine behind a graph debugger (a page that shows each node's input and output), a
per-node tool a language model calls with new settings, and a test that pins a pipeline node by node.
It is not a second executor: every node's op sits behind a probe, and the probed step list is handed to
the same kernel a plain run uses, so what the trace shows is what a run does. Rationale:
[architecture.md §22](architecture.md#22-one-record-through-the-kernel-probed-tracer-2026-09-28).

**Node names.** A `Stream`'s ops are `ops[0]`, `ops[1]`, … — their position in the ops list, so the
same op twice is two nodes. A `FlowGraph`'s steps are their document keys (`scaled`, `mask`, …). Every
method that takes a node name refuses an unknown one and lists the nodes there are.

## A first run

Twenty lines over four recordstream ops: scale a `uint8` image to `0..1`, threshold it into a mask,
label the mask's connected regions into boxes, drop the mask.

```python
import numpy as np
from recordstream import Image, Label, Stream
from recordstream.flow import Tracer
from recordstream.ops.numpy import ConnectedComponents, Scale, Threshold
from recordstream.ops.structure import DropField

pixels = np.zeros((16, 16), dtype=np.uint8)
pixels[2:6, 3:9], pixels[10:14, 10:15] = 200, 90              # a bright patch and a dim one
record = {"image": Image(pixels), "class": Label("patches")}

tracer = Tracer(Stream(ops=[Scale(), Threshold(low_level=0.5), ConnectedComponents(), DropField(key="mask")]))
print(tracer.names)
tracer.run(record, until="ops[2]")                            # pauses BEFORE the labelling
print(tracer.statuses)
print(tracer.to_dict()["nodes"][1]["output"]["mask"])          # what ops[1] wrote, summarised
tracer.resume()
print(tracer.result[0]["boxes"])                               # one box: the dim patch is below 0.5
tracer.rerun_from("ops[1]", low_level=0.3)                     # ops[1] rebuilt; ops[1..3] recompute
print(tracer.generations)
print(tracer.value("ops[2]", "boxes"))                         # two boxes now
```

What it printed (a logging banner from the environment left out):

```
['ops[0]', 'ops[1]', 'ops[2]', 'ops[3]']
{'ops[0]': 'ok', 'ops[1]': 'ok', 'ops[2]': 'paused', 'ops[3]': 'not reached'}
{'type': 'Mask', 'shape': [16, 16], 'dtype': 'bool', 'true_fraction': 0.0938}
Boxes(boxes=[(3, 2, 9, 6)], labels=None, scores=None, canvas=(16, 16), extras={}, classes=None)
{'ops[0]': 0, 'ops[1]': 1, 'ops[2]': 1, 'ops[3]': 1}
Boxes(boxes=[(3, 2, 9, 6), (10, 10, 15, 14)], labels=None, scores=None, canvas=(16, 16), extras={}, classes=None)
```

Line by line. `until="ops[2]"` paused BEFORE the labelling: `ops[0]` and `ops[1]` ran, `ops[2]` is
`paused` with its input recorded and its op not called, `ops[3]` was never reached. The mask is
described, not dumped: 24 of 256 pixels set is a `true_fraction` of 0.0938. `resume()` ran to the end
and the result holds one box — the dim patch (90/255 = 0.35) is below the threshold of 0.5.
`rerun_from("ops[1]", low_level=0.3)` rebuilt the threshold with the new level and recomputed
`ops[1]`, `ops[2]` and `ops[3]`, which now carry generation 1; `ops[0]` keeps generation 0 and did
not run again. `value("ops[2]", "boxes")` is the real `Boxes` object from the rerun: two boxes.

A flow is traced the same way, with its steps named by key:

```python
from recordstream import FlowGraph

tracer = Tracer(FlowGraph(flow={"scaled": Scale(), "mask": Threshold(low_level=0.5), "boxes": ConnectedComponents()}))
tracer.names                  # ['scaled', 'mask', 'boxes']
tracer.run(record).statuses   # {'scaled': 'ok', 'mask': 'ok', 'boxes': 'ok'}
```

Nothing is parsed or built when the tracer is constructed — `Tracer(graph)` stores the graph, and the
first use (`names`, `check`, `run`) parses it. A flow document whose `!class:` markers are still
unbuilt is built at that moment, not before.

## The trace as JSON (`to_dict()`)

`to_dict()` is plain JSON — `json.dumps` needs no encoder. The top level, after the run above with
`Tracer(..., where="patches.yaml")`:

```
{'where': 'patches.yaml', 'check': <4 rows>, 'nodes': <4 entries>, 'paused_at': None, 'result': <1 record>, 'total_ms': 0.081}
```

One node entry — `nodes[1]`, the threshold, after the first run (lists of numbers folded onto one
line; the values are verbatim):

```json
{
  "node": "ops[1]",
  "op": "Threshold",
  "status": "ok",
  "generation": 0,
  "ms": 0.014,
  "params": {
    "low_level": 0.5,
    "high_level": null,
    "low_op": ">",
    "high_op": "<",
    "field": "",
    "output": "mask"
  },
  "bound": {},
  "input": {
    "image": {"type": "Image", "layout": "HWC", "shape": [16, 16], "dtype": "float32", "min": 0.0, "max": 0.7843137383460999},
    "class": {"type": "Label", "value": "patches"}
  },
  "output": {
    "image": {"type": "Image", "layout": "HWC", "shape": [16, 16], "dtype": "float32", "min": 0.0, "max": 0.7843137383460999},
    "class": {"type": "Label", "value": "patches"},
    "mask": {"type": "Mask", "shape": [16, 16], "dtype": "bool", "true_fraction": 0.0938}
  },
  "changed": ["mask"],
  "removed": [],
  "in_place": false
}
```

| key | what it holds |
| --- | --- |
| `node`, `op` | the node's name and its op's class name (`null` for a fan-in step without an op) |
| `status` | one of the five statuses below |
| `generation` | `0` after `run`; `rerun_from` gives the rebuilt node and every node after it the next number |
| `ms` | the op call's wall time in milliseconds (`null` while paused) |
| `params` | the node's CURRENT constructor-parameter values — what a rerun starts from |
| `bound` | the values a flow's `bind:` set on the op before the call (always `{}` for a `Stream`) |
| `input`, `output` | each record entry summarised (see below); no `output` while paused |
| `changed`, `removed` | entries the op added or replaced (compared by identity), entries it left out |
| `in_place` | `true` when the op edited the record it was given instead of returning a new one |
| `error` | the exception, as `Type: message` — present only when the node raised |

An array is described, never dumped: `{type, shape, dtype, min, max}`, a boolean array as
`true_fraction`, a complex one as `abs_min` / `abs_max`, with a `non_finite` count when there is one;
an array of eight elements or fewer lists its `values`. An item's declared attributes sit beside the
description (`layout` for an `Image`; `canvas` and `boxes` for `Boxes`), a `Label` is
`{"type": "Label", "value": "patches"}`, and a torch tensor is described by shape, dtype and device
without being moved.

A node never reached carries four keys — `{"node": "ops[3]", "op": "DropField", "status": "not
reached", "generation": null}`. A paused node has `"status": "paused"`, `"ms": null`, its `params`,
`bound` and `input`, and no `output`, `changed`, `removed` or `in_place` yet.

## The five statuses

| status | meaning |
| --- | --- |
| `not reached` | the run has not got there: before any run, after a pause before an earlier node, after an earlier node raised |
| `paused` | the run stopped BEFORE this node — its input is recorded, its op was not called |
| `ok` | the op ran and returned a record |
| `dropped` | the op ran and returned `None` (the record was filtered out); the nodes after it stay `not reached` and `result` is `[]` |
| `error` | the op raised; the message is under `error`; the nodes after it stay `not reached` |

## What each call does

| call | what happens |
| --- | --- |
| `Tracer(graph, where="", copy_snapshots=False)` | stores the arguments; the graph is parsed and its ops built on first use. `where` prefixes every refusal (`patches.yaml:ops[1]: …`) so a page can locate it |
| `names` | the node names in schedule order |
| `check(seed)` | the static check (next section): one JSON row per node, or a refusal before anything runs |
| `run(seed, until=None)` | `check`, then the seed through every node; the previous trace is cleared first. `until=<node>` pauses BEFORE that node: its input is recorded, its op is not called, the nodes after it are `not reached`, `result` is `None` |
| `step()` | from a pause: runs the paused node and pauses before the next one; on the last node the run completes |
| `resume()` | from a pause: runs to the end |
| `rerun_from(node, **params)` | rebuilds the node through its constructor from its current values with `params` written over them, checks it, and recomputes from that node on; the nodes before keep their outputs and generation and are not run again. Without `params` the node reruns unchanged |
| `value(node, entry, side="output")` | the REAL object under `entry` in the node's output — or, with `side="input"`, in what it received. Never a summary |
| `statuses`, `generations` | `{node: status}` and `{node: generation}` for every node (`None` until reached) |
| `paused_at`, `result` | the node the run paused before (or `None`); what the graph yielded, `None` until a run completes |
| `report` | the rows of the last `check` |
| `to_dict()` | the trace as plain JSON, arrays summarised |
| `copy_snapshots=True` | every input and output snapshot is a deep copy instead of the object the kernel handed the node — the remedy for a chain whose ops edit records in place (see the snapshots section) |

## The check before the run

`run` begins with `check(seed)`: each node's declared needs are checked against what reaches it — the
seed's entries plus what the nodes whose records reach it declare they produce. That is the by-name
chain check (`recordstream.ops.contract.check_chain`) made graph-aware: a node is checked over the
nodes on its `from:` line and its `merge_from:` steps, in schedule order, so a flag raised three
nodes earlier is carried to the gate that requires it — checked one node at a time, every gated node
would be refused — while a fork's sibling branch does not count, because its entries never arrive. A
`bind:` reference is not part of that lineage: it hands one value to a parameter, not a record.

The rows for the run above:

```json
[
  {"node": "ops[0]", "op": "Scale", "available": ["class", "image"], "consumes": null, "produces": null, "complete": true, "verdict": "ok"},
  {"node": "ops[1]", "op": "Threshold", "available": ["class", "image"], "consumes": ["NDArrayItem"], "produces": ["Mask"], "complete": true, "verdict": "unverifiable",
   "note": "Threshold declares the TYPES it consumes and produces, not the record entries — nothing can be checked by name here; the run shows what it wrote"},
  {"node": "ops[2]", "op": "ConnectedComponents", "available": ["class", "image"], "consumes": ["Mask"], "produces": ["Boxes"], "complete": false, "verdict": "unverifiable",
   "note": "ConnectedComponents declares the TYPES it consumes and produces, not the record entries — nothing can be checked by name here; the run shows what it wrote"},
  {"node": "ops[3]", "op": "DropField", "available": ["class", "image"], "consumes": null, "produces": null, "complete": false, "verdict": "ok"}
]
```

Three verdicts. `ok`: the node's needs are met by name. `unverifiable`: the node declares the TYPES
it consumes and produces (a `Transform`'s tuple, as `Threshold` and `ConnectedComponents` do) rather
than record entries, so nothing can be checked by name — the run shows what it wrote. Every node
after such a node is checked on incomplete knowledge (`complete: false`): a refusal there is reported
in the row as `unverifiable` with the message under `note`, never raised. `refused`: an unmet need on
complete knowledge — the row carries the message under `error` and `check` raises it.

A refusal is raised BEFORE any node runs, located at the node. An op that declares by name — here an
[`Algorithm`](algorithm.md) reading `image` — fed a seed that spells the entry `picture`:

```python
from recordstream.algorithm import Algorithm, Input, Output, Param

class BackgroundLevel(Algorithm):
    percentile: float = Param(default=25.0, doc="Percentile rank across the per-row medians.")
    image: Image = Input(doc="The image to read.")
    background: float = Output(doc="The background level.")

    def compute(self):
        per_row = np.median(np.asarray(self.image), axis=1)
        return {"background": float(np.percentile(per_row, self.percentile))}

tracer = Tracer(Stream(ops=[Scale(), BackgroundLevel()]), where="patches.yaml")
tracer.check({"picture": Image(pixels), "class": Label("patches")})
```

```
ChainContractError: patches.yaml:ops[1]: BackgroundLevel needs the record entry 'image', which nothing before it produces — the chain has class, picture at that point (an op that DOES write it must declare it in `produces`)
```

Nothing ran: `tracer.statuses` is `{'ops[0]': 'not reached', 'ops[1]': 'not reached'}`. With the
entry spelled `image` the same chain checks clean — the `BackgroundLevel` row reads
`"consumes": {"image": "Image"}, "produces": {"background": "*"}, "verdict": "ok"`.

## Reruns: the constructor is the authority

`rerun_from(node, **params)` builds the op again as `type(op)(**{**current, **params})`: the node's
CURRENT constructor-parameter values, with the new ones written over. Two things follow.

A value the host set after construction survives — a viewer that writes its window into an op by
setting the attribute gets a rerun that keeps that window. And the constructor validates: a value it
refuses is refused before anything in the trace changes. Measured on the run above:

```python
tracer.rerun_from("ops[1]", low_op="~")
```

```
ValidationError: 1 validation error for ThresholdConfig
low_op
  Input should be '>' or '>=' [type=literal_error, input_value='~', input_type=str]
```

After the refusal `tracer.to_dict()` is equal to what it was before, `generations` is still
`{'ops[0]': 0, 'ops[1]': 0, 'ops[2]': 0, 'ops[3]': 0}`, and the live node still runs with `low_op: ">"`.
A parameter the constructor does not have is refused the same way:

```
tracer.rerun_from("ops[1]", nosuch=1)
ValidationError: 1 validation error for ThresholdConfig
nosuch
  Extra inputs are not permitted [type=extra_forbidden, input_value=1, input_type=int]
```

A node that raised at run time is recorded as `error` and a rerun with a corrected value recovers.
A threshold told to read an entry that is not there:

```python
tracer = Tracer(Stream(ops=[Scale(), Threshold(low_level=0.5, field="nosuch"), ConnectedComponents()]), where="patches.yaml")
tracer.run(record)
```

```
ValueError: Threshold: field 'nosuch' not in record (keys: ['image', 'class'])
```

The error propagates, and the trace keeps it: `tracer.statuses` is
`{'ops[0]': 'ok', 'ops[1]': 'error', 'ops[2]': 'not reached'}`, `tracer.result` is `None`, and
`tracer.to_dict()["nodes"][1]["error"]` reads
`ValueError: Threshold: field 'nosuch' not in record (keys: ['image', 'class'])`. Then
`tracer.rerun_from("ops[1]", field="image").statuses` is `{'ops[0]': 'ok', 'ops[1]': 'ok', 'ops[2]': 'ok'}`
with `generations` `{'ops[0]': 0, 'ops[1]': 1, 'ops[2]': 1}` — `ops[0]` was not run again.

## Snapshots are references; the `in_place` flag

By default a node's input snapshot is the very object the kernel handed it, and its output snapshot is
what it returned — no copy. That keeps the trace cheap (measured for the design record: one
1024x1024 float32 record through four nodes retains 5 MiB by reference and 37 MiB deep-copied), and
it has one blind spot: an op that edits the record it was given, instead of returning a new one, makes
the input snapshot show the edited record. The trace flags exactly that case by comparing the
record's entries before and after the call by identity:

```python
def blank(record):                                  # overwrites 'image' in the record it was given
    record["image"] = Image(np.zeros_like(np.asarray(record["image"])))
    return record

nodes = Tracer(Stream(ops=[Scale(), blank])).run(record).to_dict()["nodes"]
[(n["node"], n["op"], n["in_place"]) for n in nodes]
# [('ops[0]', 'Scale', False), ('ops[1]', 'function', True)]
```

With references, `value("ops[0]", "image", side="input")` on a `Tracer(Stream(ops=[blank]))` shows the
zeros the op wrote (its maximum is `0.0`); with `Tracer(Stream(ops=[blank]), copy_snapshots=True)` the
input snapshot keeps what the node received (maximum `200.0`). Turn copies on for a chain whose entries
carry `in_place: true`; leave them off otherwise.

## Every refusal

Each message is located: `where:node:` when the tracer was given a `where` (here `patches.yaml`) and a
node is involved, `node:` alone without a `where`, and `Tracer:` when no node is involved either.

| you do | you get |
| --- | --- |
| `check(seed)` / `run(seed)` with an entry a node needs missing from the seed | `ChainContractError: patches.yaml:ops[1]: BackgroundLevel needs the record entry 'image', which nothing before it produces — the chain has class, picture at that point (an op that DOES write it must declare it in `produces`)` — nothing ran |
| `run(record, until="nosuch")` | `KeyError: "patches.yaml: no node named 'nosuch' — the nodes are ['ops[0]', 'ops[1]']"` |
| `step()` or `resume()` when nothing is paused | `RuntimeError: patches.yaml: nothing is paused — run(seed, until=<node>) first` |
| `rerun_from("ops[1]", low_op="~")` — a value the constructor refuses | `ValidationError: 1 validation error for ThresholdConfig / low_op / Input should be '>' or '>=' [type=literal_error, input_value='~', input_type=str]` — the trace is untouched |
| `rerun_from("ops[1]", nosuch=1)` — a parameter the constructor does not have | `ValidationError: 1 validation error for ThresholdConfig / nosuch / Extra inputs are not permitted [type=extra_forbidden, input_value=1, input_type=int]` |
| `rerun_from("ops[3]")` on a node the run never got to | `RuntimeError: ops[3]: 'ops[3]' was never reached (its status is 'not reached') — run() first` |
| `rerun_from("nosuch")` | `KeyError: "Tracer: no node named 'nosuch' — the nodes are ['ops[0]', 'ops[1]', 'ops[2]', 'ops[3]']"` |
| a node that raises while running | the op's own error propagates — `ValueError: Threshold: field 'nosuch' not in record (keys: ['image', 'class'])` — the node is `error`, the nodes after it `not reached`, `result` is `None` |
| `value("ops[1]", "nosuch")` | `KeyError: "patches.yaml:ops[1]: 'ops[1]' has no output entry 'nosuch' — it has ['class', 'image', 'mask']"` |
| `value("ops[1]", "mask", side="both")` | `ValueError: ops[1]: side must be one of ('input', 'output'), got 'both'` |
| `value("ops[1]", "mask")` before any run | `RuntimeError: ops[1]: 'ops[1]' was never reached — nothing is recorded for it` |
| `value("ops[2]", "mask")` on the node the run paused before | `RuntimeError: ops[2]: 'ops[2]' has no output yet (its status is 'paused')` — its `side="input"` is there |
| `Tracer(FlowGraph()).names` — a flow graph with no flow | `ValueError: FlowGraph.flow is not set — provide a flow mapping or FlowStep list.` |
| `Tracer(42).names` | `TypeError: Tracer: expected a Stream or a FlowGraph, got int` |
| a flow whose `from:` names a later step, on first use | `ValueError: flow step 'late': from: 'early' does not name an EARLIER step (document order is the schedule; steps so far: [])` |
| `rerun_from` on a node after a 1→N expanding node | `RuntimeError: ops[1]: 'ops[1]' is a 1→N expanding node and the trace keeps only its last branch — rerun from 'ops[1]' or earlier instead` |

## Limits

* A [1→N expanding node](graph.md#expanding-1n-steps) runs the rest of the graph once per child, and
  the trace keeps one entry per node, so the nodes after it show the LAST branch only; a rerun from a
  node after it is refused (the last row above), a rerun from the expanding node itself works.
* A `bind:` or `merge_from:` that fails inside the kernel — before the node's op is called —
  propagates out of `run` with the node still `not reached`; it is not attributed to a node.
* A node that declares types rather than entry names is `unverifiable` to the check, and so are its
  flags and reports; the run shows what it did.

Related: [graph.md](graph.md) for `flow:` documents and the kernel the tracer runs on,
[graph-contract.md](graph-contract.md) for what a whole graph must deliver, [algorithm.md](algorithm.md)
for declaring an op's inputs and outputs by name.
