# RecordStream

**RecordStream** is a high-performance, functional data processing engine built for modern Machine Learning pipelines. It provides a clean, fluent API for streaming and transforming data from any source while maintaining strict compatibility with PyTorch and Hugging Face.

Part of the **Modular Quartet**: `Loggair`, `Confluid`, `Liquifai`, and `RecordStream`.

## 🚀 Key Features

-   **A record is a plain dict:** the [record model](https://github.com/Gearlux/recordstream/blob/main/docs/record-model.md) — a `dict` of typed values (`Image`, `Mask`, `Boxes`, `Label`, `MultiLabel`, …), each owning its own metadata, with key names carrying meaning (`"image"`, `"mask"`, `"bboxes"`). No wrapper container, no role tags.
-   **Libraries run AS-IS:** bare [albumentations and torchvision `transforms.v2`](https://github.com/Gearlux/recordstream/blob/main/docs/augmentation.md) transforms drop straight into any ops list — the engine invokes each op family natively (one call = one joint draw across image/mask/boxes). No adapter classes anywhere.
-   **Type-dispatched native ops:** a `Transform` samples its parameters once per record and applies a per-type kernel to every value it handles — teach an existing op a new value type with one `@MyOp.kernel(NewType)` registration.
-   **Graph pipelines:** readable [`flow:` documents](https://github.com/Gearlux/recordstream/blob/main/docs/graph.md) of named steps — `from:` forks, `merge_from:` merges, `bind:` feeds one step's value into another's parameter. An `ops:` list is the same engine's linear spelling; both parse to one step graph.
-   **Debug a graph:** [`Tracer`](https://github.com/Gearlux/recordstream/blob/main/docs/trace.md) runs one record through a `Stream` or a `FlowGraph` on the engine's own kernel — stop before a node, see what every node received and produced, rerun one node with a new value and recompute only from there.
-   **High Performance:** Native multiprocess support via `.parallel(workers=N)` using the safe `spawn` context. At most `window × N` records are in flight (`.parallel(workers=N, window=2)`, `2` the default) and each is dropped once yielded, so memory stays flat however long the source; records come out in source order, so widen `window` when a few records take far longer than the rest. The `Parallel` op takes the same two settings in a config (`!class:recordstream.ops.parallel.Parallel` with `workers: 4` and `window: 2`). [1→N expanding ops](https://github.com/Gearlux/recordstream/blob/main/docs/kinds.md#1n-expanding-ops-iterable-only-pipelines) flatten in every route.
-   **Advanced Storage:** HDF5, Zarr and Directory backends with matching read-back sources and [metadata-only querying](https://github.com/Gearlux/recordstream/blob/main/docs/storage.md#queryable-metadata-recordstreamstoragequery) — filter stored datasets without loading a single array.
-   **Passive Introspection:** ops declare the value types they [handle / consume / produce](https://github.com/Gearlux/recordstream/blob/main/docs/record-model.md) and are discoverable by category for visual editors and schema generators.
-   **Algorithms:** declare what an op is tuned by, reads and computes (`Param` / `Input` / `Output`) and write `compute()`. The [record handling, the constructor, the settings schema and the chain contract](https://github.com/Gearlux/recordstream/blob/main/docs/algorithm.md) are derived from those declarations.
-   **100% Reproducibility:** Entire pipelines are serializable via **Confluid** manifests.

## 🛠 Quick Start

One pipeline mixing a **bare albumentations Compose** (image + mask + boxes move together in one draw), a **bare torchvision v2 transform**, and a **native op** — no wrappers (mirrors [`examples/record_pipeline.py`](https://github.com/Gearlux/recordstream/blob/main/examples/record_pipeline.py)):

```python
import albumentations as A
import numpy as np
from recordstream import Stream, Image, Label, Mask, as_transform

records = [
    {
        "image": Image(rng.random((16, 20, 3)).astype(np.float32)),   # typed: knows its layout
        "mask": Mask((rng.random((16, 20)) > 0.5).astype(np.uint8)),
        "bboxes": [[2, 3, 6, 7]],                                     # albumentations vocabulary
        "labels": ["drone"],
        "class": Label("drone_x", classes=["noise", "drone_x"]),      # typed: knows its vocab
        "gain_db": -3.0,                                              # a scalar is just another key
    }
    for rng in (np.random.default_rng(i) for i in range(100))
]

stream = Stream(
    source=records,
    ops=[
        A.Compose(                                    # bare albumentations — as-is
            [A.HorizontalFlip(p=0.5)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        ),
        A.GaussNoise(p=1.0),                          # image only (its own kwarg vocabulary)
        as_transform(lambda d: d - 0.5, handles=(Image,)),   # native: a plain function op
    ],
).parallel(workers=4)

for record in stream:
    print(record["image"].shape, record["class"].value)   # image+mask+boxes flipped together
```

The same ops list in Confluid YAML — bare library transforms are ordinary `!class:` nodes:

```yaml
ops:
  - !class:albumentations.HorizontalFlip
    p: 0.5
  - !class:albumentations.GaussNoise
    p: 1.0
  - !class:recordstream.ops.numpy.Threshold
    low_level: 0.5
```

### Toggling a branch from the CLI (`Enable`)

Wrap any stretch of an ops list in `Enable` to switch the whole chain on or off from one flag.
The toggle is the declared `enabled` parameter; `name` identifies the wrapper so several of them
toggle independently:

```yaml
ops:
  - !class:recordstream.ops.numpy.Threshold {low_level: 0.5}
  - !class:recordstream.ops.enable.Enable
    name: visualize          # ← names THIS wrapper; scopes its CLI flag
    enabled: false           # ← off by default; the chain below is skipped
    ops:
      - !class:recordstream.ops.image.ConvertToImage {}
      - !class:recordstream.ops.debug.PrintRecordOp {}
```

```bash
recordstream run pipeline.yaml --visualize.enabled true   # this wrapper only
recordstream run pipeline.yaml --visualize.enabled+       # polarity shorthand → True
recordstream run pipeline.yaml --enabled false            # broadcast: every Enable off
```

Inner ops are not materialized until the wrapper first fires, so gating an expensive chain with
`enabled: false` costs nothing at startup. In Python the same wrapper is one call —
`Enable(ops=[...], name="visualize", enabled=False)` — which is what lets a visual editor or a
generated tool schema set the toggle too (see [docs/architecture.md](https://github.com/Gearlux/recordstream/blob/main/docs/architecture.md#6-every-knob-is-a-declared-parameter--the-enable-toggle-2026-07-27)).

### Inference as an op (`ModelPredict`)

A pipeline can carry its own inference: `recordstream.ops.predict.ModelPredict` runs any
callable model wrapper on each record and stamps the prediction back as a record field —
a class `Label`, `Boxes`, an int class mask, or the restored image, by `kind`. The model's
heavy work (build the network, load `checkpoint_path`) happens in its `solidify()`, called
lazily on the first record; the op itself imports no ML framework.

```yaml
pipeline: !class:recordstream.core.stream.Stream
  source: !class:recordstream.sources.huggingface.HuggingFaceSource {path: ylecun/mnist, split: test}
  ops:
    - !class:recordstream.ops.image.ConvertToImage {width: 224, height: 224}
    - !class:recordstream.ops.predict.ModelPredict
        model: !class:<your model wrapper> {checkpoint_path: runs/checkpoints/mnist/last.ckpt}
        kind: classification        # or detection / segmentation / restoration
```

A viewer reads the stamped `predict*` fields back as layers; `recordstream run` executes
the same document offline.

### Stating a pipeline's interface (`RecordContract`)

`recordstream.ops.contract.RecordContract` is a pass-through op that asserts what each
record carries at the point in the chain where it sits — an executable, visible interface
statement. One class serves both boundary roles, decided by position: the **first** op in
a chain states what the host must feed (input contract), the **last** states what the
pipeline guarantees to deliver (output contract). `fields` maps a record key to a
registered item type name (`"*"` = present with any type, for boundaries past a
conversion that emits plain values); `name` labels the boundary in errors:

```yaml
pipeline: !class:recordstream.core.stream.Stream
  source: !class:recordstream.sources.huggingface.HuggingFaceSource {path: ylecun/mnist, split: test}
  ops:
    - !class:recordstream.ops.contract.RecordContract
        name: classification output
        fields: {image: Image, class: Label}
```

A violating record fails loudly at the boundary — naming the contract, the record
ordinal, the offending entry, and what the record does carry — instead of surfacing
later as an empty result:

```
ContractError: classification output: record #0 has no entry 'class' (expected Label); present: image[Image]
```

A consuming workspace or visual editor seeds the contract into a graph so the required
record shape is declared before the first node is wired; the exported document then
enforces the same contract when it runs offline.

### Checking that a chain holds together (`check_chain`)

`RecordContract` answers *"what must reach this point?"*. A chain of many small ops raises a
different question — *"does this chain hold together at all?"* — and getting it wrong is **silent**:
an op that reads an entry an earlier op was supposed to write does not raise, it returns the record
unchanged, so the pipeline runs to completion and answers nothing.

So an op MAY declare its interface as class attributes, and `check_chain` reads them off the op list
once, before the first record: A visual editor may draw a gate as a wire from the producer's outcome and write `requires` back into the document on export — the document form stays the one shown here.

```python
class MeasureSymbolClock:
    consumes = {"signal": "Signal", "inst_freq": "InstFreq"}   # same {key: item type} vocabulary
    produces = {"symbol_clock": "SymbolClock"}                 # as RecordContract.fields
    reports  = "clock"        # its key in an analysis report; "" marks a transform
    flags    = ()             # the boolean findings it raises

class DecodeBleAdvertising:
    def __init__(self, requires: str = "") -> None:
        self.requires = requires        # the ONE flag that gates this op
```

```python
from recordstream.ops.contract import check_chain

check_chain(ops, provided={"signal"}, where="view_bte.yaml")
# ChainContractError: view_bte.yaml: MeasureSymbolClock needs the record entry 'inst_freq',
# which nothing before it produces — MeasureInstantaneousFrequency produces it, but LATER in
# the chain — move it before MeasureSymbolClock
```

Four things are refused: an unmet `consumes`, a `requires` naming a flag nothing raises (or one
raised only later), two ops declaring one flag, and two ops reporting under one name.

Declaring is **opt-in** — an op with none of these attributes is checked for nothing, so existing
chains are unaffected. `flag_producers(ops)` gives `{flag: index of the op that raises it}`, total by
construction, so *"which op decided this branch?"* always has an answer.

### Tracing one record through a graph (`Tracer`)

`check_chain` answers *"does this chain hold together?"* before a run. The next question comes from a
debugger, a per-node tool or an LLM re-running one node: *"what did each node receive and produce,
and what changes downstream if this parameter changes?"* `Tracer` (`recordstream.flow`) runs ONE
record through a `Stream` or a `FlowGraph` on the engine's own kernel and keeps a snapshot per node:

```python
from recordstream.flow import Tracer

tracer = Tracer(graph)                        # a Stream (nodes ops[0], ops[1], …) or a FlowGraph (its step keys)
tracer.check(seed)                            # static: refuses an unmet need BEFORE anything runs, located at the node
tracer.run(seed, until="gated")               # pauses BEFORE 'gated' — everything so far recorded
tracer.step(); tracer.resume()                # one node on; to the end
tracer.rerun_from("floor", percentile=75.0)   # rebuilds ONE node through its constructor, recomputes from there
tracer.value("gated", "mask")                 # the real array, only when asked
tracer.to_dict()                              # JSON: per node status, ms, params, bound values, summarised entries
```

A refused need is located: `ops[0]: BackgroundLevel needs the record entry 'image', which nothing
before it produces — the chain has class, picture at that point`. A rerun starts from the node's
CURRENT parameter values (what a host set after construction survives) and goes through the
constructor, so a value it refuses — `rerun_from("floor", percentile=150.0)` →
`ValidationError: percentile Input should be less than or equal to 100` — leaves the trace untouched,
and the nodes before it keep their generation and are not run again. Snapshots are references (one
1024x1024 record through four nodes keeps 5 MiB; `copy_snapshots=True` keeps 37 MiB), and an op that
edits its record in place is flagged `in_place`. Measured over 200 records of 64x64 through four ops:
0.103 ms per record plain, 0.161 ms traced (0.027 ms of it the static check). Walkthrough:
[docs/trace.md](https://github.com/Gearlux/recordstream/blob/main/docs/trace.md).

### A flow used as one op (`Subgraph`)

A `Subgraph` is a `flow:` mapping written inside the step that uses it. The outer flow sees ONE step;
its inside runs on the same kernel as any flow, on the record that step receives:

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

On a 120x160 image with two light rectangles on a dark background this yields a mask with mean
`0.2855`, exactly as the same three steps written flat. `result:` names the inner step whose record
leaves the subgraph (blank = the last one). The tracer lists the inside:
`Tracer(graph).names` is `['prep', 'prep/grey', 'prep/scaled', 'mask']`, and
`run(record, until="prep/scaled")` pauses inside it.

Every mistake in a subgraph written as a flow step's op or as an `ops:` list member is refused **before the first record**, in words, naming the outer step (a subgraph held inside another op's slot — `Enable(ops=[…])`, `RandomApply(op=…)`, `Pipeline(transforms=[…])` — is opened only when that op first calls it):

| you write | you get |
| --- | --- |
| `result: nosuch` | `flow step 'prep' (a subgraph): Subgraph: result 'nosuch' does not name an inner step (the steps are: ['grey', 'scaled']) (at demo.yaml:3:9)` |
| an inner step with `from: grey`, where `grey` is an OUTER step | `… Subgraph: step 'mask' reads 'grey' (from: 'grey'), which is not a step inside this subgraph — a step inside a subgraph reads only the record the subgraph receives and the steps before it inside …` |
| an outer step with `bind: {value: measure.level}`, `level` a step inside `measure` | `flow step 'stamp': bind value='measure.level' reads 'level' of 'measure', which is a subgraph — a step outside a subgraph cannot read a value of a step inside it; move that step out of the subgraph` |
| a 1→N expanding op inside (here the test op `SgSplit`) | `… Subgraph: step 'split' (SgSplit) is a 1→N expanding op, and a subgraph returns one record per record it receives — move the step out of the subgraph` |
| a step named `prep/grey` | `flow: step name 'prep/grey' may not contain '/' (reserved for the nodes inside a subgraph …)` |

A subgraph's `consumes`, `produces` and `flags` are read off its inner ops, so `check_chain` and the
tracer's check see it as the steps it holds: a flag raised inside reaches a gate after it. `produces` and `flags`
count only the inner steps on the lineage of `result` — the record the subgraph returns.

**Using one subgraph twice** — three ways:

1. **Copy the block.** A second `op: !class:…Subgraph` with the same `steps:` is an independent copy.
2. **Keep it as a template.** A steps file (next point) kept aside and inserted as a copy wherever it
   is needed; nothing links the copies back.
3. **Include one steps file.** Move the steps to `prep.steps.yaml` (a bare steps mapping) and write
   `steps: {include: prep.steps.yaml}` in each use — every use gets its own op objects:

   ```yaml
   flow:
     read: {}
     prep:
       op: !class:recordstream.flow.subgraph.Subgraph
         steps:
           include: prep.steps.yaml
         result: scaled
     prep2:
       from: read
       op: !class:recordstream.flow.subgraph.Subgraph
         steps:
           include: prep.steps.yaml
   ```

YAML anchors and `!ref:` are not reuse spellings: an anchor merged into `flow:` always runs first
whatever line it is written on, and a `!ref:` makes every use ONE object, so a setting written on one
use changes the other. Full guide: [docs/graph.md](https://github.com/Gearlux/recordstream/blob/main/docs/graph.md#subgraphs-a-flow-used-as-one-op);
why it is an op: [docs/architecture.md](https://github.com/Gearlux/recordstream/blob/main/docs/architecture.md).

### Writing an op as an algorithm (`Algorithm`)

An algorithm declares its settings, inputs and outputs, and never sees a record:

```python
import numpy as np
from recordstream import Algorithm, Image, Input, Output, Param

class BackgroundLevel(Algorithm):
    percentile: float = Param(default=25.0, doc="Percentile rank across the per-row medians.")
    image: Image = Input(doc="The image to read.")
    background: float = Output(doc="The background level.")

    def compute(self):
        per_row = np.median(np.asarray(self.image, dtype=np.float64), axis=1)
        return {"background": float(np.percentile(per_row, self.percentile))}

image = Image(np.full((4, 6), 10.0), layout="HWC")
BackgroundLevel(percentile=30.0).run(image=image)                 # {'background': 10.0}
BackgroundLevel()({"image": image, "exposure_ms": 20})            # the record + 'background'
BackgroundLevel(keys={"image": "photo"})({"photo": image})        # read from another entry
BackgroundLevel().consumes                                        # {'image': 'Image'} — for check_chain
```

Used as an op, it reads each input from the record entry of the same name and writes each output
the same way; `keys` names other entries, and `Output(replaces="boxes")` writes back where an input
was read. Inputs are found by name only, so `check_chain` knows before a run what a pipeline needs.
Full guide: [docs/algorithm.md](https://github.com/Gearlux/recordstream/blob/main/docs/algorithm.md).

### A training set of valid examples (`DrawSource`)

A generator (an `Algorithm` with no record inputs) gives the same record every time. `DrawSource`
draws its settings anew per record. The draws run top to bottom, and each picks only among the values
the generator itself accepts given the draws above it — so every record is a valid example and the
generator's rules are never restated:

```python
from recordstream.draws import Choice, Repeat, Uniform
from recordstream.sources import DrawSource

source = DrawSource(
    generator=Traffic(),          # an Algorithm whose settings hold a Road with lanes
    count=1000,
    seed=7,
    draws=[
        Choice(field="road.width"),                                   # every value its Literal allows
        Choice(field="road.closed", values=[None, 0], weights=[3, 1]),
        Repeat(field="road.lanes", count=(0, 8), each=[Choice(field="position")]),  # while there is room
        Uniform(field="speed", low=30, high=120),
    ],
)
source[0]["load"]                 # the generator's output
source[0]["settings"]             # its drawn settings file: confluid.load() rebuilds that one example
```

Record `i` depends only on `seed`, `i` and the draws. A list of draws that cannot be met stops with
the generator's own reason (`DrawRefused: road.width: none of [2, 4] is accepted after nothing drawn —
Road: a lane off the road …`). The draws are `Choice`, `Uniform`, `Grid` (a number among whole multiples
of a step, linear, log or power, written with its unit: `low: 1MHz, high: 2.5MHz, step: 100kHz`), `Span`
and `Repeat`. Full guide: [docs/sources.md](https://github.com/Gearlux/recordstream/blob/main/docs/sources.md#drawing-a-generators-settings-drawsource);
why the generator judges: [docs/architecture.md](https://github.com/Gearlux/recordstream/blob/main/docs/architecture.md).

### Declaring the class vocabulary (`ClassNamesOutput`, `ClassNamesScan`)

`RecordContract` states what each RECORD carries. A classification pipeline usually has
something to say about the DATASET too — the ordered class list a consumer needs to show a
label as a name rather than an integer. `ClassNamesOutput` is that statement. It holds one
thing, the list, and derives nothing:

```yaml
class_names_output: !class:recordstream.ops.contract.ClassNamesOutput
  names: [cat, dog]                           # in class-id order
```

The list gets there one of three ways, each chosen explicitly:

| route | how |
|---|---|
| state them | type `names: [cat, dog]` |
| take them from a source that knows them | wire the source's `class_names` output into `names` — `HuggingFaceSource` reads its `ClassLabel` names from the dataset metadata, no rows read |
| take them from a walker | wire a `ClassNamesScan`'s `class_names` output into `names` |

`ClassNamesScan` derives a vocabulary by walking a source's label column — for a source whose
format does not describe one (a folder reader, a CSV):

```python
from recordstream import Label
from recordstream.ops.contract import ClassNamesOutput, ClassNamesScan

records = [{"class": Label(value="dog")}, {"class": Label(value="cat")}, {"class": Label(value="dog")}]
output = ClassNamesOutput(names=ClassNamesScan(source=records).class_names)
output.class_names  # ['cat', 'dog']
```

The walk goes through [key projection](https://github.com/Gearlux/recordstream/blob/main/docs/projection.md)
so a projection-aware source is asked
only for the label column, and it reports sorted-unique stringified values — the same ordering
`LabelMap.fit` uses. It is deliberately a **separate class**: a walk costs seconds per thousand
records, so it happens because someone asked for it, never as a silent fallback inside
`ClassNamesOutput`.

**In a document, a wire is a literal.** A config document cannot reference one object's
attribute from another — `names: !ref:scan.class_names` is refused when the document loads — so
in a document the vocabulary is always the list itself. A visual editor that lets you draw the
wire evaluates it when it saves the graph and writes the resulting list into `names`.

## 📚 Documentation

| Page | Covers |
|---|---|
| [docs/record-model.md](https://github.com/Gearlux/recordstream/blob/main/docs/record-model.md) | The record data model: a plain dict of typed values, type-dispatched ops and kernels, mixing libraries as-is, custom item types, engines, storage layout |
| [docs/algorithm.md](https://github.com/Gearlux/recordstream/blob/main/docs/algorithm.md) | Writing an op as an `Algorithm`: `Param` / `Input` / `Output` slots, `compute()`, entry names (`keys`), replacing outputs, what tools derive (`algorithm_spec`, `consumes` / `produces`, the settings schema) |
| [docs/kinds.md](https://github.com/Gearlux/recordstream/blob/main/docs/kinds.md) | Writing ops (kernels, `field=`, type-changing ops), the collate registry (`collate_records`) + its read-back (`batch_values` / `batch_tensor` / `batch_metadata`), the Keras `RecordSequence` adapter, 1→N expanding ops |
| [docs/graph.md](https://github.com/Gearlux/recordstream/blob/main/docs/graph.md) | `flow:` documents + the `FlowGraph` engine, `ops:` as the linear spelling of the same step graph, subgraphs (`Subgraph` — a flow used as one op, and using one twice), expanding (1→N) steps, `Stream.from_ops_yaml`, tracing one record (`Tracer`) |
| [docs/graph-contract.md](https://github.com/Gearlux/recordstream/blob/main/docs/graph-contract.md) | what ONE graph must deliver — outputs (all required), the records of a delivered stream, `delivered:` written by the editor, the `classes` slot rule, and every refusal `check()` gives |
| [docs/trace.md](https://github.com/Gearlux/recordstream/blob/main/docs/trace.md) | Tracing one record through a `Stream` or a `FlowGraph` (`Tracer`): the static check, pausing before a node, `step` / `resume`, rerunning one node through its constructor, looking inside a subgraph (`prep/grey`), the JSON trace, every refusal |
| [docs/sources.md](https://github.com/Gearlux/recordstream/blob/main/docs/sources.md) | `HuggingFaceSource`, `FilesSource`, `DatasetSplit` train/val/test views, `RangeSource`, `ConcatSource`, `DrawSource` (a generator's settings drawn per record), Confluid `!ref:` sharing, dataset identity (`dataset_uri` / `dataset_url`) |
| [docs/storage.md](https://github.com/Gearlux/recordstream/blob/main/docs/storage.md) | HDF5 / Zarr / Directory sinks & sources (`typedrecord-v1`), array-valued item attributes, the `SupportsMetadataScan` protocol + `MetadataFilterSource` querying |
| [docs/projection.md](https://github.com/Gearlux/recordstream/blob/main/docs/projection.md) | Key projection (`SupportsProjection`), lazy key walks (`iter_key`), one-peek `first_value`, `num_classes`, the fittable `LabelMap`, class-balance weights |
| [docs/predictions.md](https://github.com/Gearlux/recordstream/blob/main/docs/predictions.md) | The model boundary: prediction-output contracts (`ClassificationOutput` & co), `ensure_record_dataset`, the `PredictionsSink` protocol + the classification sink |
| [docs/image.md](https://github.com/Gearlux/recordstream/blob/main/docs/image.md) | Generic value→image conversion (`ConvertToImage`, `normalize_to_uint8`), mask→class-id conversion (`ConvertToMask`), channel layout / value range / element type (`ConvertMode`, `Scale`, `ToType`), array introspection helpers |
| [docs/configure.md](https://github.com/Gearlux/recordstream/blob/main/docs/configure.md) | Per-record op parameters (`ConfigureOp` and a flow step's `bind:`) |
| [docs/runnable.md](https://github.com/Gearlux/recordstream/blob/main/docs/runnable.md) | Runnables (`run()` + `recordstream run`), the `@entrypoint` task/role markers + `run_entrypoint` dispatch with a worked example, `TorchRunner` / `ProgressReporting` |
| [docs/workflow.md](https://github.com/Gearlux/recordstream/blob/main/docs/workflow.md) | Workflow combinators (`Sequence`/`Conditional`/`Switch` + predicates): resume-safe multi-stage pipelines as ONE document |
| [docs/augmentation.md](https://github.com/Gearlux/recordstream/blob/main/docs/augmentation.md) | Augmentation via bare albumentations / torchvision `transforms.v2` — the op-family dispatch, key vocabulary, bbox recipes, seeding |
| [docs/architecture.md](https://github.com/Gearlux/recordstream/blob/main/docs/architecture.md) | Architecture decision records — the *why* behind non-obvious mechanisms (e.g. why collation is a pluggable registry) |

## 🧭 Scope: a modality-neutral engine

RecordStream deliberately contains **no domain-specific code** — every op, source and sink in this package is meaningful for any modality (arrays, tensors, images, generic metadata). Domain packages build on it and keep their own vocabulary:

- Signal/waveform items and ops (spectrograms, FFT windows, recording formats) live in the domain package, which registers its item types into the same registries.
- Task-specific trainers, collates and models live in their consuming projects.

## 🌐 Ecosystem Integration

RecordStream is designed to sit between your data catalog and your training loop, acting as the high-performance "glue" for ML pipelines:

- **Hugging Face** for community datasets and Arrow/Parquet loading — `HuggingFaceSource` turns a `datasets.Dataset` into record dicts of typed values with full metadata traceability, and [names the dataset it reads](https://github.com/Gearlux/recordstream/blob/main/docs/sources.md#identifying-a-dataset) so a run record can point at it (see [docs/sources.md](https://github.com/Gearlux/recordstream/blob/main/docs/sources.md)).
- **Confluid** for configuration: every pipeline is a YAML document, every op a `!class:` node — including bare library transforms — every run reproducible.
- **PyTorch**: `Stream` and `FlowGraph` implement the `Dataset` protocol (`__len__`/`__getitem__`/`.batch`/`.parallel`) and plug straight into a `DataLoader` with a [registry collate](https://github.com/Gearlux/recordstream/blob/main/docs/kinds.md#batching--collate_records--the-collate-registry-recordstreamcollate) (`collate_records` is the default).
- **Keras 3**: no `DataLoader` exists to do the batching, so [`RecordSequence`](https://github.com/Gearlux/recordstream/blob/main/docs/kinds.md#keras-recordsequence--the-batching-half-the-framework-leaves-to-you) is the `keras.utils.PyDataset` half — row order, slicing, per-epoch reshuffle, `collate_records` — and a `transform` callable supplies the batch shape, exactly as `collate_fn` does for torch.
- **Augmentation libraries**: [albumentations](https://albumentations.ai) and torchvision `transforms.v2` transforms run **as-is** in any ops list — the engine speaks each library's native convention (kwarg vocabulary vs dict walk), so there is nothing to wrap (see [docs/augmentation.md](https://github.com/Gearlux/recordstream/blob/main/docs/augmentation.md)).

## 🔧 Installation

```bash
pip install recordstream
```

RecordStream is on PyPI as a pre-release; pip installs it without `--pre`, because no final release
exists yet. Leave `--pre` off: it lets pip take pre-releases of *every* dependency too (a pydantic beta,
for one).

The core engine is **numpy**, and installs no ML framework. A framework arrives only with the extra
that needs it:

| Extra | Provides |
|---|---|
| `torch` | The pieces that genuinely produce tensors — the `ToTensor` op, `batch_tensor`, and the `classification_output` / `segmentation_output` builders |
| `keras` | `recordstream.keras` — the `RecordSequence` `PyDataset` adapter and the `KERAS_BACKEND` ordering. Keras 3 is an API, so this names no compute engine; it runs on whichever of torch / TensorFlow / JAX you have |

```bash
pip install "recordstream[torch]"
```

Everything else works without either. A `Stream` is map-style (`__len__`/`__getitem__`), so a
`DataLoader` still accepts one directly on a torch install; `batch_values`, `multi_hot` and the
class-balance statistics return numpy, so a non-torch backend converts in one line. Reaching for
`recordstream.ops.ToTensor` without the extra raises an `ImportError` naming it.

## 📄 License

MIT
