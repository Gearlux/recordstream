# Graph contracts — what one graph takes from its host and must deliver back

A consuming workspace opens several graphs — one that delivers its records, one that turns
dropped files into records, one that consumes the records it labelled — and each must deliver
something different. `recordstream.ops.contract.GraphContract` states that per graph, as DATA:
a map of named, typed slots. A visual editor draws every graph's boundary from it (a root node
whose input sockets are the `outputs`; an input node whose output sockets are the `inputs`),
its export writes what the author wired into `delivered`, and the host refuses a graph that
delivers less than it promised — in the same words, because `check()` is the ONE verification
both of them call.

[`RecordContract`](../README.md) states what each RECORD carries at one point of a chain; a
`GraphContract` states what a whole GRAPH hands back. The two meet in `records`: the graph
contract appends a `RecordContract` as the delivered stream's last op.

## What a contract declares

| key | meaning | example |
| --- | --- | --- |
| `name` | the graph, as a refusal names it | `classification source` |
| `inputs` | `{slot: kind}` the HOST provides when it runs the graph | `{files: "List[str]"}` |
| `outputs` | `{slot: kind}` the graph must deliver — EVERY entry is required | `{stream: Stream, classes: "List[str]"}` |
| `records` | `{entry: item type}` every record of a delivered stream carries | `{input: Image, target: Label}` |
| `optional` | `{slot: kind}` the graph MAY deliver — only where the host has its own answer when it is left unwired | `{classes: "List[str]"}` |
| `delivered` | `{slot: value}` — what the drawn graph wired into each output, and into each `records` entry the key it is read from; written by the editor, read by the host | `{stream: !ref:stream, classes: [cat, dog], input: image}` |

A kind is one of the graph slot kinds — `Stream`, `Source`, `Sink`, `Op`, `Record`, `List[str]`,
`str`, `int`, `float`, `bool` — or any registered item type name (`Image`, `Label`, `Boxes`, …).
There is no optional output: what a graph may or may not deliver is simply not declared.

## A source graph: records plus their class vocabulary

The workspace declares the contract; the author wires a Stream and the class names; the saved
document carries the contract with its `delivered` map:

```yaml
data: !class:recordstream.sources.huggingface.HuggingFaceSource {path: ylecun/mnist, split: train}
stream: !class:recordstream.core.stream.Stream
  source: !ref:data
source: !class:recordstream.ops.contract.GraphContract
  name: classification source
  outputs: {stream: Stream, classes: "List[str]"}
  records: {image: Image, class: Label}
  delivered:
    stream: !ref:stream
    classes: !ref:data          # the source declares its own class names — or type them: [cat, dog]
```

The host reads it through three calls:

```python
contract = confluid.load(text)["source"]
contract.check()                 # every refusal below, or nothing
stream = contract.stream()       # a NEW Stream: the delivered one's source and ops, the records contract appended
contract.class_names             # ['0', '1', …, '9'] — read from the wired source's declaration
```

`stream()` never rewrites the delivered Stream (it is the document's object); the returned
Stream carries `class_names` too, so the vocabulary travels with the records.

## Wiring a record entry: the root's names, the source's keys

A host reads records by the root's names (`input`, `target`); a source writes its own (a
HuggingFace source writes `image` and `class`, and declares them in `produces`). A visual editor
draws each `records` entry as an input of the root and each declared entry of a source as an
output; a wire from `image` into `input` is saved as the key it is read from:

```yaml
data: !class:recordstream.sources.huggingface.HuggingFaceSource {path: ylecun/mnist}
stream: !class:recordstream.core.stream.Stream
  source: !ref:data
contract: !class:recordstream.ops.contract.GraphContract
  name: classification source
  outputs: {stream: Stream, classes: "List[str]"}
  records: {input: Image, target: Label}
  delivered:
    stream: !ref:stream
    classes: !ref:data
    input: image                # the root's `input` is the source's `image`
    target: class
```

`stream()` hands every record on with those entries under the root's names — the host sees
`input` and `target`, the other entries ride along — and `wired_entries()` answers
`{"input": "image", "target": "class"}`. No rename step is drawn: the wire says it. An entry
nothing is wired into is read under its own name, so a chain that already writes `input` needs
no wire.

The same refusal the outputs get covers a wired entry: an entry named like an output, or like a
constructor parameter of a delivered object, is refused at declaration time (confluid would push
the key into that parameter).

## A slot the host can do without (`optional`)

Every entry of `outputs` is required. A slot the host has its own answer for is declared apart,
under `optional`: a graph of dropped files may type its own class list, and when it leaves the slot
unwired the host uses its source graph's. Wired, an optional slot is checked like an output (an
empty typed list is refused); unwired, it is never missing.

```yaml
class_list: !class:recordstream.ops.contract.ClassNamesOutput {names: [cat, dog]}
contract: !class:recordstream.ops.contract.GraphContract
  name: classification files
  inputs: {files: "List[str]"}
  outputs: {stream: Stream}
  optional: {classes: "List[str]"}
  records: {input: Image}
  delivered:
    stream: !ref:stream
    classes: !ref:class_list        # leave this line out and the host answers with its own list
    input: image
```

A slot declared both under `outputs` and under `optional` is refused, and so is an optional slot
named like a constructor parameter of a delivered object (the broadcast rule below).

## A graph with host inputs

An input is something the host has and the graph needs — the files a user dropped, the
records to annotate, a viewer's window. It is declared beside the outputs and is not checked
by `check()` (the host fills it when it runs the graph):

```yaml
files: !class:recordstream.sources.files.FilesSource {}     # the host fills `files` at run time
stream: !class:recordstream.core.stream.Stream
  source: !ref:files
  ops:
    - !class:recordstream.ops.formats.ReadFile {}
source: !class:recordstream.ops.contract.GraphContract
  name: files graph
  inputs: {files: "List[str]"}
  outputs: {stream: Stream}
  delivered:
    stream: !ref:stream
```

## The class vocabulary slot is `classes`

A source graph delivers its class names under the output slot **`classes`** (kind
`List[str]`), which takes either form:

| you wire | the document keeps | `class_names` reads |
| --- | --- | --- |
| a typed list — `classes: [cat, dog]` | the list | `['cat', 'dog']` |
| a producer that declares `class_names` — a HuggingFace source, a `ClassNamesOutput`, a `ClassNamesScan` | the producer, so where the names came from stays visible | its declaration, as strings |

The slot is deliberately NOT named `class_names`, and the contract refuses that name. confluid
resolves `delivered:` as an ordinary mapping: a key in it that a SIBLING's constructor takes as
a parameter is pushed into that sibling, and `class_names` is a constructor parameter of
`Stream`. Measured 2026-09-27 on the same document, differing in one word:

```yaml
names: !class:recordstream.ops.contract.ClassNamesOutput {names: [cat, dog]}
stream: !class:recordstream.core.stream.Stream
  source: [{a: 1}]
source: !class:recordstream.ops.contract.GraphContract
  name: classification source
  outputs: {stream: Stream, class_names: "List[str]"}     # CON — collides with Stream(class_names=…)
  delivered:
    stream: !ref:stream
    class_names: !ref:names
```

```
ConstructionError: Failed to construct Stream at <unicode string>:2:9: 1 validation error for StreamConfig
```

The `ClassNamesOutput` landed in `Stream(class_names=…)`, which takes a list of strings. With a
LITERAL list under `class_names:` the document loads — and the list is silently pushed into the
Stream (`Stream.class_names == ['cat', 'dog']`, though the Stream never declared them). Renamed:

```yaml
  outputs: {stream: Stream, classes: "List[str]"}         # PRO — no delivered object takes `classes`
  delivered:
    stream: !ref:stream
    classes: !ref:names
```

```
loaded; delivered = {stream: Stream, classes: ClassNamesOutput}; class_names == ['cat', 'dog']; the Stream's own class_names is None
```

The same collision can hit any slot, so the rule is general: an output slot may not share its
name with a constructor parameter of ANY delivered object. `check_declaration()` refuses it
naming the parameter and the object; with nothing delivered yet (the editor's draw-time check)
there is no object to collide with, so the check bites when the graph is applied.

## Refusals

Every refusal is a `ContractError` and names the graph first. They are checked in this order,
so the first one a reader sees is the one to act on:

| you did | it says |
| --- | --- |
| declared a kind outside the vocabulary | `classification source: output 'x' declares the type 'Widget', which is neither a graph slot type (Stream, Source, Sink, Op, Record, List[str], str, int, float, bool) nor a registered item type` |
| named an output like a parameter of a delivered object | `classification source: output 'class_names' shares its name with a parameter of Stream (delivered as 'stream'); confluid would push the slot's value into that parameter — rename the slot (the class vocabulary slot is 'classes')` |
| the same, for any other parameter | `files: output 'source' shares its name with a parameter of Stream (delivered as 'stream'); confluid would push the slot's value into that parameter — rename the slot` |
| left an output unwired | `classification source: 'classes' is not delivered — wire it in the graph (this graph delivers: stream, classes)` |
| wired a Stream nothing feeds | `classification source: the Stream delivered as 'stream' has no source — connect a Source to it` |
| wired a source where a Stream is expected | `classification source: 'stream' is a list, not a Stream — put a Stream between them` |
| typed an empty list of names | `classification source: 'classes' resolved to [] — wire a class_names output (a HuggingFace source has one) or a Value list into it` |
| typed something that is not a list | `classification source: 'classes' is a str, not a list of names` |
| wired a producer that declares no class names | `classification source: 'classes' is delivered by a FilesSource, which declares no class names — wire a source that does (a HuggingFace source has one) or type the names as a list` |
| asked a sink graph for its stream | `sink: this graph delivers no stream (it delivers: sink)` |
| declared a slot both as an output and as optional | `g: 'classes' is declared both as an output and as an optional slot — give it one place` |
| declared a record entry with an output's name | `g: 'stream' is declared both as an output and as a record entry — they are wired into the same place, so give them different names` |
| wired a record entry to something that is not a key | `classification source: the record entry 'input' is wired to 3 — it names the entry it is read from, a word like 'image'` |

One more is raised where the stream is READ, not where the graph is applied — the `records`
contract runs as the delivered stream's last op, so a record missing a declared entry is refused
the moment it is produced:

```
classification source: record #0 has no entry 'input' (expected Image); present: note[str]
```

and, for an entry wired to a key the records lack:

```
classification source: record #0 has no entry 'image' (wired into 'input', expected Image); present: picture[Image], class[Label]
```

## See also

- [graph.md](graph.md) — `flow:` documents and the `FlowGraph` engine the graph itself is written in
- [sources.md](sources.md) — the sources a graph wires, including which declare their own `class_names`
- [architecture.md](architecture.md) — the rationale behind the contract mechanisms
