# Algorithms — declare the settings, inputs and outputs; the op is derived

An **algorithm** states three things and computes:

- its **params** — the settings it is tuned by;
- its **inputs** — the values it reads;
- its **outputs** — the values it computes.

It never sees a record. Run as a recordstream op, it reads its inputs from the record entries with
the same names and writes its outputs back the same way. The base class `Algorithm` does that once,
for every algorithm. The same three declarations also give you the constructor, the settings schema
a visual editor or tool generator reads, the output sockets, and what the chain checker needs.

Why the mechanism exists and which alternatives were rejected: [architecture.md §21](architecture.md#21-an-algorithm-declares-its-params-inputs-and-outputs-the-op-is-derived-algorithm-2026-09-27).

## Writing one

```python
from typing import Any, Dict

import numpy as np
from annotated_types import Interval
from confluid import configurable
from typing_extensions import Annotated

from recordstream import Algorithm, Image, Input, Output, Param

Percent = Annotated[float, Interval(ge=0.0, le=100.0)]


@configurable(category="op", group="image")
class BackgroundLevel(Algorithm):
    """Estimate an image's background level: a low percentile of its per-row medians."""

    percentile: Percent = Param(default=25.0, doc="Percentile rank in [0, 100] across the per-row medians.")
    image: Image = Input(doc="The image to read.")
    background: float = Output(doc="The background level, in the image's own units.")

    def compute(self) -> Dict[str, Any]:
        per_row = np.median(np.asarray(self.image, dtype=np.float64), axis=1)
        return {"background": float(np.percentile(per_row, self.percentile))}
```

| Slot | Spelling | What it is |
|---|---|---|
| param | `name: type = Param(default=..., doc=...)` | A keyword constructor argument: `BackgroundLevel(percentile=30.0)`. Leave out `default` to make it required. |
| input | `name: type = Input(doc=...)` | Read from the record entry `name`. `Input(default=None)` makes it optional. |
| output | `name: type = Output(doc=...)` | Written to the record entry `name`. `Output(replaces="an_input")` writes it back where that input was read. |

`compute()` reads the params and inputs as `self.<name>` and **returns every output by name**.
Don't write `__init__`; it is generated from the params, plus one more argument, `keys`
([below](#which-record-entry-keys)).

## Running it without a record

```python
pixels = np.full((4, 6), 10.0)
pixels[0, :] = 90.0                       # one bright row
image = Image(pixels, layout="HWC")

level = BackgroundLevel(percentile=30.0)
level.run(image=image)                    # {'background': 10.0}
level.background                          # 10.0 — the last computed value (None before a run)
```

`run()` puts the inputs on a copy of the object, so the configured instance never holds a record's
values between calls.

## Running it as an op

An algorithm is an ordinary op: `record -> record`. It sits in an `ops:` list, a `Pipeline`, a
`flow:` step or a `Stream`, like any other op.

```python
out = level({"image": image, "exposure_ms": 20})
sorted(out)                               # ['background', 'exposure_ms', 'image']
out["background"]                         # 10.0
```

The incoming record is not changed; the op returns a new one with the outputs added.

```yaml
ops:
  - !class:mypkg.BackgroundLevel
    percentile: 30
```

## Which record entry: `keys`

By default an input is read from the entry with its own name, and an output is written to the entry
with its own name. `keys` changes that, per slot:

```python
BackgroundLevel(keys={"image": "photo", "background": "level"})({"photo": image}).keys()
# dict_keys(['photo', 'level'])
```

```yaml
- !class:mypkg.BackgroundLevel
  percentile: 30
  keys: {image: photo, background: level}
```

The same class can therefore run twice in one chain on two different entries.

## Replacing what was read: `Output(replaces=...)`

Some algorithms exist to replace the value they read: a filter over boxes, a re-tuned signal. The
class says so once, and writing back becomes the default:

```python
@configurable(category="op", group="boxes")
class KeepBoxesOnMask(Algorithm):
    """Drop the boxes whose centre pixel is OFF in a mask."""

    mask: Mask = Input(doc="The mask to test against.")
    boxes: Boxes = Input(doc="The boxes to filter.")
    kept: Boxes = Output(replaces="boxes", doc="The boxes whose centre is ON.")

    def compute(self) -> Dict[str, Any]:
        keep = [b for b in self.boxes.boxes if self.mask[(b[1] + b[3]) // 2, (b[0] + b[2]) // 2]]
        return {"kept": Boxes(boxes=keep, canvas=self.boxes.canvas)}
```

It follows `keys`: when the boxes are read from another entry, the result goes back into that entry.

```python
mask = Mask(np.zeros((10, 10), bool)); mask[:5, :5] = True
record = {"mask": mask, "detections": Boxes(boxes=[[0, 0, 4, 4], [6, 6, 9, 9]], canvas=(10, 10))}
out = KeepBoxesOnMask(keys={"boxes": "detections"})(record)
sorted(out)                               # ['detections', 'mask'] — no second box entry
out["detections"].boxes                   # [[0, 0, 4, 4]]
```

`keys` on the output itself (`keys: {kept: filtered}`) still wins, for when you want both.
`replaces` must name an input; anything else is refused when the class is defined.

## Inputs are found by name, never by type

An input is read from exactly one entry: its own name, or the one `keys` names. It is **not** looked
up by type when that entry is missing, even if the record holds exactly one value of the right type.
The reason is that the chain checker ([below](#what-tools-read)) must be able to say, before a run,
which entries a pipeline needs. A type lookup would decide that only while the pipeline runs.

A pipeline whose records call the entry something else says so once. Either set `keys` on the op, or
put a `RenameField` at the start of the chain:

```yaml
ops:
  - !class:recordstream.ops.structure.RenameField {src: input, dst: image}
  - !class:mypkg.BackgroundLevel {}
```

## When the record is wrong

Every message names the input, the entry and the fix:

| Case | Message |
|---|---|
| The entry is stored under another name | `BackgroundLevel needs the input 'image' (Image) from the record entry 'image', which this record does not carry (it has: photo) — if it is stored under another name, set keys: {image: <entry>}` |
| The entry has the wrong item type | `BackgroundLevel: the input 'image' must be of type Image, but the record entry 'image' is of type ndarray` |
| A typo in `keys` | `BackgroundLevel: keys names ['imgae'], which are not inputs or outputs; those are ['background', 'image']` |
| `compute()` forgets an output | `BackgroundLevel.compute() returned []; it must return exactly ['background']` |

Only inputs declared as a registered item type (`Image`, `Mask`, `Boxes`, `Label`, a domain item)
are type-checked. A `float` input accepts a numpy `float32` as it is.

## In a flow graph

An output is also readable off the step after it ran, so a later step can bind it with the existing
`bind:` grammar ([graph.md](graph.md)):

```yaml
flow:
  level: !class:mypkg.BackgroundLevel {percentile: 30}
  bright:
    op: !class:recordstream.ops.numpy.Threshold {field: image, output: bright}
    bind:
      low_level: level.background     # the value level computed for THIS record
```

With the image above, `bright` is a `Mask` that is on for the bright row only:
`[[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]`.

## What tools read

Nothing below is written by the author; it is derived from the three declarations.

**The chain checker** (`check_chain`, [architecture.md §16](architecture.md#16-an-op-declares-its-interface-and-a-chain-is-checked-before-it-runs-check_chain-2026-09-05))
reads `consumes` and `produces`, which follow `keys` and `replaces`:

```python
BackgroundLevel().consumes                                 # {'image': 'Image'}
BackgroundLevel().produces                                 # {'background': '*'}
KeepBoxesOnMask(keys={"boxes": "detections"}).produces     # {'detections': 'Boxes'}

check_chain([BackgroundLevel()], provided=["photo"], where="pipeline.yaml")
# ChainContractError: pipeline.yaml: BackgroundLevel needs the record entry 'image', which nothing
# before it produces — the chain has photo at that point (...)
```

Optional inputs are not in `consumes`. An output that is not an item type is `'*'`.

**The settings schema** (`confluid.to_pydantic`, what a form, a visual editor or a generated tool
schema reads) lists the params and `keys`, with their range marks and docs:

```python
{"percentile": {"default": 25.0, "minimum": 0.0, "maximum": 100.0}, "keys": {"default": None}}
```

**The output sockets** (`confluid.output_specs`):

```python
[{'name': 'background', 'type': 'float', 'description': "The background level, in the image's own units."}]
```

**The whole interface** (`algorithm_spec`), for a tool that wants all three kinds of slot:

```python
from recordstream import algorithm_spec

spec = algorithm_spec(BackgroundLevel)
[(s.name, s.default, s.doc) for s in spec.params]
# [('percentile', 25.0, 'Percentile rank in [0, 100] across the per-row medians.')]
[(s.name, s.annotation.__name__, s.required, s.doc) for s in spec.inputs]
# [('image', 'Image', True, 'The image to read.')]
[(s.name, s.annotation.__name__, s.doc) for s in spec.outputs]
# [('background', 'float', "The background level, in the image's own units.")]
```

**Saving the configuration** (`confluid.dump`) writes the settings only:

```yaml
_target_: BackgroundLevel
percentile: 30.0
keys:
  image: photo
```

> confluid's `input_specs()` means something else by "input": the **constructor** arguments. For an
> algorithm it lists `percentile` and `keys`, like for any other class. The record inputs are read
> with `algorithm_spec`.

**The type checker** sees the generated constructor. mypy reports `BackgroundLevel(percentil=30.0)`
(a typo), `BackgroundLevel(percentile="high")` (a wrong type) and `BackgroundLevel(image=...)` (an input
is not a setting), and accepts `BackgroundLevel()` and `BackgroundLevel(percentile=30.0, keys={...})`.

## Rules

- **Don't write `__init__`.** It is generated from the params, and defining one is refused.
- **`compute()` returns every output, and nothing else.** A missing or extra name is refused.
- **Outputs are read-only.** `self.background` after a run is the last computed value; assigning to it
  fails. That is deliberate: confluid treats a settable property as a setting.
- **A slot may not reuse a name the base class uses:** `run`, `compute`, `consumes`, `produces`, `keys`.
- **A subclass inherits the slots** and may add more; its constructor gains the new params.
- **A list or dict default is copied per instance**, so two instances never share it.

## When to use something else

- An op that applies to **every value of a type** in the record ("brighten every `Image`") is a
  `Transform` with kernels ([record-model.md](record-model.md#ops--type-dispatch-with-once-per-record-parameters)).
- A **library transform** (albumentations, torchvision `transforms.v2`) runs as it is.
- An op that **drops records, expands one record into many, or composes other ops** is a plain callable
  op or a composer (`Pipeline`, `Enable`, `RandomApply`).

Everything that reads named entries and writes named entries is an algorithm.
