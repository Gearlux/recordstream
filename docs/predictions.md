# The model boundary: datasets in, predictions out

RecordStream owns the data on both sides of a model: the dataset a trainer consumes, and the
prediction it emits. This page covers the three surfaces that sit on that boundary.

## `ensure_record_dataset` — normalize a wired dataset slot

A config can wire a dataset slot to a `Stream`, another torch `Dataset`, a bare source, or a plain
list of records. Normalize once, and the rest of the pipeline (target detection, label encoding,
collate, metrics) can assume record items unconditionally:

```python
from recordstream import RecordSource, ensure_record_dataset

dataset = ensure_record_dataset(self.train_set)   # -> a map-style Dataset of records
```

A `Stream` comes back **as-is** — identity matters, because a label-encoding Stream carries its
`class_names` and re-wrapping would lose it. Anything else is wrapped in a `Stream`, which makes it
both map-style and record-yielding.

`RecordSource` is the contract it enforces (`Dataset | Iterable[Record]`), named once so a consumer
annotates `Optional[Lazy[RecordSource]]` instead of inventing its own union.

## Prediction-output contracts (`recordstream.outputs`)

What a model's eval-mode `forward` returns, declared as typed dicts so metrics, sinks and
visualizers read known keys instead of guessing whether a tensor is logits, probabilities, or
argmax'd class ids:

| Contract | Keys |
|---|---|
| `ClassificationOutput` | `logits` `[B, C]`, `probs` `[B, C]`, `class_idx` `[B]` |
| `DetectionOutput` | `boxes` `[N, 4]` xyxy absolute pixels, `scores` `[N]`, `labels` `[N]` |
| `SegmentationOutput` | `logits` `[B, C, H, W]`, `probs` `[B, C, H, W]`, `mask` `[B, H, W]` |

That guess is not hypothetical: two independently-written detector wrappers agree that `boxes` is
xyxy in absolute pixels only because `DetectionOutput` says so.

Each contract is **generic in the array type**, so the same declaration describes a torch run and a
numpy/TF/JAX one:

```python
from recordstream import ClassificationOutput

def predict(self, x) -> ClassificationOutput[np.ndarray]: ...   # a non-torch backend
```

A bare `ClassificationOutput` means "whatever array type". The **builders** are the only torch part
— `softmax` and `argmax` are library calls, not type declarations:

```python
from recordstream import classification_output

def predict_step(self, batch, batch_idx):
    return classification_output(self(x))     # logits -> the full contract
```

Detection deliberately has no builder: its boxes come from the detector's own interface, so the
dict is built inline at the call site rather than invented from logits.

## Predictions sinks (`recordstream.predictions`)

A predict/test loop calls `predictions_sink.write(prediction, metadata)` once per record and
`close()` at the end. `PredictionsSink` is that contract as a `@runtime_checkable` Protocol —
annotate the slot with it rather than `Any`, so a use site can't call `.write` on something that is
still a deferred marker.

`ClassificationPredictionsSink` is the classification implementation: read `probs` / `class_idx`,
resolve the class id to a label, build a top-k list, and thread a record through your ops.

```yaml
predictions_sink: !class:recordstream.predictions.ClassificationPredictionsSink
  class_names: !ref:class_id_to_label      # {0: "bird", 1: "cat", ...}
  top_k: 5
  confidence_threshold: 0.0                 # skip predictions below this top-1 probability
  ops:
    - !class:recordstream.ops.sink.RecordSinkOp
      sink: !class:mypkg.JsonSink() { path: ./predictions }
```

Each written record carries the original metadata plus `predicted_class_id`,
`predicted_class_label`, `predicted_confidence` and `predicted_top_k`, under a single `"metadata"`
key. Diagnostics identify a record by its **ordinal** in the run — the sink is modality-neutral, so
it never assumes a metadata key exists.

Like every configurable here it is **zero-arg constructible**: `ops` is required to *run*, not to
*build*, so the non-empty check fires on the first `write` rather than in `__init__`.

### Why this is a second sink protocol

`recordstream.storage.base.DataSink` takes a whole record (`write(record)`) and is what
`RecordSinkOp` adapts into an op chain. A predictions sink instead receives the model's output plus
the metadata of the record it came from, and builds the record itself — the two halves arrive
separately because a model emits a batch while the sink contract is per-record. Keep the two
distinct; collapsing them is tracked as a deliberate decision, not something to do by drift.
