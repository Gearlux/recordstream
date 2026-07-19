# Field projection, class counting & label maps (`sampleflux.projection` / `sampleflux.labels`)

## Field projection

Walking a source for a single field (the classic case: counting classes from *targets*) shouldn't pay to build the fields you don't need. `sampleflux.projection` adds an opt-in protocol plus lazy helpers:

```python
from sampleflux import project, iter_targets, num_classes
from sampleflux import ProjectionField  # Literal["input", "target", "metadata"]

# A source MAY implement SupportsProjection (`project(fields)`) to skip building
# unrequested fields — e.g. an image dataset reads only the label column for a
# target-only walk, never decoding an image.
for sample in project(my_source, ("target",)):
    ...                       # sample.input is None; sample.target populated

labels = list(iter_targets(my_source))     # lazy
n = num_classes(my_source)                 # max(class_id) + 1 — always walks
```

The field set is a **closed `Literal`**, `ProjectionField`, not a bare `str` — so a typo is a type error, and a UI / form-spec / MCP schema enumerates the choices straight from the annotation instead of hard-coding a parallel list:

```python
from typing import get_args
get_args(ProjectionField)        # ('input', 'target', 'metadata')
```

Sources that don't implement `SupportsProjection` still work via a correct full-iteration fallback (just without the skip-decode speedup). `num_classes` is a free function, not a `Flux` method: integer class-id semantics are classification-specific, so the task-agnostic engine doesn't advertise it.

## `LabelMap` — fittable name↔id encoding

When a dataset's `target` is a class **name** rather than an integer id, `LabelMap` turns it into the pinned encoding the `EncodeTargetOp` / `DecodeTargetOp` need — the *fittable* companion to those ops. Fit it once (sklearn `LabelEncoder`, deterministic sorted ordering), persist it in the `class_names.json` format, and reload it at eval/predict so every stage shares one ordering:

```python
from sampleflux import LabelMap, Flux

lm = LabelMap.fit(iter_targets(train_source))   # {"bird": 0, "cat": 1, "dog": 2}
lm.num_classes        # 3
lm.label_names        # ["bird", "cat", "dog"]  (id -> name)
lm.save("class_names.json")                     # marainer's class_names.json format

encoded = Flux(source=train_source, ops=[lm.encode_op()])   # targets are now ints

# Later, at eval time — reload the SAME ordering instead of refitting:
lm2 = LabelMap.load("class_names.json")
```

`LabelMap.fit` is the *only* place a mapping is derived from data; everywhere downstream the mapping is pinned, so train / eval / predict never disagree. `scikit-learn` backs `fit` (lazy-imported).
