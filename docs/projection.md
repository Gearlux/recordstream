# Key projection, class counting & label maps (`recordstream.projection` / `recordstream.labels`)

## Key projection

Walking a source for a single record key (the classic case: counting classes from the label key) shouldn't pay to build the values you don't need. `recordstream.projection` adds an opt-in protocol plus lazy helpers, all **key-addressed** — any subset of record keys:

```python
from recordstream import project, iter_key, num_classes

# A source MAY implement SupportsProjection (`project(keys)`) to skip building
# unrequested values — e.g. an image dataset reads only the label column for a
# class-count walk, never decoding an image.
for record in project(my_source, ("class",)):
    ...                       # partial records carrying only the "class" entry

labels = list(iter_key(my_source, "class"))   # lazy; a Label unwraps to .value, a MultiLabel to
                                              # its .values LIST, other items to their payload,
                                              # plain values verbatim
n = num_classes(my_source, key="class")       # max(class_id) + 1 — always walks
```

A **deferred** source — a `!class:` marker straight out of a config — is materialized first, so a caller never has to know which entry point flows and which doesn't (flowing a live object is a no-op). Sources that don't implement `SupportsProjection` still work via a correct full-iteration fallback (just without the skip-decode speedup); `Stream.project(keys)` is the engine's implementation — it runs the op chain, then keeps only the requested keys. `num_classes` is a free function, not a `Stream` method: integer class-id semantics are classification-specific, so the task-agnostic engine doesn't advertise it.

## `LabelMap` — fittable name↔id encoding

When a dataset's label is a class **name** rather than an integer id, `LabelMap` turns it into the pinned encoding the `EncodeTarget` / `DecodeTarget` ops need — the *fittable* companion to those ops. Fit it once (deterministic `sorted(set(...))` ordering), persist it in the `class_names.json` format, and reload it at eval/predict so every stage shares one ordering:

```python
from recordstream import LabelMap, Stream, iter_key

lm = LabelMap.fit(iter_key(train_source, "class"))   # {"bird": 0, "cat": 1, "dog": 2}
lm.num_classes        # 3
lm.class_names        # ["bird", "cat", "dog"]  (id -> name)
lm.save("class_names.json")                     # {"class_names": [...], "num_classes": N}

encoded = lm.encode(train_source)             # a Stream whose "class" Labels carry int ids
# (the long form, when you need to pin the field or tolerate unknowns:
#  Stream(source=train_source, ops=[lm.encode_op(ignore_unknown=True)]))

# Later, at eval time — reload the SAME ordering instead of refitting:
lm2 = LabelMap.load("class_names.json")
```

`LabelMap.fit` is the *only* place a mapping is derived from data; everywhere downstream the mapping is pinned, so train / eval / predict never disagree. `fit` is pure stdlib — no ML library is pulled in to sort a set of names.

### Multi-label targets and `to_ids`

`fit` accepts a `MultiLabel` (or any sequence) just as readily as a single `Label`, so a
multi-label dataset builds its vocabulary from the same call — every distinct member becomes
one class:

```python
from recordstream import LabelMap, MultiLabel

lm = LabelMap.fit([MultiLabel(["cat", "dog"]), MultiLabel(["bird"])])
lm.class_names        # ["bird", "cat", "dog"]
```

`to_ids(target)` is the one accessor a consumer needs — it always returns a **list of int class
ids**, whatever it is handed, so there is no name-vs-id and no single-vs-multi branch at the call
site:

```python
lm.to_ids(Label("cat"))              # [1]      a name  -> its id
lm.to_ids(Label(1))                  # [1]      an ALREADY-encoded id passes through
lm.to_ids(MultiLabel(["cat", "dog"]))  # [1, 2]
lm.to_ids("dog")                     # [2]      a bare value works too
```

Because encoded ids pass through untouched, `LabelMap().to_ids(...)` (an *empty* map) is a valid
way to normalize an already-encoded dataset to id lists — useful for counting classes or
class-frequency statistics without fitting anything.

## Class-balance weights (`class_counts` / `inverse_frequency_weights`)

Training on a skewed label distribution biases a model toward the majority class. The remedy is
per-class weights, and *deriving* them is a statistic over the labels — so it lives here, beside
the `LabelMap` that normalizes the targets:

```python
from recordstream import LabelMap, inverse_frequency_weights, iter_key

targets = list(iter_key(train_source, "class"))     # walk ONCE — see the note below
weights = inverse_frequency_weights(targets, num_classes=3, label_map=lm)
# array([0.667, 2.0, 1.0], dtype=float32)   rare classes weigh more
```

`w[c] = total / (num_classes * count[c])`, so a class at exactly the mean frequency gets `1.0`. A
class observed **zero** times gets `0.0` (never infinity), an id outside `[0, num_classes)` is
ignored rather than raising, and the whole call returns `None` when nothing was counted — so "no
weights" stays distinguishable from "all-zero weights". `class_counts(...)` exposes the raw
histogram if you want it.

Three things the signature is deliberate about:

- **It takes already-walked targets, not a source.** A trainer typically walks the target stream
  once and reuses that single pass to fit the `LabelMap`, derive `num_classes`, *and* weigh the
  classes. A convenience that walked internally would quietly double the passes over the dataset.
- **Every target shape works**, because `LabelMap.to_ids` normalizes it: a `Label`, a `MultiLabel`
  (counting for every class it names), a bare id with no map at all, a name with a fitted map.
- **It returns numpy**, like everything in `recordstream.batch` except `batch_tensor`. What a
  framework does with the vector is its own convention — `torch.nn` takes a `weight` tensor in the
  loss constructor, Keras takes `class_weight` on `fit()` — so that last step belongs to the
  consuming trainer, not here.
