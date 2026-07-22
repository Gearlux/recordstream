# The transform taxonomy, multi-type carriers & expanding ops (`sampleflux.kinds`)

## What an op processes, how it's called

A **sample** is the triple `(input, target, metadata)`; the classic AI tuple is the **pair** `(input, target)`. A transform declares — via its `__call__` signature alone — exactly which *slice* of the triple it processes, and the engine binds that view and merges the result back (untouched fields preserved):

| scope | without metadata | with metadata |
|---|---|---|
| input only | `input` — the bare value | `input_meta` — `InputMeta(input, metadata)` |
| target only | `target` — the bare value | `target_meta` — `TargetMeta(target, metadata)` |
| both | `pair` — `(input, target)` / `Pair` | `sample` — the full `Sample` |
| metadata only | — | `metadata` — the bare dict (`m: dict` / `MetaDict`) |

Each scope works in **two calling styles** — packed (one argument) or unpacked (the fields as separate arguments) — and unpacked arguments COMBINE freely: each parameter binds its own view (annotation first, then the name, then the classic `f(input, target, metadata)` positional defaults):

```python
class A:  # bare input value — any array/tensor/dict; target+metadata pass through
    def __call__(self, x: Input): return x / 255.0            # Annotated[T, INPUT] keeps a real T

class B:  # the classic AI signature, unpacked
    def __call__(self, input, target): return aug(input), target

class C:  # input with its metadata, unpacked (2nd arg named `metadata`/`meta`)
    def __call__(self, input, metadata): return crop(input, metadata["roi"]), metadata

class D:  # packed named view
    def __call__(self, v: TargetMeta) -> TargetMeta: return TargetMeta(encode(v.target), v.metadata)

class E:  # the full triple, unpacked
    def __call__(self, input, target, metadata): return input, target, {**metadata, "seen": True}

class F:  # today's classic — completely unchanged
    def __call__(self, sample: Sample) -> Sample: ...

class G:  # COMBINED views: input WITH its metadata + target WITH its metadata
    def __call__(self, im: InputMeta, tm: TargetMeta):
        return InputMeta(aug(im.input), im.metadata), TargetMeta(remap(tm.target), tm.metadata)

class H:  # metadata-only transform
    def __call__(self, m: dict) -> dict: return {**m, "canonical": True}
```

Detection rules: arity counts **required** parameters (optional extras don't change anything); 3 args → unpacked `sample`; 2 args → `input_meta`/`target_meta` when the 2nd is named `metadata`/`meta` (or annotated `dict`), first-arg name `target` selects the target side, else the `pair`; 1 arg → the annotation (`Sample`, `tuple`/`Pair`, `InputMeta`/`TargetMeta`, `Input`/`Target` marks; untyped = **any** — exactly today's behavior). `op_contract(op)` exposes the result — `OpContract(accepts, produces, expands, style, bindings)`, where `bindings` lists each unpacked parameter's scope in order (e.g. `("input_meta", "target_meta")`) and `accepts` is the grid summary of the covered fields — the vocabulary a visual editor can surface as socket types. Escape hatches: `SAMPLE_KIND_IN`/`SAMPLE_KIND_OUT`/`CALL_STYLE`/`EXPANDS` class attrs.

Merge-back: `None` drops the sample; a returned `Sample` takes over; otherwise only the declared fields update (a `pair` op keeps metadata; an `input` op keeps target+metadata; the meta variants receive the *actual* metadata dict, so in-place mutation propagates). The views are real NamedTuples (`Pair`/`InputMeta`/`TargetMeta`), recognized by `Sample.from_any`/`classify_carrier` *before* the generic tuple rule, flow natively under `Flux(native=True)`, and have default collates.

## Multi-type carriers & the collate registry (`sampleflux.collate`)

Pipelines can carry more than `Sample` triplets: **`Flux(native=True)`** (opt-in) keeps each carrier's own kind — a metadata-free **pair** (`(image, label)`, `(tensor, mask)`, `(tensor, coco_dict)`) or a bare **value** — and adapts every op via its introspected contract:

```python
from confluid import configurable
from sampleflux import Flux, Sample, op_contract

@configurable
class NormalizePair:                          # a pair-native op — no metadata anywhere
    def __call__(self, pair: tuple) -> tuple:
        img, label = pair
        return img / 255.0, label

@configurable
class StampOp:                                # a classic Sample op — unchanged
    def __call__(self, sample: Sample) -> Sample: ...

flux = Flux(source=[(img_a, 3), (img_b, 7)], ops=[NormalizePair(), StampOp()], native=True)
# NormalizePair receives the raw pair; StampOp receives a PROMOTED Sample view
# (promotion is one-way and sticky, so op-written metadata is never dropped).

op_contract(NormalizePair())   # OpContract(accepts='pair', produces='pair', expands=False)
```

Detection reads the `__call__` annotations (`Sample` → sample-op, `tuple[...]` → pair-op, untyped → works-on-anything — **untyped ops behave exactly as today**); the class attrs `SAMPLE_KIND_IN` / `SAMPLE_KIND_OUT` / `EXPANDS` override detection where introspection can't see. `native=False` (the default) coerces everything to `Sample` exactly as before — no consumer changes.

**Collation** is a pluggable registry keyed by representation:

```python
from sampleflux import collate, get_collate, register_collate

batch = collate(list(flux))                   # dispatches on the detected kind
@register_collate("yolo")                     # task aliases are additive
def yolo_collate(items): ...
loader = DataLoader(flux, collate_fn=get_collate("yolo"))
```

Defaults: `"sample"` (stacked input/target + list-form batched metadata — the `is_batched` convention), `"pair"` (`(stacked_inputs, stacked_targets)`), `"value"`, and the view forms `"input_meta"`/`"target_meta"`. Consumer collates (classification/segmentation/detection) register additively and keep their own conventions. The string keys primarily target the MCP tool surface (JSON-serializable, enumerable collate selection) — in Python, passing the function directly stays the normal path; the full rationale is recorded in [architecture.md](architecture.md#batching-is-two-stage-collation-is-a-pluggable-registry-samplefluxcollate-2026-07-17).

## 1→N expanding ops (iterable-only pipelines)

An op may return **several** carriers — a windowing op splitting one capture into N windows is just a generator-returning op:

```python
from typing import Iterator

@configurable
class SlidingWindowOp:
    def __call__(self, sample: Sample) -> Iterator[Sample]:
        for w in sliding_windows(sample.input, self.size, self.stride):
            yield sample._replace(input=w)
```

Expansion is detected from the return annotation (`Iterator[...]` / `Iterable[...]` / `List[...]`; or the explicit `EXPANDS = True` marker) and flattened in every iteration route — sequential, spawn-parallel, and streamed — depth-first, so sibling order matches the nested-loop intuition. Each child continues through the remaining ops with its own (shallow-copied) Context; a child filtered to `None` just drops.

A pipeline containing an expanding op is **ITERABLE-ONLY**: `len(flux)` / `flux[i]` raise a clear `TypeError` (the expanded length is unknowable up front). Iterate it, wrap it in a torch `IterableDataset`, window at the source for random access, or materialize with `list(flux)`. `FlowGraph` steps are strictly 1→1 (a named step has one result) — expanding pipelines belong to the `Flux` engine.
