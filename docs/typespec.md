# Type specs — `ACCEPTS` / `PRODUCES` (`sampleflux.typespec`)

`sampleflux.typespec` describes *what flows through a `Sample`* and lets ops declare what they accept/produce, so tools (connection validators, visual config editors, schema generators) can filter which ops may connect. It is flexible by design — N-dimensional arrays across numpy/torch/tensorflow, **per-axis bounded ranges**, dtype families, images, and arbitrary Python types — and anything left unspecified defaults to `Any`.

```python
from sampleflux.typespec import SampleType, ArrayType, Dim, PythonType, UnionType

# "a 2-D float array whose first axis is 1–10, second axis any size"
ArrayType(shape=(Dim.range(1, 10), Dim.any("N")), dtype="floating")
ArrayType.parse("3 h w", dtype="float32", framework="torch")  # jaxtyping-style shorthand
ArrayType.image("CHW", channels=3, dtype="float32", framework="torch")  # an image convenience
```

`dtype`, `framework`/`frameworks`, and the image `layout` are **closed `Literal`s**, not bare strings — a typo is a type error and a UI / connection-validator enumerates the choices via `typing.get_args(...)`:

- `Dtype` — concrete names (`"float32"`, `"int64"`, …); `DtypeFamily` — relaxed families (`"floating"`, `"numeric"`, …); `DtypeSpec = Dtype | DtypeFamily` is the `dtype` field type.
- `Framework = Literal["numpy", "torch", "tensorflow"]`, `ImageLayout = Literal["CHW", "HWC"]`.

Authored dtypes must be canonical names; aliases / casing (`"double"`, `"FLOAT32"`) and exotic platform dtypes (`float128`) are runtime-only conveniences normalized by `canonical_dtype` — the single boundary where arbitrary input crosses into the typed domain.

## Declaring an op's contract

Use the class attributes `ACCEPTS` / `PRODUCES` (each a `SampleType`; both default to `Any`, so annotating is optional and backward-compatible). No base class — transforms stay plain callables:

```python
@configurable
class StandardizeOp:
    ACCEPTS = SampleType(input=UnionType((ArrayType(dtype="numeric"), PythonType("PIL.Image.Image"))))
    PRODUCES = SampleType(input=ArrayType(dtype="floating", frameworks={"numpy"}))
    def __call__(self, sample): ...
```

Matching is asymmetric: `consumer.accepts(producer)` is strict (used at runtime against a concrete inferred type); `compatible(consumer, producer)` is permissive (used at edit time — `Any`/unknown on either side passes).

## A Sample's own type

A `Sample`'s type comes from `sample.describe()` — it returns a type stored in the reserved metadata keys `__features__` (a `datasets.Features` dict) + `__spec__` (sidecar refinements), or infers one from the live data; attach a stored type with `sample.with_type(SampleType(...))`. Default pipelines stamp nothing, so metadata stays byte-identical and serialization is untouched.
