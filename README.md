# DataFlux

**DataFlux** is a high-performance, functional data processing engine built for modern Machine Learning pipelines. It provides a clean, fluent API for streaming and transforming data from any source while maintaining strict compatibility with PyTorch and Hugging Face.

Part of the **Modular Quartet**: `LogFlow`, `Confluid`, `Liquify`, and `DataFlux`.

## 🚀 Key Features

-   **Functional Purity:** Transforms are simple Python callables. No complex base classes required.
-   **Standardized Sample Triplet:** Standardizes on `(input, target, metadata)` for full traceability.
-   **High Performance:** Native multiprocess support via `.parallel(workers=N)` using the safe `spawn` context.
-   **Advanced Storage:** Built-in support for high-performance backends:
    -   **HDF5**: Clean, efficient read/write.
    -   **Zarr**: Cloud-native, concurrent storage (Group and Batch modes).
    -   **Directory**: Robust concurrent writing for irregular data lengths.
-   **Passive Introspection:** Automatically generates JSON manifests for visual orchestration in **FluxStudio**.
-   **100% Reproducibility:** Entire pipelines are serializable via **Confluid** manifests.

## 🎯 Design Goals & Requirements

### Stream Engine
- **Functional API:** Provide a lazy, chainable pipeline API (`map`, `filter`, `batch`).
- **Standardized Samples:** Use the `Sample(input, target, metadata)` triplet as the primary data unit.
- **Parallel Execution:** Support high-performance multiprocess execution via `.parallel(workers=N)` using the `spawn` context.

### Storage
- **High-Performance Sinks:** Native support for HDF5 (sequential), Zarr (concurrent), and Directory (irregular) storage.
- **JointFlux Pattern:** Support aggregating multiple heterogeneous data sources into a single stream, preserving per-source transform chains.

### Metadata & Discovery
- **Passive Introspection:** Automatically discover available tools and ops for serialized manifests.
- **Discovery Categories:** `@configurable` classes are tagged with a confluid `category` (sources `HuggingFaceSource`/`DatasetSplit` → `source`, engines `Flux`/`JointFlux` → `engine`, concrete `Sample→Sample` ops → `op`, storage **sinks** `HDF5Sink`/`ZarrGroupSink`/`ZarrBatchSink`/`DirectorySink` → `sink` (FluxStudio surfaces them as `DatasetProcessor` sink nodes; their read-back **sources** stay UNcategorised); `FilterOp`/`WrappedOp` are deliberately UNcategorised) so tools like navigaitor's `list_configurable_classes(category=...)` enumerate them by kind.
- **Serialization Symmetry:** Ensure full-pipeline states are serializable and reconstructible via Confluid.

## 🛠 Quick Start

```python
import numpy as np
from dataflux.core import Flux

# 1. Define a simple transformation
def normalize(data: np.ndarray, mean: float = 0.0):
    return data - mean

# 2. Build a pipeline
raw_data = [np.random.randn(10) for _ in range(100)]

flux = Flux(raw_data) \
    .map(normalize, mean=0.5) \
    .filter(lambda s: s.input.mean() > 0) \
    .parallel(workers=4)

# 3. Collect or stream
for sample in flux:
    print(sample.input.shape)
```

## 🏷 Type Specs

`dataflux.typespec` describes *what flows through a `Sample`* and lets ops declare what they accept/produce, so tools like FluxStudio can filter which nodes may connect. It is flexible by design — N-dimensional arrays across numpy/torch/tensorflow, **per-axis bounded ranges**, dtype families, images, and arbitrary Python types — and anything left unspecified defaults to `Any`.

```python
from dataflux.typespec import SampleType, ArrayType, Dim, PythonType, UnionType

# "a 2-D float array whose first axis is 1–10, second axis any size"
ArrayType(shape=(Dim.range(1, 10), Dim.any("N")), dtype="floating")
ArrayType.parse("3 h w", dtype="float32", framework="torch")  # jaxtyping-style shorthand
ArrayType.image("CHW", channels=3, dtype="float32", framework="torch")  # an image convenience
```

`dtype`, `framework`/`frameworks`, and the image `layout` are **closed `Literal`s**, not bare strings — a typo is a type error and a UI / connection-validator enumerates the choices via `typing.get_args(...)`:

- `Dtype` — concrete names (`"float32"`, `"int64"`, …); `DtypeFamily` — relaxed families (`"floating"`, `"numeric"`, …); `DtypeSpec = Dtype | DtypeFamily` is the `dtype` field type.
- `Framework = Literal["numpy", "torch", "tensorflow"]`, `ImageLayout = Literal["CHW", "HWC"]`.

Authored dtypes must be canonical names; aliases / casing (`"double"`, `"FLOAT32"`) and exotic platform dtypes (`float128`) are runtime-only conveniences normalized by `canonical_dtype` — the single boundary where arbitrary input crosses into the typed domain.

**Declare an op's contract** with the class attributes `ACCEPTS` / `PRODUCES` (each a `SampleType`; both default to `Any`, so annotating is optional and backward-compatible). No base class — transforms stay plain callables:

```python
@configurable
class StandardizeOp:
    ACCEPTS = SampleType(input=UnionType((ArrayType(dtype="numeric"), PythonType("PIL.Image.Image"))))
    PRODUCES = SampleType(input=ArrayType(dtype="floating", frameworks={"numpy"}))
    def __call__(self, sample): ...
```

Matching is asymmetric: `consumer.accepts(producer)` is strict (used at runtime against a concrete inferred type); `compatible(consumer, producer)` is permissive (used at edit time — `Any`/unknown on either side passes). A `Sample`'s own type comes from `sample.describe()` — it returns a type stored in the reserved metadata keys `__features__` (a `datasets.Features` dict) + `__spec__` (sidecar refinements), or infers one from the live data; attach a stored type with `sample.with_type(SampleType(...))`.

## 🌀 Fourier Transform (`FourierOp` / `InverseFourierOp` / shift ops)

A small **1-D FFT toolkit**, each op in a numpy variant (`dataflux.ops.numpy`, on `np.ndarray`) and a torch variant (`dataflux.ops.torch`, on `torch.Tensor`); the flat `from dataflux.ops import …` resolves to the torch one (the package's torch-default convention, like `RescaleOp`):

- **`FourierOp`** — the 1-D discrete Fourier transform (`numpy.fft.fft` / `torch.fft.fft`).
- **`InverseFourierOp`** — its inverse (`…fft.ifft`), back to the time domain.
- **`FftShiftOp`** / **`IfftShiftOp`** — center the zero-frequency component, and undo it (`…fft.fftshift` / `ifftshift`).

`FourierOp` / `InverseFourierOp` accept **real *and* complex** signals; the raw transform is always complex (take `.real` downstream if you started real). With a unit `scaling` (see **Windowing & spectral units** below) `FourierOp` may instead emit a *real* power/density spectrum, so it declares a permissive `PRODUCES` (complex **or** floating). The shift ops are pure, dtype-preserving bin rearrangements (no FFT), so they work on any array — including an already-computed 2-D spectrogram.

```python
import numpy as np
from dataflux.sample import Sample
from dataflux.ops.numpy import FourierOp, InverseFourierOp, FftShiftOp

x = np.array([1.0, 2.0, 3.0, 4.0])                # real signal
spectrum = FourierOp()(Sample(input=x)).input      # complex128, == np.fft.fft(x)

xc = np.array([1 + 2j, 3 - 1j, 0j, -2 + 1j])       # complex signal — also supported
FourierOp(n=8, axis=-1, norm="ortho")(Sample(input=xc))  # zero-pad to 8, orthonormal scaling

# Round-trip (forward then inverse recovers the input):
recovered = InverseFourierOp()(FourierOp()(Sample(input=x))).input.real  # ≈ x

# Center the spectrum for display — two equivalent ways:
centered = FftShiftOp()(FourierOp()(Sample(input=x)))   # explicit, composable
centered = FourierOp(shift=True)(Sample(input=x))       # the one-node convenience flag
```

Parameters mirror `numpy.fft.fft` / `torch.fft.fft`: `n` (output length — zero-pad/truncate), `axis` (numpy) / `dim` (torch) — the single transform axis, default the last, so a `[B, N]` batch transforms per row — and `norm`, a closed `Literal["backward", "ortho", "forward"]` (use the **same** `norm` on the inverse to round-trip). Dtype promotion follows each framework: real `float32`/`complex64` → `complex64`, `float64`/integer/`complex128` → `complex128` (numpy) or `complex64` for integer (torch); the torch FFT/IFFT ops promote half precision (`float16`/`bfloat16`) to `float32` first because torch's FFT rejects it. Both transform ops take a `shift` flag — `FourierOp(shift=True)` applies `fftshift` **after** the transform, `InverseFourierOp(shift=True)` applies `ifftshift` **before** it — so the two invert each other exactly (the standalone `FftShiftOp`/`IfftShiftOp` are the same logic, decoupled, for centering arrays that didn't come from `FourierOp`).

### Windowing & spectral units (`WindowOp` / `SpectrumScalingOp` / `FourierOp(window=…, scaling=…)`)

A raw FFT is **uncalibrated** — to read a spectrum in real units you must taper the signal with a *window* (to control spectral leakage) and divide out the window's gain. DataFlux ships this as two composable ops plus options on `FourierOp` (numpy **and** torch variants). The window + unit math lives in **`dataflux.windows`** (pure numpy; `get_window` / `scale_spectrum` / the `WindowName` + `SpectrumScaling` Literals).

- **`WindowOp(window=…)`** — multiplies the signal by a taper and **stashes the correction** (`window_sum` `S1=Σw`, `window_sum_sq` `S2=Σw²`, `window_enbw_bins`, `window_coherent_gain`) into the metadata for a later scaling step. Windows: `boxcar` (rectangular/none), `bartlett`, `hann`, `hamming`, `blackman`, `blackmanharris`, `nuttall`, `flattop`, `kaiser`, `tukey`, `gaussian` — parametrized ones take `window_param` (Kaiser β / Tukey α / Gaussian σ); `periodic=True` (default) is the DFT-even form correct for FFT analysis.
- **`SpectrumScalingOp(scaling=…)`** — turns a spectrum into physical units, reading `S1`/`S2` from the metadata (rectangular `S1=S2=N` if no window was applied):

  | `scaling`     | output            | formula                | units    |
  |---------------|-------------------|------------------------|----------|
  | `"none"`      | complex (raw)     | `X`                    | —        |
  | `"amplitude"` | complex           | `X / S1`               | V        |
  | `"power"`     | real              | `|X|² / S1²`           | V²       |
  | `"density"`   | real              | `|X|² / (Fs·S2)`       | V²/Hz    |

  `density` uses `sample_rate` (Hz) → falls back to `metadata["samplerate"]` → `1.0` (per normalized frequency). `one_sided=True` folds a real signal's spectrum to one side (keep `0…N/2`, double the interior bins).

- **`FourierOp(window=…, scaling=…, sample_rate=…)`** folds all three into one node. The default (`window="boxcar"`, `scaling="none"`) is byte-for-byte the old behaviour. Calibrated `scaling` assumes the unscaled transform, so combining it with a non-`"backward"` `norm` raises.

```python
from dataflux.ops.numpy import FourierOp, WindowOp, SpectrumScalingOp

# one node — Hann-windowed power-spectral density in dBW/Hz-ready units:
psd = FourierOp(window="hann", scaling="density", sample_rate=122.88e6)(sample).input

# …is exactly the explicit, composable chain:
psd = SpectrumScalingOp(scaling="density", sample_rate=122.88e6)(
    FourierOp()(WindowOp(window="hann")(sample))
).input
```

A unit-amplitude tone reads `amplitude` ≈ its amplitude and `power` ≈ amplitude²; `power` and `density` differ by the window's equivalent noise bandwidth in Hz (`Fs·S2/S1²`) — the calibration that makes a windowed FFT match a reference analyzer.

## 🔎 Field Projection & Class Counting

Walking a source for a single field (the classic case: counting classes from
*targets*) shouldn't pay to build the fields you don't need. `dataflux.projection`
adds an opt-in protocol plus lazy helpers:

```python
from dataflux import project, iter_targets, num_classes
from dataflux import ProjectionField  # Literal["input", "target", "metadata"]

# A source MAY implement SupportsProjection (`project(fields)`) to skip building
# unrequested fields — e.g. an image dataset reads only the label column for a
# target-only walk, never decoding an image.
for sample in project(my_source, ("target",)):
    ...                       # sample.input is None; sample.target populated

labels = list(iter_targets(my_source))     # lazy
n = num_classes(my_source)                 # max(class_id) + 1 — always walks
```

The field set is a **closed `Literal`**, `ProjectionField`, not a bare `str` —
so a typo is a type error, and a UI / form-spec / MCP schema enumerates the
choices straight from the annotation instead of hard-coding a parallel list:

```python
from typing import get_args
get_args(ProjectionField)        # ('input', 'target', 'metadata')
```

Sources that don't implement `SupportsProjection` still work via a correct
full-iteration fallback (just without the skip-decode speedup). `num_classes` is
a free function, not a `Flux` method: integer class-id semantics are
classification-specific, so the task-agnostic engine doesn't advertise it.

### `LabelMap` — fittable name↔id encoding

When a dataset's `target` is a class **name** rather than an integer id, `LabelMap` turns it into
the pinned encoding the `EncodeTargetOp` / `DecodeTargetOp` need — the *fittable* companion to
those ops. Fit it once (sklearn `LabelEncoder`, deterministic sorted ordering), persist it in the
`class_names.json` format, and reload it at eval/predict so every stage shares one ordering:

```python
from dataflux import LabelMap, Flux

lm = LabelMap.fit(iter_targets(train_source))   # {"bird": 0, "cat": 1, "dog": 2}
lm.num_classes        # 3
lm.label_names        # ["bird", "cat", "dog"]  (id -> name)
lm.save("class_names.json")                     # marainer's class_names.json format

encoded = Flux(source=train_source, ops=[lm.encode_op()])   # targets are now ints

# Later, at eval time — reload the SAME ordering instead of refitting:
lm2 = LabelMap.load("class_names.json")
```

`LabelMap.fit` is the *only* place a mapping is derived from data; everywhere downstream the
mapping is pinned, so train / eval / predict never disagree. `scikit-learn` backs `fit` (lazy-imported).

## 🖼 Image Conversion (`dataflux.ops.image`)

The single, modality-agnostic "any value → image" layer — generic so every
project (waivefront spectrograms, any dataset preview, FluxStudio) reuses one
implementation. Domain-specific rendering (overlays, signal plots) stays in the
consuming package.

```python
from dataflux.ops.image import ConvertToImageOp, value_to_image

# Op: sample.input (2-D map / CHW tensor / PIL / bool mask) -> PIL image.
op = ConvertToImageOp(
    colormap="viridis",   # closed `Colormap` Literal -> dropdown in FluxStudio, enum in navigaitor
    width=1024, height=512,  # exact resize when both > 0; else bound longest side by max_size
    flip_vertical=True,      # e.g. a spectrogram stores row 0 = f_min but display wants f_max on top
)
sample = op(sample)          # also publishes image_width_px / image_height_px to metadata

# Library function for ad-hoc previews (PIL / tensor / ndarray / mask -> (H, W, 3) uint8):
rgb = value_to_image(some_value, colormap="magma", max_size=512)

# NormalizeToUint8Op: the standalone min-max value -> uint8 quantization step
# (decoupled from colormap / PIL). vmin/vmax default None = per-array auto-contrast;
# set them to pin a fixed scale across samples (out-of-range values clamp).
from dataflux.ops.image import NormalizeToUint8Op

sample = NormalizeToUint8Op()(sample)                       # auto per-array min/max
sample = NormalizeToUint8Op(vmin=-80.0, vmax=0.0)(sample)   # fixed dB window across a dataset
u8 = NormalizeToUint8Op.normalize_to_uint8(arr, vmin=-80.0, vmax=0.0)  # the backing @staticmethod
```

`Colormap` / `COLORMAPS` / `value_to_image` / `sample_to_image` are re-exported
from `waivefront.visualizers` for backward compatibility. Pillow is a runtime
dependency; matplotlib is imported lazily (only non-`gray` colormaps need it).

## 📦 Storage Integration

DataFlux makes it easy to move data between different formats:

```python
from dataflux.storage.hdf5 import HDF5Source
from dataflux.storage.zarr import ZarrGroupSink

# Stream from HDF5 to Zarr in parallel
Flux.from_source(HDF5Source("input.h5")) \
    .parallel(workers=8) \
    .map(heavy_op) \
    .to_sink(ZarrGroupSink("output.zarr"))
```

### Sinks and their matching sources

Every sink has a source that reads its layout back into `Sample` triplets:

| Backend | Sink | Source | Round-trips |
|---|---|---|---|
| HDF5 (sequential) | `HDF5Sink` | `HDF5Source` | input + target + metadata |
| Zarr group (one group / sample) | `ZarrGroupSink` | `ZarrGroupSource` | input + target + metadata |
| Zarr batch (one stacked array) | `ZarrBatchSink` | `ZarrBatchSource` | input only (uniform shape) |
| Directory (one dir / sample) | `DirectorySink` | — | — |

```python
from dataflux.storage.zarr import ZarrGroupSink, ZarrGroupSource

Flux(samples).to_sink(ZarrGroupSink("ds.zarr", overwrite=True))
for sample in ZarrGroupSource("ds.zarr"):   # input/target as before, metadata from .zattrs
    ...
```

### Array-valued metadata (e.g. segmentation masks)

`HDF5Sink` stores scalar/string metadata as HDF5 **attributes**, but HDF5 caps
attribute size — a large array (a segmentation mask, a per-sample weight map) put
in `Sample.metadata` would overflow that limit. So **array-valued metadata
(`np.ndarray` / `torch.Tensor`) is written as its own dataset** under a per-sample
group `{prefix}_meta/<key>`, and `HDF5Source` merges it back into `Sample.metadata`
on read. This is fully backward-compatible: files written before this layout (no
`{prefix}_meta` group) read exactly as before.

```python
sample = Sample(input=iq, target=label, metadata={"mask": mask_2d, "snr": 12.0})
Flux([sample]).to_sink(HDF5Sink("ds.h5", overwrite=True))
loaded = next(iter(HDF5Source("ds.h5")))
loaded.metadata["mask"]   # the full array, byte-exact (not a truncated repr)
loaded.metadata["snr"]    # scalar, via attributes as before
```

## ✂️ Train / Val / Test Splitting

`DatasetSplit` partitions any indexable source (implementing `__len__` and `__getitem__`) into reproducible **train / val / test** views. It is a `source` (`category="source"`) — it yields `Sample`s and is wired into a trainer's `source:` slot — and it applies no ops, so it's a source, not an engine.

**Property API (preferred).** Configure **one** `DatasetSplit` with a `seed` and the held-out fraction(s), then read the three cached views off it — `split.train` / `split.val` / `split.test`:

```python
from dataflux import DatasetSplit
split = DatasetSplit(source=src, val_fraction=0.1, test_fraction=0.1, seed=42)
split.train   # ≈80% — the remainder      split.val   # ≈10%      split.test  # ≈10%
```

The views are disjoint and complementary, computed once over a single deterministic shuffle (cached), so the underlying source is consumed once. In Confluid YAML they're reachable by **attribute reference** — `!ref:my_split.train` / `.val` / `.test`. All three refs resolve to the *same* `DatasetSplit` instance, so the upstream source is loaded **exactly once**:

```yaml
hf_train: !class:dataflux.sources.HuggingFaceSource()
  path: mnist
  split: train

my_split: !class:dataflux.sources.DatasetSplit()
  source: !ref:hf_train
  val_fraction: 0.1
  test_fraction: 0.1
  seed: 42

train_set: !class:dataflux.core.Flux() { source: !ref:my_split.train }
val_set:   !class:dataflux.core.Flux() { source: !ref:my_split.val }
test_set:  !class:dataflux.core.Flux() { source: !ref:my_split.test }
```

Omit `test_fraction` for a plain two-way train/val split; omit both fractions and `train` is the whole source (`val`/`test` empty).

**Select-one API.** Passing `split` makes the `DatasetSplit` *itself* iterate that one view (`split=None` ⇒ `train`), so it's directly usable as a single `source:`. `split` is the closed `Literal["train", "val", "test"]`, exported as `dataflux.SplitName`.

```yaml
val_set: !class:dataflux.sources.DatasetSplit()
  source: !ref:hf_train
  split: val
  val_fraction: 0.1
  seed: 42
```

### Range & concatenation sources

- **`RangeSource(source, start, end)`** — a contiguous index slice `[start:end)` over a source (negatives count from the end; clamped). The plain-slice counterpart to `DatasetSplit`.

    ```yaml
    first_half: !class:dataflux.sources.RangeSource()
      source: !ref:hf_train
      start: 0
      end: 5000
    ```

- **`ConcatSource(sources)`** — joins multiple indexable sources into one longer indexable source (the indexable counterpart to `JointFlux`, which is iteration-only). Because it's indexable, a `ConcatSource` can itself be wrapped by `DatasetSplit` / `RangeSource`.

    ```yaml
    combined: !class:dataflux.sources.ConcatSource()
      sources:
        - !ref:train_main
        - !ref:extra_shard
    ```

**HuggingFace native slicing** (alternative, no DataFlux split needed): `split: "train[:90%]"` / `"train[90%:]"` on two `HuggingFaceSource`s.

> **Note on `!ref:`** — Confluid `!ref:` resolves to the same live object as the referenced key (including attribute refs like `!ref:my_split.train`), so a single `HuggingFaceSource` is loaded once and shared. Use `!clone:` when you want an independent deep copy instead.

> **Lazy & zero-arg construction** — `HuggingFaceSource` follows the workspace lazy-init convention: the constructor does no work (no network), so `HuggingFaceSource()` is valid and building one is free. The dataset is downloaded only on first access to the read-only `.dataset` property (cached thereafter; reset `_dataset` to reload), and `.resolved_metadata_features` (the `"*"` expansion) is derived lazily from the loaded columns. `path` is therefore optional at construction and validated lazily — accessing `.dataset` with an empty `path` raises a clear `ValueError`.

## 🔁 Reattach an ops-only YAML (`Flux.from_ops_yaml`)

A `{ops: [!class:…()]}` document — e.g. one exported from a FluxStudio canvas (`fluxstudio export …`) — can be attached to any source:

```python
from dataflux import Flux
from dataflux.sources import HuggingFaceSource

flux = Flux.from_ops_yaml("ops.yaml", source=HuggingFaceSource(path="mnist"))
```

The helper **materializes** the deferred `!class:` markers before attaching (via `confluid.materialize`) — necessary because `confluid.load` leaves markers nested under a mapping key deferred, and a `Flux` rejects deferred markers at iteration by design. The manual equivalent is `Flux(source=src, ops=confluid.materialize(confluid.load("ops.yaml")["ops"]))`.

## 🎛 Per-sample op parameters (`ConfigureOp` / `CaptureOutputOp`)

Some op parameters are only known *per sample*. Two composable ops cover this — both are what FluxStudio emits when you wire a value into an op parameter on the canvas:

- **`ConfigureOp(ops, target, param, key)`** — runs `ops` on the sample as a side-branch; the chain's final `sample.input` is written to `metadata[key]` and injected as `target.<param>`, then `target` is applied. Use it when the value is *derived from the sample itself* (e.g. a threshold from the sample's own max).
- **`CaptureOutputOp(op, output|captures, key)`** — applies `op`, then records one or more of its `@output` attribute values into `metadata[key]`. The value is captured from the **actual run**, so it works for *stochastic* outputs (a random draw) that can't be recomputed. It reads through a `.target` wrapper, so it composes with `ConfigureOp`.

Together they express "feed one op's runtime `@output` into a later op's parameter" — capture the output, then unstash it into the parameter per sample:

```yaml
ops:
  # NoiseFloorOp draws an SNR each call; capture it into metadata.
  - !class:dataflux.ops.capture.CaptureOutputOp
    op: !class:waivefront.torchsig.processing.NoiseFloorOp {}
    output: applied_snr_db
    key: __captured_snr
  # …then inject the captured value into a later op's `noise_power_db` per sample.
  - !class:dataflux.ops.configure.ConfigureOp
    ops:
      - !class:dataflux.ops.stash.UnstashInputOp { key: __captured_snr }
    target: !class:waivefront.torchsig.processing.NoiseFloorOp {}
    param: noise_power_db
```

## 🔗 Paired Join (Binary ↔ Annotations)

`AnnotationJoinSource` joins a data `DataSource` (e.g. raw binary samples) with a sidecar mapping-shaped annotation store via a key function. It generalises the common "I have data, and I have a sidecar file of annotations that covers some of it" pattern — typically re-attaching a LabelStudio export back onto the raw samples for training. Three join policies cover the scenarios we actually see in ML research:

| Policy | Iterates | Use case |
|---|---|---|
| `left_outer` (default) | Every data sample; attaches annotation when the key matches | Process everything, use labels where available |
| `inner` | Only data samples whose key is in the store | Train/evaluate on the labeled subset |
| `right_driven` | Every key in the annotation store; resolves the data sample via `data_resolver(key, data)` | Very sparse labels where full-data enumeration is costly |

```yaml
data: !class:waivefront.rfuav.data.source.RFUAVSource()
  root: /Volumes/Data/RFUAV
  window_samples: 1000000

labels: !class:annotaide.store.JSONFileAnnotationStore()
  path: /Volumes/Data/RFUAV-labels

paired: !class:dataflux.paired.AnnotationJoinSource()
  data: !ref:data
  annotations: !ref:labels
  key_fn: "waivefront.rfuav.keys:sample_window_key"
  policy: left_outer
```

Annotation records are **flattened into `Sample.metadata`**, so a detection record `{bboxes, labels, scores}` shows up as three independent metadata keys. Two `metadata` keys are always populated: `annotated: bool` and `annotation_key: str`. Optional `prefix` and `store_full_under` parameters shape the layout. The parameters are typed, not `Any`: `data` is an `Iterable[Any]` (any source), `annotations` is an `AnnotationStore` (a read-mapping `key → record` — a `dict` or annotaide's `JSONFileAnnotationStore` both qualify), and `policy` is a fixed `Literal["left_outer", "inner", "right_driven"]`. Both the store shape and the policy are validated at construction.

### Coarser-granularity keys (broadcast and slicing)

`key_fn` is free to return a coarser key than the sample granularity. When multiple data samples map to the same key, they all look up the same record:

- **Without `extract_fn`** — the record is broadcast identically into every matching sample's metadata (e.g. a scalar pack-level class label inherited by every window of that pack).
- **With `extract_fn`** — the record is projected per sample. The callable is invoked as `extract_fn(record, sample) -> dict | None`; returning `None` marks the sample unannotated (and filters it under `policy="inner"`). Use this when a pack-level annotation carries time-ranged content that must be trimmed to each window's bounds.

Multi-granularity joins (e.g. pack-level + window-level annotations merged together) compose by chaining `AnnotationJoinSource` instances — the output of one is itself a `DataSource` that the next can consume.

### Callable resolution

`key_fn`, `extract_fn`, and `data_resolver` all accept either a callable **or** a `"module:function"` string path resolved through `dataflux.discovery.resolve_callable`. The string form is what survives YAML round-trip via Confluid.

See [`examples/paired_annotations.py`](examples/paired_annotations.py) for a runnable end-to-end walkthrough of all four scenarios.

## 🌐 Ecosystem Integration

DataFlux is designed to sit between your data catalog and your training loop, acting as the high-performance "glue" for ML pipelines.

### Hugging Face (Community & Standardized Datasets)
-   **Use Hugging Face for:** Accessing community datasets and leveraging the `datasets` library for efficient Arrow/Parquet loading.
-   **Integration:** Use DataFlux to transform `datasets.Dataset` objects into standardized `Sample` triplets, ensuring metadata traceability that often goes missing in simple dictionary-based records.
-   **`metadata_features` (which columns ride along on `Sample.metadata`):** `None` / `[]` keep none (the default); an explicit list keeps exactly those columns; and the sentinel **`"*"`** (or `["*"]`) keeps **every column except `input_feature` / `target_feature`** — the full-traceability option, resolved against the dataset's real columns at load. It stays opt-in so existing configs are unchanged.

```yaml
hf_train: !class:dataflux.sources.HuggingFaceSource()
  path: mnist
  input_feature: image
  target_feature: label
  metadata_features: ["*"]   # keep every other column as metadata (here: none extra beyond hf_path/hf_split)
```

### DataFlux (The Functional Engine)
-   **Use DataFlux for:** The "inner loop" of your experiment. When you need high-performance multiprocess streaming, per-sample metadata preservation, and 100% reproducible pipelines via **Confluid** serialization.

## 🔧 Installation

```bash
pip install git+https://github.com/Gearlux/dataflux.git@main
```

## 📄 License

MIT
