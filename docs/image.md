# Image conversion (`recordstream.ops.image`)

The single, modality-agnostic "any value → image" layer — generic so every consuming project (spectrogram previews, dataset browsers, GUI viewers) reuses one implementation. Domain-specific rendering (overlays, signal plots) stays in the consuming package.

```python
from recordstream.ops.image import ConvertToImage, value_to_image

# Op: an array-bearing record value (2-D map / CHW tensor / PIL / bool mask) -> an Image item.
op = ConvertToImage(
    colormap="viridis",   # closed `Colormap` Literal -> enumerable in GUIs / schemas
    width=1024, height=512,  # exact resize when both > 0; else bound longest side by max_size
    flip_vertical=True,      # e.g. a spectrogram stores row 0 = f_min but display wants f_max on top
    field="spec",            # source key; blank picks the first array-bearing value
    output="image",          # key the HWC-uint8 Image item is written to
)
record = op(record)          # adds record["image"]; the pixel dimensions live in its array shape

# Library function for ad-hoc previews (PIL / tensor / ndarray / mask -> (H, W, 3) uint8):
rgb = value_to_image(some_value, colormap="magma", max_size=512)

# normalize_to_uint8: the standalone min-max value -> uint8 quantization step
# (decoupled from colormap / PIL). vmin/vmax default None = per-array auto-contrast;
# set them to pin a fixed scale across records (out-of-range values clamp).
from recordstream.ops.image import normalize_to_uint8

u8 = normalize_to_uint8(arr)                          # auto per-array min/max
u8 = normalize_to_uint8(arr, vmin=-80.0, vmax=0.0)    # fixed dB window across a dataset
```

`record_to_image(record, ...)` renders a record's first array-bearing (2-D / 3-D) value the same way — the ad-hoc whole-record preview for viewer tooling. Pillow is a runtime dependency; matplotlib is imported lazily (only non-`gray` colormaps need it).

## Per-channel standardization (`Normalize`)

The normalization node between an image conversion and a model — the same math as the
albumentations transform of the same name (`(x - mean*max_value) / (std*max_value)`), with
the ImageNet statistics as defaults:

```yaml
ops:
  - !class:recordstream.ops.image.ConvertToImage {width: 224, height: 224}
  - !class:recordstream.ops.image.Normalize {}      # ImageNet mean/std over uint8 input
  - !class:recordstream.ops.torch.ToTensor {normalize: false}
```

A bare `!class:albumentations.Normalize` in a YAML `ops:` list computes the identical
result; this op exists so a **drawn** pipeline has a node for the step. Output keeps the
item type in float32; a 2-D map with per-channel statistics is refused by name (convert it
first, or pass single-element `mean`/`std`).

## Boxes (`ConvertToBoxes` / `ConvertFromBoxes`)

The record model speaks ONE box format by contract — the `Boxes` item: absolute-pixel,
half-open `[x0, y0, x1, y1]` (y down), with `labels`, `scores`, `classes` and `canvas`
beside the rows. That single target is what keeps every consumer interoperable, so the
conversion pair varies only the OTHER side:

```yaml
ops:
  - !class:recordstream.ops.image.ConvertToBoxes    # any source layout -> the canonical
    field: class          # a HF `objects` dict (seen through its Label wrapper),
    format: xywh          # a bare [N, 4] array, or a mis-made Boxes; rows in COCO
                          # xywh / cxcywh / xyxy, `normalized: true` for [0, 1] rows
  - !class:recordstream.ops.image.ConvertFromBoxes  # the canonical -> a sink's layout
    format: xywh          # e.g. write reviewed annotations back in the dataset's shape
    container: objects    # {'bbox', 'category', 'score'} — or `array` for rows only
```

`ConvertToBoxes` stamps the image's `(H, W)` as the canvas and carries a class vocabulary
when given one (or when the source item already holds it); a trainer wanting normalized
cxcywh converts at its own sink/collate — never by storing non-canonical rows in a `Boxes`.

## Masks (`ConvertToMask`)

The segmentation counterpart, and the same shape of op — read one field, write a differently-typed item under `output`. A segmentation dataset ships its target as a greyscale/paletted PNG whose pixel values *are* the class ids (an Oxford-IIIT Pet trimap, Cityscapes label ids, a VOC segmentation map); this turns that payload into the `int64` `[H, W]` `Mask` every per-pixel loss expects.

```python
from recordstream.ops.image import ConvertToMask

op = ConvertToMask(
    field="segmentation_mask",  # source key; blank picks the first array/PIL-bearing value
    output="mask",              # key the int64 Mask item is written to
)
```

It converts and **nothing else**, because the rest of the chain is ops that already exist:

| you want | use |
| --- | --- |
| remap the ids (a 1-based trimap → 0-based) | `FormulaOp(field="mask", formula="a - 1")` |
| remap through a lookup table (Cityscapes id → trainId) | `EncodeTarget` |
| resize / augment it **together with the image** | a bare `albumentations` transform in the same ops list |
| drop the source column | `DropField(key="segmentation_mask")` |

That fourth row is why `output` defaults to `"mask"`: it is albumentations' own key vocabulary, so the engine's op-family dispatch hands `image` **and** `mask` to one call — a single joint draw moves both, and the `Mask` type survives the round trip. An image-only transform (`Normalize`) still touches the image alone.

```yaml
# the target half of a segmentation `preprocess` chain
- !class:recordstream.ops.image.ConvertToMask {field: segmentation_mask, output: mask}
- !class:recordstream.ops.formula.FormulaOp   {field: mask, formula: a - 1}
- !class:recordstream.ops.structure.DropField {key: segmentation_mask}
- !class:albumentations.Resize                {height: 224, width: 224}   # image AND mask
```

`int64` is not a knob: a class-id map is integer by definition, and it is what `torch.nn.CrossEntropyLoss` requires (it rejects int32 with *"expected target dtype to be Long or Byte, but got Int"*). Libraries that cast on the way past — albumentations returns int32 — are corrected at the model boundary with `batch_tensor(batch, "mask", dtype=torch.int64)`, where the caller names the contract. An RGB-encoded mask is **refused** rather than collapsed: picking one of three channels is a decision the op must not make silently.

## Introspection helpers

Pure library functions (not ops) also live here, backing viewer tooling: `select_channel` (reduce an array/tensor to a 2-D float32 map for one channel; negative = mean across channels), `channel_count`, `array_histogram` (finite-only binning + summary stats, JSON-safe), `confusion_matrix_payload` / `confusion_matrices_payload` (render payloads for every confusion-matrix-shaped entry in a metrics result), and `draw_text` (text → `(H, W, 3)` uint8 image with word-wrap and 9-grid anchoring, plus the closed `TextPosition` Literal).
