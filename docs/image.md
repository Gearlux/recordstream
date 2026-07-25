# Image conversion (`sampleflux.ops.image`)

The single, modality-agnostic "any value → image" layer — generic so every consuming project (spectrogram previews, dataset browsers, GUI viewers) reuses one implementation. Domain-specific rendering (overlays, signal plots) stays in the consuming package.

```python
from sampleflux.ops.image import ConvertToImage, value_to_image

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
from sampleflux.ops.image import normalize_to_uint8

u8 = normalize_to_uint8(arr)                          # auto per-array min/max
u8 = normalize_to_uint8(arr, vmin=-80.0, vmax=0.0)    # fixed dB window across a dataset
```

`sample_to_image(record, ...)` renders a record's first array-bearing (2-D / 3-D) value the same way — the ad-hoc whole-record preview for viewer tooling. Pillow is a runtime dependency; matplotlib is imported lazily (only non-`gray` colormaps need it).

## Introspection helpers

Pure library functions (not ops) also live here, backing viewer tooling: `select_channel` (reduce an array/tensor to a 2-D float32 map for one channel; negative = mean across channels), `channel_count`, `array_histogram` (finite-only binning + summary stats, JSON-safe), `confusion_matrix_payload` / `confusion_matrices_payload` (render payloads for every confusion-matrix-shaped entry in a metrics result), and `draw_text` (text → `(H, W, 3)` uint8 image with word-wrap and 9-grid anchoring, plus the closed `TextPosition` Literal).
