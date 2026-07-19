# Image conversion (`sampleflux.ops.image`)

The single, modality-agnostic "any value → image" layer — generic so every consuming project (spectrogram previews, dataset browsers, GUI viewers) reuses one implementation. Domain-specific rendering (overlays, signal plots) stays in the consuming package.

```python
from sampleflux.ops.image import ConvertToImageOp, value_to_image

# Op: sample.input (2-D map / CHW tensor / PIL / bool mask) -> PIL image.
op = ConvertToImageOp(
    colormap="viridis",   # closed `Colormap` Literal -> enumerable in GUIs / schemas
    width=1024, height=512,  # exact resize when both > 0; else bound longest side by max_size
    flip_vertical=True,      # e.g. a spectrogram stores row 0 = f_min but display wants f_max on top
)
sample = op(sample)          # also publishes image_width_px / image_height_px to metadata

# Library function for ad-hoc previews (PIL / tensor / ndarray / mask -> (H, W, 3) uint8):
rgb = value_to_image(some_value, colormap="magma", max_size=512)

# NormalizeToUint8Op: the standalone min-max value -> uint8 quantization step
# (decoupled from colormap / PIL). vmin/vmax default None = per-array auto-contrast;
# set them to pin a fixed scale across samples (out-of-range values clamp).
from sampleflux.ops.image import NormalizeToUint8Op

sample = NormalizeToUint8Op()(sample)                       # auto per-array min/max
sample = NormalizeToUint8Op(vmin=-80.0, vmax=0.0)(sample)   # fixed dB window across a dataset
u8 = NormalizeToUint8Op.normalize_to_uint8(arr, vmin=-80.0, vmax=0.0)  # the backing @staticmethod
```

`Colormap` / `COLORMAPS` / `value_to_image` / `sample_to_image` are re-exported from `waivefront.visualizers` for backward compatibility. Pillow is a runtime dependency; matplotlib is imported lazily (only non-`gray` colormaps need it).

## Introspection helpers

Pure library functions (not ops) also live here, backing viewer tooling: `select_channel` (reduce an array/tensor to a 2-D float32 map for one channel; negative = mean across channels), `channel_count`, `array_histogram` (finite-only binning + summary stats, JSON-safe), `confusion_matrix_payload` / `confusion_matrices_payload` (render payloads for every confusion-matrix-shaped entry in a metrics result), and `draw_text` (text → `(H, W, 3)` uint8 image with word-wrap and 9-grid anchoring, plus the closed `TextPosition` Literal).
