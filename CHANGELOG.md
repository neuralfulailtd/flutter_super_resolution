## 1.1.1

* Documented where to download compatible models: added an AI Zoo link to the
  README, pre-filtered for `type=upscale` and Flutter compatibility.

## 1.1.0

* **Much faster output decoding.** The model's output tensor is now read as a
  zero-copy `Float32List` view over ONNX Runtime's native buffer (via
  `GetTensorMutableData`) instead of `OrtValue.value`, which allocated a nested
  `List` of *boxed* doubles — hundreds of thousands of heap objects per tile.
  This removes the dominant post-inference cost and its GC pressure. A guarded
  fallback to the old `.value` path is kept for forward-compatibility.
* **Seamless symmetric tile blending.** Tiles are now cross-faded with a
  raised-cosine (Hann) window feathered on *all* overlapping edges. Because the
  rising and falling ramps form a partition of unity, adjacent tiles blend
  perfectly — replacing the previous one-sided linear ramp that left a slight
  discontinuity at trailing seams. Implemented as a running weighted-average, so
  memory stays at `8 × output` bytes (no separate float accumulator).
* **Faster tensor packing.** Input normalisation uses a precomputed 256-entry
  uint8→float lookup table instead of a per-pixel division; per-tile edge blend
  weights are precomputed once per tile rather than per pixel.
* **Lower per-tile latency.** Dropped the per-tile `Future.delayed(Duration.zero)`
  GC yield — `runAsync` already runs inference on a separate isolate (freeing the
  UI isolate), and the FFI fast-path no longer produces per-tile garbage to
  collect. The cached `GetTensorMutableData` binding is resolved once at init.
* Default `maxOutputMemoryMB` lowered to 128 (mobile-appropriate; matches the
  example app and documented ~2048×2048 output budget).

## 1.0.5

* Fixed tiling seams: tile boundaries now fall in the middle of the overlap
  region (half-overlap stitching) so each tile contributes only from its
  well-contextualised interior.
* Fixed edge tiles: they are now zero-padded to `tileSize`×`tileSize` before
  inference, preventing shape-mismatch errors on fixed-input ONNX models.
* Fixed `ui.Image` memory leak: per-tile images are now `dispose()`d after
  being drawn onto the canvas.
* Fixed `OrtSessionOptions` resource leak: `release()` is now called in a
  `finally` block so native options objects are freed on error too.
* Optimised tensor decoding: channel lists are pre-cast once per row instead
  of once per pixel, significantly reducing dynamic dispatch in the hot loop.
* Added `inputName` constructor parameter to support models whose input node
  is not named `'input'`.
* Simplified `_prepareInputTensor` — width/height are now read directly from
  the image, removing redundant parameters.
* Fixed example app: replaced non-existent `ui.ImageForRenderObject` with
  `RawImage`; moved upscaler initialisation out of the button handler so the
  model is loaded once rather than on every tap; improved error reporting via
  `SnackBar`.
* Fixed test file: removed scaffold `Calculator` placeholder; added tests for
  constructor defaults, parameter validation, and lifecycle.

## 0.0.1

* Initial release.
