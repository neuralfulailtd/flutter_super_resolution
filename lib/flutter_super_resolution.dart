library flutter_super_resolution;

import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' as math;
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
// Internal bindings are imported directly so the model's output tensor can be
// read as a zero-copy Float32List view over native memory (see
// [_outputAsFloat32View]). The public `.value` getter instead allocates a
// nested List of *boxed* doubles per tile — hundreds of thousands of heap
// objects that dominate post-processing time. The FFI fast-path is guarded and
// falls back to `.value` if the internal layout ever changes.
// ignore: implementation_imports
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;

typedef ProgressCallback = void Function(double progress, String message);

/// Dart signature for ORT's `GetTensorMutableData(OrtValue*, void**)`.
typedef _GetTensorDataFn = bg.OrtStatusPtr Function(
    ffi.Pointer<bg.OrtValue>, ffi.Pointer<ffi.Pointer<ffi.Void>>);

class FlutterUpscaler {
  OrtSession? _session;

  // Pre-allocated to avoid per-inference-call alloc/free overhead.
  OrtRunOptions? _runOptions;

  // Pre-allocated tile buffer reused across all tiles in a single upscale call.
  // Eliminates ~3 MB GC allocation per tile on 512-px tile grids.
  Float32List? _tileBuffer;

  // 256-entry uint8→float lookup table (value / 255). Replaces a per-pixel,
  // per-channel floating-point division in the input packing hot loop.
  final Float32List _u8ToFloat = Float32List(256);

  // Cached FFI binding for GetTensorMutableData, resolved once at init so the
  // hot path never re-walks the ORT API struct.
  _GetTensorDataFn? _getTensorData;

  /// Tile size in source pixels. Larger = fewer ORT calls but more RAM per
  /// tile. 128 is safe on most mobile devices; use 64 for very old iPhones.
  final int tileSize;

  /// Overlap between adjacent tiles in source pixels. Tiles share this many
  /// pixels of context and their outputs are cross-faded over the corresponding
  /// output-space zone with a raised-cosine window so that seams are invisible.
  final int overlap;

  /// Name of the model's input node (default: `'input'`).
  final String inputName;

  /// Maximum memory budget for the CPU output buffers in megabytes.
  ///
  /// The tiled path pre-allocates two CPU buffers whose combined size is
  /// `8 × finalWidth × finalHeight` bytes. If that would exceed this limit
  /// an [Exception] is thrown rather than risking an OOM crash.
  /// Default is 128 MB — handles up to ~2048×2048 output (1024×1024 src at ×2).
  final int maxOutputMemoryMB;

  FlutterUpscaler({
    this.tileSize = 128,
    this.overlap = 16,
    this.inputName = 'input',
    this.maxOutputMemoryMB = 128,
  }) : assert(overlap < tileSize, 'overlap must be smaller than tileSize') {
    for (int i = 0; i < 256; i++) {
      _u8ToFloat[i] = i / 255.0;
    }
  }

  // ---------------------------------------------------------------------------
  // Initialisation
  // ---------------------------------------------------------------------------

  /// Load an ONNX model from a Flutter asset bundle path.
  Future<void> initializeModel(String modelPath) async {
    OrtEnv.instance.init(level: OrtLoggingLevel.error);
    _resetForNewSession();
    final opts = OrtSessionOptions();
    try {
      _configureSessionOptions(opts);
      final data = await rootBundle.load(modelPath);
      _session = OrtSession.fromBuffer(data.buffer.asUint8List(), opts);
    } catch (e) {
      throw Exception('Failed to load model from asset "$modelPath": $e');
    } finally {
      opts.release();
    }
  }

  /// Load an ONNX model from an absolute filesystem path.
  Future<void> initializeModelFromFile(String filePath) async {
    OrtEnv.instance.init(level: OrtLoggingLevel.error);
    _resetForNewSession();
    final opts = OrtSessionOptions();
    try {
      _configureSessionOptions(opts);
      _session =
          OrtSession.fromBuffer(await File(filePath).readAsBytes(), opts);
    } catch (e) {
      throw Exception('Failed to load model from file "$filePath": $e');
    } finally {
      opts.release();
    }
  }

  void _resetForNewSession() {
    _session?.release();
    _session = null;
    _runOptions?.release();
    _runOptions = OrtRunOptions();
    _tileBuffer = Float32List(3 * tileSize * tileSize);
    _getTensorData = _resolveGetTensorData();
  }

  /// Resolve ORT's `GetTensorMutableData` once. Returns null if the binding is
  /// unavailable, in which case the slower `.value` path is used.
  _GetTensorDataFn? _resolveGetTensorData() {
    try {
      return OrtEnv.instance.ortApiPtr.ref.GetTensorMutableData
          .asFunction<_GetTensorDataFn>();
    } catch (_) {
      return null;
    }
  }

  void _configureSessionOptions(OrtSessionOptions opts) {
    // Log which execution providers ONNX Runtime has compiled in.
    final available = OrtEnv.instance.availableProviders();
    // ignore: avoid_print
    print(
        '[FlutterUpscaler] Available providers: ${available.map((p) => p.value).toList()}');

    // On Apple platforms, register CoreML first so ONNX dispatches supported
    // ops to the Neural Engine / GPU. Unsupported ops fall back to CPU
    // automatically — no manual op partitioning needed.
    if (Platform.isMacOS || Platform.isIOS) {
      try {
        // enableOnSubgraph lets CoreML claim static-shape subgraphs (Conv,
        // Gemm, etc.) even when the top-level input has dynamic dims.
        // Without this flag CoreML rejects the entire graph if any input
        // shape is dynamic, and all ops fall back to CPU.
        opts.appendCoreMLProvider(CoreMLFlags.enableOnSubgraph);
        // ignore: avoid_print
        print(
            '[FlutterUpscaler] CoreML provider registered (enableOnSubgraph)');
      } catch (e) {
        // ignore: avoid_print
        print('[FlutterUpscaler] CoreML unavailable: $e — CPU only');
      }
    }
    // CPU threads cover ops not handled by CoreML.
    // Scales to P-core count on Apple Silicon; diminishing returns beyond 8.
    final intraOp = Platform.numberOfProcessors.clamp(4, 8);
    opts.setIntraOpNumThreads(intraOp);
    opts.setInterOpNumThreads(2);
    opts.setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Upscale [sourceImage] by integer factor [scale].
  ///
  /// For images larger than [tileSize], the tiled path is used automatically.
  /// All CPU buffers are freed as soon as the result is returned.
  Future<ui.Image?> upscaleImage(
    ui.Image sourceImage,
    int scale, {
    bool useTiling = true,
    ProgressCallback? onProgress,
  }) async {
    if (_session == null) {
      throw StateError('Model not initialised. Call initializeModel first.');
    }
    try {
      final smallEnough =
          sourceImage.width <= tileSize && sourceImage.height <= tileSize;
      if (!useTiling || smallEnough) {
        return await _processFullImage(sourceImage, scale, onProgress);
      }
      return await _processByTiles(sourceImage, scale, onProgress);
    } catch (e) {
      throw Exception('Error upscaling image: $e');
    }
  }

  /// Release the ONNX session and free native memory.
  void dispose() {
    _session?.release();
    _session = null;
    _runOptions?.release();
    _runOptions = null;
    _tileBuffer = null;
    _getTensorData = null;
  }

  // ---------------------------------------------------------------------------
  // Full-image path
  // ---------------------------------------------------------------------------

  Future<ui.Image> _processFullImage(
    ui.Image src,
    int scale,
    ProgressCallback? onProgress,
  ) async {
    onProgress?.call(0.0, 'Reading image…');
    final bytes = await src.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (bytes == null) throw Exception('Failed to read source image bytes');
    final rgba = bytes.buffer.asUint8List();

    onProgress?.call(0.15, 'Preparing tensor…');
    final tensor = _rgbaToTensor(rgba, src.width, src.height);

    onProgress?.call(0.25, 'Running neural network…');
    final outW = src.width * scale;
    final outH = src.height * scale;

    // The output tensor is consumed (copied into RGBA bytes) inside [_infer],
    // before ORT releases it — the Float32List view is only valid until then.
    final pixels = await _infer(
      tensor,
      outH,
      outW,
      (chw, rowStride, planeStride) =>
          _chwToPixels(chw, rowStride, planeStride, outW, outH),
    );
    tensor.release();

    onProgress?.call(0.85, 'Decoding output…');
    final image = await _decode(pixels, outW, outH);

    onProgress?.call(1.0, 'Done.');
    return image;
  }

  // ---------------------------------------------------------------------------
  // Tiled path — CPU accumulation buffer, no per-tile GPU textures
  // ---------------------------------------------------------------------------
  //
  // Two CPU byte arrays are pre-allocated once and filled in place:
  //   outputPixels  : Uint8List   — running RGBA output (4 bytes / pixel)
  //   blendWeights  : Float32List — accumulated blend weight (4 bytes / pixel)
  //
  // Each tile's output is blended directly into outputPixels with a running
  // weighted-average. That running average is algebraically identical to the
  // "accumulate weighted sum, then divide by total weight" scheme but uses half
  // the memory (no separate float RGB accumulator). The GPU sees a single
  // upload at the very end, regardless of tile count.
  //
  // Total memory = 8 × finalWidth × finalHeight bytes.
  // The constructor's maxOutputMemoryMB guard prevents OOM for huge outputs.

  Future<ui.Image> _processByTiles(
    ui.Image src,
    int scale,
    ProgressCallback? onProgress,
  ) async {
    final finalW = src.width * scale;
    final finalH = src.height * scale;

    // ── Memory guard ─────────────────────────────────────────────────────────
    final neededMB = finalW * finalH * 8 ~/ (1024 * 1024);
    if (neededMB > maxOutputMemoryMB) {
      throw Exception(
        'Output ($finalW×$finalH px) needs ~${neededMB}MB but '
        'maxOutputMemoryMB is $maxOutputMemoryMB. '
        'Reduce scale/input size or raise maxOutputMemoryMB.',
      );
    }

    // ── Single source readback ───────────────────────────────────────────────
    onProgress?.call(0.0, 'Reading source image…');
    final srcBytes = await src.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (srcBytes == null) throw Exception('Failed to read source image bytes');
    // Wrapped in a nullable so we can null it out after the last tile,
    // allowing the GC to reclaim the source bytes while we finish compositing.
    Uint8List? srcRgba = srcBytes.buffer.asUint8List();

    // ── Output buffers (CPU, no GPU involved until the final upload) ──────────
    final outputPixels = Uint8List(finalW * finalH * 4);
    final blendWeights = Float32List(finalW * finalH);

    final step = tileSize - overlap;
    final bz = overlap * scale; // blend-zone width in output pixels
    final numX = (src.width / step).ceil();
    final numY = (src.height / step).ceil();
    final total = numX * numY;

    // Raised-cosine (Hann) ramp LUT rising 0→1 across the blend zone. Two such
    // ramps — one rising on a tile's leading edge, one falling on its trailing
    // edge — form a partition of unity, giving a seamless symmetric cross-fade.
    final ramp = _buildRampLut(bz);

    // Padded output tile dimensions (input is always zero-padded to tileSize).
    final outPad = tileSize * scale;

    for (int gy = 0; gy < numY; gy++) {
      for (int gx = 0; gx < numX; gx++) {
        final srcX = gx * step;
        final srcY = gy * step;
        final actualW = math.min(tileSize, src.width - srcX);
        final actualH = math.min(tileSize, src.height - srcY);

        final tileIdx = gy * numX + gx;
        onProgress?.call(
          0.05 + 0.90 * tileIdx / total,
          'Tile ${tileIdx + 1}/$total…',
        );

        final tileOutW = actualW * scale;
        final tileOutH = actualH * scale;

        // Separable per-edge blend weights for this tile. A leading edge is
        // feathered only if a neighbour precedes it; a trailing edge only if
        // one follows. Edge (image-border) tiles taper on their inner sides
        // exactly like interior tiles.
        final wxArr = _edgeWeights(
            tileOutW, bz, ramp, /*lead*/ gx > 0, /*trail*/ gx < numX - 1);
        final wyArr = _edgeWeights(
            tileOutH, bz, ramp, /*lead*/ gy > 0, /*trail*/ gy < numY - 1);

        // Slice tile directly from source bytes — no Canvas, no GPU reads.
        final tensor = _sliceTileToTensor(
            srcRgba, src.width, srcX, srcY, actualW, actualH);

        await _infer(tensor, outPad, outPad, (chw, rowStride, planeStride) {
          _blendTileIntoBuffer(
            chw,
            rowStride,
            planeStride,
            outputPixels,
            blendWeights,
            finalW,
            finalH,
            srcX * scale,
            srcY * scale,
            tileOutW,
            tileOutH,
            wxArr,
            wyArr,
          );
          return null;
        });
        tensor.release();
      }
    }

    // Release the source byte reference as early as possible.
    srcRgba = null;

    // Set alpha channel to fully opaque (blend only touches RGB).
    for (int i = 3; i < outputPixels.length; i += 4) {
      outputPixels[i] = 255;
    }

    onProgress?.call(0.97, 'Uploading result…');
    final image = await _decode(outputPixels, finalW, finalH);

    onProgress?.call(1.0, 'Done.');
    return image;
  }

  /// Build a raised-cosine ramp of length [bz] rising smoothly from 0 to 1.
  /// `ramp[k] = ½(1 − cos(π·(k+½)/bz))`. Returns an empty list when [bz] ≤ 0.
  Float32List _buildRampLut(int bz) {
    if (bz <= 0) return Float32List(0);
    final lut = Float32List(bz);
    for (int k = 0; k < bz; k++) {
      lut[k] = 0.5 - 0.5 * math.cos(math.pi * (k + 0.5) / bz);
    }
    return lut;
  }

  /// Per-column (or per-row) blend weights for one tile edge dimension.
  ///
  /// Interior samples weigh 1.0. Within [bz] pixels of a feathered leading edge
  /// the weight follows the rising [ramp]; within [bz] of a feathered trailing
  /// edge it follows the mirrored (falling) ramp. Because the rising and
  /// falling raised-cosine ramps sum to 1, adjacent tiles cross-fade seamlessly.
  Float32List _edgeWeights(
    int len,
    int bz,
    Float32List ramp,
    bool leadTaper,
    bool trailTaper,
  ) {
    final a = Float32List(len);
    for (int i = 0; i < len; i++) {
      double w = 1.0;
      if (leadTaper && bz > 0 && i < bz) {
        w *= ramp[i];
      }
      if (trailTaper && bz > 0 && i >= len - bz) {
        w *= ramp[len - 1 - i];
      }
      a[i] = w;
    }
    return a;
  }

  /// Blend one tile's CHW float output (flat [chw], NCHW plane order) into the
  /// running CPU buffer using a running weighted-average.
  ///
  ///   newPixel = old·(1−α) + tilePixel·α,   α = w / (prevWeight + w)
  ///
  /// where `w = wxArr[tx]·wyArr[ty]` is the separable raised-cosine window.
  void _blendTileIntoBuffer(
    Float32List chw,
    int rowStride,
    int planeStride,
    Uint8List pixels,
    Float32List weights,
    int finalW,
    int finalH,
    int outX,
    int outY,
    int tileOutW,
    int tileOutH,
    Float32List wxArr,
    Float32List wyArr,
  ) {
    final gBase = planeStride;
    final bBase = 2 * planeStride;

    for (int ty = 0; ty < tileOutH; ty++) {
      final globalY = outY + ty;
      if (globalY >= finalH) break;

      final wy = wyArr[ty];
      final srcRow = ty * rowStride;
      final dstRow = globalY * finalW;

      for (int tx = 0; tx < tileOutW; tx++) {
        final globalX = outX + tx;
        if (globalX >= finalW) break;

        final w = wxArr[tx] * wy;
        if (w <= 0.0) continue;

        final i = dstRow + globalX;
        final prevW = weights[i];
        final totalW = prevW + w;
        final alpha = w / totalW;
        final ia = 1.0 - alpha;

        final o = srcRow + tx;
        final p = i << 2;
        pixels[p] = _mix(pixels[p], chw[o], ia, alpha);
        pixels[p + 1] = _mix(pixels[p + 1], chw[gBase + o], ia, alpha);
        pixels[p + 2] = _mix(pixels[p + 2], chw[bBase + o], ia, alpha);
        weights[i] = totalW;
      }
    }
  }

  /// Weighted mix of an existing 0–255 byte with a fresh 0–1 float sample,
  /// clamped back to a byte.
  int _mix(int old, double sample, double ia, double alpha) {
    final v = sample <= 0.0
        ? 0.0
        : sample >= 1.0
            ? 255.0
            : sample * 255.0;
    final r = old * ia + v * alpha;
    return r <= 0.0
        ? 0
        : r >= 255.0
            ? 255
            : (r + 0.5).toInt();
  }

  // ---------------------------------------------------------------------------
  // Tensor construction (CPU-only, no GPU)
  // ---------------------------------------------------------------------------

  /// Convert a full image's raw RGBA bytes to a CHW Float32 ORT tensor.
  OrtValueTensor _rgbaToTensor(Uint8List rgba, int width, int height) {
    final pixelCount = width * height;
    final input = Float32List(3 * pixelCount);
    final lut = _u8ToFloat;
    final gBase = pixelCount;
    final bBase = pixelCount * 2;
    for (int i = 0; i < pixelCount; i++) {
      final p = i * 4;
      input[i] = lut[rgba[p]];
      input[gBase + i] = lut[rgba[p + 1]];
      input[bBase + i] = lut[rgba[p + 2]];
    }
    return OrtValueTensor.createTensorWithDataList(
        input, [1, 3, height, width]);
  }

  /// Slice a [tileSize]×[tileSize] patch directly from source RGBA bytes and
  /// pack it into a CHW Float32 ORT tensor. No Canvas, no GPU round-trip.
  /// Edge tiles (actualW/H < tileSize) are zero-padded at no extra cost
  /// because Dart zero-initialises Float32List.
  OrtValueTensor _sliceTileToTensor(
    Uint8List srcRgba,
    int srcWidth,
    int tileX,
    int tileY,
    int actualW,
    int actualH,
  ) {
    final pixelCount = tileSize * tileSize;
    // Reuse pre-allocated buffer. Zero-fill handles edge-tile padding at no
    // extra cost. ONNX copies the data into native memory on tensor creation,
    // so it is safe to overwrite this buffer on the next tile.
    final input = _tileBuffer!;
    input.fillRange(0, input.length, 0.0);
    final lut = _u8ToFloat;
    final gBase = pixelCount;
    final bBase = pixelCount * 2;

    for (int ty = 0; ty < actualH; ty++) {
      final srcRowBase = ((tileY + ty) * srcWidth + tileX) * 4;
      final dstRowBase = ty * tileSize;
      for (int tx = 0; tx < actualW; tx++) {
        final p = srcRowBase + tx * 4;
        final i = dstRowBase + tx;
        input[i] = lut[srcRgba[p]];
        input[gBase + i] = lut[srcRgba[p + 1]];
        input[bBase + i] = lut[srcRgba[p + 2]];
      }
    }

    return OrtValueTensor.createTensorWithDataList(
        input, [1, 3, tileSize, tileSize]);
  }

  // ---------------------------------------------------------------------------
  // Inference + output access
  // ---------------------------------------------------------------------------

  /// Run the model on [input] and hand the output's CHW float data to
  /// [consume] before the output tensor is released.
  ///
  /// [consume] receives `(chw, rowStride, planeStride)` where `chw` is the flat
  /// NCHW buffer, `rowStride == outW`, and `planeStride == outW·outH`. The
  /// buffer may be a zero-copy view over native memory, so [consume] must fully
  /// read what it needs synchronously and must not retain the buffer.
  Future<R> _infer<R>(
    OrtValueTensor input,
    int outH,
    int outW,
    R Function(Float32List chw, int rowStride, int planeStride) consume,
  ) async {
    // Reuse pre-allocated OrtRunOptions — avoids native alloc/free per tile.
    List<OrtValue?>? outputs;
    try {
      outputs = await _session!.runAsync(_runOptions!, {inputName: input});
      if (outputs == null || outputs.isEmpty || outputs[0] == null) {
        throw Exception('Model returned no output');
      }
      final out = outputs[0]!;
      final count = 3 * outH * outW;
      final chw = _outputAsFloat32View(out, count) ??
          _nestedToFloat32(out.value as List, outH, outW);
      return consume(chw, outW, outH * outW);
    } finally {
      outputs?.forEach((v) => v?.release());
    }
  }

  /// Zero-copy [Float32List] view over the output tensor's native float buffer,
  /// valid only until [out] is released. Returns null (triggering the
  /// [_nestedToFloat32] fallback) if the FFI fast-path is unavailable.
  Float32List? _outputAsFloat32View(OrtValue out, int elementCount) {
    final getData = _getTensorData;
    if (getData == null) return null;
    try {
      final pp = calloc<ffi.Pointer<ffi.Void>>();
      try {
        OrtStatus.checkOrtStatus(getData(out.ptr, pp));
        final dataPtr = pp.value;
        if (dataPtr == ffi.nullptr) return null;
        return dataPtr.cast<ffi.Float>().asTypedList(elementCount);
      } finally {
        calloc.free(pp);
      }
    } catch (_) {
      return null;
    }
  }

  /// Fallback: flatten the nested `List<List<List<List<double>>>>` returned by
  /// `OrtValue.value` into a contiguous NCHW [Float32List].
  Float32List _nestedToFloat32(List value, int outH, int outW) {
    final out = Float32List(3 * outH * outW);
    final batch = value[0] as List;
    int idx = 0;
    for (int c = 0; c < 3; c++) {
      final plane = batch[c] as List;
      for (int y = 0; y < outH; y++) {
        final row = plane[y] as List;
        for (int x = 0; x < outW; x++) {
          out[idx++] = (row[x] as num).toDouble();
        }
      }
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Output decoding (full-image path)
  // ---------------------------------------------------------------------------

  /// Convert a flat NCHW float buffer to packed RGBA bytes.
  Uint8List _chwToPixels(
    Float32List chw,
    int rowStride,
    int planeStride,
    int width,
    int height,
  ) {
    final pixels = Uint8List(width * height * 4);
    final gBase = planeStride;
    final bBase = 2 * planeStride;
    int idx = 0;
    for (int y = 0; y < height; y++) {
      final row = y * rowStride;
      for (int x = 0; x < width; x++) {
        final o = row + x;
        pixels[idx++] = _toByte(chw[o]);
        pixels[idx++] = _toByte(chw[gBase + o]);
        pixels[idx++] = _toByte(chw[bBase + o]);
        pixels[idx++] = 255;
      }
    }
    return pixels;
  }

  /// Clamp a 0–1 float to a 0–255 byte with rounding.
  int _toByte(double v) => v <= 0.0
      ? 0
      : v >= 1.0
          ? 255
          : (v * 255.0 + 0.5).toInt();

  Future<ui.Image> _decode(Uint8List pixels, int width, int height) {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
        pixels, width, height, ui.PixelFormat.rgba8888, completer.complete);
    return completer.future;
  }
}