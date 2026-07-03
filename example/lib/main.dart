import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui' as ui;
import 'package:flutter_super_resolution/flutter_super_resolution.dart';

void main() => runApp(const SuperResolutionDemoApp());

class SuperResolutionDemoApp extends StatelessWidget {
  const SuperResolutionDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Super Resolution Demo',
      home: _DemoPage(),
    );
  }
}

class _DemoPage extends StatefulWidget {
  const _DemoPage();

  @override
  State<_DemoPage> createState() => _DemoPageState();
}

class _DemoPageState extends State<_DemoPage> {
  ui.Image? _original;
  ui.Image? _upscaled;
  double _progress = 0.0;
  String _progressMessage = '';
  bool _isProcessing = false;

  // Kept alive so the model is not reloaded on every button tap.
  FlutterUpscaler? _upscaler;

  @override
  void initState() {
    super.initState();
    _loadDefaultImage();
    _initUpscaler();
  }

  @override
  void dispose() {
    _upscaler?.dispose();
    // Dispose native ui.Image handles to release GPU/CPU memory.
    _original?.dispose();
    _upscaled?.dispose();
    super.dispose();
  }

  Future<void> _loadDefaultImage() async {
    try {
      final data = await rootBundle.load('assets/sample_image.jpg');
      final codec =
          await ui.instantiateImageCodec(data.buffer.asUint8List());
      final frame = await codec.getNextFrame();
      codec.dispose();
      if (mounted) {
        setState(() {
          _original?.dispose(); // release the previous image if any
          _original = frame.image;
        });
      } else {
        frame.image.dispose(); // widget was removed before we could use it
      }
    } catch (e) {
      _showError('Failed to load sample image: $e');
    }
  }

  Future<void> _initUpscaler() async {
    try {
      // tileSize: 64 on older iPhones (6/7/SE gen 1) to keep per-tile RAM low.
      // maxOutputMemoryMB: combined size of the two CPU output buffers;
      // 128 MB handles up to ~2048×2048 output (1024×1024 src at 2×).
      final upscaler = FlutterUpscaler(
        tileSize: 128,
        overlap: 16,
        maxOutputMemoryMB: 128,
      );
      await upscaler.initializeModel('assets/super_resolution_model.onnx');
      if (mounted) {
        setState(() => _upscaler = upscaler);
      } else {
        upscaler.dispose();
      }
    } catch (e) {
      _showError('Failed to load SR model: $e');
    }
  }

  Future<void> _upscaleImage() async {
    if (_original == null || _upscaler == null) return;

    setState(() {
      _isProcessing = true;
      _progress = 0.0;
      _progressMessage = '';
    });

    try {
      final result = await _upscaler!.upscaleImage(
        _original!,
        2,
        onProgress: (progress, message) {
          if (mounted) setState(() {
            _progress = progress;
            _progressMessage = message;
          });
        },
      );

      if (result == null) return;

      if (mounted) {
        setState(() {
          _upscaled?.dispose(); // release the previous upscaled image
          _upscaled = result;
        });
      } else {
        result.dispose(); // widget unmounted mid-operation
      }
    } catch (e) {
      _showError('Upscaling failed: $e');
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  /// Shows a [SnackBar]. Schedules via [addPostFrameCallback] when called
  /// from [initState] (before the first frame renders) so the [Scaffold] is
  /// guaranteed to be in the tree.
  void _showError(String message) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text(message)));
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Super Resolution Demo')),
      body: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 24),
            if (_original != null)
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _imageCard('Original', _original!),
                  if (_upscaled != null) _imageCard('Upscaled ×2', _upscaled!),
                ],
              ),
            if (_isProcessing)
              Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    LinearProgressIndicator(value: _progress),
                    const SizedBox(height: 8),
                    Text(_progressMessage,
                        style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: (_isProcessing || _upscaler == null || _original == null)
                  ? null
                  : _upscaleImage,
              child: Text(_isProcessing ? 'Processing…' : 'Upscale Image'),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _imageCard(String label, ui.Image image) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 4),
        RawImage(image: image, width: 200, height: 200, fit: BoxFit.contain),
        const SizedBox(height: 4),
        Text('${image.width}×${image.height}',
            style: const TextStyle(fontSize: 11, color: Colors.grey)),
      ],
    );
  }
}
