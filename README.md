# Flutter Super Resolution

A Flutter library for upscaling images using ONNX Runtime neural networks with advanced tiling support.

## Platform Support

- Windows
- macOS
- iOS
- Android (Only ARM devices)

## Features

- High-quality image upscaling using machine learning models
- Supports full image and tiled processing
- Customizable tile size and overlap
- Progress tracking during upscaling
- Optimized for mobile devices

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_super_resolution: ^1.0.0
  onnxruntime: ^latest_version
```

## Downloading Models

To download compatible super-resolution models, visit the AI Zoo:

[https://www.neuralfulai.com/aizoo?type=upscale&compatibility=flutter](https://www.neuralfulai.com/aizoo?type=upscale&compatibility=flutter)

This link is pre-filtered to show models with **type: upscale** that are **compatible with Flutter**.

## Usage

```dart
final upscaler = FlutterUpscaler(
    tileSize: 128,   // Optional: Customize tile processing
    overlap: 8       // Optional: Prevent tile seams
);

// Initialize model from assets
await upscaler.initializeModel('assets/super_resolution_model.onnx');

// Upscale image
final upscaledImage = await upscaler.upscaleImage(
sourceImage,
scale: 2,        // Scale factor (2x, 4x, etc.)
onProgress: (progress, message) {
print('$message: ${(progress * 100).toStringAsFixed(1)}%');
}
);
```

## Parameters

- `tileSize`: Size of processing tiles (default: 128)
- `overlap`: Pixel overlap between tiles to reduce seam artifacts (default: 8)

## Methods

- `initializeModel(modelPath)`: Load ONNX model from Flutter assets
- `initializeModelFromFile(filePath)`: Load ONNX model from device file
- `upscaleImage(image, scale)`: Upscale image with optional progress tracking
- `dispose()`: Release model resources

## Requirements

- Flutter SDK
- ONNX Runtime Flutter plugin
- Pre-trained ONNX super-resolution model

## Performance Tips

- Use smaller tile sizes for memory-constrained devices
- Adjust overlap to balance processing quality and speed

## License

MIT License