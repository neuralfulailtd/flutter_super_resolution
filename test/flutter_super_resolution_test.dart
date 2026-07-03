import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_super_resolution/flutter_super_resolution.dart';

void main() {
  group('FlutterUpscaler constructor', () {
    test('defaults are applied', () {
      final u = FlutterUpscaler();
      expect(u.tileSize, 128);
      expect(u.overlap, 16);
      expect(u.inputName, 'input');
      expect(u.maxOutputMemoryMB, 128);
    });

    test('custom parameters are stored', () {
      final u = FlutterUpscaler(
          tileSize: 256, overlap: 16, inputName: 'x', maxOutputMemoryMB: 64);
      expect(u.tileSize, 256);
      expect(u.overlap, 16);
      expect(u.inputName, 'x');
      expect(u.maxOutputMemoryMB, 64);
    });

    test('asserts when overlap >= tileSize', () {
      expect(() => FlutterUpscaler(tileSize: 64, overlap: 64),
          throwsA(isA<AssertionError>()));
      expect(() => FlutterUpscaler(tileSize: 64, overlap: 65),
          throwsA(isA<AssertionError>()));
    });

    test('smallest valid overlap is tileSize - 1', () {
      expect(() => FlutterUpscaler(tileSize: 64, overlap: 63), returnsNormally);
    });
  });

  group('FlutterUpscaler lifecycle', () {
    test('dispose before init does not throw', () {
      expect(() => FlutterUpscaler().dispose(), returnsNormally);
    });

    test('double dispose does not throw', () {
      final u = FlutterUpscaler();
      u.dispose();
      expect(() => u.dispose(), returnsNormally);
    });

    // upscaleImage requires a real ui.Image, which cannot be constructed in
    // pure unit tests without a Flutter engine. The StateError is thrown
    // synchronously at the start of the async function body (before the image
    // parameter is accessed), so passing null as dynamic is safe here.
    test('upscaleImage throws StateError when model not loaded', () async {
      final u = FlutterUpscaler();
      // expectLater correctly awaits the rejected Future — unlike throwsStateError
      // with a non-awaited async lambda, which would be a false positive.
      await expectLater(
        () => u.upscaleImage(null as dynamic, 2),
        throwsStateError,
      );
    });
  });
}
