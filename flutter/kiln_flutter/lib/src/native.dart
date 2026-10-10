import 'dart:ffi';
import 'dart:io';

import 'kiln_bindings_generated.dart';

/// Loads libkiln for the current platform.
///
/// Android: bundled by the plugin's Gradle build as `libkiln.so`.
/// macOS / Linux (desktop dev): set [libraryPathOverride] to a host build of
/// libkiln, e.g. `<repo>/build/host/libkiln.dylib`, before first use.
class KilnNative {
  KilnNative._();

  static String? libraryPathOverride;
  static KilnBindings? _bindings;

  static KilnBindings get bindings => _bindings ??= KilnBindings(_open());

  static DynamicLibrary _open() {
    final override = libraryPathOverride;
    if (override != null) return DynamicLibrary.open(override);
    if (Platform.isAndroid) return DynamicLibrary.open('libkiln.so');
    if (Platform.isMacOS) return DynamicLibrary.open('libkiln.dylib');
    if (Platform.isLinux) return DynamicLibrary.open('libkiln.so');
    if (Platform.isIOS) return DynamicLibrary.process();
    throw UnsupportedError('kiln: unsupported platform ${Platform.operatingSystem}');
  }
}
