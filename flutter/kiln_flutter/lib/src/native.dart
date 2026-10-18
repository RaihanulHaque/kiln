import 'dart:ffi';
import 'dart:io';

import 'kiln_bindings_generated.dart';

/// Loads libkiln for the current platform.
///
/// Android: bundled by the plugin's Gradle build as `libkiln.so`, built for
/// armv8.2 with dot product and fp16. Phones without those instructions (Armv8.0,
/// e.g. Snapdragon 680) get `libkiln_armv8.so`, built for plain armv8.0.
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
    if (Platform.isAndroid) {
      return DynamicLibrary.open(_androidHasArmv82() ? 'libkiln.so' : 'libkiln_armv8.so');
    }
    if (Platform.isMacOS) return DynamicLibrary.open('libkiln.dylib');
    if (Platform.isLinux) return DynamicLibrary.open('libkiln.so');
    if (Platform.isIOS) return DynamicLibrary.process();
    throw UnsupportedError('kiln: unsupported platform ${Platform.operatingSystem}');
  }

  /// True when the CPU has what `armv8.2-a+dotprod+fp16` needs: the dot product
  /// (`asimddp`), vector half precision (`asimdhp`) and scalar half precision (`fphp`).
  /// Android lists these in the `Features` line of /proc/cpuinfo. If the file cannot
  /// be read, answer false: the baseline library runs everywhere, just slower.
  static bool _androidHasArmv82() {
    try {
      final cpuinfo = File('/proc/cpuinfo').readAsStringSync();
      final line = RegExp(r'^Features\s*:(.*)$', multiLine: true).firstMatch(cpuinfo);
      if (line == null) return false;
      final flags = line.group(1)!.split(RegExp(r'\s+')).toSet();
      return flags.containsAll(const ['asimddp', 'asimdhp', 'fphp']);
    } on FileSystemException {
      return false;
    }
  }
}
