# kiln_flutter

Flutter bindings for [kiln](../../README.md), an on-device LLM runtime that is honest about
what it runs on. Android and iOS.

```dart
final kiln  = await Kiln.create();
final model = await kiln.loadModel('/path/to/model.gguf');
print(model.info);          // qwen35 Q4_K_M, 0.75B, via llama.cpp/cpu, thinking

final ctx = await model.createContext();
final gen = ctx.chat([KilnChatMessage.user('Why is the sky blue?')],
    enableThinking: true, maxReasoningTokens: 512);   // budget: small models rarely stop thinking

await for (final piece in gen.pieces) {
  if (piece.isReasoning) showThinking(piece.text); else showAnswer(piece.text);
}
final stats = await gen.stats;  // tok/s, cached prefix, reasoning tokens, why it stopped
gen.cancel();                   // from anywhere, any time
```

Everything blocking runs on a background isolate; the UI thread only receives pieces. You may
await `stats` without listening to `pieces`. Pass the full history to `chat` every turn and
kiln reuses whatever prefix the KV cache already holds.

## Status

| platform | native | verified |
|---|---|---|
| Android arm64 | `libkiln.so` built by Gradle + CMake from this repo | Pixel 8: 10 of 10 integration tests, speed run |
| iOS 15+ | `kiln.xcframework` from `scripts/build_ios.sh`, vendored by the podspec, Metal | iPhone 16 Pro: 7 of 9 before a Dart fix, re-run pending |

Desktop (macOS/Linux) works for development by passing a host build:
`Kiln.create(libraryPath: '<repo>/build/host/libkiln.dylib')`.

## Example app

`example/` is a chat UI: device probe, model picker with in-app download, thinking toggle for
models that support it, collapsible reasoning, per-reply stats, stop button. Get a model
onto an Android device with the downloader or

```bash
adb push model.gguf /sdcard/Android/data/dev.kiln.kiln_flutter_example/files/
```

## Integration test

```bash
cd example && flutter test integration_test -d <device-id>
```

Downloads a 229 MB model on first run. Covers probe, bad files, info, template, single turn,
KV reuse on turn 2, cancel, busy-context guard, context overflow, and Qwen3.5 thinking
channels (fetches Qwen, 533 MB). `--dart-define=KILN_SPEED=true` adds a 3-run speed test per
model. iOS needs Rosetta on Apple-silicon Macs for Flutter's device bridge.

## Building the native side

- **Inside this repo**: nothing to do. `src/CMakeLists.txt` points at the repo root.
- **For pub.dev**: `scripts/vendor_flutter.sh` copies the core and the pinned llama.cpp into
  `src/core/` (35 MB) so the package builds alone; `--check` proves it by building the example
  APK with the repo root hidden.
- **iOS**: `scripts/build_ios.sh` produces `ios/Frameworks/kiln.xcframework` (device + simulator;
  `--device-only` to skip the simulator slice). Rebuild it after any change to the C core.

## Regenerating bindings

The C header is the contract. After changing `include/kiln/kiln.h`:

```bash
dart run ffigen --config ffigen.yaml
```
