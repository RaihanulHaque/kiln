# kiln_flutter

Flutter bindings for [kiln](../../README.md), an on-device LLM runtime that is honest about
what it runs on. Android first. The native core is built from this repo by the plugin's
Gradle/CMake step, so there is one source of truth.

```dart
final kiln  = await Kiln.create();
final model = await kiln.loadModel('/path/to/model.gguf');
print(model.info);          // qwen35 Q4_K_M, 0.75B, via llama.cpp/cpu, thinking

final ctx = await model.createContext();
final gen = ctx.chat([KilnChatMessage.user('Why is the sky blue?')], enableThinking: true);

await for (final piece in gen.pieces) {
  if (piece.isReasoning) showThinking(piece.text); else showAnswer(piece.text);
}
final stats = await gen.stats;  // tok/s, cached prefix, reasoning tokens, why it stopped
gen.cancel();                   // from anywhere, any time
```

Everything blocking runs on a background isolate; the UI thread only receives pieces.
Pass the full history to `chat` every turn and kiln reuses whatever prefix the KV cache
already holds.

## Example app

`example/` is a chat UI that shows the device probe, what actually loaded, a thinking toggle
for models that support it, collapsible reasoning, and per-reply stats. Get a model onto the
device with the in-app downloader or

```bash
adb push model.gguf /sdcard/Android/data/dev.kiln.kiln_flutter_example/files/
```

## Desktop development

On macOS or Linux, build the host core (`cmake --preset host && cmake --build --preset host`)
and pass its path: `Kiln.create(libraryPath: '<repo>/build/host/libkiln.dylib')`. iOS is not
wired up yet.

## Regenerating bindings

The C header is the contract. After changing `include/kiln/kiln.h`:

```bash
dart run ffigen --config ffigen.yaml
```
