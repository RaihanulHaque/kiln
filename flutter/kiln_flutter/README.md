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

## Decision models

A decision model does not write text. It reads a state and one typed question, and returns a
probability for every option in a single pass (about half a second on a Pixel 8). Supported:
LFM2 d1-omni with a text state.

```dart
final model = await kiln.loadModel('/path/to/d1-omni-600M-Q8_0.gguf');
print(model.info.isDecisionModel);   // true
final ctx = await model.createContext();
const mail = 'My blender arrived broken. I want my money back today.';

final refund = await ctx.decide(KilnQuestion.yesNo('Is the customer asking for a refund?'), state: mail);
print(refund.yes);        // true
print(refund.value);      // 0.997, the probability of yes

final team = await ctx.decide(
    const KilnQuestion.choice('Which team should handle this?', [
      KilnOption('billing', 'Payments, refunds and invoices'),
      KilnOption('shipping', 'Delivery status and lost parcels'),
    ]),
    state: mail);
print(team.bestKey);        // billing
print(team.probabilities);  // one per option, in order, summing to 1

final urgency = await ctx.decide(
    KilnQuestion.score('How urgent is this?', ['Can wait', 'Today', 'Blocking the customer now']),
    state: mail);
print(urgency.value);     // expected level, 0 to 2
```

`chat` on a decision model and `decide` on a chat model both throw a `KilnException` with
`KILN_ERR_NOT_SUPPORTED`. The d1-omni weights are under Liquid AI's `lfm1.0` license, not MIT.

## Status

| platform | native | verified |
|---|---|---|
| Android arm64 | `libkiln.so` built by Gradle + CMake from this repo | Pixel 8: 16 of 16 integration tests, speed run |
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
KV reuse on turn 2, cancel, busy-context guard, context overflow, Qwen3.5 thinking
channels (fetches Qwen, 533 MB) and the d1-omni decision model (fetches it, 407 MB). `--dart-define=KILN_SPEED=true` adds a 3-run speed test per
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
