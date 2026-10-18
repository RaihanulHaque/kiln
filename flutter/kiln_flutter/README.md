# kiln_flutter

Flutter bindings for [kiln](https://github.com/RaihanulHaque/kiln), an on-device LLM runtime that
is honest about what it runs on. Android (arm64) today; iOS is in progress.

<p align="center"><img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/kiln_promo.gif" width="600" alt="kiln: a chat reply streams while a decision and an embedding search run beside it, then the measured numbers"></p>

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

## Install

```bash
flutter pub add kiln_flutter
```

Then set `minSdk = 29` in `android/app/build.gradle.kts`, put a GGUF model on the device, and
run on a real arm64 Android phone. The first build compiles the native library and takes two to
three minutes. The
[getting-started guide](https://github.com/RaihanulHaque/kiln/blob/main/docs/GETTING_STARTED.md)
walks through every step, from an empty project to a first chat and a first decision.

To track the repo instead of a release, depend on it from GitHub:

```yaml
dependencies:
  kiln_flutter:
    git:
      url: https://github.com/RaihanulHaque/kiln.git
      path: flutter/kiln_flutter
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

The example app has a Decide screen that uses this. `chat` on a decision model and `decide` on a chat model both throw a `KilnException` with
`KILN_ERR_NOT_SUPPORTED`. The d1-omni weights are under Liquid AI's `lfm1.0` license, not MIT.

## Embedding models

An embedding model does not write text either. It turns a text into a list of numbers, a
vector, and texts with a similar meaning get similar vectors. That is what search over the
user's own notes needs: embed every note once, embed the query, and sort by similarity.

```dart
final model = await kiln.loadModel('/path/to/embeddinggemma-2-UD-Q4_K_XL.gguf');
print(model.info.isEmbeddingModel);  // true
print(model.info.nEmbd);             // 768 numbers per vector
final ctx = await model.createContext();

final notes = ['Knead the dough and let it rise for an hour.', 'The train leaves at nine.'];
final vectors = [for (final n in notes) await ctx.embed(n, kind: KilnEmbedKind.document)];

final query = await ctx.embed('how do I bake bread', kind: KilnEmbedKind.query);
for (var i = 0; i < notes.length; i++) {
  print('${query.similarity(vectors[i]).toStringAsFixed(3)}  ${notes[i]}');   // 0.72 and 0.57
}
```

`vectors[i].vector` is a `Float32List` you can store; `KilnEmbedding.cosine(a, b)` compares two
stored vectors. `dimensions: 256` keeps a shorter vector, for models trained for it. A text
longer than the context throws instead of being cut, so split long documents into pieces.

Some embedding models also take **images and audio** and place them in the same space as text,
so a typed query can find a photo or a voice note. They need a second file, the projector:

```dart
await model.loadProjector('/path/to/mmproj-embeddinggemma-2-Q8_0.gguf');
final photo = await ctx.embedMedia([KilnMedia.file('/path/to/holiday.jpg')]);
final voice = await ctx.embedMedia([KilnMedia.bytes(wavBytes)]);
final query = await ctx.embed('a lake between mountains', kind: KilnEmbedKind.query);
print(query.similarity(photo));
```

On a Pixel 8 a short text takes 30 to 80 ms, a small image about 2.5 s and a 3-second audio
clip about half a second; the projector adds roughly 0.6 GB of RAM. Tested with
EmbeddingGemma 2 (text, images, audio; Apache 2.0) and nomic-embed-text v1.5 (text; Apache 2.0).

## Speech to text

A speech model writes down what is said in a recording. Two of the three supported families
come as a single file in whisper.cpp's format and need nothing else:

```dart
final model = await kiln.loadModel('/path/to/ggml-parakeet-tdt-0.6b-v3-q8_0.bin');
print(model.info.isSpeechModel);   // true
final ctx = await model.createContext();

final t = ctx.transcribe(KilnMedia.file('/path/to/voice_note.wav'));   // wav, mp3 or flac
t.pieces.listen((words) => print(words));     // a sentence at a time
final stats = await t.stats;
print('${stats.audioSeconds} s of audio in ${stats.ms} ms, ${stats.speed}x real time');
```

Parakeet TDT 0.6B v3 (669 MB, 25 European languages) made the fewest mistakes of the small
models on English and is the fastest; Whisper base (60 MB) knows 99 languages and reports
which one it heard in `stats.language`. Qwen3-ASR 0.6B works too, as a GGUF plus its projector
(`loadProjector`), but on a Pixel 8 it is slower than the clip it transcribes.

Each call takes one clip, from a file or its bytes. The example app records with the `record`
package to a 16 kHz wav and hands the file over.

## Text to speech

The other direction. Pocket TTS (Kyutai, English) comes as a text model plus a projector, and
imitates the voice of a short recording:

```dart
final model = await kiln.loadModel('/path/to/pocket-tts-en.gguf');
await model.loadProjector('/path/to/mmproj-pocket-tts-en.gguf');   // generatesAudio: true
final ctx = await model.createContext();

final s = ctx.speak('Your parcel arrives on Thursday.', voice: KilnMedia.file('/path/to/me.wav'));
s.pieces.listen((p) => play(p.samples, p.sampleRate));   // about a second at a time
final stats = await s.stats;
print('${stats.audioSeconds} s of speech, ${stats.speed}x real time');
```

Or `await s.wav` for the whole utterance as wav bytes. On a Pixel 8 a 3-second sentence takes
3.4 to 4.1 s (0.7 to 0.9x real time; about a second of it is encoding the voice). Speaking does not touch the chat held in the context.

## Status

| platform | native | verified |
|---|---|---|
| Android arm64 | `libkiln.so` built by Gradle + CMake from this repo; `libkiln_armv8.so` is loaded instead on phones without dot product (Armv8.0) | Pixel 8: 26 integration tests (4 skipped: the speed and media groups, which passed at 0.3.1) (0.5.0); Galaxy S26 Ultra: 26 of 26 at 0.3.1 |
| iOS 15+ | not in the pub.dev package; from a clone, `kiln.xcframework` via `scripts/build_ios.sh`, Metal | **In progress, not supported yet.** iPhone 16 Pro at v0.1.0: 7 of 9 integration tests; not re-run for v0.2.0 |

Desktop (macOS/Linux) works for development by passing a host build:
`Kiln.create(libraryPath: '<repo>/build/host/libkiln.dylib')`.

## Example app

<p>
<img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_library.png" width="170" alt="Model library"> <img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_chat_thinking.png" width="170" alt="Thinking, streamed"> <img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_chat_reply.png" width="170" alt="Chat reply with stats"> <img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_decide.png" width="170" alt="Decide screen"> <img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_search_media.png" width="170" alt="Search screen with notes, photos and audio"> <img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_search_result.png" width="170" alt="A typed question finds a spoken clip"> <img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_transcribe.png" width="170" alt="Transcribe screen: two clips written down by Parakeet at 9x real time"> <img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_speak.png" width="170" alt="Speak screen: Pocket TTS says a sentence in the sample voice"> <img src="https://raw.githubusercontent.com/RaihanulHaque/kiln/main/docs/img/app_library_light.png" width="170" alt="Light theme">
</p>

`example/` is a small chat app in one file, with a light and a dark theme that follow the system.

- **Model library**: what the probe found on the device, the models already on it, and three
  small models to download with one tap (progress, cancel, delete). Any other GGUF by link.
- **Chat**: streaming replies with basic markdown, a Think switch on models that support it,
  reasoning that folds away once the answer starts, speed and cache reuse under each reply,
  stop, copy, and new chat.
- **Decide**: opens instead of the chat when the loaded model is a decision model (LFM2
  d1-omni is in the download list). Type a text and a yes / no, choice or score question, or
  use one of three presets, and see the answer with a probability bar per option.
- **Search**: opens when the loaded model is an embedding model (EmbeddingGemma 2 is in the
  download list). A list of notes and a question; the notes are ranked by meaning. With the
  model's encoders switched on (a 555 MB download, one tap) you can also add photos and audio
  clips from the phone, and they are ranked in the same list as the notes.

`--dart-define=KILN_THEME=light` or `dark` forces a theme. Besides the downloader, a model can
be copied to an Android device with

```bash
adb push model.gguf /sdcard/Android/data/dev.kiln.kiln_flutter_example/files/
```

## Integration test

```bash
cd example && flutter test integration_test -d <device-id>
```

Downloads a 229 MB model on first run. Covers probe, bad files, info, template, single turn,
KV reuse on turn 2, cancel, busy-context guard, context overflow, Qwen3.5 thinking
channels (fetches Qwen, 533 MB), the d1-omni decision model (fetches it, 407 MB) and the
EmbeddingGemma 2 embedding model (176 MB). `--dart-define=KILN_MEDIA=true` adds the image and
audio test, which fetches a 555 MB projector. `--dart-define=KILN_SPEED=true` adds a 3-run speed test per
model.

## Building the native side

- **From pub.dev**: nothing to do. The package carries the C core and the pinned llama.cpp, and
  Gradle builds them.
- **Inside this repo**: nothing to do. `src/CMakeLists.txt` points at the repo root.
- **Publishing**: `scripts/vendor_flutter.sh` copies the core and the pinned llama.cpp into
  `src/core/` (35 MB) so the package builds alone; `--check` proves it by building the example
  APK with the repo root hidden.
- **iOS**: `scripts/build_ios.sh` produces `ios/Frameworks/kiln.xcframework` (device + simulator;
  `--device-only` to skip the simulator slice). Rebuild it after any change to the C core.

## Regenerating bindings

The C header is the contract. After changing `include/kiln/kiln.h`:

```bash
dart run ffigen --config ffigen.yaml
```
