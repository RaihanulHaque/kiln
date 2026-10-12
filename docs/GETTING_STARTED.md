# Getting started with kiln in a Flutter app

From an empty Flutter project to a model answering on a phone. About ten minutes, most of it
the first native build.

What you need:

- A Flutter version that ships Dart 3.12 or newer (tested with Flutter 3.47), with the Android
  SDK and NDK installed (`flutter doctor` is green for Android)
- A real **arm64 Android phone** running Android 10 or newer. The x86 emulator is not supported.
- Around 250 MB free on the phone for the smallest model

iOS is in progress and not supported yet; see [iOS](#ios).

## 1. Add the dependency

```bash
flutter pub add kiln_flutter
flutter pub add path_provider http    # to find a folder for the model file, and to download it
```

That is all: nothing to clone. The package carries the native code, and the first build
compiles it.

To follow the repo instead of a release, depend on it from GitHub:

```yaml
dependencies:
  kiln_flutter:
    git:
      url: https://github.com/RaihanulHaque/kiln.git
      path: flutter/kiln_flutter
```

## 2. Two Android settings

In `android/app/build.gradle.kts`, raise the minimum Android version to 10 (API 29):

```kotlin
android {
    defaultConfig {
        minSdk = 29
    }
}
```

Without it the build stops with a message asking for it.

If your app downloads the model, it needs the internet permission in
`android/app/src/main/AndroidManifest.xml`, above `<application>`:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

## 3. Get a model onto the phone

kiln runs models in the GGUF format and does not ship one. Your app either downloads a model on
first launch or you copy one over USB while developing. Three small ones that are tested:

| model | size | good for |
|---|---|---|
| [LFM2.5 350M Q4_K_M](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf) | 229 MB | fastest, short answers |
| [Gemma 3 270M Q8_0](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf) | 292 MB | tiny and quick |
| [Qwen3.5 0.8B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf) | 533 MB | smartest, can think first |

Each model has its own license; read it before you ship one.

Download it once into a folder your app owns:

```dart
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

Future<File> fetchModel(String url) async {
  final dir = (await getExternalStorageDirectory()) ?? await getApplicationDocumentsDirectory();
  final file = File('${dir.path}/${Uri.parse(url).pathSegments.last}');
  if (file.existsSync()) return file;

  // Write to a .part file and rename at the end, so a dropped connection never leaves
  // half a model that would be loaded next time. Mobile connections drop: try three times.
  final part = File('${file.path}.part');
  for (var attempt = 1; ; attempt++) {
    final client = http.Client();
    try {
      final resp = await client.send(http.Request('GET', Uri.parse(url)));
      if (resp.statusCode != 200) throw HttpException('HTTP ${resp.statusCode}');
      await resp.stream.pipe(part.openWrite());
      if (resp.contentLength != null && part.lengthSync() != resp.contentLength) {
        throw const HttpException('download was cut short');
      }
      return await part.rename(file.path);
    } catch (_) {
      if (part.existsSync()) part.deleteSync();
      if (attempt == 3) rethrow;
    } finally {
      client.close();
    }
  }
}
```

While developing you can skip the download and copy a file over USB. Run the app once so its
folder exists, then:

```bash
adb push model.gguf /sdcard/Android/data/<your.application.id>/files/
```

## 4. First chat

```dart
import 'package:kiln_flutter/kiln_flutter.dart';

final kiln = await Kiln.create();            // once per app
print(await kiln.probeDevice());              // Google Tensor G3 (arm64-v8a), 9 cores (5 fast), 7.9 GB RAM

final file = await fetchModel(
    'https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf');
final model = await kiln.loadModel(file.path);
print(model.info);                            // lfm2 Q4_K - Medium, 0.35B params, 0.23 GB, via llama.cpp/cpu

final ctx = await model.createContext(params: const KilnContextParams(nCtx: 4096));

final history = <KilnChatMessage>[const KilnChatMessage.user('Why is the sky blue? One sentence.')];
final gen = ctx.chat(history);

final reply = StringBuffer();
await for (final piece in gen.pieces) {
  if (!piece.isReasoning) reply.write(piece.text);   // update your UI here
}
print(await gen.stats);                       // tok/s, cached tokens, why it stopped
```

Things to know:

- **Nothing blocks the UI.** Loading and generating run on a background isolate; your code
  only receives pieces.
- **Keep the history and pass all of it every turn.** Add the reply as
  `KilnChatMessage.assistant(reply.toString())`, add the next user message, and call `chat`
  again. kiln re-reads only what is new.
- **One generation per context at a time.** A second `chat` while one is running throws.
- **`gen.cancel()` stops a reply.** `gen.stats` still completes, with `stoppedByUser` set.
- **Free what you load.** `await ctx.dispose(); await model.dispose();` before loading another
  model. A 0.8B model holds about 900 MB of RAM.
- **Errors are `KilnException`s** with a `status` and a `message`: a file that is not a model,
  a prompt longer than the context, and so on.

## 5. Thinking models

Some models can reason before they answer. `model.info.supportsThinking` tells you. The
reasoning arrives on its own channel, so you can show it separately or hide it:

```dart
final gen = ctx.chat(history,
    enableThinking: true,
    maxReasoningTokens: 512,                                  // budget, see below
    sampling: const KilnSamplingParams(repeatPenalty: 1.0));

await for (final piece in gen.pieces) {
  if (piece.isReasoning) showThinking(piece.text); else showAnswer(piece.text);
}
```

Always set `maxReasoningTokens` with small models. Qwen3.5 0.8B almost never stops thinking by
itself; with a budget, kiln closes the thinking after that many tokens and the model answers.
`stats.reasoningTruncated` tells you when that happened. Turning the repeat penalty off for
thinking turns makes the answer loop less often.

## 6. Decision models

A decision model does not write text. It reads a text and one question and returns a
probability for every possible answer, in a few hundred milliseconds. Use it to classify,
route or rate something. Supported today: LFM2
[d1-omni 600M](https://huggingface.co/LiquidAI/d1-omni-600M-GGUF/resolve/main/d1-omni-600M-Q8_0.gguf)
(407 MB, Liquid AI's `lfm1.0` license).

```dart
final model = await kiln.loadModel(d1File.path);
print(model.info.isDecisionModel);   // true
final ctx = await model.createContext();
const mail = 'My blender arrived broken. I want my money back today.';

final refund = await ctx.decide(KilnQuestion.yesNo('Is the customer asking for a refund?'), state: mail);
print(refund.yes);                   // true
print(refund.value);                 // 0.997, the probability of yes

final team = await ctx.decide(
    const KilnQuestion.choice('Which team should handle this?', [
      KilnOption('billing', 'Payments, refunds and invoices'),
      KilnOption('shipping', 'Delivery status and lost parcels'),
    ]),
    state: mail);
print(team.bestKey);                 // billing
print(team.probabilities);           // one per option, summing to 1

final urgency = await ctx.decide(
    KilnQuestion.score('How urgent is this?', ['Can wait', 'Today', 'Blocking the customer now']),
    state: mail);
print(urgency.bestKey);              // Today
```

`chat` on a decision model and `decide` on a chat model both throw a `KilnException`.

## 7. Embedding models: search by meaning

*Needs kiln_flutter 0.3.0. pub.dev still has 0.2.0, so until 0.3.0 is published use the GitHub
dependency from step 1 for this part.*

An embedding model turns a text into a list of numbers, a vector. Texts that mean similar
things get similar vectors, even when they share no words. Embed your notes once, embed what
the user types, and sort by similarity: that is search by meaning, fully on the phone.
Tested: [EmbeddingGemma 2](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-UD-Q4_K_XL.gguf)
(176 MB, Apache 2.0).

```dart
final model = await kiln.loadModel(embedFile.path);
print(model.info.isEmbeddingModel);  // true
final ctx = await model.createContext();

final notes = ['Knead the dough and let it rise for an hour.', 'The train leaves at nine.'];
final vectors = [for (final n in notes) await ctx.embed(n, kind: KilnEmbedKind.document)];

final query = await ctx.embed('how do I bake bread', kind: KilnEmbedKind.query);
for (var i = 0; i < notes.length; i++) {
  print('${query.similarity(vectors[i]).toStringAsFixed(3)}  ${notes[i]}');
}
```

- Use `KilnEmbedKind.document` for what is searched and `KilnEmbedKind.query` for what the
  user types. The model was trained with a different prefix for each, and kiln adds it.
- `vectors[i].vector` is a `Float32List`. Save it next to the note, and compare saved vectors
  later with `KilnEmbedding.cosine(a, b)`. Vectors from different models cannot be compared.
- A text longer than the context (2048 tokens by default) throws `KILN_ERR_CONTEXT_FULL`.
  Split long documents into paragraphs and embed each one.

With the model's projector file
([mmproj](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/mmproj-Q8_0.gguf),
555 MB) the same model also embeds images and audio into the same space, so a typed query can
find a photo or a voice note:

```dart
await model.loadProjector(projectorFile.path);
final photo = await ctx.embedMedia([KilnMedia.file('/path/to/holiday.jpg')]);
final query = await ctx.embed('a lake between mountains', kind: KilnEmbedKind.query);
print(query.similarity(photo));
```

An image takes a few seconds on a phone processor and audio about a sixth of its length, so
embed media in the background when it is added, not while the user is searching.

## 8. Build and run

```bash
flutter run --release
```

When you build a file to hand out, build it for arm64 only. kiln's native library exists for
arm64 alone, so the 32-bit and x86 parts of a default build could not run it:

```bash
flutter build apk --release --target-platform android-arm64
flutter build appbundle --release --target-platform android-arm64
```

The first build compiles the native code and takes two to three minutes; later builds are quick.
Measure speed with a release build on a cool phone: a phone that has been working hard for a
few minutes slows its processor down and can run at half speed.

## A complete app to read

[`flutter/kiln_flutter/example`](../flutter/kiln_flutter/example) is a full app in one file:
model library with downloads, chat with thinking, a Decide screen and a Search screen. To run it:

```bash
git clone --recursive https://github.com/RaihanulHaque/kiln.git
cd kiln/flutter/kiln_flutter/example
flutter run --release
```

## iOS

**In progress. Not supported yet.**

kiln built and ran on an iPhone 16 Pro at v0.1.0, using the phone's GPU. For v0.2.0 it has not
been run again, so the pub.dev package is Android only.

If you want to try it anyway, on a Mac with Xcode and a clone of this repo:

```bash
scripts/build_ios.sh          # builds flutter/kiln_flutter/ios/Frameworks/kiln.xcframework
```

Add `ios: {ffiPlugin: true}` back under `platforms` in the plugin's `pubspec.yaml`, then depend
on the plugin by path (`kiln_flutter: {path: <clone>/flutter/kiln_flutter}`).
Expect rough edges, and please open an issue with what you find.

## When something goes wrong

| what you see | cause and fix |
|---|---|
| Build asks for `minSdkVersion 29` | Set `minSdk = 29`, step 2. |
| `Failed to load dynamic library 'libkiln.so'` | The device or emulator is not arm64. Use a real arm64 phone, and build with `--target-platform android-arm64`. |
| `KilnException(KILN_ERR_MODEL_LOAD)` | The file is missing, not a GGUF, or a download that was cut short. Check its size and download it again. |
| The app is killed while loading | The model does not fit in RAM. Try a smaller model or a smaller quantization. |
| `KilnException(KILN_ERR_CONTEXT_FULL)` | The conversation is longer than `nCtx`. Start a new chat with `ctx.reset()` or raise `nCtx`. |
| The model never answers with thinking on | Set `maxReasoningTokens`, step 5. |
| `KILN_ERR_NOT_SUPPORTED` from `chat`, `decide` or `embed` | The model is of another kind. Check `model.info.isDecisionModel` and `isEmbeddingModel`. |
| Much slower than the numbers in the README | Debug build, or a hot phone. Use `--release` and let the phone cool. |
| Bold markers like `**this**` in the output | Models write markdown. Render it, or see `_Markdown` in the example app. |

## What is and is not verified

kiln only claims what was run. As of v0.2.0: Pixel 8 (Tensor G3) with the full test suite and
the example app. Not yet tried: Snapdragon and MediaTek phones, phones with 4 GB of RAM, and
iOS on v0.2.0. If you run it on another device, an issue with the output of `probeDevice()` and
`stats` is very welcome. The [README](../README.md) has the status table.
