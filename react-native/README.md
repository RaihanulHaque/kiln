# kiln for React Native

Chat, decision, embedding, speech-to-text and text-to-speech models on the phone, from
JavaScript / TypeScript. **Android only** for now; iOS is not built (the iOS build of kiln
itself is still open).

It is a TurboModule (React Native's New Architecture, which is the only one since RN 0.82).
The native side is a thin Kotlin layer over the kiln Kotlin library in [`../android`](../android),
which does the real work and is tested there. JS never sees a native pointer. Models and contexts
are numbers, every blocking call runs on a background thread, and streamed text arrives as events
tagged with the call they belong to.

## Where it was checked, 2026-10-15 (Linux x86_64 desktop, no Android SDK)

| what | how | result |
|---|---|---|
| The spec (`src/NativeKiln.ts`) | React Native 0.87.1's codegen parsed it and generated `NativeKilnSpec.java` | ok |
| The Kotlin module | `kotlinc` against RN 0.87.1's `react-android` classes and Android API 36's `android.jar`, with the generated spec and `../android`'s Kotlin sources | compiles, no warnings |
| The JS layer (`src/kiln.ts`) | 9 Node tests against a fake native module (macOS and Linux) | pass |
| TypeScript | `tsc --noEmit` against RN 0.87.1's types | ok |
| **An app on a phone** | a new RN 0.87.1 app (`@react-native-community/cli init`, release build, built-in Kotlin off as the template ships), this package added by path, on the Pixel 8 (Android 17), 2026-10-15 | **all pass**: chat (LFM2.5 350M answered "Paris"), cache reuse (16 of 19 prompt tokens), stop from `onPiece` (4 tokens), abort then a new chat on the same context, Parakeet transcribing a 3-second clip with live text, Pocket TTS writing 2.16 s of audio to a wav |

The test app is not in the repository; its screen ran each call once and logged the result.

## Use it

```ts
import { Kiln, greedy, isCancelled, type Message } from 'kiln-react-native';

const model = await Kiln.loadModel(`${filesDir}/Qwen3.5-0.8B-Q4_K_M.gguf`);
const ctx = await model.newContext();

const history: Message[] = [{ role: 'user', content: 'What is the capital of France?' }];
const reply = await ctx.chat(history, {
  onPiece: (p) => setText((t) => t + p.text),   // p.channel: 'answer' | 'reasoning'
});
console.log(reply.answer, `${reply.stats.genTps.toFixed(1)} tok/s`);

await ctx.close();
await model.close();
```

For the next turn, push `{ role: 'assistant', content: reply.answer }` and the new question onto
`history` and call `chat` again with the whole list. kiln reuses what is already in its cache (`reply.stats.cachedTokens`).

- **Stop quietly:** return `false` from `onPiece`. The promise resolves with `stats.stoppedByUser`.
  Pieces reach JS as events, so the phone may have written a few more tokens by the time the
  stop arrives.
- **Cancel one call:** pass `signal` from an `AbortController`. The promise rejects and
  `isCancelled(e)` is true. It stops only that call, at its next piece (or before it starts);
  another call on the same context carries on.
- **Cancel whatever runs:** `ctx.cancel()` stops the running call on that context, also in the
  middle of reading a long prompt.
- **Errors** reject with `e.code` set to kiln's status (`"MODEL_LOAD"`, `"CONTEXT_FULL"`, ...),
  or `"INVALID_STATE"` for a closed handle.

The other kinds of model. Each needs its own model file and context:

```ts
// speech to text (Whisper or Parakeet: no projector needed)
const asr = await Kiln.loadModel(parakeetPath);
const asrCtx = await asr.newContext();
const t = await asrCtx.transcribe({ path: wavPath }, { onText: (w) => setLive((s) => s + w) });

// embeddings: similarity of two normalised vectors is their dot product
const emb = await Kiln.loadModel(embeddingGemmaPath);
await emb.loadProjector(embeddingGemmaProjectorPath);   // only for images / audio
const embCtx = await emb.newContext();
const q = await embCtx.embed('when does the train leave', { kind: 'query' });
const img = await embCtx.embedMedia(null, [{ path: photoPath }]);   // a path, or base64 bytes

// text to speech: load the model, then its projector, then make the context; the audio is a wav file
const tts = await Kiln.loadModel(ttsPath);
await tts.loadProjector(ttsProjectorPath);
const voiceCtx = await tts.newContext();
const speech = await voiceCtx.speak('Hello there.', `${cacheDir}/hello.wav`, { voice: { path: voiceWav } });

// decision models: a probability per answer, in one pass
const d1 = await Kiln.loadModel(d1OmniPath);
const d1Ctx = await d1.newContext();
const d = await d1Ctx.decide('my parcel never came', { id: 'refund', type: 'yes_no', instructions: 'Is the customer asking for a refund?' });
```

## Install (inside this repository, for now)

The Android build compiles the kiln core and the Kotlin library from `../android`, so the
package only works from a checkout of this repository. Publishing to npm needs a vendored copy,
like the Flutter plugin's `scripts/vendor_flutter.sh`. In an app:

```bash
npm install /path/to/kiln/react-native
```

npm links the folder (a symlink in `node_modules`), and Metro does not follow a link out of the
app on its own. Tell it in the app's `metro.config.js`, and resolve `react` / `react-native` from
the app only, so the bundle has one copy of each:

```js
const path = require('path');
const fs = require('fs');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const kiln = fs.realpathSync(path.join(__dirname, 'node_modules/kiln-react-native'));
const escape = (p) => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

module.exports = mergeConfig(getDefaultConfig(__dirname), {
  watchFolders: [kiln],
  resolver: {
    nodeModulesPaths: [path.join(__dirname, 'node_modules')],
    blockList: [new RegExp(`^${escape(path.join(kiln, 'node_modules'))}/.*`)],
  },
});
```

Set `minSdkVersion = 29` in the app's `android/build.gradle`, then build as usual. Autolinking
finds `KilnPackage`, and the app's build runs codegen on `src/NativeKiln.ts`. Requirements: React
Native 0.87 (tested with 0.87.1, New Architecture; older versions were not tried), and an
**arm64-v8a** phone with minSdk 29. kiln is built for 64-bit ARM only, so an x86_64 emulator fails
with `UnsatisfiedLinkError`. Model files must be in a place the app can read, such as its files
directory.

## Things to know

- **Close what you open.** `model.close()` rejects while one of its contexts is still open.
- **One call at a time per context.** A second call waits for the first. Use two contexts to work
  in parallel.
- **Big results cross the bridge as plain arrays.** An embedding is 768 numbers, which is fine.
  Audio is not sent to JS at all: `speak` writes a wav file and returns its path, so play it with
  any audio library.
- **Every streamed piece is one event.** At 100 tokens a second that is 100 small events a second.
  This has not been measured on a phone yet.

## Tests

```bash
npm install
npm test          # the JS layer against a fake native module
npm run typecheck
```

`npm run codegen` writes the schema codegen makes of the spec to `build-codegen/schema.json`.
That's a quick check after changing `src/NativeKiln.ts`.

## After changing `kiln.h`

Change it in `../android` first (JNI, `Native.kt`, Kotlin types), then the spec in
`src/NativeKiln.ts`, `KilnModule.kt`, and `src/kiln.ts`. Bump `version` in `package.json` with
the rest.

## Next steps

1. Vendoring and an npm release, which also removes the Metro setup above.
2. Embeddings and decision models in an app (the JS tests and the Kotlin library cover them; the
   app run did chat, speech to text and text to speech).
3. iOS, once kiln's own iOS build is done.
