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
| The JS layer (`src/kiln.ts`) | 7 Node tests against a fake native module | pass |
| TypeScript | `tsc --noEmit` against RN 0.87.1's types | ok |
| **An app on a phone** | | **not run** |

So no code here has run on Android yet. The native work it calls (`dev.kiln`) passed 12 tests
on the desktop JVM against real models. What remains is a test app on the Pixel 8 (see
[Next steps](#next-steps)).

## Use it

```ts
import { Kiln, greedy, isCancelled } from 'kiln-react-native';

const model = await Kiln.loadModel(`${filesDir}/Qwen3.5-0.8B-Q4_K_M.gguf`);
const ctx = await model.newContext();

const history = [{ role: 'user' as const, content: 'What is the capital of France?' }];
const reply = await ctx.chat(history, {
  onPiece: (p) => setText((t) => t + p.text),   // p.channel: 'answer' | 'reasoning'
});
console.log(reply.answer, `${reply.stats.genTps.toFixed(1)} tok/s`);

await ctx.close();
await model.close();
```

For the next turn, push the reply and the new question onto `history` and call `chat` again
with the whole list. kiln reuses what is already in its cache (`reply.stats.cachedTokens`).

- **Stop quietly:** return `false` from `onPiece`. The promise resolves with `stats.stoppedByUser`.
- **Cancel:** pass `signal` from an `AbortController`, or call `ctx.cancel()`. The promise rejects,
  and `isCancelled(e)` is true.
- **Errors** reject with `e.code` set to kiln's status (`"MODEL_LOAD"`, `"CONTEXT_FULL"`, ...),
  or `"INVALID_STATE"` for a closed handle.

The other kinds of model:

```ts
// speech to text (Whisper or Parakeet: no projector needed)
const t = await ctx.transcribe({ path: wavPath }, { onText: (w) => setLive((s) => s + w) });

// embeddings: similarity of two normalised vectors is their dot product
const q = await ctx.embed('when does the train leave', { kind: 'query' });

// images: a path, or the file's bytes as base64
const img = await ctx.embedMedia(null, [{ path: photoPath }]);

// text to speech: load the model, then its projector, then make the context; the audio is a wav file
const tts = await Kiln.loadModel(ttsPath);
await tts.loadProjector(ttsProjectorPath);
const voiceCtx = await tts.newContext();
const speech = await voiceCtx.speak('Hello there.', `${cacheDir}/hello.wav`, { voice: { path: voiceWav } });

// decision models: a probability per answer, in one pass
const d = await ctx.decide('my parcel never came', { id: 'refund', type: 'yes_no', instructions: 'Is the customer asking for a refund?' });
```

## Install (inside this repository, for now)

The Android build compiles the kiln core and the Kotlin library from `../android`, so the
package only works from a checkout of this repository. Publishing to npm needs a vendored copy,
like the Flutter plugin's `scripts/vendor_flutter.sh`. In an app:

```bash
npm install /path/to/kiln/react-native
```

Then build the app as usual. Autolinking finds `KilnPackage`, and the app's build runs codegen on
`src/NativeKiln.ts`. Requirements: React Native 0.80 or newer (New Architecture), and an
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

1. A small test app on the Pixel 8: `npx @react-native-community/cli init`, add this package by
   path, and chat, transcribe and speak with the test models. Check that tokens stream smoothly.
2. Vendoring and an npm release.
3. iOS, once kiln's own iOS build is done.
