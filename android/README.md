# kiln for Kotlin / Android

A Kotlin API for kiln on Android: chat, decision, embedding, speech-to-text and text-to-speech
models on the phone. It is an Android library (`.aar`) holding one native library,
`libkiln_jni.so`, which contains the JNI glue and the kiln core (llama.cpp, whisper.cpp) linked
statically. There is no separate `libkiln.so` and no `libc++_shared.so`.

**Where it was run, 2026-10-15:**

| what | where | result |
|---|---|---|
| Kotlin API + JNI glue, host tests (`src/test`) | Linux x86_64 desktop JVM (Ryzen 5 8500G, OpenJDK 17, Kotlin 2.3.20), `libkiln_jni.so` built for the desktop | **12 tests pass**, also under `-Xcheck:jni` with no warnings |
| Gradle build of the `.aar` (AGP 9.0.1, Gradle 9.1, NDK 28.2) | **not run**: that machine has no Android SDK, and its network was too slow to fetch one | |
| On a phone (`src/androidTest`) | **not run** | |

On the Ryzen the results match the C and Rust sides: refund P(yes) 0.9964, embedding scores
0.921 / 0.513, the same greedy reply, and Parakeet heard "The train to London leaves at 9 in the
morning." in 174 ms.

So the Kotlin code and the C++ glue were tested, but not the Android packaging. On the Mac with the
Pixel 8, run `./gradlew :kiln:assembleRelease :kiln:connectedAndroidTest`, then check that
`llvm-readelf -d` on the `.so` inside the `.aar` lists no `libomp.so` and no `libc++_shared.so`.

## Use it

```kotlin
import dev.kiln.*

KilnModel.load("/data/user/0/your.app/files/Qwen3.5-0.8B-Q4_K_M.gguf").use { model ->
    model.newContext().use { ctx ->
        val history = mutableListOf(Message.user("What is the capital of France?"))
        val stats = ctx.chat(history) { piece ->
            print(piece.text)          // piece.channel: ANSWER or REASONING
            true                       // false stops the reply
        }
        println("\n${stats.generatedTokens} tokens at ${stats.genTps} tok/s")
    }
}
```

Every call blocks until it is done, so don't call it on the main thread. In an app, use the
Flow, which runs the work on `Dispatchers.IO` and cancels it when the collector stops:

```kotlin
viewModelScope.launch {
    ctx.chatFlow(history).collect { event ->
        when (event) {
            is Piece -> reply += event.text
            is GenStats -> tokPerSec = event.genTps
        }
    }
}
```

The other kinds of model:

```kotlin
// speech to text (Whisper or Parakeet: no projector needed)
val t = ctx.transcribeCollect(Media.File(wavPath))

// embeddings: similarity of two normalised vectors is their dot product
val q = ctx.embed("when does the train leave", EmbedOptions.QUERY)

// text to speech: load the model, then its projector, then make the context
KilnModel.load(ttsPath).use { tts ->
    tts.loadProjector(ttsProjectorPath)
    tts.newContext().use { c ->
        val speech = c.speakCollect("Hello there.", SpeakOptions(voice = Media.File(voiceWav)))
        // speech.samples: mono floats at speech.sampleRate (24000 for Pocket TTS)
    }
}

// decision models: a probability per answer, in one pass
val d = ctx.decide("my parcel never came", Question("wants_refund", QuestionType.YES_NO, "Is the customer asking for a refund?"))
```

## Things to know

- **Close what you open.** `KilnModel` and `KilnContext` hold native memory the garbage collector
  does not see. Use `use { }`, or call `close()`. A model refuses to close while one of its
  contexts is still open.
- **One call at a time per context.** A second call waits for the first. `cancel()` works from
  any thread, and the running call throws `KilnException` with `Status.CANCELLED`. Returning
  `false` from a callback stops quietly instead (`stoppedByUser`).
- **An exception thrown in your callback** stops the generation and comes out of the call
  unchanged. The context stays usable.
- **Text pieces are whole characters.** The C callback can split a character such as "é" across
  two tokens. The binding holds the first half back.
- **Strings cross JNI as UTF-8 bytes**, not `jstring`. JNI's own string functions use "modified
  UTF-8", which encodes an emoji as two 3-byte halves, and kiln would then tokenize the wrong bytes.
- **Model files must be readable by the app**, for example under `context.filesDir`, or a file
  the app downloaded. Android does not let apps read `/data/local/tmp`.
- **arm64-v8a only**, minSdk 29, the same as the Flutter plugin. The CPU baseline is
  `armv8.2-a+dotprod+fp16`.
- **R8 / ProGuard:** `consumer-rules.pro` keeps the names native code looks up. Apps need nothing extra.

## Build and test

```bash
cd android
./gradlew :kiln:assembleRelease          # kiln/build/outputs/aar/kiln-release.aar (needs the Android SDK and NDK 28.2)
./gradlew :kiln:testDebugUnitTest        # host tests: builds libkiln_jni for this computer, runs on the desktop JVM
./gradlew :kiln:connectedAndroidTest     # on a phone over adb
```

Without Gradle (how the Ryzen run was done): build the glue with
`cmake -S android/kiln/src/main/cpp -B build/jni-host && cmake --build build/jni-host --target kiln_jni`,
compile `src/main/kotlin` and `src/test/kotlin` with `kotlinc` against kotlinx-coroutines-core and
JUnit 4, and run `org.junit.runner.JUnitCore dev.kiln.ModelsTest` with
`-Djava.library.path=build/jni-host`. Add `-Xcheck:jni`: it catches JNI misuse that a debug
Android app would crash on.

The host tests use the C suite's models (`scripts/fetch_models.sh --media --speech` from the
repository root). A test whose model is missing is skipped. They cover chat (streaming, greedy
repeat, cache reuse, stopping), cancel from another thread, an exception in the callback, the
Flow, thinking, the decision model, text and image embeddings, Whisper and Parakeet, and Pocket
TTS, whose audio is read back by Parakeet. They also check that the Kotlin default values equal kiln's.

## After changing `kiln.h`

Unlike the Rust and Flutter bindings, nothing here is generated. Update three places:
`src/main/cpp/kiln_jni.cpp` (the C call), `Native.kt` (the `external fun`), and `Types.kt` / the
class that uses it. Then run the host tests.

## Not done yet

- Publishing to Maven Central.
- A sample app. The Flutter example app already covers the phone UI.
- The log callback. `Kiln.setLogLevel` works, and on Android kiln writes its log to logcat.
