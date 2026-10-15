# Changelog

## Unreleased

- **Hybrid models reuse the system prompt.** An app that sends the same system prompt with a
  new user message each time (extraction, classification) got no reuse on Qwen3.5 and LFM2:
  their only checkpoint sat at the end of the whole history. kiln now also checkpoints after
  the system prompt. Qwen3.5 0.8B reuses 25 of the second question's 41 prompt tokens (0
  before), the same through C, Rust, Kotlin, Flutter and React Native, on the M4 Pro and the
  Pixel 8 (Android 17); 520 checks on both. With the Hishab Qwen3.5 extractor on the Pixel 8:
  146 of 167 prompt tokens reused, prompt eval 2.9 s to 0.8 s.

- **Rust bindings** in `rust/`: `kiln-sys` (raw functions generated from `kiln.h` by bindgen;
  `scripts/gen_rust_bindings.sh` regenerates them) and `kiln`, a safe API (`Model`, `Context`,
  `CancelHandle`, streamed text to a closure in whole UTF-8 characters, errors as `Result`).
  The build script compiles libkiln with CMake and links it statically, or links a prebuilt one
  from `KILN_LIB_DIR`. No change to the C ABI. Run on Linux x86_64 (Ryzen 5 8500G, Rust 1.99)
  and macOS (M4 Pro, Metal, Rust 1.94) on 2026-10-15: 11 tests over the C suite's models pass
  on both, and on Linux the decision and embedding numbers match `kiln-cli`. Building libkiln
  from Cargo stops with a message on Android and Windows, where it was not tried. Not on
  crates.io yet.
- **Kotlin / Android library** in `android/`: one `libkiln_jni.so` (JNI glue with the core linked in
  statically, c++_static) and a Kotlin API (`KilnModel`, `KilnContext`, `chatFlow` as a coroutine
  Flow, `KilnException`). Strings cross JNI as UTF-8 byte arrays, not `jstring`. No change to the
  C ABI. The Kotlin code and the glue were run on the desktop JVM against a desktop build of the
  same `.so` (Ryzen 5 8500G, 2026-10-15): 12 tests pass, also under `-Xcheck:jni`, which found a
  JNI call made after a callback threw (fixed). On the M4 Pro the same 12 pass through Gradle, the
  `.aar` builds with the NDK (its `.so` needs neither `libomp.so` nor `libc++_shared.so`), and the
  device test passes on the Pixel 8 (Android 17): load, probe and a chat with LFM2.5 350M.
- **React Native module** in `react-native/` (Android): a TurboModule (`src/NativeKiln.ts`) whose
  Kotlin side calls the Kotlin library above, and a TypeScript API (`Kiln.loadModel`, `KilnModel`,
  `KilnContext` with `chat` streaming to `onPiece`, `AbortSignal` cancelling, `transcribe`,
  `embed`, `decide`, `speak` to a wav file). A new React Native 0.87.1 app with this package added
  by path ran on the Pixel 8 (Android 17): chat, cache reuse, stop, abort, Parakeet speech to text
  and Pocket TTS text to speech all worked; the app needs a Metro setting for the linked folder
  (in `react-native/README.md`). 9 tests of the JS layer pass against a fake native module.
  Builds only inside this repository (not on npm).

## 0.5.0

Text to speech. The C ABI changed: two new functions (`kiln_speak`, `kiln_speak_options_default`),
two new structs (`kiln_speak_options`, `kiln_speak_stats`), a new callback type
(`kiln_audio_cb`), and two fields at the end of `kiln_projector_info` (`generates_audio`,
`sample_rate`), so that struct grew. Submodules unchanged (llama.cpp 781dbc5ac, whisper.cpp 0c7be696).

- **`kiln_speak` turns text into audio**, in the voice of a short reference clip. A text-to-speech
  model is a small text model plus a projector, loaded like Qwen3-ASR (`kiln_model_load`, then
  `kiln_model_load_projector`; `kiln_projector_info.generates_audio` reads true). The audio
  comes to a callback in pieces of about a second as it is made (mono floats, 24 kHz), so
  playback can start before the sentence is finished. Each call uses a voice context of its
  own, so the conversation in the context is kept. `kiln_speak_stats` gives the length, the
  time reading the text and the voice, the time generating, and whether the model ended the
  sentence itself; a fixed `seed` makes the same text give the same audio. This is llama.cpp's
  audio-generation helper (the path of `llama-tts`), which llama.cpp marks experimental.
- Tested with **Pocket TTS** (Kyutai, English, CC-BY-4.0; 159 MB backbone and 60 MB projector in
  llama.cpp's format from EryriLabs/pocket-tts-GGUF) and, on the desktop, with **Qwen3-TTS 1.7B**
  (ggml-org, Q4_K_M, 1.5 GB for both files).
- `kiln-cli --speak "text" --voice clip.wav --wav out.wav`; `kiln-bench` speaks a stock
  sentence when the projector generates audio (series `speak`, the tokens column holds the
  audio length in ms); `scripts/fetch_models.sh --speech` fetches Pocket TTS; `scripts/bench.sh`
  pairs it with its projector and the train clip as the voice.
- The test suite checks the result the honest way: the generated audio is written down again
  with Parakeet and the words compared.

Speeds for "The train to London leaves at nine in the morning." (2.7 s of speech), voice from
the 3-second test clip, whole call:

| model | M4 Pro (Metal) | Pixel 8 (CPU, 5 threads pinned) |
|---|---|---|
| Pocket TTS, English | 0.22 s (12x real time) | 3.4 to 4.1 s (0.7 to 0.9x real time) |
| Qwen3-TTS 1.7B, Q4_K_M | 2.0 s (1.6x) | not tried: 1.5 GB of weights |

On the Pixel 8 that is a little slower than the speaker, so an app should let the first
piece or two arrive before playing, or keep sentences short. Of the 3.4 s, about 0.8 to 0.95 s
is encoding the 3-second voice reference (counted in `prompt_ms`; a cache per voice would
remove it) and 1.5 to 2.4 s is the frames. Without OpenMP and pinning (llama.cpp's own
`llama-tts` built the old way) the Pixel took 5 to 15 s for the same sentence, getting slower
each run: the flow decoder is many small graphs, the same trap the Parakeet decoder fell into
in 0.4.0.

Verified with `kiln-tests` on 2026-10-13: 505 checks on macOS (M4 Pro, Metal), Android
(Pixel 8) and Linux arm64 (Ubuntu 24.04 in Docker on the M4 Pro); 26 integration tests (4 skipped) on the Pixel 8 through the Flutter plugin.

Added after the release, 2026-10-14: 505 checks on Linux x86_64 (Ryzen 5 8500G, CPU), the
first run there with the whisper.cpp backend. On that desktop Parakeet writes down the 3-second
clip in 0.16 s and a 17-second clip in 0.9 s, and Pocket TTS speaks the sentence above
(2.7 s of speech) in 0.6 s.

Not run: iOS. Known gaps: one language per Pocket TTS file (English is the one
fetched; French, German, Italian, Portuguese, Spanish exist); no streaming playback in the
example app (it collects the pieces, then plays); Qwen3-TTS not run on a phone; the voice
reference is re-encoded on every call (about 0.75 s on the Pixel 8), a cache per voice
would remove that.


## 0.4.0

Speech to text, with three model families: Whisper, Parakeet and Qwen3-ASR. The C ABI changed:
two new functions (`kiln_transcribe`, `kiln_transcribe_options_default`), two new structs
(`kiln_transcribe_options`, `kiln_transcribe_stats`), and one new field at the end of
`kiln_model_info` (`is_speech_model`), so that struct grew: code built against 0.3.x headers
must be rebuilt. The llama.cpp submodule is unchanged (781dbc5ac); whisper.cpp (0c7be696) is a
new submodule.

- **A second engine, whisper.cpp, runs Whisper and Parakeet.** `kiln_model_load` takes their
  files (`ggml-base-q5_1.bin`, `ggml-parakeet-tdt-0.6b-v3-q8_0.bin`: whisper.cpp's own format,
  not GGUF) like any model; the audio encoder is built in, so no projector, and
  `kiln_model_info.is_speech_model` reads true. Everything but `kiln_transcribe` answers
  `KILN_ERR_NOT_SUPPORTED` on them. whisper.cpp pins the same ggml as llama.cpp (0.26.0) and
  builds against the copy already in libkiln, so there is one ggml. The backend is
  `src/backends/whispercpp/`; backends now say which file formats they read (`Backend::can_load`),
  so a file in another format fails with one clear message instead of one try per backend.
  Parakeet is the most accurate of the three on English and the fastest (`docs/ASR_COMPARISON.md`);
  Whisper base is the smallest file (60 MB) and knows 99 languages. Both always take the most
  likely word. Whisper reports the language it heard; Parakeet does not. Whisper reads a
  language hint and refuses one it does not know (`KILN_ERR_INVALID_ARG`); Parakeet ignores it,
  as it ignores `prompt` and `max_tokens`. Whisper works in 30-second windows, so a long
  recording needs no cutting; Parakeet takes the clip whole.

- **`kiln_transcribe` writes down what is said in an audio clip.** A speech model is two
  files: a small text model and an audio encoder (its `mmproj`). Load the model, load the
  encoder with `kiln_model_load_projector`, and call `kiln_transcribe` with a wav, mp3 or flac
  file (a path, or the bytes). The words are streamed to the same callback `kiln_chat` uses.
  `kiln_transcribe_stats` gives the length of the clip, the time in the encoder, in reading
  and in writing, and the language the model heard. `kiln_transcribe_options` takes a language
  hint, another instruction, and a token limit.
- Tested with **Qwen3-ASR 0.6B** (Q8_0, 805 MB, with its 214 MB encoder; Apache 2.0). On the
  Linux desktop (Ryzen 5 8500G, CPU, 8 threads) a 3-second clip takes 0.6 to 0.8 s and a
  17-second clip 3.0 to 3.4 s, about 5 times faster than real time, with 1.5 GB of memory in
  use. The desktop was in normal use during these runs; they are not a quiet-machine
  benchmark. A clip with no speech in it gives an empty text.
- The text is what llama.cpp's `llama-server` returns from `/v1/audio/transcriptions` for the
  same file, when both run with the same number of threads (compared on four clips at 6
  threads and on the 17-second one at 8; the token counts of the prompts match too). Between 6 and 8 threads one headline in that clip
  changes from "Men Walk on Moon" to "Men walk on moon", in the server and in kiln alike:
  processors round differently with a different split of the work, and two near-equal
  spellings can swap. The words do not change.
- Qwen3-ASR starts its answer with `language English<asr_text>`. kiln removes that header and
  reports the language in the stats.
- `kiln_model_load_projector` now also works on a chat-style model (before: embedding models
  only). It still returns `KILN_ERR_NOT_SUPPORTED` on a decision model.
- A transcription clears the context: a conversation held in it is forgotten, and the audio
  leaves nothing behind for the next `kiln_chat`.
- Fix: a context smaller than 512 tokens kept a batch size of 512 where llama.cpp had lowered
  it. Harmless for chat, where the prompt has to fit in the context anyway; it would have
  aborted on audio.
- `kiln-cli`: `--transcribe clip.wav` (repeatable) and `--language`. `kiln-bench`: a chat-style
  model given `--mmproj` and `--media` is benchmarked as a speech model (series
  `transcribe-<file>`); `scripts/bench.sh` does that for Qwen3-ASR.
- `scripts/fetch_models.sh --speech` downloads the speech model and its encoder (1.0 GB).
- Internal: the projector code moved from `embedding.h` to `media.h`, shared by embeddings and
  speech; the sampling loop of `generate` is now a method that `transcribe` also uses.

Verified with `kiln-tests` on 2026-10-13: 476 checks on macOS (M4 Pro, Metal; whisper.cpp uses
Metal there too) and on Android (Pixel 8, CPU); 398 on Linux x86_64 (Ryzen 5 8500G, CPU)
before the whisper.cpp backend and its 78 checks were added. Without the speech models the
count is 349 (334 before, plus 8 for the transcript header and 7 for the new error paths).

Speeds for the 3.1-second test clip, whole call including reading the file:

| model | M4 Pro (Metal) | Pixel 8 (CPU) |
|---|---|---|
| Parakeet TDT 0.6B v3, q8_0 (669 MB) | 53 to 68 ms | 0.24 to 0.37 s |
| Whisper base, q5_1 (60 MB) | 84 ms | 3.0 s |
| Qwen3-ASR 0.6B, Q8_0 (805 MB + 214 MB encoder) | 130 ms | 3.8 s, slower than the clip |

Two things were needed to get there on the phone. The engine's threads inherit the calling
thread's core set, so the call runs under `AffinityScope` (as the mtmd encoders do): pinned,
Parakeet's encoder is a steady 0.3 s; unpinned it swung between 0.4 and 7 s. And the Android
build now uses OpenMP (the NDK's `libomp.a`, linked statically): without it ggml starts fresh
threads for every graph, and Parakeet's decoder computes dozens of tiny graphs per clip,
which took 1 to 13 s on the Pixel 8 with no two runs alike; with OpenMP's persistent workers
the decoder takes 25 to 35 ms. whisper.cpp's own `parakeet-cli` and `whisper-cli`, built for
Android without OpenMP, showed the same swings, so this is not a kiln bug but a build choice
that matters on phones. Chat is unchanged by the switch: LFM2.5 350M, three runs alternating the two builds on
the same phone, 97.0 / 89.5 / 85.1 tok/s without OpenMP and 96.5 / 88.5 / 85.7 with
(`-DKILN_ANDROID_OPENMP=OFF` builds the old way). Qwen3-ASR spends 2.5 s of its 3.8 s in the audio encoder. So on a Pixel 8: Parakeet
keeps up with the speaker with room to spare, Whisper base only just, Qwen3-ASR does not.

Sustained use (`KILN_BENCH_ONLY=ggml scripts/bench.sh`, bursts of 5 calls back to back,
`docs/bench/pixel8_ggml-*`): Parakeet averaged 0.28 s per 3-second clip for the first three
bursts and 0.8 to 1.5 s once the phone reached 82 °C; Whisper base 3.5 to 7.5 s throughout.

The Flutter plugin has `transcribe` (see its CHANGELOG), run on the Pixel 8.

Not run: iOS. Linux x86 has not been re-run since the whisper.cpp backend was added (Linux
arm64 in Docker: see the README's status table).

Known gaps: one clip per call; files or file bytes only (no raw microphone samples, so an app
records to a wav first); nothing is streamed in while the person is still speaking; a
recording longer than the context has to be cut by the app; only the Q8_0 file was tried.

## 0.3.1

No ABI change: code built against 0.3.0 headers keeps working.

First run on a Snapdragon: Galaxy S26 Ultra (Snapdragon 8 Elite Gen 5), 2026-10-12. 319 C
checks, 26 Flutter integration tests, the example app and the benchmarks. See
`docs/devices/galaxy-s26-ultra.md`.

- **Core selection on chips with no little cores.** kiln dropped "the slowest group of cores",
  which on a 2 + 6 chip like the Snapdragon 8 Elite Gen 5 left 2 threads and six strong cores
  idle. The slowest group is now kept when the kernel rates it at half the strongest core or
  more (`cpu_capacity`), and where every core is fast kiln runs on all but two. On the S26 Ultra
  that is 6 threads instead of 2. Under sustained load LFM2.5 and Qwen3.5 chat are 18 to 24%
  faster, Gemma 3 270M is about level (4% slower), and a decision or an embedding takes 35 to
  45% less time. `kiln_device_info.cpu_big_cores` and `cpu_fast_mask` report
  the new choice. Chips with little cores are not affected: the Pixel 8 was re-run on
  2026-10-13 and still gets 5 threads on the same cores (its little cores are rated 182 of
  1024), with 334 checks passing. No ABI change.
- `kiln-bench` takes temperatures from Android's thermal service first; the S26 Ultra has a raw
  sensor stuck at 105 °C.
- `scripts/bench.sh`: `KILN_BENCH_SKIP_HOST`, `KILN_BENCH_DEVICE_TAG`, `KILN_BENCH_DEVICE_ENV`.
- A trial of the OpenCL backend on the Adreno 840 is written up in the device profile; it is
  not a supported build.

- `kiln-bench` measures decision and embedding models as well as chat models. It reads the kind
  from the model: questions per burst for a decision model, notes per burst for an embedding
  model, and with `--mmproj` / `--media` an image and an audio clip. One line per run goes to
  `<out>.runs.csv`, and the charts get a speed-per-run panel.
- `scripts/bench.sh` runs every model in `models/` that way, skips projector files, and waits
  for the phone's cores to be uncapped before each model.
- Benchmarks re-run on the Pixel 8 at v0.3.0 for all model kinds; see `docs/bench`.
- `kiln-bench` fixes: it exits non-zero when a call fails, stops its sampler before any early
  exit (a failed projector load used to abort the process), refuses to run when it cannot write
  the per-run file, and adds `sample_run` to that file. `bench.sh`: `KILN_BENCH_ONLY` ignores
  case, and paths with spaces work.

Verified with `kiln-tests` (334 checks: 319 plus 15 for core selection) on macOS (M4 Pro,
Metal) and Android (Pixel 8); the Galaxy S26 Ultra ran the 319 before the unit checks existed
and the 15 afterwards. Flutter integration test: 26 tests on the Galaxy S26 Ultra with the
core-selection fix, and 26 on the Pixel 8 at 0.3.1. The Pixel 8 run was done on 2026-10-13,
after the release (the phone was locked before it), together with a fresh app that installed
0.3.1 from pub.dev and ran an embedding search on that phone. Not run on Linux or iOS.

## 0.3.0

The C ABI changed: `kiln_model_info` has two new fields at the end (its size is unchanged),
and there are four new functions (`kiln_embed`, `kiln_embed_media`, `kiln_model_load_projector`,
and the `kiln_*_default` helpers for their options) with their structs and the
`kiln_embed_kind` enum.

- Embedding models: `kiln_embed` turns a text into a vector, and
  `kiln_model_info.is_embedding_model` / `n_embd` say when to use it and how long the vector
  is. `kiln_embed_options.kind` (query or document) adds the task prefix the model was trained
  with, `dimensions` shortens the vector, `normalize` scales it to length 1.
  A text longer than the context returns `KILN_ERR_CONTEXT_FULL`; nothing is cut silently.
- Images and audio: `kiln_model_load_projector` loads a model's image / audio encoders (its
  `mmproj` file) and `kiln_embed_media` embeds an image, an audio clip, or text mixed with
  them, into the same space as text. A text query can then find a photo or a voice note.
  Images: jpg, png, bmp, gif. Audio: wav, mp3, flac. No video.
  `kiln_projector_params.image_max_tokens` (default 280) scales large photos down: a
  2000-pixel screenshot took 21.7 s at the model's own limit and 3.2 s at 280.
- The library now includes llama.cpp's `mtmd` (libkiln.so on Linux: 10.7 MB to 12.5 MB).
- `kiln_chat` and `kiln_generate` on an embedding model return `KILN_ERR_NOT_SUPPORTED`.
  Before, kiln ran such a model as a chat model and printed unused tokens.
- `kiln-cli`: `--embed`, `--query`, `--dim`, and `--mmproj`, `--media`, `--image-tokens`.
- `scripts/fetch_models.sh` downloads the two text embedding test models (176 MB and 84 MB);
  `--media` adds the encoders for the image and audio tests (0.9 GB). `tests/media` holds
  small generated test images and speech clips.

Tested models:

| model | inputs | files |
|---|---|---|
| nomic-embed-text v1.5 | text | 84 MB (`Q4_K_M`) |
| EmbeddingGemma 2 | text, images, audio | 176 MB (`UD-Q4_K_XL`) + 555 MB projector (`Q8_0`) |
| jina-embeddings-v5-omni-nano | text, images | 157 MB (`Q4_K_M`) + 203 MB vision projector (`F16`) |

For all three the vectors are identical to the ones llama.cpp gives at the same commit
(`llama-embedding` for text, `llama-server` for images, audio and mixed input). On a
16-question search test the 4-bit EmbeddingGemma 2 file picked the same answers as the
full-precision one (15 of 16). Text queries found the right one of ten images and audio clips
in 8 of 8 tries with EmbeddingGemma 2, and the right one of six images in 5 of 5 with Jina.

Limits to know: Jina's card says exact agreement with its PyTorch version needs Jina's fork
of llama.cpp; kiln matches upstream llama.cpp and was not compared with PyTorch. Jina's audio
projector (1.3 GB) was not tried. Jina v5 omni is licensed CC BY-NC 4.0 (non-commercial);
EmbeddingGemma 2 and nomic are Apache 2.0.

Fixed before release: on Android the image and audio encoders ran on a single core. kiln pins
the thread that calls it to one core, and the encoder's own threads inherited that. A small
image took 14.3 s on a Pixel 8 and now takes about 2.5 s; a 3-second audio clip went from 9.6 s
to about 0.5 s. Text was not affected.

Verified with `kiln-tests`: 319 checks on macOS (M4 Pro, Metal) and on Android (Pixel 8, CPU);
319 on Linux arm64 (Ubuntu 24.04 in Docker, GCC 13, OpenMP), also with `KILN_CPU_MASK=3`, and
319 on Linux x86_64 in the same container emulated on the Mac (AVX2);
317 on Linux x86 (Ryzen 5 8500G, CPU) before two checks were added, not re-run there since.
The Flutter plugin has `embed`, `embedMedia` and `loadProjector` with regenerated bindings:
23 integration tests on the Pixel 8, including images and audio. Not run on iOS.

Speed with EmbeddingGemma 2 at 4 bits:

| input | Ryzen 5 8500G | Pixel 8 | M4 Pro (Metal) |
|---|---|---|---|
| short sentence | 15 ms | 30 to 80 ms | 13 ms |
| small image (81 media tokens) | 0.7 s | about 2.5 s | 0.2 s |
| 3-second audio clip | 0.2 s | about 0.5 s | 0.15 s |

## 0.2.0

The C ABI changed: `kiln_chat_options`, `kiln_gen_stats` and `kiln_model_info` grew, so code
built against 0.1.0 headers must be rebuilt.

- Hybrid models (Qwen3.5, LFM2) reuse the KV cache across turns through state checkpoints.
  On a 6-turn Qwen3.5 0.8B chat, prompt time went from 3232 ms to 1251 ms in total.
- Thinking budget: `kiln_chat_options.max_reasoning_tokens` closes the think block after N
  reasoning tokens so the model answers; `kiln_gen_stats.reasoning_truncated` reports it.
- Decision models: `kiln_decide` and `kiln_model_info.is_decision_model`. LFM2 d1-omni with a
  text state; no images or audio. In the Flutter plugin as `KilnContext.decide`.
- Fixed a crash when `KILN_CPU_MASK` had fewer CPUs than the default thread count (OpenMP builds).
- Build: the llama.cpp submodule is initialised at configure time if it is missing; Android
  scripts find the NDK on Linux; `scripts/fetch_models.sh` downloads the test models.
- `kiln-cli`: `--think-budget`, and `--state` / `--ask` / `-o` for decision models.
- `tools/barrier-lab`: a small experiment showing how ggml's per-op barrier behaves.

Verified with `kiln-tests` (154 checks) on Linux x86 (Ryzen 5 8500G, CPU), macOS (M4 Pro,
Metal) and Android (Pixel 8, CPU). The Flutter plugin was run on the Pixel 8: 16 integration
tests, including `maxReasoningTokens` / `reasoningTruncated` and `decide`; its bindings are regenerated
with ffigen. Not re-run on iOS. Benchmarks were not repeated; a back-to-back run of the 0.1.0
and 0.2.0 builds on a warm Pixel 8 gave the same speed (50 to 57 tok/s, LFM2.5 350M).

Also fixed after the version bump: `scripts/android-test.sh` and `android-run.sh` exited
silently on macOS (the Linux NDK path did not exist, and `pipefail` ended the script); the
Flutter integration test downloads models to a `.part` file with retries.

llama.cpp submodule is initialised at configure time if it is missing; Android
  scripts find the NDK on Linux; `scripts/fetch_models.sh` downloads the test models.
- `kiln-cli`: `--think-budget`, and `--state` / `--ask` / `-o` for decision models.
- `tools/barrier-lab`: a small experiment showing how ggml's per-op barrier behaves.

Verified on Linux x86 (Ryzen 5 8500G, CPU) only: 154 checks in `kiln-tests`. Not re-run on
Android, iOS or macOS. The Flutter changes (`maxReasoningTokens`, `reasoningTruncated`) are
untested and the generated bindings were edited by hand.

llama.cpp submodule: b11548 (781dbc5ac).

## 0.1.0

First release. C core, llama.cpp backend, device probe, streaming chat with a reasoning
channel, KV prefix reuse, cancel, CLI and benchmark tools, Flutter plugin.
