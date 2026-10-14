# kiln

On-device LLM runtime (chat, decision, embedding, speech-to-text and text-to-speech models): a C core
(`include/kiln/kiln.h`) over llama.cpp, plus whisper.cpp for Whisper and Parakeet, with a device probe,
CLI and benchmark tools, and a Flutter plugin. Targets are Android, iOS and desktop.
`docs/CODEBASE.md` is the file-by-file map; `docs/CONCEPTS.md` explains the ideas in plain words.

## Rules for this repo

- **No Claude attribution, anywhere.** No `Co-Authored-By` or session lines in commits, no
  "Generated with Claude Code" footer in PRs, no model names, and no `claude` in branch names.
  Commits are authored by the repo owner (the local git identity). Use branch names like
  `fix/...`, `feat/...`, `docs/...`.
- Work on a branch and open a PR. The owner merges, unless they ask you to.
- Explain things to the owner in simple words with a concrete example. They are learning the
  low-level side; avoid jargon without a definition.
- Docs state what was verified and where. Do not write that something works on a platform it
  was not run on. `README.md` has a status table and `CHANGELOG.md` records where each version
  was tested.
- A change to a public struct or function in `kiln.h` is an ABI change: bump the minor version
  (CMake project, `KILN_VERSION_*`, `kiln_version()` in `src/core/error.cpp`, the Flutter
  `pubspec.yaml` and podspec) and add a `CHANGELOG.md` entry.

## Build and test

```bash
git submodule update --init --recursive      # CMake also does this if the submodule is empty
cmake --preset host && cmake --build --preset host -j
scripts/fetch_models.sh --media --speech     # test GGUFs into models/ (~1.7 GB, plus 0.9 GB of image / audio encoders and 1.9 GB for the speech models)
./build/host/kiln-tests models               # 505 checks at 0.5.0 (349 without --speech); without --media the image / audio tests are skipped
./build/host/kiln-cli -m models/<file>.gguf --chat
```

Linux from the Mac (arm64, GCC, OpenMP), with Docker Desktop running:

```bash
docker run --rm -v "$PWD":/kiln -w /kiln ubuntu:24.04 bash -c 'apt-get update -qq && \
  apt-get install -y -qq cmake g++ make git ca-certificates >/dev/null && \
  cmake -S . -B build/linux-arm64 -DCMAKE_BUILD_TYPE=Release >/dev/null && \
  cmake --build build/linux-arm64 -j 10 | tail -1 && ./build/linux-arm64/kiln-tests models | tail -1'
```

Android: `cmake --preset android-arm64`, then `scripts/android-test.sh`. iOS: `scripts/build_ios.sh`
(macOS only). The NDK is looked up in the macOS and Linux default locations, or `ANDROID_NDK_HOME`.

## Things that are easy to get wrong

- **Linux and Android builds use OpenMP; macOS does not.** A threadpool smaller than
  `cp.n_threads` crashes under OpenMP. Keep the context's thread count and the pool size equal.
  Android links the NDK's `libomp.a` statically (`OpenMP_omp_LIBRARY` and `-static-openmp` in
  `CMakeLists.txt`); check `llvm-readelf -d libkiln.so` shows no `libomp.so` after touching that.
  It was turned on for the whisper.cpp decoders: without OpenMP ggml starts fresh threads for
  every graph, and Parakeet's decoder computes dozens of tiny ones per clip, which on a Pixel 8
  took 1 to 13 s with wild swings; with OpenMP's persistent workers the same clip takes 0.3 s.
- **`kiln-bench` prints one line per run on stderr** (`chat,1,tok,...`, `decide,1,call,...`,
  `embed-text,...`). The samples go to stdout by default; `-o x.csv` writes them to that file
  and the per-run lines to `x.runs.csv` as well. It exits non-zero when a call fails. It picks the workload from the kind of model. `scripts/bench.sh` skips
  projector files and pairs embedding models with theirs; `KILN_BENCH_ONLY=<substring>` limits it.
- **Benchmark only on a quiet machine.** A video call cut tok/s by a third on unchanged code,
  and on 2026-10-13 Spotlight indexing made Metal tok/s swing between 100 and 177 for the same
  build. When a number looks like a regression, alternate the old and new build before believing it.
- **On the phone, check memory as well as heat.** With a large model loaded Android evicts its
  pages when another app wants memory; RSS falling during a run is the sign, and calls get
  several times slower. The bench samples `mem_avail_mb` and RSS for this.
- **`KILN_CPU_MASK=<hex>`** forces a core set, `KILN_NO_AFFINITY=1` disables pinning.
- **Comparing answers with and without the KV cache needs `n_batch = 1`.** Batched and
  single-token decode round differently, so greedy output can differ with no bug present.
- **Hybrid models (Qwen3.5, LFM2) cannot rewind part of their state.** Cache reuse for them
  goes through checkpoints saved at `RenderedPrompt::stable_prefix`; see `LlamaContext`.
- **Qwen3.5 0.8B rarely stops thinking by itself.** Use `max_reasoning_tokens`. The repeat
  penalty makes the answer loop more often.
- **Decision models (`kiln_decide`) are a port of llama.cpp's server code**, which the library
  does not contain. To check a change, build `llama-server` from the submodule and compare
  `/v1/systemone` one question per request; the probabilities should match to six decimals.
- **Embedding models (`kiln_embed`) need the whole text in one batch**, so the context is
  built with `n_batch = n_ubatch = n_ctx`. To check a change, build `llama-embedding` from the
  submodule and compare with `kiln-cli --embed`; the numbers should be identical. A model's
  task prefixes come from its card and live in `EmbeddingHead::init`.
- **Images and audio (`kiln_embed_media`) use llama.cpp's `mtmd` library**, built into libkiln
  (`LLAMA_BUILD_MTMD`). The reference for them is `llama-server`, not `llama-embedding`;
  `docs/CODEBASE.md` has the request. The server picks a random media marker per run: read
  it from `/props`.
- **Speech models (`kiln_transcribe`) are a chat turn with an audio clip attached.** The reference
  is `llama-server`'s `/v1/audio/transcriptions` (`docs/CODEBASE.md` has the request), not
  `llama-mtmd-cli`, which puts the audio before the instruction and can give another spelling.
  Compare with the same `-t` on both sides: between 6 and 8 threads "Men Walk on Moon" became
  "Men walk on moon" in the server and in kiln alike. Tests compare words, not capitals.
- **Whisper and Parakeet run through a second engine, whisper.cpp** (`src/backends/whispercpp/`),
  not llama.cpp: they are whole speech models with their own decoders, and their files are
  ggml's older format (`lmgg` magic, `.bin`), not GGUF. `Backend::can_load` sniffs the first bytes,
  so each backend sees only its own format. whisper.cpp must pin the same ggml as llama.cpp
  (both 0.26.0 today): its CMake takes the `ggml` target a parent already made. When bumping
  either submodule, check `ggml/CMakeLists.txt` in both. The engines are not thread-safe per
  model: `SpeechModel::mutex()` serialises calls from different contexts.
- **miniaudio is compiled twice into libkiln**, once by llama.cpp's mtmd and once by the
  whisper.cpp backend, both with `MA_API static`; two symbols escape that and are renamed with
  `#define` before the include (`ma_atomic_global_lock`, `ma_android_sdk_version`). A new
  duplicate-symbol link error after a llama.cpp bump means another one appeared.
- **Qwen3-ASR is not a usable voice path on a Pixel 8**: 3.8 s for a 3-second clip, 2.5 s of it in
  the encoder. Parakeet is (0.3 s); Whisper base keeps up only just (3.0 s, half of it the
  encoder: Whisper always encodes a 30-second window). `kiln_model_info.is_speech_model` is
  false for Qwen3-ASR (it is a chat-style model plus projector); the example app lists only
  the other two.
- **Text to speech (`kiln_speak`) is llama.cpp's audio-generation helper** (`mtmd_helper_gen_audio_*`,
  the path of `llama-tts`), which llama.cpp marks experimental: check `tools/mtmd/mtmd-helper.h`
  and `tools/tts/tts.cpp` after every submodule bump. The backbone must return hidden states,
  so `LlamaContext` makes a second llama context with `embeddings = true` on the first call
  (`tts_ctx_`), sharing the pinned threadpool; the chat context is never touched. The helper
  is kept between calls (`gen_`): making one copies the whole token table, 300 ms on the Mac CPU.
  The audio reaches the callback by calling the helper's `get_output` every 12 frames and
  handing over the new samples. The reference for correctness is a round trip: the test writes
  the audio to a wav and has Parakeet read it back. Pocket TTS files in llama.cpp's format come
  from EryriLabs/pocket-tts-GGUF; the Serveurperso ones are for another project and do not load.
  A fixed seed repeats the audio on the Mac and the Pixel 8; a Linux arm64 run differed by three
  frames between two calls, so no test may demand sample-exact repeats.
- **llama.cpp changes a small context behind your back.** Asked for `n_ctx = 32` it gave a
  context of 256 and a batch size below 52. Read both back (`llama_n_ctx`, `llama_n_batch`)
  instead of trusting what was asked for; a batch larger than `llama_n_batch` aborts inside `mtmd`.
- **A large image is slow far beyond its size.** 1080 image tokens took 21.7 s on the desktop;
  `image_max_tokens` defaults to 280. Test with a phone-sized photo, not only small images.
- **A GGUF's training context can be wrong.** EmbeddingGemma 2 states 262144 tokens; its card
  says 8192. Do not size a context from that number.
- **`kiln_bindings_generated.dart` is generated.** After changing `kiln.h`, run
  `dart run ffigen --config ffigen.yaml` in `flutter/kiln_flutter`; do not edit it by hand.
- **`flutter/kiln_flutter/src/core` wins over the repo source** when it exists (it is the
  vendored copy made by `scripts/vendor_flutter.sh`, git-ignored). Delete it after a publish
  check, or the plugin builds an old core against new bindings.
- **pub publishes by ignore rules, not by what is on disk.** `src/core` and `ios/Frameworks`
  are git-ignored in `flutter/kiln_flutter/.gitignore`, and `.pubignore` next to it replaces
  that file for pub so they are uploaded. A rule in the root `.gitignore` still applies to pub:
  an unanchored `models/` there once hid llama.cpp's `src/models`. `scripts/pub_files.py`
  lists what pub would upload.
- **Threads started inside a kiln call inherit one core.** The pinned threadpool ties the
  calling thread to a single core (it is worker 0). Anything that starts its own threads from
  that thread, as the mtmd image / audio encoders do, lands on that one core unless the caller
  is widened first: see `AffinityScope` in `media.h`. `top -H` on the device shows it.
- **Reference numbers from one processor are not exact on another.** A 4-bit model's vectors
  differ by up to 0.009 per number between x86, the Pixel 8 and the M4 GPU. Tests that compare
  with llama.cpp's numbers allow 0.015; tests that rank two texts need a real margin.
- **The Pixel 8's screen is damaged (2026-10).** adb, `scripts/android-test.sh`, the Flutter
  integration test and `adb exec-out screencap` all still work; only physical taps do not.
- **Core selection lives in `src/probe/fast_cores.h`** as pure functions with unit tests
  (`test_core_selection`). A new chip layout gets a test there with the values the phone
  reports (`cpuinfo_max_freq`, `cpu_capacity`) before the rule is touched.
- **`flutter test integration_test` overwrites `build/app/outputs/flutter-apk/app-debug.apk`**
  with a test-harness build that sits on the splash screen. Run `flutter build apk` again
  before installing the example app by hand.
- **A borrowed phone gets everything in one session**: tests, bench, Flutter test, app, and a
  saved profile in `docs/devices/`. Clean up after: uninstall the example app and remove
  `/data/local/tmp/kiln` (and `/data/local/tmp/llama.cpp`, the OpenCL kernel cache, if a GPU
  build was run).
- **GPU builds for Android are a trial only.** The OpenCL recipe and its two traps (vendor
  library path, static C++ runtime) are in `docs/devices/galaxy-s26-ultra.md`.
- **A shell glob that matches nothing makes `ls` fail**, and with `set -o pipefail` that ends
  the script without a message. This broke the Android scripts on macOS once.
- **Quantization: do not design a format before reading `docs/QUANT.md`.** On the Pixel 8,
  Q4_0 is 20% faster than Q4_K_M (109 against 90 tok/s) and an importance matrix recovers most
  of its quality; i8mm and KleidiAI change nothing for generation. `scripts/quant_eval.sh`
  reproduces the table (KL divergence from F16 on the host, kiln-bench on the phone).
- `third_party/llama.cpp` has its own `CLAUDE.md`/`AGENTS.md`; those are for that project, not this one.

## Not yet verified (as of 0.5.0)

0.5.0 (text to speech, `kiln_speak`, Pocket TTS): `kiln-tests models` gives 505 passed on macOS
(M4 Pro, Metal), Android (Pixel 8) and Linux arm64 (Docker) on 2026-10-13. Flutter `speak` with
26 integration tests (4 skipped: the speed and media groups, which passed at 0.3.1) on the Pixel 8, and the example app's Speak screen. Still open:

- Linux x86 (the Ryzen): done on 2026-10-14, 505 passed (the first run there with whisper.cpp and TTS).
- iOS, as before.
- Qwen3-TTS on a phone (1.5 GB); the other Pocket TTS languages; streaming playback in the
  example app; caching the encoded voice between calls (0.75 s per call on the Pixel 8).

## Earlier (0.4.0)

0.4.0 (speech to text, `kiln_transcribe`: Whisper and Parakeet through whisper.cpp, Qwen3-ASR
through llama.cpp): `kiln-tests models` gives 476 passed on macOS (M4 Pro, Metal) and Android
(Pixel 8) on 2026-10-13, after `scripts/fetch_models.sh --media --speech`; 398 on Linux x86
(Ryzen 5 8500G) before the whisper.cpp backend was added. The Flutter plugin has `transcribe`
(25 integration tests (4 skipped: the speed and media groups, which passed at 0.3.1) on the Pixel 8). Still open:

- Linux x86: done at 0.5.0 (see above). Linux arm64 in Docker gave 476 passed at 0.4.0.
- iOS: not built, not run; `scripts/build_ios.sh` does not know about whisper.cpp yet.
- Raw microphone samples (the example app records to a wav first), a 4-bit Qwen3-ASR file
  (ggml-org publishes Q8_0 and bf16 only), Whisper small / large and Parakeet's other
  quantisations (only base q5_1 and TDT 0.6B v3 q8_0 were run).
- `docs/ASR_COMPARISON.md` (accuracy of Whisper, Moonshine, Parakeet, Qwen3-ASR) is one Linux
  run whose raw files were lost; it has the steps to repeat it, not run on the Mac.
- pub.dev: `kiln_flutter` 0.4.0 was published on 2026-10-13 from the `v0.4.0` tag; a fresh app installed
  it from pub.dev and transcribed with Parakeet on the Pixel 8 (312 ms for the 3-second clip).
  0.5.0 was published on 2026-10-13 from the `v0.5.0` tag; a fresh app installed it from pub.dev, spoke with Pocket TTS on the Pixel 8
  (2.7 s of speech in 4.3 s) and Parakeet read the words back.

The notes below are from 0.3.1 and earlier.

Checked on 2026-10-11: `kiln-tests models` gives 154 passed on Linux x86 (Ryzen 5 8500G),
macOS (M4 Pro, Metal) and Android (Pixel 8). The Flutter integration test passes on the Pixel 8
(16 tests, including `decide`) with ffigen-generated bindings. Still open:

- iOS: the items already listed in `docs/RELEASE.md`.
- M4 Pro chat benchmarks: still the v0.1.0 figures. Re-run `scripts/bench.sh` on a quiet Mac.
  (Pixel 8 chat, and decision and embedding models on both, were run at v0.3.0 on 2026-10-13.)
- pub.dev: `kiln_flutter` 0.4.0 was published on 2026-10-13 (0.3.1 and 0.3.0 the same day, 0.2.0 on 2026-10-12), Android only. A published version
  cannot be changed: any fix needs a new version number. `scripts/publish_flutter.sh` publishes.

0.3.0 (embedding models, `kiln_embed` and `kiln_embed_media`): `kiln-tests models` gives 319
passed on macOS (M4 Pro, Metal) and Android (Pixel 8) on 2026-10-13, after
`scripts/fetch_models.sh --media`; 317 on Linux x86 before two checks were added. The Flutter
plugin has `embed` / `embedMedia` / `loadProjector` and is at 0.3.0: 23 integration tests on
the Pixel 8 (`--dart-define=KILN_MEDIA=true` for the image and audio test). Still open for it:

- Linux x86: done at 0.4.0 (see above). Linux arm64 in Docker on the Mac gave 319 passed at 0.3.0, and so did emulated x86_64
  (add `--platform linux/amd64`, about four times slower); see the command above.
- iOS, as for everything else.

Known gaps: speech models take one clip per call, from a file or file bytes, and the clip has to fit in the context; embedding models take one input per call; no video, no raw pixels or samples
(files or file bytes only); images and audio are for embedding models only, not chat. Decision models take a text state only (no images or audio,
only the lfm2-d1-omni family); the probe counts hyperthreads as cores and trusts max frequency
to tell cores apart (`docs/CPU.md` §8).
