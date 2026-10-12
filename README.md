<p align="center">
  <img src="docs/brand/kiln-lockup.png" width="260" alt="kiln">
</p>

<p align="center"><b>On-device LLM runtime for Android and desktop, with iOS in progress. Honest about what it runs on.</b></p>

<p align="center">
  <img src="docs/img/kiln_promo.gif" width="720" alt="kiln: ember ignites, a streaming reply with live tok/s, then the numbers">
</p>

Every mobile LLM library promises "runs on phones" and backs it with a device list the
maintainers happened to own. kiln is built the other way round:

- **The slow path is the promise.** llama.cpp on CPU with a Q4 model is the guaranteed tier.
  It works on any arm64 device with enough RAM.
- **Acceleration is detected, never assumed.** Metal on Apple silicon today; NPU backends
  later. Enabled only when the probe finds them, never silently.
- **The API tells you what actually happened.** Which backend loaded, how much memory it
  took, real tokens per second for every call, how much of the conversation was reused.

## Status

v0.3.0 adds embedding models (text, images and audio). It was run on 2026-10-12 and
2026-10-13 on the Linux desktop, the MacBook, a Linux container and the Pixel 8. The iPhone row is still from
v0.1.0, and the chat benchmarks were not repeated since v0.1.0:

| platform | backend | what passed |
|---|---|---|
| Pixel 8 (Tensor G3, Android 16), v0.3.0 | llama.cpp CPU, 5 threads pinned | 319 C checks, 23 Flutter integration tests; benchmarks at v0.1.0 |
| iPhone 16 Pro (A18 Pro, iOS 26.6), v0.1.0 | llama.cpp Metal | **In progress, not supported yet.** 7 of 9 Flutter integration tests at v0.1.0; not re-run since |
| MacBook Pro M4 Pro, v0.3.0 | llama.cpp Metal | 319 C checks; benchmarks at v0.1.0 |
| Ryzen 5 8500G desktop (Linux x86_64), v0.3.0 | llama.cpp CPU | 317 C checks, before two checks were added on the Mac; not re-run since |
| Linux arm64 and emulated x86_64 (Ubuntu 24.04 in Docker on the M4 Pro), v0.3.0 | llama.cpp CPU, OpenMP | 319 C checks on each; arm64 also with `KILN_CPU_MASK=3` |

Models verified end to end: LFM2.5 350M, Qwen3.5 0.8B (thinking on and off), Gemma 3 270M,
the decision model LFM2 d1-omni 600M, and the embedding models EmbeddingGemma 2 (text, images,
audio), nomic-embed-text v1.5 (text) and jina-embeddings-v5-omni-nano (text, images). All of
them through the C API; chat, `decide`, `embed` and `embedMedia` (EmbeddingGemma 2) through the
Flutter plugin on the Pixel 8.
**iOS is in progress.** It built and ran on an iPhone at v0.1.0, but it has not been re-run for
v0.2.0 and is not part of the pub.dev package. Use Android for now.

Not yet: Snapdragon or MediaTek phones, 4 GB RAM devices, x86 emulators.
`kiln_flutter` 0.3.0 is on pub.dev as an Android-only package. The
[release checklist](docs/RELEASE.md) lists what is still open.

## Quick start

### Flutter

[`kiln_flutter`](https://pub.dev/packages/kiln_flutter) is on pub.dev:

```bash
flutter pub add kiln_flutter
```

Set `minSdk = 29` and run on an arm64 Android phone. **[Getting started](docs/GETTING_STARTED.md)**
walks through every step, from an empty project to a first chat and a first decision.

```dart
final kiln  = await Kiln.create();
final model = await kiln.loadModel('/path/to/model.gguf');
print(model.info);                       // qwen35 Q4_K_M, 0.75B, via llama.cpp/cpu, thinking

final ctx = await model.createContext();
final gen = ctx.chat([KilnChatMessage.user('Why is the sky blue?')], enableThinking: true);
await for (final p in gen.pieces) {
  p.isReasoning ? showThinking(p.text) : showAnswer(p.text);
}
print(await gen.stats);                  // 36.8 tok/s, 21 of 40 prompt tokens cached, [eos]
gen.cancel();                            // from anywhere, any time
```

Package: [flutter/kiln_flutter](flutter/kiln_flutter/README.md). Everything blocking runs on a
background isolate. The example app in `flutter/kiln_flutter/example` is a full chat UI with
a model downloader, thinking toggle, collapsible reasoning and per-reply stats, plus a Decide
screen for decision models and a Search screen for embedding models.

<p>
<img src="docs/img/app_library.png" width="180"> <img src="docs/img/app_chat_thinking.png" width="180"> <img src="docs/img/app_chat_reply.png" width="180"> <img src="docs/img/app_decide.png" width="180"> <img src="docs/img/app_search_media.png" width="180"> <img src="docs/img/app_search_result.png" width="180"> <img src="docs/img/app_library_light.png" width="180">
</p>

### C

```c
kiln_model*   m; kiln_model_load("model.gguf", NULL, &m);
kiln_context* c; kiln_context_create(m, NULL, &c);

bool on_piece(kiln_channel ch, const char* piece, int32_t tok, void* user) {
    fputs(piece, stdout);                // ch: KILN_CHANNEL_ANSWER or KILN_CHANNEL_REASONING
    return true;                         // false = stop
}

kiln_chat_message history[] = {{"user", "Hi"}};
kiln_gen_stats st;
kiln_chat(c, history, 1, NULL, NULL, on_piece, NULL, &st);   // pass the full history each turn
```

```bash
git clone --recursive https://github.com/RaihanulHaque/kiln && cd kiln
cmake --preset host && cmake --build --preset host -j        # macOS / Linux
./build/host/kiln-cli --probe
./build/host/kiln-cli -m model.gguf --chat                   # interactive, shows cache reuse per turn

cmake --preset android-arm64 && cmake --build --preset android-arm64 -j
scripts/android-run.sh model.gguf -n 64                      # push to a phone over adb and run
scripts/build_ios.sh                                         # xcframework for the Flutter plugin
```

The Android presets and scripts find the NDK under `~/Library/Android/sdk/ndk` (macOS) or
`~/Android/Sdk/ndk` (Linux); set `ANDROID_NDK_HOME` if it lives elsewhere.

Already cloned without `--recursive`? Run `git submodule update --init --recursive` (the CMake
configure step also tries this automatically when `third_party/llama.cpp` is empty).

## What makes it honest

**Threads are pinned to the fast cores.** The probe groups cores by clock, drops the slowest
tier and binds one ggml worker to each remaining core. On a Pixel 8 that is the difference
between 21 tok/s and 3 tok/s once the phone is warm: unpinned, Android parks workers on the
little cores and every matmul waits for the slowest one. [Sweep](docs/bench/threads/README.md).

**Templates are the model's own.** The GGUF's jinja template runs through llama.cpp's minja
engine, so system prompts, Gemma's role rules and Qwen's `enable_thinking` switch all behave
as the model authors intended. No per-model code in kiln.

**Reasoning is a channel.** Think tags are stripped and each piece is tagged answer or
reasoning, with `reasoning_tokens` in the stats, so a UI can show a collapsible thinking
block and count tokens truthfully. Small reasoning models often never stop thinking, so
`max_reasoning_tokens` sets a budget: when it is reached kiln closes the think block and the
model answers. `reasoning_truncated` in the stats says when that happened.

**Multi-turn reuses the KV cache.** Pass the whole history; kiln evaluates only the suffix
that changed. Hybrid models (LFM2, Qwen3.5) cannot rewind partial state, so kiln saves a
checkpoint of it at the end of each turn's history and rewinds to that when the template
rewrites the reply that follows. Reported in `cached_tokens`.

**Stats are per call.** Prompt and generation tok/s, cache hits, why it stopped. On a phone
the number falls as the chip heats; kiln lets the app see that instead of hiding it.

**Decision models answer without generating.** A model such as LFM2 d1-omni reads a state and
a typed question (yes / no, pick one, rate on a scale) and returns a probability per option in
one forward pass. `kiln_decide` runs it; `kiln_model_info.is_decision_model` says when to use
it. Text state only for now, and C API only (not yet in the Flutter plugin).

```c
kiln_decision_option teams[] = {{"billing", "Payments and refunds"}, {"shipping", "Delivery problems"}};
kiln_decision_question q = {"team", KILN_QUESTION_CHOICE, "Which team should handle this?", teams, 2};
float probs[2]; kiln_decision_answer a;
kiln_decide(ctx, "My parcel never arrived.", &q, probs, 2, &a);   // teams[a.best], probs[], a.confidence
```

**Embedding models turn text into a vector.** Texts with a similar meaning get similar
vectors, which is what search over the user's own notes or documents needs. `kiln_embed` runs
one; `kiln_model_info.is_embedding_model` says when to use it and `n_embd` how long the vector
is. Tell kiln whether a text is the query or a document and it adds the prefix the model was
trained with:

```c
float q[768], d[768];
kiln_embed_options o = kiln_embed_options_default();
o.kind = KILN_EMBED_QUERY;
kiln_embed(ctx, "how do I reset my password", &o, q, 768, NULL);
o.kind = KILN_EMBED_DOCUMENT;
kiln_embed(ctx, "Steps to recover account access: open Settings, then Security.", &o, d, 768, NULL);
float similarity = 0;                       // vectors have length 1, so the dot product is the cosine
for (int i = 0; i < 768; ++i) similarity += q[i] * d[i];
```

Some embedding models also take images and audio and put them in the same space as text, so
a text query can find a photo or a voice note. They need a second file, the projector:

```c
kiln_model_load_projector(m, "mmproj-embeddinggemma-2-Q8_0.gguf", NULL, NULL);
kiln_media photo = {"holiday.jpg", NULL, 0};          // a path, or the bytes of the file
kiln_embed_media(ctx, NULL, &photo, 1, NULL, d, 768, NULL);
// compare d with the vector of the query "a lake between mountains", as above
```

Storing and searching the vectors is the app's job. C API only for now, measured on a Linux
desktop CPU with EmbeddingGemma 2 at 4 bits: about 15 ms for a short sentence, 0.7 s for a
small image, 0.2 s for a 3-second audio clip, and about 1 GB of memory once the projector is
loaded (0.4 GB without it).

## Measured

Generation tok/s, 3 to 8 runs back to back of the same job. "cold" is the first run,
"warm" the last. Details, charts and raw CSVs in [docs/bench](docs/bench/README.md).

| model | quant | resident | Pixel 8 cold → warm | iPhone 16 Pro | M4 Pro Metal |
|---|---|---|---|---|---|
| LFM2.5 350M | Q4_K_M | 0.23 GB | 111 → 90 | runs, not benchmarked | 210 to 360 |
| Qwen3.5 0.8B | Q4_K_M | 0.52 GB | 38 → 31 | not run | 140 to 147 |
| Gemma 3 270M | Q8_0 | 0.29 GB | 80 → 71 | not run | 118 to 166 |

Sustained generation on the Pixel 8 holds the CPU cluster at 80 °C and steps the fast cores
from 2.48 GHz to 1.69 GHz within a minute; token rate follows. Any phone benchmark shorter
than a minute reports the cold number.

<img src="docs/bench/pixel8_Qwen3.5-0.8B-Q4_K_M.png" width="720">

### Decision and embedding models

These do not generate text, so tokens per second does not apply. What matters in an app is the
time for one call. Measured with `kiln-cli` on 2026-10-13; the first call after loading a model
is slower (up to about 3 times) and is not counted. Times are for the whole call; for images
and audio most of it is the encoder (about 2.5 s and 0.6 s on the Pixel 8). They are not part of `kiln-bench` and have
no charts or sustained-load runs yet.

| what one call does | model | Pixel 8 | M4 Pro Metal |
|---|---|---|---|
| answer one yes / no question about a short message (56 tokens) | LFM2 d1-omni 600M, Q8_0 | 190 to 235 ms | 38 ms |
| embed a sentence (18 tokens) | EmbeddingGemma 2, UD-Q4_K_XL | 40 to 60 ms | 10 to 13 ms |
| embed a small image (81 media tokens) | the same, with its Q8_0 projector | 2.8 s | 95 ms |
| embed a 3-second audio clip (78 media tokens) | the same | 0.9 s | 43 ms |

In practice on the phone: sorting a message into one of a few classes feels instant, a search
query over notes that are already embedded takes well under a tenth of a second, and photos
and audio are slow enough that an app should embed them in the background when they are added.

## Repository

```
include/kiln/kiln.h          the C API; bindings are generated from this file only
src/core/                    C API implementation, backend registry, error plumbing
src/probe/                   device probe: SoC, cores, clocks, RAM, ISA features
src/backends/llamacpp/       the guaranteed-tier backend
tools/kiln-cli, kiln-bench   test harness and resource benchmark
tools/barrier-lab            standalone experiment: how ggml's per-op barrier reacts to slow cores
tests/                       C test suite (host and device)
flutter/kiln_flutter/        Flutter plugin, example app, integration test
scripts/                     model download, adb runners, benchmarks, vendoring, iOS build, brand assets
docs/                        concepts, codebase walkthrough, architecture, glossary, benchmarks
third_party/llama.cpp        pinned submodule
```

## Tests

```bash
scripts/fetch_models.sh                                      # the six test GGUFs into models/ (~1.7 GB); --media adds 0.9 GB
./build/host/kiln-tests models                               # C suite on the host
scripts/android-test.sh                                      # C suite on a phone over adb
cd flutter/kiln_flutter/example && flutter test integration_test -d <device>
```

The C suite needs the six test GGUFs in `models/` (`scripts/fetch_models.sh`); missing ones are skipped. The Flutter
integration test downloads what it needs. Add `--dart-define=KILN_SPEED=true` for a
per-model speed run through the plugin.

## Docs

Start with [CONCEPTS](docs/CONCEPTS.md), then the [codebase walkthrough](docs/CODEBASE.md).
Also: [architecture and roadmap](docs/ARCHITECTURE.md), [CPU notes](docs/CPU.md),
[glossary](docs/GLOSSARY.md), [benchmarks](docs/bench/README.md),
[release checklist](docs/RELEASE.md).

## License

MIT.
