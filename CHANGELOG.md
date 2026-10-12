# Changelog

## Unreleased

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
  the new choice. Chips with little cores are not affected (unit-tested with a Tensor G3
  layout; the Pixel 8 itself was not re-run). No ABI change.
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
