# Changelog

## 0.3.0

The C ABI changed: `kiln_model_info` has two new fields at the end (its size is unchanged),
and there is one new function with its two structs.

- Embedding models: `kiln_embed` turns a text into a vector, and
  `kiln_model_info.is_embedding_model` / `n_embd` say when to use it and how long the vector
  is. `kiln_embed_options.kind` (query or document) adds the task prefix the model was trained
  with, `dimensions` shortens the vector, `normalize` scales it to length 1.
  A text longer than the context returns `KILN_ERR_CONTEXT_FULL`; nothing is cut silently.
- `kiln_chat` and `kiln_generate` on an embedding model return `KILN_ERR_NOT_SUPPORTED`.
  Before, kiln ran such a model as a chat model and printed unused tokens.
- `kiln-cli`: `--embed`, `--query` and `--dim`.
- `scripts/fetch_models.sh` downloads the two embedding test models (176 MB and 84 MB).

Tested with EmbeddingGemma 2 (`UD-Q4_K_XL`, 176 MB) and nomic-embed-text v1.5 (`Q4_K_M`,
84 MB), text only. The vectors are identical to the ones llama.cpp's `llama-embedding` tool
gives at the same commit. On a 16-question search test the 4-bit EmbeddingGemma 2 file picked
the same answers as the full-precision one (15 of 16).

Verified with `kiln-tests` (222 checks) on Linux x86 (Ryzen 5 8500G, CPU) only. Not run on
macOS, Android or iOS. The Flutter plugin has no `embed` yet: its bindings have to be
regenerated with ffigen, and its package version is unchanged until then. Images and audio
for EmbeddingGemma 2 (the `mmproj` file) are not supported.

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
