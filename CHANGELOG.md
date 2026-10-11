# Changelog

## 0.2.0

The C ABI changed: `kiln_chat_options`, `kiln_gen_stats` and `kiln_model_info` grew, so code
built against 0.1.0 headers must be rebuilt.

- Hybrid models (Qwen3.5, LFM2) reuse the KV cache across turns through state checkpoints.
  On a 6-turn Qwen3.5 0.8B chat, prompt time went from 3232 ms to 1251 ms in total.
- Thinking budget: `kiln_chat_options.max_reasoning_tokens` closes the think block after N
  reasoning tokens so the model answers; `kiln_gen_stats.reasoning_truncated` reports it.
- Decision models: `kiln_decide` and `kiln_model_info.is_decision_model`. LFM2 d1-omni with a
  text state; no images or audio, not in the Flutter plugin.
- Fixed a crash when `KILN_CPU_MASK` had fewer CPUs than the default thread count (OpenMP builds).
- Build: the llama.cpp submodule is initialised at configure time if it is missing; Android
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
