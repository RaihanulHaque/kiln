# Glossary

Terms used across kiln's API, CLI output, tests and benchmarks. If a number in a README
confuses you, it should be here. If it is not, add it.

## Generation

- **prompt tokens**: tokens in the fully rendered prompt (system + history + the new user turn +
  the assistant prefix the template adds). `kiln_gen_stats.prompt_tokens`.
- **cached tokens**: how many of those prompt tokens were already in the KV cache from the
  previous turn and did not need re-evaluating. `cached_tokens`. 0 on the first turn, and 0 on
  every turn for hybrid models whose template rewrites history (see *prefix reuse*).
- **prompt eval / prompt tok/s**: the phase where the model reads the prompt. Fast because it is
  batched. `prompt_ms`, `prompt_tps`; tok/s counts only the non-cached tokens.
- **generate / gen tok/s**: the phase where the model produces output one token at a time. This
  is the number people mean by "speed". `gen_ms`, `gen_tps`.
- **generated tokens**: everything the model emitted, including reasoning. `generated_tokens`.
- **reasoning tokens**: the subset of generated tokens that went to the reasoning channel.
  `reasoning_tokens`. Billable, visible in the think block, not part of the answer.
- **piece**: one fragment of text handed to the callback. Usually one token, sometimes less than
  one UTF-8 character; concatenate before decoding.
- **channel**: `KILN_CHANNEL_ANSWER` or `KILN_CHANNEL_REASONING`. Which stream a piece belongs to.
  Think tags themselves are stripped and never delivered.
- **EOS / EOG**: end-of-sequence / end-of-generation token. `stopped_by_eos` means the model
  chose to stop. If false and `stopped_by_user` is false, the run hit `max_tokens` or the
  context filled up.
- **max_tokens**: upper bound on generated tokens for one call. 0 = unbounded.
- **greedy**: temperature ≤ 0, always pick the most likely token. Deterministic, but small
  models loop forever in reasoning mode with it. Tests use it for answers, not for thinking.

## Context and memory

- **context / n_ctx**: the window of tokens the model can attend to, prompt plus output.
  `kiln_context_params.n_ctx`; 0 = 4096. Memory cost grows with it.
- **KV cache**: per-layer key/value tensors for every token in the context. What makes the next
  token cheap to produce. Lives inside a `kiln_context`.
- **prefix reuse**: `kiln_chat` compares the new prompt's tokens with what the KV cache holds
  and only evaluates the suffix that differs. Transformers (Gemma, Llama) reuse freely.
  Recurrent / hybrid models (LFM2, Qwen3.5, Mamba) cannot drop a partial tail of their state,
  so on a divergence kiln rewinds to the newest **checkpoint** (a saved copy of that state at
  the end of a turn's history) and re-evaluates from there.
- **reset**: `kiln_context_reset` clears the KV cache. Next call starts from nothing.
- **context full**: `KILN_ERR_CONTEXT_FULL`. Either the prompt alone does not fit, or
  generation ran out of room; `generated_tokens` tells you which.
- **RSS**: resident set size, the process's physical memory. For kiln that is roughly the
  model file (mmap'd, counted once touched) + KV cache + ggml compute buffers + the runtime.
- **mmap**: the model file is mapped, not read, so the OS can page it in lazily and drop clean
  pages under pressure. `use_mmap` in model params.

## Device and threading

- **fast cores / big cores**: cores outside the slowest frequency cluster. On a Tensor G3 that is
  1x Cortex-X3 + 4x A715 (the 4x A510 are excluded). `cpu_big_cores`, `cpu_fast_mask`.
- **affinity / pinned threadpool**: kiln binds ggml worker threads to the fast cores. Without it
  Android schedules workers onto little cores and every barrier waits for the slowest one.
- **threads**: worker thread count. Default = number of fast cores, capped at 8.
- **probe**: `kiln_probe_device`: SoC name, cores, RAM, ARM ISA features, GPU presence.
- **dotprod / i8mm / fp16 / sve**: ARM ISA extensions that speed up quantised matmuls. The
  Android build baseline is `armv8.2-a+dotprod+fp16`; i8mm and sve are detected but not yet
  used (runtime dispatch is on the roadmap).
- **backend**: which engine ran the model. `llama.cpp/cpu`, `llama.cpp/metal`, later `qnn`.
  `KILN_BACKEND_AUTO` picks CPU on Android and Metal on Apple silicon.

## Thermals (benchmarks)

- **cluster temperature**: the CPU die region, reported per cluster (BIG / MID / LITTLE). Hits
  ~80 °C within a second of sustained generation on the Pixel 8 and is then held there by
  lowering the clock.
- **skin temperature**: the phone's outside. Rises slowly. This is what the OS throttles on.
- **OS throttle status**: Android `ThermalStatus` 0 to 6. On the Pixel 8 status 1 begins at
  39 °C skin, 2 at 43 °C, 3 at 45 °C.
- **run N**: the N-th repetition of the benchmark job. Only the benchmark repeats; the app
  answers each prompt once. The repeats imitate a multi-message chat with heat as the only
  variable: run 1 is cold, run 5 is hot. See
  [bench/README.md](bench/README.md#why-the-same-prompt-runs-5-times).

## Packaging

- **xcframework**: Apple's bundle of one static library per platform slice (device arm64,
  simulator arm64). `scripts/build_ios.sh` makes kiln's; the podspec vendors it.
- **vendoring**: copying the C core and the pinned llama.cpp into the Flutter package so it
  builds without the monorepo. `scripts/vendor_flutter.sh`; required for pub.dev.
- **integration test**: a Flutter test that runs inside the real app on a real device, as
  opposed to a unit test on the host. `example/integration_test/`.

## Models and files

- **GGUF**: the single-file model format llama.cpp reads. Weights + tokenizer + chat template +
  metadata. kiln accepts nothing else today.
- **quant / Q4_K_M / Q8_0**: how the weights are compressed. Q4_K_M ≈ 4.5 bits per weight,
  good default for phones. Q8_0 ≈ 8.5 bits, closer to full quality, twice the memory and
  bandwidth.
- **arch**: model family as named in the GGUF (`lfm2`, `qwen35`, `gemma3`). Decides the
  compute graph and whether the model is recurrent / hybrid.
- **chat template**: the jinja program inside the GGUF that turns a message list into the
  exact text the model was trained on. kiln executes it faithfully via llama.cpp/minja.
- **thinking budget**: `max_reasoning_tokens` in `kiln_chat_options`. After that many reasoning
  tokens kiln closes the think block itself and the model goes on to answer. 0 = unlimited.
  `reasoning_truncated` in the stats reports that it was used.
- **enable_thinking**: template switch for reasoning models. `supports_thinking` in model info
  is true only when flipping it actually changes the rendered prompt.
- **decision model**: a model that scores the options of a typed question (noul, choice, score)
  in one forward pass instead of generating text. `kiln_decide`, `is_decision_model`. LFM2
  d1-omni is the supported family.
- **speech model / ASR**: a model that writes down what is said in an audio clip (automatic
  speech recognition). `kiln_transcribe`. Whisper and Parakeet are whole speech models in one
  file (through whisper.cpp; `is_speech_model`); Qwen3-ASR is a small text model plus an audio
  encoder (through llama.cpp, with a projector).
- **real-time factor**: seconds of audio divided by the seconds it took to write them down.
  5 means five times faster than the speaker; below 1 the app cannot keep up.
- **hybrid model**: mixes attention layers with recurrent / state-space layers (LFM2, Qwen3.5).
  Fast and memory-light per token, but their state cannot be partially rewound.
