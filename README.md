# kiln

**On-device LLM runtime for Android and beyond. Honest about what it runs on.**

Every mobile LLM library promises "runs on phones" and backs it with a device list the
maintainers happened to own. kiln is built the other way round:

- **The slow path is the promise.** llama.cpp on CPU with a Q4 model is the guaranteed tier.
  It works on any arm64 device with enough RAM, full stop.
- **Acceleration is detected, never assumed.** GPU and NPU backends are enabled only when the
  runtime probe finds them, and silently fall back when they fail.
- **The API tells you what actually happened.** Which backend loaded, how much memory it took,
  real prompt and generation tokens per second. No README can lie about that.

Status: **v0.1.0.** C core, CLI harness, and a test suite that passes on macOS and on a Pixel 8
with LFM2.5, Qwen3.5, and Gemma 3. The C header is frozen at this tag. Flutter bindings are next.

## Layout

```
include/kiln/kiln.h      stable C ABI - bindings are generated from this file only
src/core/                C API impl, error plumbing, backend registry
src/probe/               CPU / RAM / ISA feature probe (macOS, Linux, Android)
src/backends/llamacpp/   guaranteed-tier backend (CPU + Metal/Vulkan via ggml)
tools/kiln-cli/          test harness: load a GGUF, run a prompt, print the truth
third_party/llama.cpp    pinned submodule
```

## Build (host, macOS or Linux)

```bash
git clone --recursive https://github.com/<you>/kiln && cd kiln
cmake --preset host && cmake --build --preset host -j
./build/host/kiln-cli --probe
./build/host/kiln-cli -m path/to/model.gguf -p "Why is the sky blue?"
```

Sample output on an M4 Pro with a 350M model, showing why the stats matter
(CPU beats the GPU on a model this small):

```
backend:  llama.cpp/cpu (gpu layers: 0)
model:    lfm2 Q4_K - Medium, 0.35B params, 0.23 GB resident
prompt:   20 tok in 17 ms (1163.0 tok/s)
generate: 18 tok in 35 ms (517.4 tok/s) [eos]
```

## Build (Android arm64)

Needs an NDK under `~/Library/Android/sdk/ndk` or `ANDROID_NDK_HOME` set.

```bash
cmake --preset android-arm64 && cmake --build --preset android-arm64 -j
scripts/android-run.sh path/to/model.gguf -n 64      # pushes to a connected device via adb
```

Measured on a Pixel 8 (Tensor G3, 1x X3 + 4x A715 + 4x A510) with the same 350M Q4_K_M model:

| threads | affinity | generate tok/s |
|---|---|---|
| 1 | none | 61 |
| 5 | none | 22 to 32 |
| 5 (auto) | pinned to the 5 fast cores | 108 to 111 |

Without affinity, more threads is *slower*: Android schedules ggml workers onto the A510 little
cores and every barrier waits for the slowest one. kiln's probe groups cores by max frequency,
drops the slowest cluster, and pins a ggml threadpool to the rest. This is the kind of thing
that makes "runs on phones" true or false.

### Results so far

Generate tok/s, Q4_K_M, CPU path with pinned threads on the Pixel, Metal on the Mac.

| model | params | resident | Pixel 8 (Tensor G3) | M4 Pro Metal | M4 Pro CPU |
|---|---|---|---|---|---|
| LFM2.5 350M | 0.35B | 0.23 GB | 108 to 111 | 173 | 517 |
| Qwen3.5 0.8B | 0.75B | 0.52 GB | 41 to 42 | 124 | 173 |

Reasoning models: Qwen3.5's own template defaults to thinking **off** by emitting an empty
`<think>` block, but llama.cpp's built-in template matcher reduces it to plain chatml and the
model then thinks whenever it feels like it, burning your token budget on drafts. kiln's
`kiln_chat_options.enable_thinking` (default false) reproduces the template's intent.

Targets `android-29`, `arm64-v8a`, baseline ISA `armv8.2-a+dotprod+fp16`. Vulkan is
available as the `android-arm64-vulkan` preset and is experimental: on most phone GPUs it is
not faster than the CPU path for small models. Measure before you ship it.

## API in one screen

```c
kiln_model* m;   kiln_model_load("model.gguf", NULL, &m);
kiln_context* c; kiln_context_create(m, NULL, &c);

bool on_piece(kiln_channel ch, const char* piece, int32_t tok, void* user) {
    // ch is KILN_CHANNEL_ANSWER or KILN_CHANNEL_REASONING; think tags never reach you
    fputs(piece, stdout);
    return true;                                   // false = stop
}

kiln_chat_message history[] = {{"user", "Hi"}};
kiln_chat_options opts = kiln_chat_options_default();   // enable_thinking = false
kiln_gen_stats st;
kiln_chat(c, history, 1, &opts, NULL, on_piece, NULL, &st);
// st.cached_tokens tells you how much of the conversation was reused from the KV cache.
// Pass the full history every turn; kiln diffs it against what the cache already holds.
// kiln_cancel(c) from any thread stops a running call.

kiln_model_info info; kiln_model_get_info(m, &info);   // backend_name, model_bytes, supports_thinking ...
```

### Chat templates and reasoning

kiln runs the model's own jinja template through llama.cpp's minja engine, not the
heuristic built-in matcher. That means system prompts, role alternation rules (Gemma), and
`enable_thinking` all behave the way the model authors intended. Verified: LFM2.5, Qwen3.5,
Gemma 3. Anything with a sane jinja template should work; file an issue with the GGUF if not.

Reasoning models stream on two channels. The think tags themselves are stripped and the
`reasoning_tokens` count is in the stats, so a UI can show a collapsible "thinking" block and
bill tokens honestly.

### Multi-turn and the KV cache

`kiln_chat` keeps the tokens currently in the KV cache and only evaluates the suffix that
changed. Pure transformers (Gemma, Llama) get full reuse. Hybrid and recurrent models (LFM2,
Qwen3.5, Mamba) cannot drop a partial tail of their state, so kiln falls back to a full
re-evaluation whenever the history diverges before the end of the cache. With Qwen3.5 that
happens every turn because its template strips the previous think block from history. Correct,
but slower; state checkpoints are on the roadmap.

## Tests

```bash
cmake --build --preset host && ./build/host/kiln-tests models     # host
scripts/android-test.sh                                            # device via adb
```

Needs the three test GGUFs in `models/` (LFM2.5-350M Q4_K_M, Qwen3.5-0.8B Q4_K_M,
Gemma-3-270M-it Q8_0). Missing models are skipped. The suite covers probe, bad files,
tokenizer round trip, template rendering, single and multi-turn chat, callback stop,
cross-thread cancel, reset, thinking on/off channel separation, and context overflow in both
the prompt and generation phases.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and the roadmap.

## License

MIT.
