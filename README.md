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

Status: **v0.1, pre-alpha.** C core and CLI harness work on macOS and cross-compile for Android.
Flutter bindings are next.

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

Targets `android-29`, `arm64-v8a`, baseline ISA `armv8.2-a+dotprod+fp16`. Vulkan is
available as the `android-arm64-vulkan` preset and is experimental: on most phone GPUs it is
not faster than the CPU path for small models. Measure before you ship it.

## API in one screen

```c
kiln_model* m;   kiln_model_load("model.gguf", NULL, &m);
kiln_context* c; kiln_context_create(m, NULL, &c);

kiln_chat_message msgs[] = {{"user", "Hi"}};
char prompt[4096];
kiln_apply_chat_template(m, msgs, 1, true, prompt, sizeof prompt);

kiln_gen_stats st;
kiln_generate(c, prompt, NULL, on_piece, NULL, &st);   // on_piece streams tokens
// kiln_cancel(c) from any thread stops it

kiln_model_info info; kiln_model_get_info(m, &info);   // info.backend_name, info.model_bytes ...
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and the roadmap.

## License

MIT.
