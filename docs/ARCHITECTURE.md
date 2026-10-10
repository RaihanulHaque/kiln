# Architecture

## Layers

```
 Flutter (Dart FFI)     Kotlin/Java (JNI)        <- generated from kiln.h, never hand-written
 ---------------------------------------------
 kiln.h   stable C ABI, opaque handles, status codes, callbacks
 ---------------------------------------------
 core     registry picks a backend, probe describes the device, errors are thread-local
 ---------------------------------------------
 backends llama.cpp (CPU / Metal / Vulkan)   [planned] QNN   [planned] LiteRT-LM
```

## Rules

1. **Nothing but C crosses `kiln.h`.** No exceptions, no std types, no structs with padding
   surprises. Every struct that may grow gets a `_default()` constructor so callers never
   zero-init by hand.
2. **Backends implement `kiln::Backend / Model / Context`** (see `src/core/internal.h`).
   The registry order is the priority order. llama.cpp is always last and always present.
3. **`KILN_BACKEND_AUTO` is conservative.** It picks the GPU only where that is known to be a
   win (Apple silicon). On Android it stays on CPU until a device-specific benchmark says
   otherwise. Explicit `KILN_BACKEND_GPU` is honoured or fails loudly, never silently
   downgraded.
4. **`kiln_model_info` and `kiln_gen_stats` are not optional niceties.** They are the product.
   The Flutter sample app will surface them on screen.
5. **Templates are the model's, executed faithfully.** No per-model special cases in kiln. If a
   template needs a kwarg (`enable_thinking`), it goes through `kiln_chat_options`.
6. **Reasoning is a channel, not a string hack.** `ReasoningSplitter` watches for the template's
   own think tags (from `common_chat_params`) and handles tags split across token pieces.

## Threading

- `kiln_generate` blocks the calling thread. Bindings run it on a worker and marshal pieces
  back to the UI thread.
- `kiln_cancel` is safe from any thread. It flips an atomic that is checked between tokens and
  inside ggml via the abort callback, so a long prompt eval also stops promptly.
- One `kiln_context` is single-consumer. Many contexts may share one `kiln_model`.

## Roadmap

- [x] C core, llama.cpp backend, probe, CLI, host + Android builds
- [x] Clamp reported `n_gpu_layers_used` to real layer count; map `mtl` -> `metal`
- [x] Fast-core detection + pinned ggml threadpool on Android (Pixel 8: 61 -> 108 tok/s)
- [ ] Validate the "highest core index = biggest core" assumption on Snapdragon and MediaTek
- [x] Jinja chat templates via llama.cpp common/minja; `enable_thinking` detected empirically
- [x] Reasoning / answer channels in `kiln_token_cb`, tags stripped, split-across-pieces safe
- [x] `kiln_chat` with KV prefix reuse; recurrent-model fallback; EOG cached so turns chain
- [x] Test suite (host + adb), 127 checks green on M4 Pro and Pixel 8
- [x] v0.1.0: header frozen
- [x] kiln-bench: RSS / CPU / freq / thermal over time, host + device, charts (docs/bench)
- [ ] Thermal-aware threading: test 4 vs 5 threads for *sustained* tok/s on Pixel 8
- [ ] GPU utilisation in kiln-bench on Metal
- [ ] KV state checkpoints (`llama_state_seq_*`) so hybrid models reuse cache across turns
- [ ] Tool calling: common already parses it; expose `kiln_tool` + a TOOL_CALL channel
- [ ] Runtime ISA dispatch (i8mm / sve) instead of one baseline build
- [ ] Context-length auto-cap from available RAM in the probe
- [x] Flutter plugin (`kiln_flutter`) via `ffigen`, worker isolate, example chat app verified on Pixel 8
- [ ] iOS: xcframework build of the core and a podspec that vendors it
- [ ] Example app: throttle setState per piece (UI rebuild costs ~15% tok/s on Pixel 8)
- [ ] Opt-in benchmark upload from the sample app -> real device compatibility table
- [ ] QNN backend for Snapdragon NPUs, gated on probe + model artifact availability
- [ ] JNI binding for Kotlin/Java
- [ ] Model manager: download, verify, cache, choose variant per device
