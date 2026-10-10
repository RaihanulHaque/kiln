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

## Threading

- `kiln_generate` blocks the calling thread. Bindings run it on a worker and marshal pieces
  back to the UI thread.
- `kiln_cancel` is safe from any thread. It flips an atomic that is checked between tokens and
  inside ggml via the abort callback, so a long prompt eval also stops promptly.
- One `kiln_context` is single-consumer. Many contexts may share one `kiln_model`.

## Roadmap

- [x] C core, llama.cpp backend, probe, CLI, host + Android builds
- [ ] Clamp reported `n_gpu_layers_used` to real layer count; map `mtl` -> `metal`
- [ ] Context-length auto-cap from available RAM in the probe
- [ ] Flutter plugin (`kiln_flutter`) via `ffigen`, with a sample chat app that shows stats
- [ ] Opt-in benchmark upload from the sample app -> real device compatibility table
- [ ] QNN backend for Snapdragon NPUs, gated on probe + model artifact availability
- [ ] JNI binding for Kotlin/Java
- [ ] Model manager: download, verify, cache, choose variant per device
