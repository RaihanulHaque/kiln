# kiln

On-device LLM runtime: a C core (`include/kiln/kiln.h`) over llama.cpp, with a device probe,
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
scripts/fetch_models.sh                      # four test GGUFs into models/ (~1.4 GB)
./build/host/kiln-tests models               # 154 checks at 0.2.0; a missing model is skipped
./build/host/kiln-cli -m models/<file>.gguf --chat
```

Android: `cmake --preset android-arm64`, then `scripts/android-test.sh`. iOS: `scripts/build_ios.sh`
(macOS only). The NDK is looked up in the macOS and Linux default locations, or `ANDROID_NDK_HOME`.

## Things that are easy to get wrong

- **Linux builds use OpenMP; macOS and Android builds do not.** A threadpool smaller than
  `cp.n_threads` crashes only on Linux. Keep the context's thread count and the pool size equal.
- **`kiln-bench` prints its per-run result lines on stderr** and the samples on stdout:
  `kiln-bench ... 2>&1 >/dev/null | grep -E '^[0-9],[0-9]+,'`.
- **Benchmark only on a quiet machine.** A video call cut tok/s by a third on unchanged code.
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
- **`kiln_bindings_generated.dart` is generated.** After changing `kiln.h`, run
  `dart run ffigen --config ffigen.yaml` in `flutter/kiln_flutter`; do not edit it by hand.
- **`flutter/kiln_flutter/src/core` wins over the repo source** when it exists (it is the
  vendored copy made by `scripts/vendor_flutter.sh`, git-ignored). Delete it after a publish
  check, or the plugin builds an old core against new bindings.
- **A shell glob that matches nothing makes `ls` fail**, and with `set -o pipefail` that ends
  the script without a message. This broke the Android scripts on macOS once.
- `third_party/llama.cpp` has its own `CLAUDE.md`/`AGENTS.md`; those are for that project, not this one.

## Not yet verified (as of 0.2.0)

Checked on 2026-10-11: `kiln-tests models` gives 154 passed on Linux x86 (Ryzen 5 8500G),
macOS (M4 Pro, Metal) and Android (Pixel 8). The Flutter integration test passes on the Pixel 8
(16 tests, including `decide`) with ffigen-generated bindings. Still open:

- iOS: the items already listed in `docs/RELEASE.md`.
- Benchmarks on 0.2.0. Run them on a cool phone: after the test suites the Pixel 8 capped its
  fast cores at 1.07 of 2.37 GHz and LFM2.5 ran at 55 tok/s instead of about 105.
- `v0.2.0` is not tagged.

Known gaps: decision models take a text state only (no images or audio,
only the lfm2-d1-omni family); the probe counts hyperthreads as cores and trusts max frequency
to tell cores apart (`docs/CPU.md` §8).
