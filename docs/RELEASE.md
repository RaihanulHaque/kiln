# Release checklist

What has to be true before a version is tagged, and before `kiln_flutter` goes to pub.dev.

## Before tagging the core

- [ ] `cmake --build --preset host && ./build/host/kiln-tests models` green on macOS
- [ ] `scripts/android-test.sh` green on at least one Snapdragon, one Tensor, one MediaTek
      device (today: Tensor G3 only)
- [ ] `kiln-cli --chat` manual run with each supported family: LFM2, Qwen3.5 (think on/off),
      Gemma 3; and one new GGUF the suite has never seen
- [ ] `scripts/bench.sh` run; numbers in `docs/bench` refreshed if the core changed
- [ ] Header diff reviewed; any ABI change bumps the minor version and is listed in CHANGELOG
- [ ] llama.cpp submodule tag recorded in CHANGELOG

## Before publishing kiln_flutter

Not yet done, and required:

- [ ] Vendor the native sources into the package (`include/`, `src/`, root `CMakeLists.txt`,
      llama.cpp at the pinned tag) so the package builds without the parent repo
- [ ] `flutter build apk --release` from a clean clone of the package alone
- [ ] Integration test in `example/integration_test/` that loads a small model from the
      device and asserts a streamed reply, stats, cancel, and a second turn with cache reuse
- [ ] Run the example on a 4 GB RAM device with a 0.8B model; confirm graceful
      `KILN_ERR_OUT_OF_MEMORY` rather than a kill
- [ ] Background/foreground the app mid-generation; rotate the screen; confirm no crash and
      cancel works
- [ ] Load a corrupt file and a non-GGUF file from the UI; confirm the error surfaces
- [ ] `dart pub publish --dry-run` clean: README, CHANGELOG, LICENSE, pubspec description,
      no `example/build`, no models
- [ ] iOS: either supported (xcframework + podspec) or `platforms:` in pubspec lists Android
      only so pub.dev does not advertise what does not exist
- [ ] One external person has followed the README from zero to a running app

## Known limitations to state in the release notes

- Android arm64 only; no x86_64 emulator build; no iOS
- CPU backend only on Android (Vulkan preset is experimental and unmeasured)
- Hybrid models re-evaluate the full prompt when history diverges before the cache end
- Tool calling not exposed
- ISA baseline is dotprod+fp16; i8mm/SVE present on many phones are not used yet
