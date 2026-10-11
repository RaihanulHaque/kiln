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

- [x] Vendor the native sources into the package (`scripts/vendor_flutter.sh`) so it builds
      without the parent repo. 2026-10-12: `--check` builds the example APK from exactly the
      files pub would upload (1152 files, 11 MB compressed) in a folder outside the repo, and
      that APK ran on the Pixel 8. The first dry run had shipped 75 KB with no native code,
      because `src/core` was git-ignored; `.pubignore` now handles that.
- [x] Integration test in `example/integration_test/` that loads a small model from the
      device and asserts a streamed reply, stats, cancel, and a second turn with cache reuse.
      Status 2026-10-10: Pixel 8 all 9 core tests green; iPhone 16 Pro 7 of 9 before the
      `StreamController.close()` deadlock fix, the remaining 2 unverified on iOS since.
      Speed group (`--dart-define=KILN_SPEED=true`) on Pixel 8 through Flutter: LFM2.5 350M
      104 to 111 tok/s, Qwen3.5 0.8B 33 to 37, Gemma 3 270M Q8 72 to 75, within 5% of the CLI.
- [ ] Run the example on a 4 GB RAM device with a 0.8B model; confirm graceful
      `KILN_ERR_OUT_OF_MEMORY` rather than a kill
- [ ] Background/foreground the app mid-generation; rotate the screen; confirm no crash and
      cancel works
- [ ] Load a corrupt file and a non-GGUF file from the UI; confirm the error surfaces
- [x] `dart pub publish --dry-run` clean: README, CHANGELOG, LICENSE, pubspec description,
      no `example/build`, no models. 2026-10-12: no errors, no warnings on a clean tree.
- [x] A new app can follow `docs/GETTING_STARTED.md`: done on 2026-10-12 with a fresh
      `flutter create` project and the GitHub dependency, on the Pixel 8 (by the author, not
      yet by an outside person; see the last item)
- [ ] iOS: xcframework + podspec exist and the app runs on an iPhone 16 Pro via Metal
      (A18 Pro GPU, 7 of 9 integration tests passed). Still needed before claiming support:
      rerun the suite after the Dart deadlock fix, add the simulator slice to CI
      (`scripts/build_ios.sh` without `--device-only`), fix the empty SoC name in the iOS probe
- [ ] Simulator run in CI for every PR (correctness only; the simulator cannot measure speed
      or thermals, and ggml's Metal path does not run under it)
- [ ] One external person has followed the README from zero to a running app

## Publishing, step by step

Publishing cannot be undone: a version on pub.dev stays there for good.

```bash
git status                                  # clean, on the tagged commit
scripts/build_ios.sh                        # rebuild the iOS framework from this commit (macOS)
scripts/vendor_flutter.sh --check           # vendor the core, build from pub's file set
cd flutter/kiln_flutter
dart pub publish --dry-run                  # read the file list and the warnings
dart pub publish                            # asks for confirmation and a Google sign-in
cd ../.. && rm -rf flutter/kiln_flutter/src/core
```

`ios/Frameworks/kiln.xcframework` is uploaded as it is on disk. It is a compiled copy of the
core, so an old one silently ships an old core: always rebuild it first. If iOS has not been
re-run for the version, remove `ios` from `platforms` in `pubspec.yaml` instead of shipping it.

`scripts/pub_files.py flutter/kiln_flutter` prints the exact file set pub would upload.

## Known limitations to state in the release notes

- Android arm64 only; no x86_64 emulator build; no iOS
- CPU backend only on Android (Vulkan preset is experimental and unmeasured)
- Hybrid models re-read the last assistant reply each turn (the template rewrites it), and the
  whole prompt when history diverges before the previous turn's checkpoint
- Tool calling not exposed
- Decision models: LFM2 d1-omni with a text state only; no images or audio
- ISA baseline is dotprod+fp16; i8mm/SVE present on many phones are not used yet
