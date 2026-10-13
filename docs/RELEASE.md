# Release checklist

What has to be true before a version is tagged, and before `kiln_flutter` goes to pub.dev.

**Published:** `kiln_flutter` 0.5.0 on 2026-10-13 (text to speech), from the `v0.5.0` tag, 1262 files,
5 MB compressed, 0 warnings; afterwards a fresh `flutter create` app with
`kiln_flutter: ^0.5.0` (minSdk 29) spoke the test sentence with Pocket TTS on the Pixel 8 (2.7 s of speech in 4.3 s)
and Parakeet read it back correctly. Before it: `kiln_flutter` 0.4.0 on 2026-10-13 (speech to text: Whisper, Parakeet, Qwen3-ASR), from the
`v0.4.0` tag, 1261 files, 5 MB compressed, 0 warnings; afterwards a fresh `flutter create` app with
`kiln_flutter: ^0.4.0` (minSdk 29) loaded Parakeet and transcribed the test clip in 312 ms on the Pixel 8. Before it: `kiln_flutter` 0.3.1 on 2026-10-13 (core selection on chips with no little
cores), from the `v0.3.1` tag; afterwards a fresh app installed it from pub.dev and ran an
embedding search on the Pixel 8. `kiln_flutter` 0.3.0 on 2026-10-13 (embedding models), from the
`v0.3.0` tag. Before it: `kiln_flutter` 0.2.0 on 2026-10-12, Android only, from commit c485fcb. Afterwards
a fresh `flutter create` app installed it with `flutter pub add kiln_flutter`, built, and
replied at 94 tok/s on the Pixel 8. The unchecked items below were accepted as known gaps for
that release and still apply to the next one.

## Before tagging the core

- [ ] `cmake --build --preset host && ./build/host/kiln-tests models` green on macOS
- [ ] `scripts/android-test.sh` green on at least one Snapdragon, one Tensor, one MediaTek
      device (today: Tensor G3 on a Pixel 8, and Snapdragon 8 Elite Gen 5 on a Galaxy S26 Ultra;
      no MediaTek, and no older or mid-range Snapdragon)
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
- [ ] iOS: `scripts/build_ios.sh` does not build whisper.cpp yet (added in 0.4.0), so the
      framework would lack Whisper and Parakeet; fix that before any iOS claim
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
scripts/publish_flutter.sh --dry-run        # everything except the upload
scripts/publish_flutter.sh                  # the same, then the upload; asks to confirm
```

The script refuses to run with uncommitted changes, vendors the core into the package, builds
the example from exactly the files pub would upload, shows the dry run, and removes the
vendored copy when it ends. The first real publish opens a browser for a Google sign-in.

The pub.dev package is Android only. `ios` is not declared in `pubspec.yaml`, and `.pubignore`
leaves the `ios/` folders out, so no compiled iOS framework is uploaded. To bring iOS back:
re-run it on a device at the version being published, add `ios: {ffiPlugin: true}` to the
pubspec, remove the two `ios/` lines from `.pubignore`, and rebuild the framework with
`scripts/build_ios.sh` right before publishing (it is a compiled copy of the core; an old one
silently ships an old core).

`scripts/pub_files.py flutter/kiln_flutter` prints the exact file set pub would upload.

## Known limitations to state in the release notes

- Android arm64 only; no x86_64 emulator build; iOS is in progress and not supported yet
- CPU backend only on Android (Vulkan preset is experimental and unmeasured)
- Hybrid models re-read the last assistant reply each turn (the template rewrites it), and the
  whole prompt when history diverges before the previous turn's checkpoint
- Tool calling not exposed
- Decision models: LFM2 d1-omni with a text state only; no images or audio
- ISA baseline is dotprod+fp16; i8mm/SVE present on many phones are not used yet
