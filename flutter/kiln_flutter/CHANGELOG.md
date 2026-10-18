## 0.5.1

- **Phones without the dot-product instruction no longer crash.** The plugin now also ships
  `libkiln_armv8.so`, built for plain Armv8.0 (`armv8-a`). `native.dart` loads it when
  `/proc/cpuinfo` lacks `asimddp`, `asimdhp` or `fphp`; other phones load `libkiln.so` as
  before. Verified on an OPPO CPH2819 (Snapdragon 680, 3.76 GB RAM): the example app and the
  Hishab app load `libkiln_armv8.so` and answer. Not yet verified: `libkiln.so` on a Pixel 8 or
  S26 with this version.
- `cancel()` on a generation, transcription or speech now also stops it when it is called
  before the worker isolate has started the call (right after `chat()`, for example). Such a
  cancel used to be lost and the call ran to the end. Each call has a cancel flag in native
  memory that the worker checks before starting and at every piece.
- With kiln's core change, a cancel during prompt rendering is no longer lost either.

## 0.5.0

Text to speech. Verified on a Pixel 8: 26 integration tests (4 skipped: the speed and media groups, which passed at 0.3.1), and the example app's new
Speak screen. Android arm64 only, as before.

- `KilnContext.speak(text, {voice, language, maxFrames, seed})`: says the text in the voice of a
  short recording (`KilnMedia`, a file or its bytes) and streams the audio as `KilnAudioPiece`s
  (mono floats, 24 kHz) about once a second; `KilnSpeech.wav` collects them into a wav file's
  bytes, `.stats` gives the length, the time and how many times faster than real time.
  `pcmToWav` is public for your own pieces.
- `KilnProjectorInfo.generatesAudio` and `.sampleRate`: a text-to-speech projector.
- Tested with Pocket TTS (Kyutai, English, 159 MB + 60 MB projector; CC-BY-4.0): 2.7 s of speech
  in 3.4 to 4.1 s (0.7 to 0.9x real time; about a second of it is encoding the voice) on the Pixel 8. The example app's Speak screen bundles a sample voice, records
  your own, plays the result with `audioplayers`, and has Pocket TTS in its download list.
- The integration test checks the audio by having Parakeet transcribe it. Through the plugin the
  sentence took 4 to 8 s on a Pixel 8 that had been running tests for hours (0.4 to 0.6x real
  time), 5.3 s in the example app; the C API on a rested phone does it in 3.4 to 4.1 s. Let a
  piece or two arrive before playing, or keep sentences short.

## 0.4.0

Published 2026-10-13. Speech to text. Verified on a Pixel 8 (Tensor G3): 25 integration tests (4 skipped: the speed and media groups, which passed at 0.3.1), and the example
app's new Transcribe screen (record, or pick a file). Android arm64 only, as before.

- `KilnContext.transcribe(KilnMedia audio, {language, prompt, maxTokens})`: writes down a
  wav, mp3 or flac clip (a file or its bytes) and streams the words; `KilnTranscription.text`
  collects them, `.stats` gives the clip length, the time taken, how many times faster than
  real time, and the language the model heard. `cancel()` stops it.
- Three model families. **Parakeet TDT 0.6B v3** (669 MB) and **Whisper base** (60 MB) come
  as one `.bin` file each in whisper.cpp's format; `loadModel` takes them like a GGUF,
  `KilnModelInfo.isSpeechModel` is true, and they need no projector. **Qwen3-ASR 0.6B** is a
  GGUF plus its projector (`loadProjector`), like the embedding media models, and
  `isSpeechModel` is false for it.
- Speed on the Pixel 8 for a 3-second clip: 0.3 s with Parakeet (0.8 to 1.5 s once the phone is
  hot), 3.0 s with Whisper base, 3.8 s with Qwen3-ASR (slower than the clip; its encoder takes
  2.5 s on that CPU). Through the plugin: Parakeet 1.2 s on a phone that had been running tests
  for an hour, Whisper base 7 s.
- Example app: a Transcribe screen with a microphone button (records a 16 kHz wav with the
  `record` package, needs `RECORD_AUDIO`) and a file picker; Parakeet and Whisper base in the
  download list; `.bin` files in the model list and the paste-a-link box.
- `KilnModelInfo.toString()` says "speech model"; `KilnMedia` is shared by `embedMedia` and
  `transcribe`.

## 0.3.1

- **Phones whose chip has no little cores now use more of it.** On a Galaxy S26 Ultra
  (Snapdragon 8 Elite Gen 5) kiln ran on 2 of 8 cores; it now runs 6 threads. Chat with
  LFM2.5 350M and Qwen3.5 0.8B is about 20% faster under sustained load, and a decision or an
  embedding takes 35 to 45% less time. Phones with little cores, such as the Pixel 8, are
  unchanged. `KilnDeviceInfo.cpuFastCores` reports the new choice.
- Verified on a Galaxy S26 Ultra (26 integration tests and the native test suite) and on a
  Pixel 8 (native test suite; its integration test was last run at 0.3.0). No API change.
  Added after the release: the 26 integration tests also pass on the Pixel 8 at 0.3.1.

## 0.3.0

Embedding models. Verified on a Pixel 8 (Tensor G3): 23 integration tests, including images
and audio. Android arm64 only, as before.

- `KilnContext.embed`: turn a text into a vector with an embedding model, for search by
  meaning. `kind` (query or document) adds the prefix the model was trained with, `dimensions`
  keeps a shorter vector, `normalize: false` returns the raw one.
- `KilnModel.loadProjector` and `KilnContext.embedMedia`: images and audio, alone or mixed
  with text, into the same space as text. `KilnMedia.file` or `KilnMedia.bytes`.
- `KilnEmbedding` holds the vector (`Float32List`) and the stats; `similarity` and
  `KilnEmbedding.cosine` compare two.
- `KilnModelInfo.isEmbeddingModel` and `nEmbd`.
- `chat` and `decide` on an embedding model throw `KILN_ERR_NOT_SUPPORTED`, as `embed` does on
  the other kinds.
- The native library now includes llama.cpp's image and audio encoders (`libkiln.so` grows
  from about 6 MB to about 7 MB).
- Example app: a Search screen for embedding models, with a picker for photos and audio.

## 0.2.0

First version on pub.dev. Android arm64 only; iOS is in progress and not part of the package.
Verified on a Pixel 8 (Tensor G3): 16 integration tests. Not yet tried on Snapdragon or
MediaTek phones or on devices with 4 GB of RAM.

- `chat(maxReasoningTokens:)`: thinking budget. After that many reasoning tokens kiln closes the
  think block and the model answers. `KilnGenStats.reasoningTruncated` reports when it was used.
- `KilnContext.decide`: ask a decision model (LFM2 d1-omni) a yes / no, choice or score
  question about a text state and get a probability per option. `KilnModelInfo.isDecisionModel`
  tells the two kinds of model apart.
- Hybrid models (Qwen3.5, LFM2) reuse the KV cache across turns through state checkpoints.

## 0.1.0

- First release. Android arm64, llama.cpp backend, streaming chat with reasoning channel,
  KV prefix reuse, cancel, device probe, model info and generation stats.
