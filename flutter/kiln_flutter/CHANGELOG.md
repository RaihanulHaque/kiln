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
