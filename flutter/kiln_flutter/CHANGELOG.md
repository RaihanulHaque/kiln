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
