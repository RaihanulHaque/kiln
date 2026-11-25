# Galaxy A17 5G (SM-A176B)

Samsung Exynos 1330. A mid-range phone with two fast cores and six little ones, run on
2026-11-25 with kiln 0.5.0 on llama.cpp `v1.0.0` (branch `feat/llama-cpp-1.0.0`). Everything
here was collected in one session over USB.

## Hardware, as the system reports it

| | |
|---|---|
| chip | Samsung s5e8535 (Exynos 1330) |
| Android | 16 (SDK 36), security patch 2026-09-05, 4 KB pages |
| RAM | 7.6 GB total, 3.1 to 4.2 GB available with the owner's apps running |
| cores 0 to 5 | Cortex-A55 (part 0xd05), top clock 2.0 GHz, `cpu_capacity` 252 |
| cores 6 and 7 | Cortex-A78 (part 0xd41), top clock 2.4 GHz, `cpu_capacity` 1024 |
| frequency governor | `energy_aware` |
| CPU features | fp16, dotprod (no i8mm, no SVE): the normal `libkiln.so` build fits |
| GPU | Mali (`libGLES_mali.so`, `/vendor/lib64/libOpenCL.so`); not tried |

## What kiln chose

2 fast cores (mask `0xc0`, cores 6 and 7), 2 threads. The little cores rate a quarter of a big
one, below the "at least half" rule in `src/probe/fast_cores.h`, so they are dropped.

## Tests

`kiln-tests models`: **568 passed, 0 failed** (llama.cpp v1.0.0, whisper.cpp v1.9.6, with the
changes of #43 and #44). Times from that run, one call each:

| | A17 | Pixel 8, for comparison |
|---|---|---|
| Parakeet, 3.1 s clip | 0.76 s | 0.3 s |
| Qwen3-ASR, 3.1 s clip | 3.5 s (encoder 2.0 s) | 3.8 s |
| Whisper base, 3.1 s clip | 8.9 s (encoder 4.4 s) | 3.0 s |
| Pocket TTS, 2.7 s of speech | 11.7 s (voice reference 6.3 s) | 4.3 s |
| jina image (256 image tokens) | 14.2 s (encoder 13.4 s) | |
| EmbeddingGemma 2 audio clip | 1.4 s (encoder 1.1 s) | |

Parakeet is the only speech-to-text model fast enough for live voice on this phone. Pocket TTS
is about four times slower than real time here, most of it in reading the voice sample.

## llama.cpp 1.0.0 against the version before

The same phone, `main` (llama.cpp `781dbc5`, whisper.cpp `0c7be696`) and this branch, run
alternately (old, new, old, new) so that heat hits both alike. The phone warmed during the runs:
LFM2.5 fell from about 50 to 37 tok/s in both builds.

| | old | new (v1.0.0) |
|---|---|---|
| LFM2.5 350M, prompt / generate tok/s, mean of 3 | 141 / 38.7 | 145 / 40.9 |
| Qwen3.5 0.8B, prompt / generate tok/s, mean of 3 | 54.4 / 14.5 | 54.4 / 13.3 |
| Whisper base, 3.1 s clip, 2 runs | 8.8 s, 7.4 s | 7.5 s, 7.6 s |
| Pocket TTS, same sentence, 2 runs | 8.8 s, 8.5 s | 9.6 s, 10.1 s |

Chat and Whisper are the same within the noise. Pocket TTS was slower in both pairs (about 12%,
in reading the voice and in generating alike); two pairs are too few to be sure. Worth
re-checking on a quiet, cool phone before a release.

## Which cores to use: the sweep

A ~60-token prompt and 200 generated tokens, twice per setting, `kiln-cli`, `KILN_CPU_MASK`
picks the cores. Prompt / generate tok/s.

| cores | LFM2.5 350M | Qwen3.5 0.8B |
|---|---|---|
| **2 big (`c0`, the default)** | 173 / **48** | 56 / **14** |
| 2 big + 4 little (`cf`) | 186 / 35 | 64 / 11 |
| all 8 (`ff`) | **215** / 37 | **84** / 13 |
| the 6 little only (`3f`) | 135 / 34 | 49 / 9 |

- Generating text is fastest on the two big cores alone: adding little cores makes every step
  wait for the slowest thread. The default is right for chat.
- Reading the prompt gains 25 to 50% from all eight cores. A long prompt (a document to
  summarise, a long system prompt) would gain from using all cores for the prompt and the two
  big ones for the answer; kiln has one core set for both today.
