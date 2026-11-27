# Parakeet 110M against 0.6B in kiln: accuracy and speed (phase 1)

Local experiment, nothing committed. Ryzen 5 8500G (Linux x86, AVX2), kiln 0.6.0 host build
with whisper.cpp's CPU backend and kiln's default threads, 2026-11-26.

## Models

| model | file | size | params |
|---|---|---|---|
| nvidia/parakeet-tdt-0.6b-v3 (25 languages) | `models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin` (ggml-org) | 669 MB | 0.6B |
| nvidia/parakeet-tdt_ctc-110m (English, CC-BY-4.0) | `conv/ggml-parakeet-tdt_ctc-110m-q8_0.bin` (converted here) | 123 MB | 110M |

The 110M file: `convert_tdt.py` (whisper.cpp's converter, skipping the `ctc_decoder.*` tensors),
then `parakeet-quantize ... q8_0`. Only its TDT head is used here; the CTC head is phase 2.

## The change it needed (`parakeet-bias.patch`, whisper.cpp)

The 110M model has a bias after every linear layer of the encoder (11 per layer: both feed
forwards, q / k / v / out, and the three convolutions); the 0.6B has none. whisper.cpp rejected
the file (`unknown tensor 'encoder.layers.0.feed_forward1.linear1.bias'`). The patch makes
these 11 tensors optional: created for every layer, set to nullptr when the file does not
have them, and added in the graph only when present. Old files load and give the same text.
Other differences (80 mels, d_model 512, 17 layers, 1 LSTM layer, 1024 tokens) needed nothing.

## Results (LibriSpeech, all clips, Q8_0 both)

| set | model | WER | model card | audio | time | speed | encoder share |
|---|---|---|---|---|---|---|---|
| test-clean (2620 clips, 5.4 h) | 0.6B | **2.20%** | 1.93% | 19451 s | 957 s | 20.3x real time | 94% |
| test-clean | 110M | **2.62%** | 2.4% | 19451 s | 254 s | **76.5x** | 88% |
| test-other (2939 clips, 5.3 h) | 0.6B | **3.86%** | 3.59% | 19226 s | 945 s | 20.3x | 94% |
| test-other | 110M | **5.45%** | 5.2% | 19226 s | 252 s | **76.4x** | 87% |

- The 110M is 3.8x faster and 5.4x smaller; it makes 0.4 more errors per 100 words on clean
  read speech and 1.6 more on the harder set.
- Both are about 0.25 points above their cards on both sets, the same offset for both: the
  cards use Whisper's text normalizer, `eval.py` a simple one (lower case, no punctuation).
  Many remaining "errors" are spelling only: `missus`/`misses` (22 in both), `color`/`colour`,
  `tis`/`'tis` (a leading apostrophe the normalizer keeps).
- The encoder is 87 to 94% of the time. Phase 3 (speed on the PC) starts there.

## Found on the way

- Batch norm in the conv module divides by `sqrt(var)`, without NeMo's `eps = 1e-5`. For the
  110M weights that changes a channel by up to 0.07% (smallest running_var 0.0072). Too small
  for WER; it will show in an exact comparison with NeMo. Folding the batch norm into the
  depthwise weights at load time would fix it and drop 4 ops per layer.

## Reproduce

```bash
cd build/asr
conda run -n ML python -I eval.py conv/ggml-parakeet-tdt_ctc-110m-q8_0.bin data/test-clean res/clean-110m
conda run -n ML python -I eval.py ../../models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin data/test-other res/other-0.6b
```

`data/` comes from Hugging Face `openslr/librispeech_asr` (parquet, `unpack.py`); openslr.org
and its mirrors gave 10 to 30 KB/s here.

# Phase 2: where the 110M encoder spends its time (2026-11-26)

Tool: `prof/encprof.cpp` compiles whisper.cpp's `parakeet.cpp` in, links kiln's ggml (llama.cpp's,
OpenMP, native), and times each graph node with the scheduler's eval callback. Node sums match
the plain encode time, so the per-node split costs nothing visible.

Reference: `ref/clips/*.f32` (20 test-clean clips, 1.3 to 35 s, 163 s total) and their encoder
outputs from the unchanged build in `ref/base/*.enc` (int32 n_state, n_frames, then floats).

## Threads: 6 is 16% faster than kiln's 8

| threads | 4 | 5 | 6 | 7 | 8 | 12 |
|---|---|---|---|---|---|---|
| encode, 20 clips (ms) | 2295 | 1788 | **1570-1611** | 1876 | 1849-1935 | 1743 |

6 and 8 alternated twice. The 8500G has 6 cores, 12 hardware threads; kiln counts 12 "cores"
(the known gap in docs/CPU.md section 8) and picks 8 threads, so two cores run two threads each
and every op waits for them. Q8_0 and F16 weights are the same speed at 6 threads (1591-1613
against 1614-1628 ms).

## Per-node, 6 threads

| share of encoder time | 10.6 s clip (102 ms) | 35 s clip (453 ms) |
|---|---|---|
| weight matmuls (q8_0, all layers) | 53% | 37% |
| subsampling front end (3 convs before the layers) | 23% | 16% |
| attention scores, softmax and their copies | 12% | 32% |
| other small ops (norm, add, silu, pad, roll, ssm_conv) | 6% | 10% |
| copies (cont) | 3% | 3% |
| GLU gate (sigmoid) | 3% | 3% |

- Weight matmuls run at about 410 GFLOP/s at 8 threads; they are not obviously wasteful.
- The front end is slow for its work: the first conv (1 input channel) is an im2col plus a
  matmul whose inner size is 9 (about 20 GFLOP/s); its bias add (`pre_conv_0`, broadcast over
  a [W, H, 256] tensor) costs as much as the conv; the stride-2 depthwise conv is the plain-layout
  `conv_2d_dw` kernel from the ggml PR; ReLUs are separate passes over large tensors.
- Attention grows with length squared: 12% at 10 s, 32% at 35 s.
- Rebuilding the scheduler when the length changes: 1 to 12 ms per call, about 3.5% in total.

# Phase 3a: faster subsampling front end (2026-11-26)

`parakeet-bias-frontend.patch` (whisper.cpp, uncommitted; `PARAKEET_OLD_FRONTEND=1` still picks
the old graph for comparisons). The front end now keeps channels first ([chan, freq, time]):

- conv 0 (1 input channel, 3x3, stride 2) with its bias and relu is one `ggml_custom_4d` op:
  per output pixel, each of the 9 input values times a row of 256 weights. Dot-product matmuls
  (ggml's mul_mat, llamafile's sgemm and `ggml_conv_2d_direct` alike) spend a horizontal sum
  per output value when the inner size is 9 or 16; padding to 16 for sgemm still took 4.8 ms.
- the depthwise convs run on a channels-first view (ggml's CWHN kernel), the 1x1 convs are plain
  matmuls (no im2col), and every bias is a contiguous add.
- no input rounding to f16 (`ggml_conv_2d` does that for its im2col).

Accuracy, with the F16 file against PyTorch float64 on the .nemo weights (front-end output,
one clip): old 4.8e-4 rel rms, new **1.4e-4**. With Q8_0 the encoder output moves 1 to 4%
for any small change (Q8_0 against F16 with the same graph: 4.7%), so Q8_0 is judged by WER.

| | encoder, 20 clips, 6 threads | 10.6 s clip | test-clean WER | test-clean through kiln-cli |
|---|---|---|---|---|
| before | 1550-1556 ms | 101.6 ms | 2.617% | 76.5x real time, encoder 223 s |
| after | **1350-1389 ms** | **83.3 ms** | **2.617%** (same errors) | **89.5x**, encoder 186 s |

The front end went from about 23 to about 5.5 ms on the 10.6 s clip.

# Phase 3b: old front end removed, batch norm folded (2026-11-27, M4 Pro)

Machine: MacBook Pro M4 Pro (8 performance cores), macOS, kiln host build (whisper.cpp with Metal
for `kiln-cli`; `encprof` runs the CPU backend), 8 threads.

- The `PARAKEET_OLD_FRONTEND` switch and its graph are gone; the new front end is the only one.
- The conv module's batch norm is folded into the depthwise conv when the model loads:
  s = gamma / sqrt(var + 1e-5), w' = w * s, b' = (b_dw - mean) * s + beta (b_dw = 0 for the 0.6B).
  The graph keeps `ssm_conv` + one add (`enc_%d_conv_bn`) in place of sub, sqrt, div, mul, add and
  the depthwise bias add. NeMo's eps = 1e-5 is now applied (it was missing).

| check | result |
|---|---|
| F16 encoder output, 20 reference clips, after against before | worst rel rms 3.0e-5, max abs 1.4e-3 |
| Q8_0 encoder output, after against before | rel rms 1-4% (any change does this with Q8_0; judged by WER) |
| encoder time, Q8_0, 20 clips, before and after alternated 3 times | 593-618 ms against 595-605 ms: no change |
| LibriSpeech test-clean WER, Q8_0, kiln-cli | **2.608%** (Ryzen, before: 2.617%) |
| LibriSpeech test-other WER, Q8_0, kiln-cli | **5.454%** (Ryzen, before: 5.45%) |

The before numbers for WER are the Ryzen's (phase 1 and 3a); a Mac before run was not made.

Through `kiln-cli` on the M4 Pro the whole sets ran at 196x (test-clean) and 190x (test-other) real
time, with the TDT decoder taking more time than the encoder: test-clean encoder 45.4 s, decoder
53.7 s; test-other 47.2 s and 54.1 s. On the Ryzen the encoder was 87-88%.

# Phase 3c: the TDT decoder (2026-11-27, M4 Pro)

Tool: `prof/decprof.cpp` (whole transcription per clip, best of 3, on the CPU or `--gpu`), the same
20 reference clips (163.3 s), Q8_0, 8 threads for the encoder.

## Where the decoder's time went

| | CPU | GPU (Metal) |
|---|---|---|
| encoder | 600 ms | 346 ms |
| decoder | 238 ms | 338 ms |
| joint steps (1324) | 0.100 ms each | 0.141 ms each |
| predict steps (826) | 0.128 ms each | 0.184 ms each |
| of predict: graph build / alloc / compute | 2.3 / 2.9 / 100 ms | 2.1 / 9.6 / 140 ms |

Building and allocating the graphs is cheap; running them is not. Each step is one matrix-vector
product and a few vector ops (about 1 to 3 MFLOP), so the fixed cost per graph dominates:

| decoder threads (CPU) | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| decoder, 20 clips | **86 ms** | 112 ms | 149 ms | 236 ms |

kiln's Mac build has no OpenMP, so ggml hands each small graph to its threads afresh. On the GPU
each step also pays a round trip. Metal's buffers here are not host memory (`ggml_backend_buffer_is_host`
is false), so the CPU cannot read the GPU's weights in place.

## Changes (in `parakeet-bias-frontend.patch`)

1. The decoder uses 1 thread (`PARAKEET_DEC_THREADS=n` overrides it).
2. The joint's encoder projection (`enc_w`, `enc_b`) is done for every frame at the end of the encoder
   graph (`enc_proj`), instead of once per joint step.
3. With a GPU, the decoder's weights (embedding, LSTM, joint `pred_w`/`pred_b`/`net_w`/`net_b`) are
   placed in CPU buffers, its state (LSTM, `pred_out`) in CPU memory, the projection is copied to a CPU
   tensor once per clip, and the decoder's scheduler gets the CPU backend alone. CPU-only builds place
   everything as before.

## Results

| 20 clips | before | after |
|---|---|---|
| CPU: decoder | 238 ms | **79 ms** (joint 0.021 ms, predict 0.062 ms a step) |
| CPU: whole transcription | 862 ms (189x) | **714 ms (229x)** |
| GPU: decoder | 338 ms | **74 ms** |
| GPU: whole transcription | 718 ms (228x) | **458 ms (357x)** |
| transcripts, CPU and GPU | | identical to before |

LibriSpeech through `kiln-cli` (Metal), Q8_0:

| set | WER before | WER after | speed before | speed after | decoder before | decoder after |
|---|---|---|---|---|---|---|
| test-clean | 2.608% | **2.602%** | 196x | **289x** | 53.7 s | **21.7 s** |
| test-other | 5.454% | **5.451%** | 190x | **272x** | 54.1 s | **22.0 s** |

`kiln-tests models`: 603 passed (the Parakeet 0.6B, which has no biases, and the TTS round trip
through Parakeet included).

## On a phone: Galaxy A17 5G (Exynos 1330, Android 16, OpenMP), 2026-11-27

`decprof` built with NDK r28 against kiln's Android ggml (OpenMP, `armv8.2-a+dotprod+fp16`), run on
kiln's cores for this phone (the two Cortex-A78, `taskset c0`), `-t 2`, best of 2 per clip:

```bash
G=build/android-arm64/third_party/llama.cpp/ggml/src
$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android29-clang++ -O2 -std=c++17 \
  -march=armv8.2-a+dotprod+fp16 -fopenmp -static-openmp -DPARAKEET_VERSION='"prof"' \
  -Ithird_party/whisper.cpp/include -Ithird_party/whisper.cpp/src -Ithird_party/llama.cpp/ggml/include \
  experiments/parakeet/prof/decprof.cpp -o build/asr/prof/decprof-android \
  $G/libggml.a $G/libggml-cpu.a $G/libggml-base.a -static-libstdc++
```

Decoder threads, the new code, alternated 3 times (phone at 29.6-29.8 C):

| decoder threads | round 1 | round 2 | round 3 |
|---|---|---|---|
| 1 | 413 ms | 421 ms | 419 ms |
| 2 | 393 ms | 528 ms | 427 ms |

Even: OpenMP's threads persist between graphs, so the cost of a second thread is small here. 1 stays
the default (steadier, and it leaves the other fast core to the app).

Before (patch at d4a56fa, decoder on 2 threads) against after, alternated 3 times:

| | before | after |
|---|---|---|
| decoder, 20 clips | 451-470 ms | **395-414 ms** |
| whole transcription | 7850-7958 ms | 7788-7907 ms (21x real time both) |
| transcripts | | identical |

On the phone the encoder is about 88% of the time (6.9 s of 7.9 s), so the decoder change is
small there. The 1-thread decoder on the Ryzen (OpenMP, 6 to 8 threads) is not measured.

# Phase 3d: the encoder on a phone (2026-11-27, Galaxy A17 5G)

`encprof` built for Android like `decprof` (phase 3c), run on the two Cortex-A78 (`taskset c0`,
`-t 2`), 20 reference clips, Q8_0.

## Profile, one 10.6 s clip (598 ms, 18x real time; node sums 642 ms, so about 7% profiler overhead)

| by op | share |
|---|---|
| MUL_MAT (weights and attention) | 65.8% |
| ADD (biases, residuals) | 12.2% |
| CONT | 5.9% |
| PAD | 3.7% |
| CONV_2D_DW, SILU, MUL, SOFT_MAX, SIGMOID, the front end's custom op | 1-2% each |

By category (`cats.py`): weight matmuls 50.5%, other small ops 25.3%, attention 13.8%, front end
7.4%. Per-node times include the views just before a node (a RESHAPE shows 11 ms), so single
small rows overstate.

## What does not help on the A17

| try | result (20 clips, encoder) |
|---|---|
| more cores: 2 fast + 2 little (`0xcc`, 4 threads) | 9.1-9.8 s, against 7.8-8.5 s on the 2 fast cores |
| all 8 cores (`0xff`) | 18.0-18.1 s: ggml gives each thread an equal share, so the A78s wait for the A55s |
| Q4_0 weights | 7.29 / 7.64 s against Q8_0 7.20 / 7.57 s (alternated): not faster |
| Q5_0 weights | 18.3 / 21.2 s |

The weights already sit in ggml's `CPU_REPACK` buffer (158 tensors, 105 MB; `q8_0_4x4` for
dot-product cores), so the matmuls run the best kernel ggml has for this CPU. The encoder is
compute bound, unlike chat, so halving the weight bytes buys nothing.

Accuracy of the smaller formats (LibriSpeech through kiln-cli, M4 Pro Metal):

| weights | size | test-clean | test-other |
|---|---|---|---|
| Q8_0 | 123 MB | 2.602% | 5.451% |
| Q4_0 | 66 MB | 2.817% | 5.661% |
| Q5_0 | 80 MB | 2.676% | 5.535% |

## Conv module padding in one op

The depthwise conv's input was a transpose copy, a pad, a roll and a second pad. Now one
`ggml_pad_ext` reads the transposed GLU output and pads both sides (CPU and Metal kernels read
through strides). Encoder output bit-identical (F16, 20 clips, max |d| 0). A17, alternated three
times: 6923 / 6942 / 6959 ms against 6947 / 7051 / 6995 ms, 0.3 to 1.5% faster.

