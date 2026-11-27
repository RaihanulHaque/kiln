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

