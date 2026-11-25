# How kiln's OCR was made fast

This note explains how kiln runs the OCR model OCRia-nano-1 and how it was made 3 to 5 times
faster after the first working port, with every output unchanged. It is written for someone who
knows some programming but not the low-level side: each term is explained where it first appears.

**In short.** The model came as ONNX files. kiln converts them to one GGUF file and runs them on
ggml, the engine kiln already uses for its other models. The first port gave the right answers
but was 4 to 20 times slower than ONNX Runtime. Four changes to *how the engine runs the model*
closed most of that gap: how numbers are laid out in memory, fusing small steps into one pass, a
dedicated kernel for the first layer, and reusing memory. There was no quantization, no pruning
and no retraining: the weights are the original ones, in full 32-bit precision, and every output
is within 0.000001 of ONNX Runtime's.

| | before | after |
|---|---|---|
| English test page (1400 × 528, 9 lines), Galaxy A17 5G | 0.70 s | **0.27 s** |
| the same page, Ryzen 5 8500G desktop | 0.18 s | **0.06 s** |
| difference from ONNX Runtime, any output | within 1e-6 | within 1e-6 |

Contents: [the model](#1-the-model) · [from ONNX to GGUF](#2-from-onnx-to-gguf) ·
[where the time went](#3-where-the-time-went) · [the four changes](#4-the-four-changes) ·
[what did not work](#5-what-did-not-work) · [results](#6-results) ·
[how correctness was kept](#7-how-correctness-was-kept) · [lessons](#8-lessons) ·
[reproducing](#9-reproducing-the-numbers)

---

## 1. The model

[OCRia-nano-1](https://huggingface.co/llamaindex/OCRia-nano-1) (LlamaIndex) is not one network but
a pipeline of small ones. kiln runs the OCR part: six networks, 3.9 M parameters in all.

| network | job | parameters |
|---|---|---|
| detector | looks at the whole page and marks where text is | 0.08 M |
| router | looks at each line: which alphabet, and is it upside down | 0.04 M |
| Latin recognizer | reads English and other Latin-script lines | 0.80 M |
| world recognizer | reads Bengali, Devanagari and other scripts | 0.98 M |
| RTL recognizer | reads Arabic and Hebrew | 0.71 M |
| CJK recognizer | reads Chinese, Japanese and Korean (6,214 characters) | 1.27 M |

The model card's 8.1 M also counts **LayoutNet** (4.27 M), which finds tables, headings and
columns. kiln does not run it, so none of the work below touches it.

A page goes through the pipeline like this:

```
photo ──► shrink to 1600 px ──► detector ──► one box per line
                                                 │
                     ┌───────────────────────────┘
                     ▼
          cut out each line, turn it straight
                     │
                     ▼
                  router ──► picks a recognizer (and notices upside-down pages)
                     │
                     ▼
        recognizer (Latin / world / RTL / CJK) ──► letters + confidence
```

The steps between the networks (shrinking, cutting out lines, filtering junk) are a port of
LiteParse's `ocria` engine (Apache 2.0) and were not part of the speed work: they were already a
small share of the time.

### What the networks are made of

All six are convolutional networks, built from a handful of layer types repeated many times. A
*convolution* slides a small window over an image and, at each position, mixes the numbers inside
the window with learned weights. An image inside a network has many *channels*: not just red,
green and blue, but 16, 64 or more numbers per pixel, each one a feature the network learned to
look for.

| layer | what it does |
|---|---|
| depthwise 3×3 convolution | each channel looks at its own 3×3 neighbourhood, separately from the others |
| 1×1 (pointwise) convolution | at each pixel, mixes all the channels together; it does not look at neighbours |
| bias + ReLU | adds a constant per channel, then turns negative values into zero |
| FakeQuant (Latin recognizer only) | snaps each number to one of 256 levels, as the model was trained to |
| first layer | a 3×3 convolution over the grey page, from 1 channel to 16 |

Nearly all of the time is spent in the first four. Speeding up the model means speeding up those.

---

## 2. From ONNX to GGUF

ONNX and GGUF are both file formats: a list of weights plus a description of what to do with
them. Neither is fast or slow by itself. Speed comes from the engine that reads the file.

- `scripts/convert_ocria.py` reads the six ONNX files and writes one GGUF file (16 MB) with the
  architecture name `ocria`. Each network's graph (the list of operations and which weights they
  read) is stored as JSON inside the file, and the weights as 32-bit floats.
- The Latin recognizer was stored in ONNX's 8-bit form (QDQ: weights saved as 8-bit integers plus
  a scale). The converter turns those weights back into floats, which gives exactly the values
  ONNX Runtime computes with. The 8-bit rounding of the *activations* (the numbers flowing
  between layers) is part of how the model was trained, so it is kept as a FakeQuant step.
- `src/backends/ocr/ocr_graph.cpp` reads the JSON at load time and, for each input size, builds
  the matching ggml graph (ggml's word for "a list of operations to run").

Why not just ship ONNX Runtime? It would add a second engine of several megabytes to every app,
with its own threads and its own build for each platform. With ggml, OCR shares the engine,
threads and build that kiln already has.

---

## 3. Where the time went

The first port was correct and slow. Per network, on the desktop (Ryzen 5 8500G, 8 threads):

| network | ONNX Runtime | kiln, first port |
|---|---|---|
| detector | 24 ms | 110 ms |
| Latin recognizer | 6.4 ms | 75 ms |
| world recognizer | 7.8 ms | 21 ms |
| router | 0.4 ms | 6.5 ms |

To find out why, each network was timed one operation at a time: ggml can run a slice of a graph
(`ggml_graph_view` in `ggml-impl.h`), so running the first 1, 2, 3, … operations and taking the
differences gives the cost of each one. Three findings drove everything that followed:

1. **The convolutions used ggml's slowest paths.** ggml stored images one channel at a time (all
   of channel 0, then all of channel 1, …). For that layout its depthwise convolution is a plain
   loop that does one multiplication at a time, with a bounds check around every pixel.
2. **The 8-bit rounding was two thirds of the Latin recognizer.** In ggml, FakeQuant is five
   separate operations (divide, round, clamp, subtract, multiply), and each walks through all of
   memory again. A single division of a large tensor by one number took 17 ms.
3. **The first layer was a quarter of the detector.** A convolution from 1 channel to 16 reads
   only 9 numbers per output pixel, too thin a job for the general-purpose code ggml used for it.

---

## 4. The four changes

Each change was made on its own, then checked against ONNX Runtime and timed before the next one.

### 4.1 Memory layout: channels side by side

**The problem.** A tensor's numbers sit in memory in one long row, and the order matters. ggml
kept images *channels-last-in-memory*: every pixel of channel 0, then every pixel of channel 1,
and so on. The depthwise convolution for that order is the slow scalar loop, and a 1×1
convolution has to be copied and rearranged before it can be a matrix multiply.

**The change.** Image tensors are now stored *channels-first*: the 16 or 64 channel values of one
pixel sit next to each other, then the next pixel's.

```
before (one channel at a time):   c0: p0 p1 p2 p3 …   c1: p0 p1 p2 p3 …   c2: …
after  (one pixel at a time):     p0: c0 c1 c2 …      p1: c0 c1 c2 …      p2: …
```

Two things follow:

- **A 1×1 convolution becomes one matrix multiplication** over all pixels, with no copying.
  Matrix multiplication is the operation ggml is most optimised for.
- **The depthwise convolution takes ggml's SIMD path** (`ggml_conv_2d_dw_direct` on a
  channels-first view). SIMD ("single instruction, multiple data") lets one CPU instruction work
  on 4 to 16 numbers at once; here, the same position in neighbouring channels. Its kernels
  (the learned 3×3 windows) are rearranged into the order that path expects once, at load time.

A comparison: a warehouse that stores one product per aisle makes every order a walk through
all the aisles. Storing each customer's whole order together turns it into one stop.

**Result.** On a 64-channel map of the detector's size, one depthwise 3×3 plus one 1×1 went from
**61.5 ms to 15.0 ms**. Every network got roughly twice as fast from this alone.

### 4.2 Kernel fusion: one pass instead of six

**The problem.** After each convolution came several small operations: add the bias, apply ReLU,
then, in the Latin recognizer, FakeQuant's five operations. Each is cheap to compute, but each
reads every number from memory and writes it back. Moving the numbers cost far more than the
arithmetic did.

**The change.** *Kernel fusion* (also called operator fusion) means doing several operations in
one pass over memory. kiln adds one custom ggml operation (`epilogue_rows` in `ocr_graph.cpp`)
that, for each number, adds the bias, applies ReLU and does the 8-bit rounding, then writes it
once. The graph loader decides which nodes to fuse: a bias, ReLU or FakeQuant is folded into the
convolution before it when nothing else reads the value in between.

Two details made it fast:

- The loop goes pixel by pixel and, inside, channel by channel. Because the channels sit side by
  side (4.1), the bias for each number is simply the next one in a short list: no division or
  `%` to work out which channel a number belongs to. The first version used `%` and was *slower*
  than the separate operations; the per-pixel loop is what lets the compiler use SIMD.
- Rounding *half to even* (2.5 → 2, 3.5 → 4), which is what ONNX does, uses a floating-point
  trick: `(x + 1.5·2²³) − 1.5·2²³`. Adding a large enough number pushes the fraction out of the
  float, and the CPU's own rounding mode is already half to even. It vectorises on every CPU.
  kiln is built without `-ffast-math`, so the compiler may not cancel the two additions.

The arithmetic is the same as the separate operations, in the same order, so the results do not
change.

**Result.** Latin recognizer **37 ms → 17 ms**. The other networks gain less, since only their
bias and ReLU are fused.

### 4.3 A dedicated kernel for the first layer

**The problem.** Every network starts with a 3×3 convolution from the one-channel grey image to
16 channels. ggml handled it as a general convolution: copy every 3×3 window into a matrix
(*im2col*), then multiply. With only 9 numbers per window, the copying cost more than the
multiply saved. It took about 10 ms in the detector.

**The change.** A small hand-written operation (`conv1ch_op`) for exactly this shape. For each
output pixel it reads the 9 input numbers once and computes all 16 output channels in an inner
loop the compiler vectorises, writing the result straight into the channels-first layout.

The detector's first step, turning the colour page grey, also moved out of the graph: it is done
while the page is copied in, which saves one more pass over the largest image.

**Result.** **10 ms → 2.5 ms** for the detector's first layer.

### 4.4 Reusing memory between runs

**The problem.** Each run asked the system for fresh memory for its intermediate results. The
operating system hands out memory lazily: the first time each 4 KB page is touched, it stops the
program to find and zero a page (a *page fault*). For the detector, that was a fifth of a run.

**The change.** Each network keeps its memory allocator (`ggml_gallocr`) from one run to the next
and reuses the memory, growing it only when a larger page arrives.

**Result.** Detector **42 ms → 34 ms**.

### 4.5 A smaller one: reading the output where it lies

The CJK recognizer scores 6,214 characters at every position along the line. kiln used to
rearrange that block into ONNX's order before decoding it. It now decodes it in the order it is
stored, which also reads memory front to back, the order caches handle best.

---

## 5. What did not work

**Making every convolution im2col plus a matrix multiply.** It is the usual way to speed up
convolutions, so it was tried for all of them. The detector went from 110 ms to **142 ms**:
these networks are mostly depthwise and 1×1 convolutions, where copying the windows costs more
than it saves. It was measured, found slower, and removed. Without the measurement it might
have stayed.

---

## 6. Results

Per network on the Ryzen 5 8500G (8 threads):

| network | ONNX Runtime | kiln, first port | kiln, now | gain |
|---|---|---|---|---|
| detector | 24 ms | 110 ms | **37 ms** | 3.0× |
| Latin recognizer | 6.4 ms | 75 ms | **16 ms** | 4.7× |
| world recognizer | 7.8 ms | 21 ms | **5.9 ms** | 3.6× |
| CJK recognizer | 2.7 ms | | **3.8 ms** | |
| router | 0.4 ms | 6.5 ms | **1.5 ms** | 4.3× |

The world recognizer, which reads Bengali, is now faster in kiln than in ONNX Runtime. The
detector and the Latin recognizer are still behind it.

Whole pages, `kiln-cli --ocr`, the second of two calls:

| page | machine | first port | now |
|---|---|---|---|
| English test page, 1400 × 528, 9 lines | Ryzen 5 8500G | 0.18 s | **0.06 s** |
| English test page | Galaxy A17 5G (2 big cores) | 0.70 s | **0.27 s** |
| Bengali test page, 1000 × 384, 4 lines | Galaxy A17 5G | | **0.13 s** |
| receipt photo, 600 × 600, 8 lines | Galaxy A17 5G | | **0.43 s** |

In the Lekha test app (`android/lekha`) on the same Galaxy A17, the first read after starting
the app took 309 ms for the English page and 152 ms for the Bengali page, image decoding included.

Which change applies where:

| change | applies to |
|---|---|
| channels-first layout | all six networks |
| fusion: bias + ReLU | all six |
| fusion: 8-bit rounding | the Latin recognizer, the only one with FakeQuant |
| first-layer kernel | the first layer of all six |
| memory reuse | all six; most visible on the detector, the largest image |
| output read in place | mostly the CJK recognizer |

---

## 7. How correctness was kept

Speed work is only safe when every step is checked. Two checks were run after every change:

- **Each network against ONNX Runtime.** The same input tensor went through ONNX Runtime and
  through kiln, and every output number was compared. All six agree to within 1e-6 (0.000001).
- **Whole lines against an independent crop.** For lines, OpenCV cut out the line at the corners
  kiln reported (`kiln-cli --quads`), ONNX Runtime read it, and the text and confidence matched
  kiln's.

One bug only these checks could catch: at first about 5% of the Latin recognizer's confidences
were off by up to 0.07. ONNX's 8-bit step divides by the scale and rounds half to even; kiln
multiplied by the inverse and rounded halves away from zero (ggml's `ggml_round`). Both look
reasonable; only the first matches the model. The fused operation in 4.2 does it ONNX's way.

`kiln-tests models` has 35 OCR checks (`test_ocr` in `tests/test_core.cpp`) on the English,
Bengali and upside-down test pages in `tests/media/`.

---

## 8. Lessons

1. **The format does not decide the speed; the engine does.** GGUF and ONNX hold the same
   weights. The first port and the final one read the same file.
2. **Profile before optimising.** The guess was "convolutions are slow". Profiling one operation
   at a time showed the 8-bit rounding was two thirds of a network, and a single division 17 ms.
3. **Memory layout can matter more than arithmetic.** The same multiplications got four times
   faster by storing the numbers in a different order.
4. **Moving data costs more than computing on it.** Fusion did not remove a single addition; it
   removed trips through memory.
5. **Measure the "obvious" improvement too.** im2col was the textbook answer and made things slower.

---

## 9. Reproducing the numbers

```bash
cmake --preset host && cmake --build --preset host -j
scripts/fetch_models.sh --ocr                       # downloads OCRia-nano-1 and converts it
./build/host/kiln-cli -m models/ocria-nano-1.gguf --ocr tests/media/ocr_english.png
./build/host/kiln-tests models                      # includes the 35 OCR checks
```

`kiln-cli --ocr` prints the time spent finding lines (`detect`) and reading them (`read`). Time
the second of two calls, on a quiet machine: see "Benchmark only on a quiet machine" in
`CLAUDE.md`. On a phone, `scripts/android-test.sh` runs the same tests; the Galaxy A17 numbers
and settings are in [devices/galaxy-a17-5g.md](devices/galaxy-a17-5g.md).

The per-network comparisons with ONNX Runtime were run with Python scripts that are not part of
the repository: ONNX Runtime and onnx in a virtual environment, the same input tensor written to
both sides, every output compared.

Not yet measured: macOS, iOS, and the GPU. All the numbers above are on the CPU.
