# Quantization on phones: a feasibility test

**Question:** could a kiln-specific quantization format ("KQ") make models 20% faster on a
phone CPU than the usual Q4 / Q8 files? **One run, one model, one phone:** LFM2.5 350M on a
Pixel 8 (Tensor G3), 2026-10-13, llama.cpp 781dbc5ac. Repeat it with `scripts/quant_eval.sh`.

## What limits speed on a phone

Generating a token means reading every weight once. On a phone the CPU reads memory far
slower than it multiplies, so **tokens per second is set by bytes per weight**, not by the
maths (see `docs/CONCEPTS.md`, "Why generation speed is about memory"). A format can only be
faster by (a) using fewer bytes per weight, or (b) having a kernel that turns those bytes into
numbers with less waste. A smarter format at the same bit count gains nothing on speed; it can
only gain on quality.

## What was measured

Six formats made from the same F16 file (`LFM2.5-350M-F16.gguf` from LiquidAI), three of them
also with an **importance matrix** (imatrix: a conversion-time pass over text that tells the
quantizer which weights matter more, so they get the finer scales). Quality is measured as how
far each format's next-word probabilities drift from the F16 model on wikitext-2 (mean KL
divergence; 0 = identical) and how often it picks the same top word. Speed is kiln-bench on the
Pixel 8, 3 runs of 256 tokens, the first run on a cool phone.

| format | file | mean KL from F16 | same top word | Pixel 8 tok/s (runs 1, 2, 3) |
|---|---|---|---|---|
| Q8_0 | 379 MB | 0.004 | 96.8% | 61, 57, 58 |
| Q6_K | 293 MB | 0.044 | 89.4% | 70, 64, 65 |
| Q4_K_M (the usual choice) | 229 MB | 0.483 | 67.6% | 90, 90, 76 |
| Q4_K_M + imatrix | 229 MB | **0.305** | **72.6%** | same bytes, same speed |
| Q4_0 | 219 MB | 0.838 | 58.6% | **109, 107, 106** |
| Q4_0 + imatrix | 219 MB | 0.598 | 65.3% | same bytes, same speed |
| IQ4_XS | 212 MB | 0.592 | 63.3% | 86, 68, 69 |
| IQ4_XS + imatrix | 212 MB | 0.493 | 66.8% | same bytes, same speed |

Three Android builds were compared on Q4_0 and Q4_K_M: the current one (`dotprod+fp16`), one
with **i8mm** added, and one with i8mm plus Arm's **KleidiAI** kernels. Generation speed was
the same within noise on all three (Q4_0: 109 / 109 / 111 tok/s). Prompt reading was about 8%
faster with KleidiAI (62 tokens in 185 ms against 203 ms). KleidiAI's Q8_0 run was slower and
its memory was being evicted during the run, so that figure was discarded.

## What it says

1. **The 20% exists, and it is free: Q4_0 is 20% faster than Q4_K_M** (109 against 90 tok/s)
   for 4% fewer bytes, because its simpler block layout has a leaner kernel. Nobody uses it
   because plain Q4_0 is clearly worse in quality (58.6% against 67.6% same top word).
2. **The importance matrix buys back most of that quality at no cost in speed.** Q4_0 + imatrix
   reaches 65.3%, close to plain Q4_K_M's 67.6%, at Q4_0's speed. Q4_K_M + imatrix is the best
   4-bit quality (72.6%) at Q4_K_M's speed. The imatrix is a conversion-time step on the
   computer; the phone runs a standard GGUF.
3. **Faster kernels do not help generation on this chip.** i8mm and KleidiAI change nothing
   for tokens per second, which confirms the memory-bound picture; they help prompt reading a
   little. SME on newer chips was not tested (the S26 Ultra has it).
4. **A 350M model is fragile at 4 bits.** Every 4-bit format keeps only 59 to 73% of the F16
   model's top words. Q6_K keeps 89% at 70 tok/s. Larger models lose much less; this test
   should be repeated on Qwen3.5 0.8B before any rule is made.

## What a "KQ" format would and would not do

- **A new format with the same bits per weight cannot be faster than Q4_0** on this phone;
  Q4_0 already reads the bytes about as fast as the memory allows. It could only be more
  accurate, and imatrix already does most of that inside standard GGUF.
- **Fewer bits** (3 or 2) is the only way to more speed, and there the loss in quality on
  small models is severe; IQ3 / IQ2 exist in ggml already and should be measured before
  anything is designed.
- A format of our own means our own kernels inside a fork of ggml for every CPU (Arm with and
  without i8mm / SME, x86, Apple), a conversion tool, and keeping all of it in step with
  llama.cpp. That is months of work with a ceiling close to where Q4_0 + imatrix already is.

**Recommendation.** Ship the gain that is there: a `kiln-quant` conversion recipe (F16 → GGUF
with an importance matrix, Q4_0 for speed or Q4_K_M for quality, embeddings and output kept
at 6 or 8 bits), publish the files under kiln's name, and benchmark them against the stock
Q4_K_M the way this document does. Revisit a true format of our own only after measuring
IQ3 / IQ2 and SME, and on a 1B-class model.

## Not done

One model, one phone, one text set. Wikitext is not chat; absolute perplexity is meaningless
for this instruction-tuned model (thousands), which is why KL and top-word agreement are used.
Qwen3.5 0.8B, Gemma 3, the S26 Ultra (SME), IQ3 / IQ2, and a chat-style evaluation text are
all open. Raw files: `scripts/quant_eval.sh` reproduces every number here.
