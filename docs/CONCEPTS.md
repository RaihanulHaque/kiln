# Concepts, in plain language

Read this before the glossary. It explains the ideas; the glossary defines the terms.

## Cores and threads

A **core** is a physical processor on the chip. The Pixel 8 has nine: one Cortex-X3 (fastest),
four A715 (fast), four A510 (slow, power-saving). That count never changes.

A **thread** is a unit of work that the program creates. kiln asks the math library (ggml) to
split every matrix multiplication into N slices and run them at the same time; N is the
thread count. Five threads on five cores means each core runs one slice and the work finishes
about five times sooner. Five threads on one core means the OS takes turns between them and
nothing gets faster.

So a thread count is a *request*. Which cores run those threads is the operating system's
choice, unless the program pins them.

## Pinned and unpinned

**Unpinned**: the program creates threads and lets Android decide where they run. Android's
scheduler is tuned for battery life, so it will move a thread onto a slow A510 whenever it
thinks that saves power. It also moves threads around while they are running.

**Pinned**: the program sets a CPU affinity mask on each thread: "this thread may run on core
8 only", "this one on core 7 only", and so on. The OS honours it and never migrates them.

Why it matters: ggml runs the N slices and then waits at a **barrier** for all N to finish
before the next step. The step takes as long as the *slowest* slice. One slice on an A510
makes four fast cores sit idle. On a warm Pixel 8 that difference measured 21 tok/s pinned
versus 3 tok/s unpinned, same code, same model.

kiln pins by default. The core set comes from the probe (see below) and can be overridden
with `KILN_CPU_MASK` for experiments or disabled with `KILN_NO_AFFINITY=1`.

## The probe

Before loading anything, kiln reads what the device is: SoC name, how many cores, each
core's maximum clock, RAM, which ARM instruction extensions exist, whether a GPU backend is
compiled in. From the per-core clocks it groups cores into **tiers** and drops the slowest
tier; what remains are the **fast cores**, which become the thread count and the pin set.
Everything in `kiln_device_info` comes from here and is also what the stats and the sample
app display.

## Why generation speed is about memory, not compute

To produce one token the model reads **every weight once**. A 0.8B model at 4-bit is about
0.5 GB. At 40 tok/s that is 20 GB/s of reading, which is near what a phone's RAM can deliver
to the CPU. More cores help only until the memory bus is saturated. This is why 4-bit models
are nearly twice as fast as 8-bit ones of the same size, and why adding threads has
diminishing returns.

## Heat

A phone has no fan. Within a second of sustained generation the CPU cluster reaches its
limit (about 80 °C on a Pixel 8) and the chip lowers its clock to stay there: 2.5 GHz at the
start, 1.7 GHz a minute later. Token rate falls with the clock, roughly 40 to 50% over a
minute. Separately, when the phone's *skin* gets warm (39 °C), Android starts system-wide
throttling in steps.

The consequence: a speed number from a five-second test is the cold number. A chat session
runs at the hot number. kiln reports tok/s per call so an app can see this happen; the
benchmarks report both.

## What a sweep is

Run the identical benchmark job once per setting of one knob, keep everything else fixed,
compare. The thread sweep changed only the thread count and placement (1 to 5, pinned to
the top cores, pinned to the A715s only, unpinned). "Run 1 → 5" inside each configuration
is the same prompt repeated back to back, so that heat is the only thing changing between
run 1 and run 5. See [bench/README.md](bench/README.md).

## The KV cache and multi-turn chat

While reading a prompt the model computes, per layer, a "key" and "value" vector for every
token and stores them: the **KV cache**. Producing the next token then only needs the new
token's math plus a lookup into that cache. It is why the second token is as cheap as the
hundredth.

For a chat, every turn's prompt is the whole history plus the new message. kiln compares the
new prompt's tokens with what the cache already holds and evaluates only the part that
changed, usually just the new user message. That is `cached_tokens` in the stats. Some model
families (LFM2, Qwen3.5, Mamba) keep recurrent state that cannot be partially rewound; when
the history diverges before the end of the cache, kiln clears it and re-reads the whole
prompt. Slower, still correct.

## Reasoning models and channels

Models like Qwen3.5 can "think" before answering: they emit a block between `<think>` and
`</think>`, then the answer. The template decides whether thinking is on. kiln strips the
tags and delivers each piece of text on one of two channels, reasoning or answer, so an app
can show a collapsible thinking box and count the tokens separately. With thinking off,
kiln renders the template in a way that tells the model to answer directly.

## Chat templates

Each model was trained on conversations formatted a specific way: special tokens around
roles, a fixed system prompt position, rules like "roles must alternate". The format lives
inside the GGUF as a jinja template. Get it slightly wrong and the model still produces
text, just worse. kiln executes the model's own template with a real jinja engine
(llama.cpp's minja) rather than guessing the family from a few markers, which is what makes
Gemma, Qwen and LFM all behave without per-model code.

## What "backend" means

The engine that actually runs the model. Today there is one, llama.cpp, with its CPU path on
every device and its Metal path on Apple silicon. The C API is written so other engines
(Qualcomm QNN for the NPU, Google's LiteRT-LM) can sit behind the same calls later. The
model info reports which one loaded, because a library that silently falls back is a
library that lies.
