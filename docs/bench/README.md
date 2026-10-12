# Benchmarks

These benchmarks cover chat models only (generation speed over time). Decision and embedding
models are timed per call in the main [README](../../README.md#decision-and-embedding-models);
`kiln-bench` does not run them yet.

Resource usage over time while kiln loads a model and generates. Produced by `kiln-bench`
(`tools/kiln-bench`) and `scripts/bench_plot.py`; rerun everything with `scripts/bench.sh`.

Workload: load, then 5 runs of a ~65-token prompt + 256 generated tokens with a context reset
between runs, then a cooldown. Sampler at 250 ms: process RSS, process CPU% (100 = one core),
mean frequency of the fast cores, CPU cluster and skin temperature (Android: `dumpsys
thermalservice`, since `/sys/class/thermal` is not readable by apps on Android 14+), and the
OS thermal throttle status. 2026-10-10, kiln v0.1.0, llama.cpp b11548.

## Why the same prompt runs 5 times

In the app, one prompt gives one answer. Nothing is repeated. The 5 repeats exist only in the
benchmark, to imitate a long chat session under controlled conditions.

A user chatting for a minute sends several messages back to back. Each one heats the phone a
little more, so message 5 runs on a hotter SoC than message 1 and comes out slower. To measure
that effect the bench could send 5 different prompts, but then a speed difference might come
from the prompts differing in length or difficulty. So it sends the *same* prompt 5 times:
now the only thing that changes between run 1 and run 5 is heat.

So "how fast is kiln on a Pixel 8 with Qwen3.5 0.8B?" has two honest answers: 38 tok/s for
the first message, about 31 tok/s a minute into the conversation. Quote both.

## How to read the tables

- **run**: one repetition of the job: reset the context, feed the ~65-token prompt, generate
  256 tokens. The bench does 5 runs back to back with a 0.5 s gap. Run 1 starts on a cool
  phone. Run 5 starts after 40 to 60 s of nonstop generation, on a hot phone. Nothing else
  changes between runs: same model, prompt, seed, code.
- **gen tok/s run 1 → 5**: generation speed measured in run 1, then in run 5. The gap between
  the two numbers is the cost of heat. Report the run-5 number if you want to be honest about
  what a chat session feels like.
- **fast-core MHz run 1 → 5**: average clock of the five fast cores while generating, run 1
  versus run 5. The SoC lowers the clock to hold the cluster at ~80 °C. Fewer MHz, fewer
  tok/s; the two columns move together.
- **CPU temp**: hottest CPU cluster reported by the thermal HAL, at bench start → maximum.
- **OS throttle**: Android's thermal status (0 none, 1 light, 2 moderate, 3 severe, up to 6).
  Driven by *skin* temperature, not cluster temperature; on the Pixel 8 status 1 starts at
  39 °C skin. "not within 20 s" means the run was too short to trigger it.
- **RSS after load / peak RSS**: resident memory of the process right after the model and
  context are created, and the maximum seen. Includes the mmap'd model file once touched.
- **process CPU%**: CPU time consumed per wall second, summed over threads. 100 = one core
  fully busy; 500 = five cores fully busy.
- **load**: wall time for `kiln_model_load` + `kiln_context_create`.

See [../GLOSSARY.md](../GLOSSARY.md) for the rest of the project's vocabulary.

## Pixel 8 (Tensor G3, CPU, 5 threads pinned to X3 + 4x A715)

| model | load | RSS after load | peak RSS | gen tok/s run 1 → 5 | fast-core MHz run 1 → 5 | CPU temp | OS throttle |
|---|---|---|---|---|---|---|---|
| LFM2.5 350M Q4_K_M | 0.9 s | 506 MB | 513 MB | 109 → 90 | 2480 → ~2050 | 42 → 82 °C | not within 20 s |
| Qwen3.5 0.8B Q4_K_M | 2.0 s | 1176 MB | 1200 MB | 38 → 31 | 2480 → 1690 | 45 → 82 °C | status 1 at 36 s |
| Gemma 3 270M Q8_0 | 1.6 s | 657 MB | 665 MB | 80 → 71 | 2480 → ~1900 | 45 → 83 °C | status 1 at 31 s |

![](pixel8_Qwen3.5-0.8B-Q4_K_M.png)

### What the charts say

- **CPU: all five fast cores saturated the whole time.** Process CPU sits at 480 to 500%. The
  pinned threadpool is doing exactly what it should. Little cores stay idle.
- **The CPU cluster hits ~80 °C within one second of generation starting** and stays there. The
  SoC then holds temperature by stepping frequency down: 2.48 GHz at run 1, 1.69 GHz by run 5
  on Qwen. Token rate follows: **-18% for Qwen, -18% for LFM2.5, -11% for Gemma across 60
  seconds** of sustained generation. This is before the OS even declares throttling.
- **OS throttle status flips to 1 when skin temperature crosses 39 °C**, about 30 to 36 seconds
  into sustained generation. Status 1 is mild; a long chat session would walk up the ladder
  (43 °C, 45 °C, 46.5 °C ...) and lose more.
- **Memory is flat and predictable.** RSS = model file (mmap, fully touched) + KV cache for
  n_ctx 2048 + ggml compute buffers. Qwen3.5 0.8B costs 1.2 GB resident, more than 2x its file
  size, because its hybrid architecture has large state buffers. RSS drops to tens of MB after
  free: no leaks across 5 runs.
- **Prompt eval is cheap and stable**: 120 to 490 ms for 65 tokens; not affected by thermals
  much because it is short.

### Implications for the library

1. Peak tok/s from a single short run is a lie on phones. The honest number is the run-5
   figure, and the stats API already reports per-call tok/s so an app can watch it decay.
2. 5 threads pegged at 100% is the fastest way to hit the thermal wall. Worth testing whether
   4 threads gives a higher *sustained* rate at lower temperature. On the roadmap.
3. "Available RAM" on this 8 GB phone bottomed at 1.67 GB with Qwen loaded. A 2B model at Q4
   (1.5 to 2.5 GB resident) would work but would start evicting other apps.

## M4 Pro (Metal)

| model | load | peak RSS | gen tok/s (5 runs) | process CPU |
|---|---|---|---|---|
| LFM2.5 350M Q4_K_M | 0.2 s | 365 MB | 210 to 362 | ~26% |
| Qwen3.5 0.8B Q4_K_M | 0.5 s | 761 MB | 139 to 147 | ~25% |
| Gemma 3 270M Q8_0 | 0.4 s | 468 MB | 118 to 166 | ~25% |

CPU stays at a quarter of one core: the GPU does the work and the sampler cannot see it
(no GPU counters yet). Run-to-run variance on tiny models is dispatch overhead, not thermals.
RSS decreases over time as macOS reclaims clean mmap pages.

## Files

- `<device>_<model>.csv`: raw samples. Columns: `t_ms,run,phase,rss_mb,cpu_pct,freq_mhz,temp_c,skin_c,throttle,mem_avail_mb,tokens`.
- `<device>_<model>.png`: chart.
- `summary.md`: table regenerated by `bench_plot.py`.
