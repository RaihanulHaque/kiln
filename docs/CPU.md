# CPU inference on phones: what actually limits speed

Low-level notes behind kiln's threading and probe decisions. Written so the policy can be
argued about per SoC family rather than tuned on one phone.

## 1. Decode is memory-bandwidth bound, not compute bound

Generating one token means reading **every weight once**. A 0.8B model at Q4_K_M is ~0.5 GB
of weights, so 40 tok/s = 20 GB/s of sustained reads. Phone LPDDR5 delivers roughly 25 to
70 GB/s peak and much less sustained from the CPU side. That is the ceiling.

Consequences:

- More cores help only until they saturate the memory bus. Past that, extra threads add
  barrier overhead and heat for nothing. This is why 5 vs 4 threads is an experiment, not a
  foregone conclusion.
- Smaller quants are faster almost linearly on decode (Q4 reads half what Q8 reads), which is
  why Gemma 270M Q8 (80 tok/s) is slower than LFM2.5 350M Q4 (109 tok/s) despite fewer params.
- Prompt eval is different: many tokens share one weight read, so it is compute bound and
  scales with cores and ISA width. It is also short, so it rarely matters on phones.

## 2. big.LITTLE and why the slowest core sets the pace

ggml splits every matmul across its worker threads and then **waits at a barrier** for all of
them. The matmul finishes when the slowest thread finishes. Put one worker on a Cortex-A510
and four on A715s, and the A715s idle at the barrier while the A510 grinds. Measured on the
Pixel 8: 5 unpinned threads were *slower* than 1 thread.

Rules that follow:

- Workers must run on cores of **equal speed**, or the fast ones are wasted.
- Pin them (CPU affinity) so the Android scheduler cannot migrate a worker onto a little core
  mid-matmul. Android's EAS scheduler optimises for energy, not for barrier latency.
- A faster "prime" core (X-series) in a cluster of equals is only useful if its extra speed is
  not spent waiting. With 1 X3 + 4 A715 the X3 finishes its slice early and waits. Whether to
  include it is an empirical question per SoC: it adds bandwidth and heat.

Common layouts (prime + big + little):

| SoC family | layout | notes |
|---|---|---|
| Tensor G3 (Pixel 8) | 1x X3 + 4x A715 + 4x A510 | kiln's reference; fast set = 5 |
| Snapdragon 8 Gen 2 | 1x X3 + 4x (A715/A710) + 3x A510 | similar to G3 |
| Snapdragon 8 Gen 3 | 1x X4 + 5x A720 + 2x A520 | fast set = 6; little tier tiny |
| Snapdragon 8 Elite | 2x Oryon prime + 6x Oryon | no little cores at all; "drop slowest cluster" would wrongly drop 6 cores |
| Dimensity 9200/9300 | 1x X3 + 3x A715 + 4x A510 / 4x X4 + 4x A720 | 9300 has no little tier |
| Exynos 2400 | 1x X4 + 2x A720@2.9 + 3x A720@2.6 + 4x A520 | four frequency tiers |
| Snapdragon 7 series | 4x A7xx + 4x A5xx | fast set = 4 |
| Budget (SD 4xx, Helio G) | 2x A76 + 6x A55 | fast set = 2; 6 littles may beat 2 bigs on bandwidth |

"Drop the slowest cluster" is right for the first row and wrong for several others. See §6.

## 3. ISA extensions that matter for quantised matmul

The probe reports these; ggml picks kernels based on what the **build** enables, not what the
device has, unless runtime dispatch is on.

- **dotprod** (ARMv8.2, `sdot/udot`): 4x int8 multiply-accumulate per lane. The single
  biggest win for Q4/Q8 kernels. Universal on anything from 2018 on.
- **fp16** (`FEAT_FP16`): half-precision arithmetic; used for activations and some kernels.
- **i8mm** (ARMv8.6, `smmla`): int8 matrix multiply, 2x over dotprod for Q8-type kernels in
  ggml's `GGML_CPU_ARM_ARCH=armv8.6-a+i8mm` builds. Present on A710/A715/X2+ and up; absent on
  A55/A76/A77 era cores.
- **SVE/SVE2**: scalable vectors. Present on X3/A715/A510 but ggml's SVE paths are limited;
  gains are small today.

kiln's Android build baseline is `armv8.2-a+dotprod+fp16`, chosen so one `.so` runs on
nearly everything. The Flutter plugin also ships `libkiln_armv8.so` (plain `armv8-a`) for the
phones that lack dotprod or fp16, and `native.dart` picks one from `/proc/cpuinfo`; the Kotlin
and Rust bindings have no fallback yet. The Pixel 8 has i8mm and SVE and does not use them. Fix: build the CPU backend
with `GGML_CPU_ALL_VARIANTS` (ggml compiles several kernel sets and picks at runtime from
`getauxval(AT_HWCAP)`), at the cost of a larger `.so`. On the roadmap; expect 10 to 25% on
Q8 kernels on i8mm cores, less on Q4_K.

## 4. DVFS and thermal: the clock is not a constant

Every phone runs its CPU at whatever frequency the thermal budget allows. On the Pixel 8:

- The CPU cluster reaches ~80 °C within **one second** of sustained decode and stays there.
  The governor then reduces frequency to hold that temperature: 2.48 GHz → 1.69 GHz over
  60 s with 5 threads. Token rate tracks frequency almost 1:1 because decode is bandwidth
  bound and bandwidth scales with the memory/fabric clock that drops along with it.
- Separately, Android's thermal service watches **skin** temperature and starts throttling the
  whole system at status 1 (39 °C on Pixel 8), with steps at 43/45/46.5/52/55 °C.
- So there are two regimes: the first ~10 s (cold, fast) and steady state (hot, 10 to 20%
  slower). Any number from a short run is the cold number.

Fewer threads produce less heat per token only if they also finish the token faster per joule;
otherwise the same work just takes longer. That is exactly what the thread sweep measures.

## 5. Memory layout and caches

- Weights are mmap'd. First touch page-faults them in (the slow first run), after which they
  are resident as long as the OS does not evict them. RSS after load ≈ file size + state.
- KV cache size = layers × 2 × n_ctx × kv_dim × bytes. For a 0.8B model at n_ctx 4096 it is a
  few hundred MB in fp16. Hybrid models (Qwen3.5, LFM2) replace much of it with fixed-size
  recurrent state, which is why Qwen3.5 0.8B's RSS is 2.3x its file size: large per-layer
  state buffers allocated up front.
- L2/L3 are irrelevant for decode (weights stream through once) but matter for prompt eval and
  for the recurrent state updates.

## 6. What kiln does today, and what it should do

**Today**: probe reads `cpuinfo_max_freq` and `cpu_capacity` per core, clusters by frequency,
and keeps every core faster than the slowest cluster. The slowest cluster is kept as well when
its `cpu_capacity` (the kernel's rating of a core, 1024 = strongest) is at least half of the
strongest core's: then it is not a little cluster. N = that many threads (capped at 8) are
pinned to the highest-numbered of those cores; where every core is fast, two are left free.
The rules are pure functions in `src/probe/fast_cores.h`, with unit tests.
Measured `cpu_capacity`: Pixel 8 (Tensor G3) 182 / 725 / 1024 for A510 / A715 / X3;
Galaxy S26 Ultra (Snapdragon 8 Elite Gen 5) 741 / 1024. A little core is under a fifth of the
best core; the half-way line has a wide margin on both sides.

- no-little layouts (8 Elite, Dimensity 9300): **fixed after measuring a Galaxy S26 Ultra**
  (Snapdragon 8 Elite Gen 5, 2 + 6). The frequency rule alone kept 2 cores; with the capacity
  rule it keeps all 8 and runs 6 threads, which under sustained load is 18 to 24% faster for LFM2.5 and Qwen3.5 chat,
  about level for Gemma 3 270M (4% slower), and takes 35 to 45% less time per decision and embedding. Sweep and numbers:
  [devices/galaxy-s26-ultra.md](devices/galaxy-s26-ultra.md). A Dimensity has not been tried.

Still wrong or unknown on:

- four-tier layouts (Exynos 2400): "slowest cluster" removes only the A520s; the remaining
  mix of 2.9 and 2.6 GHz A720s is unequal and will barrier-wait. Fix: group into tiers, and
  only mix tiers whose frequencies are within ~10% of each other.
- 2+6 budget layouts: 2 fast threads may lose to 6 slow ones on bandwidth. Unknown without
  measuring.
- x86 with SMT, and chips whose cores differ but report one max frequency (§8): the probe sees
  identical cores, builds no mask, does not pin, and counts hyperthreads as cores.

**Direction**: measure, don't assume. The probe gives candidate core sets (fastest cluster
only, fastest two clusters, all-but-little, everything). On first run with a model, kiln can
decode ~32 tokens with each candidate (under 5 s total), keep the winner, and cache the choice
per `(soc, model arch)` in the app's files dir. This turns every user's phone into a data
point instead of trusting a table, and it is the only approach that survives the long tail of
Android SoCs. Opt-in telemetry on top of that gives the compatibility table the README
promised.

Until that exists, `KILN_CPU_MASK` and `KILN_NO_AFFINITY` let a developer override the
policy, and `kiln-bench` plus `scripts/bench_compare.py` quantify any configuration.

### Measured on the Pixel 8 (Qwen3.5 0.8B)

Full results in [bench/threads](bench/threads/README.md). Short version: 5 pinned threads
beat every alternative cold (37 tok/s) and hot (21 tok/s); fewer threads ran *hotter* (a lone
X3 reaches 96 °C); and 5 unpinned threads on a warm phone collapsed to 2 to 5 tok/s because
the scheduler moved them to the little cores. Pinning is the policy's load-bearing part.

## 7. How to read a thread sweep

`scripts/bench_compare.py` plots, per configuration, tok/s and clock and temperature across
back-to-back runs. Look for:

- **run-1 tok/s**: raw throughput while cold. Usually favours more threads.
- **run-8 tok/s**: sustained throughput. The number a chat session actually gets.
- **drop %**: how much heat cost that configuration. A config with a lower drop may win
  sustained even if it loses cold.
- **MHz**: if two configs reach the same clock floor, the one with more useful threads wins.
  If a config with fewer threads keeps a higher clock, it is converting heat headroom into
  speed.

## 8. A desktop case: Ryzen 5 8500G (x86, SMT, two kinds of core)

Measured on Linux, 2026-10-11, with `tools/barrier-lab` and `kiln-bench`. The chip has 2 Zen 4
cores and 4 smaller Zen 4c cores, each with two hardware threads: 12 logical CPUs, where CPU
*n* and CPU *n+6* share a core.

**The frequency label does not show the difference.** `cpuinfo_max_freq` (and `lscpu`) report
5076 MHz for all 12 CPUs. The same work on one core at a time:

| core | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| ms per op (barrier-lab, compute) | 19.9 | 27.7 | 27.7 | 27.9 | 20.0 | 27.6 |

Cores 0 and 4 are about 40% faster alone, and about 25% faster with all six busy: with a fixed
1/N split, their threads spend 20% of every op waiting at the barrier. The probe groups by max
frequency, so it sees six equal cores. `acpi_cppc/highest_perf` may rank them correctly; that
was not checked.

**ggml softens the slowest-worker rule for big matmuls only.** `ggml_compute_forward_mul_mat`
hands out 16- or 64-row chunks from an atomic counter when there are at least 4 chunks per
thread, so a fast thread takes more. Below that, and for every non-matmul op, each thread gets
a fixed share and the barrier after the op waits for the slowest. In the lab (this one run on
a 4-core cloud VM, not the 8500G), one thread with 2x the work cost 15.9 ms per op with a
fixed split and 9.3 ms with chunks, against 8.9 ms with equal threads.

**Decode on this machine, Qwen3.5 0.8B Q4_K_M** (`KILN_CPU_MASK`, 3 runs of 256 tokens, quiet
machine):

| threads | gen tok/s | prompt ms (65 tok) |
|---|---|---|
| default: 8, unpinned | 44.6 | 153 |
| `11`: the 2 Zen 4 cores | 43.8 | 263 |
| `41`: one Zen 4 core, both hardware threads | 36.6 | 425 |
| `3f`: 6 cores, one thread each | 44.5 | 156 |
| `fff`: all 12 hardware threads | 32.5 | 198 |

What this shows:

- Decode hits the memory wall with two cores: 2, 6 and 8 threads all give about 44 tok/s.
- Prompt eval is compute bound, so six cores beat two (156 ms against 263 ms), slow cores included.
- Two threads on one core hurt real kernels: 16% slower than two cores, and 12 threads are 27%
  slower than 6. The lab's compute mode shows the opposite (6 to 8 threads gave 1.31x), because
  its dependent multiply chain leaves most of a core idle. ggml's kernels do not.
- Pinning made no difference on an idle desktop (44.6 unpinned, 44.5 pinned). On a busy one it
  matters a lot: with Zoom and a browser running, `fff` fell to about 6 tok/s while `3f` held 32.

So on this chip the best set is one thread per physical core, and kiln's default of 8 unpinned
threads only matches it because of the 8-thread cap. A 4-core / 8-thread laptop would get two
workers per core. The probe change: count physical cores from
`/sys/devices/system/cpu/cpu*/topology/core_id`, and do not trust max frequency to tell cores
apart. Not implemented yet.

Benchmark on a quiet machine: the default configuration dropped by a third (44 to 30 tok/s)
while Zoom and a browser were running, on unchanged code.
