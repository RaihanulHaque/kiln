// Which cores kiln runs on, and how many threads it starts. Pure functions with no system
// calls, so the rules can be tested with the core layouts of phones that are not on the desk.
#pragma once
#include <cstdint>
#include <vector>

namespace kiln {

// max_freq[i]: top clock of core i in kHz (0 = unknown). capacity[i]: the kernel's cpu_capacity
// for core i, where 1024 is the strongest core of the chip; empty when it could not be read.
// Returns the bitmask of cores to use (bit i = core i), or 0 when all cores are alike and there
// is nothing to choose.
//
// Rule 1 (big.LITTLE): cores are grouped by top clock, and the slowest group is dropped.
//   Tensor G3: 4 x 1.70 GHz (A510), 4 x 2.37 GHz (A715), 1 x 2.91 GHz (X3)  ->  the five fast ones.
// Rule 2 (no little cores): the slowest group is kept when its capacity is at least half of the
//   strongest core's.
//   Snapdragon 8 Elite Gen 5: 6 x 3.63 GHz at capacity 741, 2 x 4.74 GHz at 1024  ->  all eight.
//   Rule 1 alone left 2 threads there and six strong cores idle.
inline uint64_t fast_core_mask(const std::vector<long>& max_freq, const std::vector<long>& capacity) {
    long minf = 0, maxf = 0;
    for (long f : max_freq) if (f > 0) { if (!minf || f < minf) minf = f; if (f > maxf) maxf = f; }
    if (maxf <= minf) return 0;

    bool slow_is_strong = false;
    if (capacity.size() == max_freq.size()) {
        long cap_slow = 0, cap_max = 0;
        bool ok = true;
        for (size_t i = 0; i < capacity.size(); ++i) {
            if (capacity[i] <= 0) { ok = false; break; }
            if (capacity[i] > cap_max) cap_max = capacity[i];
            if (max_freq[i] == minf && capacity[i] > cap_slow) cap_slow = capacity[i];
        }
        slow_is_strong = ok && cap_max > 0 && cap_slow * 2 >= cap_max;
    }
    uint64_t mask = 0;
    for (size_t i = 0; i < max_freq.size() && i < 64; ++i)
        if (max_freq[i] > minf || (slow_is_strong && max_freq[i] > 0)) mask |= (1ull << i);
    return mask;
}

// Threads to start when the app does not say. cores: all cores; fast: cores in the mask (or
// `cores` when unknown); pinned: threads will be pinned to the mask.
//
// One thread per fast core, at most 8. Where every core is fast and threads are pinned, two cores
// are left free: there are no little cores for the UI thread and the rest of the system, and on a
// Galaxy S26 Ultra six pinned threads were faster than eight in every sustained run (LFM2.5 350M
// 194 vs 164 tok/s in the fifth run, Qwen3.5 0.8B 56 vs 49, Gemma 3 270M 146 vs 133).
// Threads go to the highest-numbered cores first, which keeps the strongest ones.
inline int default_thread_count(int cores, int fast, bool pinned) {
    int n = fast > 0 ? fast : cores;
    if (n <= 0) n = 4;
    if (pinned && n == cores && n > 4) n -= 2;
    return n > 8 ? 8 : n;
}

} // namespace kiln
