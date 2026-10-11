// barrier-lab: a stripped-down copy of how ggml runs a graph on N threads, for experiments.
//
// Every "op" is split across the threads, then all threads meet at a spin barrier (the same
// algorithm as ggml_barrier in ggml-cpu.c). The program reports how long each thread spent
// working vs waiting at the barrier, so you can see the slowest worker setting the pace.
//
// build: g++ -O2 -pthread -o barrier-lab tools/barrier-lab/main.cpp
// usage: ./barrier-lab [--cpus 0,1,2|none] [--mode compute|memory] [--split static|dynamic]
//                      [--ops N] [--slow F]
//   --cpus   pin thread i to the i-th listed CPU ("none" = let the OS schedule; then --threads N)
//   --mode   compute: arithmetic in registers (core-bound). memory: stream a 256 MB buffer
//            (bandwidth-bound, like LLM decode)
//   --split  static: thread i gets exactly 1/N of the work (ggml for small matrices and most
//            non-matmul ops). dynamic: threads grab 64 small chunks from an atomic counter
//            (ggml's mul_mat when there are enough chunks)
//   --slow   give thread 0 F times its share of work, to simulate a slower core
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
#if defined(__x86_64__)
#  include <immintrin.h>
static inline void cpu_relax() { _mm_pause(); }
#elif defined(__aarch64__)
static inline void cpu_relax() { __asm__ volatile("yield" ::: "memory"); }
#else
static inline void cpu_relax() {}
#endif

using clk = std::chrono::steady_clock;
static double secs(clk::duration d) { return std::chrono::duration<double>(d).count(); }

// ---- the barrier, same shape as ggml_barrier
struct Barrier {
    alignas(64) std::atomic<int> n_barrier{0};
    alignas(64) std::atomic<int> n_passed{0};
    int n;
    void wait() {
        int passed = n_passed.load(std::memory_order_relaxed);
        if (n_barrier.fetch_add(1, std::memory_order_seq_cst) == n - 1) {   // last one in
            n_barrier.store(0, std::memory_order_relaxed);
            n_passed.fetch_add(1, std::memory_order_seq_cst);                  // release everyone
            return;
        }
        while (n_passed.load(std::memory_order_relaxed) == passed) cpu_relax(); // spin
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
};

// ---- work: one "unit" is a fixed amount of either arithmetic or memory traffic
static const size_t UNITS = 4096;                  // work units per op
static const size_t BUF_FLOATS = 64u << 20;        // 256 MB
static std::vector<float> g_buf;

static float do_units(bool memory, size_t first, size_t count) {
    float acc = 0;
    if (memory) {
        const size_t per = BUF_FLOATS / UNITS;          // each unit reads its own 64 KB slice
        const float* p = g_buf.data() + first * per;
        for (size_t i = 0; i < count * per; ++i) acc += p[i];
    } else {
        float a = 1.0001f + first, b = 0.9999f;
        for (size_t i = 0; i < count * 4000; ++i) a = a * b + 0.0001f;  // dependent chain
        acc = a;
    }
    return acc;
}

int main(int argc, char** argv) {
    std::vector<int> cpus = {0, 1};
    int n_unpinned = 0, ops = 200;
    bool memory = false, dynamic = false;
    double slow = 1.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "--cpus") {
            std::string v = next(); cpus.clear();
            if (v != "none") for (char* t = strtok(&v[0], ","); t; t = strtok(nullptr, ",")) cpus.push_back(atoi(t));
        }
        else if (a == "--threads") n_unpinned = atoi(next().c_str());
        else if (a == "--mode")    memory = next() == "memory";
        else if (a == "--split")   dynamic = next() == "dynamic";
        else if (a == "--ops")     ops = atoi(next().c_str());
        else if (a == "--slow")    slow = atof(next().c_str());
        else { fprintf(stderr, "unknown arg %s (see the comment at the top of main.cpp)\n", a.c_str()); return 1; }
    }
    const bool pinned = !cpus.empty();
    const int n = pinned ? (int)cpus.size() : (n_unpinned > 0 ? n_unpinned : 2);
    if (memory) g_buf.assign(BUF_FLOATS, 1.0f);

    Barrier bar; bar.n = n;
    alignas(64) std::atomic<size_t> next_chunk{0};
    std::vector<double> busy(n, 0), waited(n, 0);
    std::vector<float> sink(n, 0);

    auto worker = [&](int ith) {
        if (pinned) {
            cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[ith], &set);
            if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0)
                fprintf(stderr, "warning: could not pin thread %d to CPU %d\n", ith, cpus[ith]);
        }
        for (int op = 0; op < ops; ++op) {
            auto t0 = clk::now();
            if (dynamic) {
                const size_t CHUNKS = 64, per = UNITS / CHUNKS;
                for (size_t c = ith; c < CHUNKS; c = next_chunk.fetch_add(1, std::memory_order_relaxed)) {
                    size_t cnt = ith == 0 ? (size_t)(per * slow) : per;   // thread 0's chunks cost more
                    sink[ith] += do_units(memory, c * per, cnt);
                }
            } else {
                size_t per = UNITS / n;
                size_t cnt = ith == 0 ? (size_t)(per * slow) : per;
                sink[ith] += do_units(memory, ith * per, cnt);
            }
            auto t1 = clk::now();
            bar.wait();
            if (dynamic && ith == 0) next_chunk.store(n, std::memory_order_relaxed);  // reset for next op
            if (dynamic) bar.wait();                                                  // everyone sees the reset
            auto t2 = clk::now();
            busy[ith] += secs(t1 - t0); waited[ith] += secs(t2 - t1);
        }
    };

    if (dynamic) next_chunk.store(n);
    auto start = clk::now();
    std::vector<std::thread> th;
    for (int i = 0; i < n; ++i) th.emplace_back(worker, i);
    for (auto& t : th) t.join();
    double wall = secs(clk::now() - start);

    printf("%d threads, %s, %s split, %s%s\n", n, memory ? "memory-bound" : "compute-bound",
           dynamic ? "dynamic" : "static", pinned ? "pinned" : "unpinned",
           slow != 1.0 ? (", thread 0 does " + std::to_string(slow) + "x work").c_str() : "");
    printf("  %.3f ms per op  (%d ops, %.2f s)\n", wall * 1000 / ops, ops, wall);
    for (int i = 0; i < n; ++i) {
        double tot = busy[i] + waited[i];
        printf("  thread %2d %-8s busy %5.1f%%  waiting at barrier %5.1f%%\n", i,
               pinned ? ("cpu " + std::to_string(cpus[i])).c_str() : "", 100 * busy[i] / tot, 100 * waited[i] / tot);
    }
    float s = 0; for (float v : sink) s += v;
    return s == 12345.f;   // keep the work from being optimised away
}
