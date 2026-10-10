// kiln-bench: resource usage over time while kiln loads and generates.
// Samples RSS, process CPU%, CPU frequency, and thermal zones on a background thread.
// Writes CSV; scripts/bench_plot.py turns it into charts.
#include "kiln/kiln.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#  include <mach/mach.h>
#  include <sys/resource.h>
#elif defined(__linux__)
#  include <sys/resource.h>
#  include <dirent.h>
#endif

using clk = std::chrono::steady_clock;
static double now_ms() { static auto t0 = clk::now(); return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

// ---------------------------------------------------------------- probes

static double rss_mb() {
#if defined(__APPLE__)
    mach_task_basic_info info; mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &cnt) == KERN_SUCCESS) return info.resident_size / 1e6;
    return 0;
#else
    std::ifstream f("/proc/self/statm"); long pages = 0, res = 0; f >> pages >> res;
    return res * 4096.0 / 1e6;
#endif
}

static double cpu_seconds() {
    rusage r{}; getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_utime.tv_usec / 1e6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec / 1e6;
}

// mean current freq (MHz) over the fast cores, 0 when unavailable
static double cpu_freq_mhz(uint64_t fast_mask) {
#if defined(__linux__)
    double sum = 0; int n = 0;
    for (int i = 0; i < 64; ++i) {
        if (fast_mask && !(fast_mask & (1ull << i))) continue;
        char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        std::ifstream f(p); long khz = 0; if (f >> khz) { sum += khz / 1000.0; ++n; }
    }
    return n ? sum / n : 0;
#else
    (void)fast_mask; return 0;
#endif
}

struct ThermalZone { std::string path, type; };
static std::vector<ThermalZone> find_thermal_zones() {
    std::vector<ThermalZone> z;
#if defined(__linux__)
    for (int i = 0; i < 128; ++i) {
        char p[96]; snprintf(p, sizeof p, "/sys/class/thermal/thermal_zone%d/type", i);
        std::ifstream f(p); std::string t; if (!(f >> t)) continue;
        std::string lower = t; for (auto& c : lower) c = (char)tolower(c);
        if (lower.find("cpu") != std::string::npos || lower.find("big") != std::string::npos ||
            lower.find("mid") != std::string::npos || lower.find("little") != std::string::npos ||
            lower.find("soc") != std::string::npos || lower.find("tsens") != std::string::npos)
            z.push_back({std::string("/sys/class/thermal/thermal_zone") + std::to_string(i) + "/temp", t});
    }
#endif
    return z;
}
static double max_temp_c(const std::vector<ThermalZone>& zones) {
    double best = 0;
    for (auto& z : zones) { std::ifstream f(z.path); long m = 0; if (f >> m) { double c = m > 1000 ? m / 1000.0 : (double)m; if (c > best && c < 150) best = c; } }
    return best;
}

// Android 14+ hides /sys/class/thermal from apps and shell. dumpsys still works for the shell
// user, so the bench (not the library) reads temperatures from it about once a second.
struct Thermal { double cpu_c = 0, skin_c = 0; int status = 0; bool ok = false; };
static Thermal read_dumpsys_thermal() {
    Thermal t;
#if defined(__ANDROID__)
    FILE* p = popen("dumpsys thermalservice 2>/dev/null", "r");
    if (!p) return t;
    char line[512]; bool hal = false;
    while (fgets(line, sizeof line, p)) {
        if (strstr(line, "Thermal Status:")) { t.status = atoi(strchr(line, ':') + 1); t.ok = true; }
        if (strstr(line, "Current temperatures from HAL")) hal = true;
        if (!hal) continue;
        const char* v = strstr(line, "mValue="); const char* ty = strstr(line, "mType=");
        if (!v || !ty) continue;
        double val = atof(v + 7); int type = atoi(ty + 6);
        if (type == 0 && val > t.cpu_c) t.cpu_c = val;       // CPU clusters
        if (type == 3 && val > t.skin_c) t.skin_c = val;     // skin
    }
    pclose(p);
#endif
    return t;
}

static double mem_available_mb() {
#if defined(__linux__)
    std::ifstream f("/proc/meminfo"); std::string k; long v; std::string u;
    while (f >> k >> v >> u) if (k == "MemAvailable:") return v / 1024.0;
#endif
    return 0;
}

// ---------------------------------------------------------------- sampler

struct Sampler {
    std::atomic<bool> stop{false};
    std::atomic<int>  phase{0};          // index into phase_names
    std::atomic<int>  tokens{0};
    std::atomic<int>  run{0};
    std::vector<const char*> phase_names{"idle", "load", "prompt", "generate", "cooldown"};
    FILE* out;
    uint64_t fast_mask;
    std::vector<ThermalZone> zones;
    int interval_ms;
    std::thread th;

    void start() {
        fprintf(out, "t_ms,run,phase,rss_mb,cpu_pct,freq_mhz,temp_c,skin_c,throttle,mem_avail_mb,tokens\n");
        th = std::thread([this] {
            double last_t = now_ms(), last_cpu = cpu_seconds(), last_therm = -1e9;
            Thermal th{};
            while (!stop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
                double t = now_ms(), c = cpu_seconds();
                double pct = (c - last_cpu) * 1000.0 / (t - last_t) * 100.0;   // 100% = one core
                last_t = t; last_cpu = c;
                if (zones.empty() && t - last_therm > 1000) { th = read_dumpsys_thermal(); last_therm = t; }
                double temp = zones.empty() ? th.cpu_c : max_temp_c(zones);
                fprintf(out, "%.0f,%d,%s,%.1f,%.1f,%.0f,%.1f,%.1f,%d,%.0f,%d\n", t, run.load(), phase_names[phase.load()],
                        rss_mb(), pct, cpu_freq_mhz(fast_mask), temp, th.skin_c, th.status, mem_available_mb(), tokens.load());
                fflush(out);
            }
        });
    }
    void finish() { stop = true; th.join(); }
};

static Sampler* g_sampler;
static bool on_piece(kiln_channel, const char*, int32_t, void*) { g_sampler->tokens++; return true; }

int main(int argc, char** argv) {
    std::string model, out_path;
    int runs = 5, n_gen = 256, cooldown_s = 5, interval_ms = 250, threads = 0;
    std::string prompt = "You are a helpful assistant. Write a detailed, multi-paragraph explanation of how a kiln works, "
                         "covering the history of kilns, the physics of heat transfer inside them, the chemistry of firing clay, "
                         "and modern electric versus wood-fired designs. Be thorough.";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) exit(2); return argv[++i]; };
        if      (a == "-m")  model = next();
        else if (a == "-o")  out_path = next();
        else if (a == "-r")  runs = atoi(next());
        else if (a == "-n")  n_gen = atoi(next());
        else if (a == "-t")  threads = atoi(next());
        else if (a == "--cooldown") cooldown_s = atoi(next());
        else if (a == "--interval") interval_ms = atoi(next());
        else { fprintf(stderr, "usage: kiln-bench -m model.gguf [-o out.csv] [-r runs] [-n tokens] [-t threads] [--cooldown s]\n"); return 2; }
    }
    if (model.empty()) return 2;
    kiln_set_log_level(KILN_LOG_ERROR);

    kiln_device_info dev; kiln_probe_device(&dev);
    Sampler s; s.out = out_path.empty() ? stdout : fopen(out_path.c_str(), "w");
    s.fast_mask = dev.cpu_fast_mask; s.zones = find_thermal_zones(); s.interval_ms = interval_ms;
    g_sampler = &s;
    fprintf(stderr, "device: %s, %d cores (%d fast), thermal: %s\n", dev.soc, dev.cpu_cores, dev.cpu_big_cores, !s.zones.empty() ? "sysfs" : read_dumpsys_thermal().ok ? "dumpsys" : "none");
    now_ms();
    s.start();
    std::this_thread::sleep_for(std::chrono::seconds(2));          // baseline

    s.phase = 1;
    double t_load = now_ms();
    kiln_model_params mp = kiln_model_params_default(); mp.n_threads = threads;
    kiln_model* m = nullptr;
    if (kiln_model_load(model.c_str(), &mp, &m) != KILN_OK) { fprintf(stderr, "load: %s\n", kiln_last_error()); return 1; }
    kiln_context_params cp = kiln_context_params_default(); cp.n_ctx = 2048;
    kiln_context* c = nullptr;
    if (kiln_context_create(m, &cp, &c) != KILN_OK) { fprintf(stderr, "ctx: %s\n", kiln_last_error()); return 1; }
    double load_ms = now_ms() - t_load;
    kiln_model_info mi; kiln_model_get_info(m, &mi);
    fprintf(stderr, "model: %s %s via %s, load %.0f ms, rss after load %.0f MB\n", mi.arch, mi.quant, mi.backend_name, load_ms, rss_mb());
    s.phase = 0;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    kiln_chat_message msgs[] = {{"user", prompt.c_str()}};
    kiln_sampling_params sp = kiln_sampling_params_default(); sp.max_tokens = n_gen; sp.seed = 1;
    double peak_rss = 0;
    fprintf(stderr, "run,prompt_tok,prompt_ms,gen_tok,gen_ms,gen_tps,rss_mb,temp_c\n");
    for (int r = 1; r <= runs; ++r) {
        kiln_context_reset(c);
        s.run = r; s.tokens = 0;
        // phase flips to "generate" on first piece; prompt eval is everything before that
        s.phase = 2;
        struct Hook { static bool cb(kiln_channel ch, const char* p, int32_t t, void* u) { if (g_sampler->phase.load() == 2) g_sampler->phase = 3; return on_piece(ch, p, t, u); } };
        kiln_gen_stats st;
        kiln_status rc = kiln_chat(c, msgs, 1, nullptr, &sp, Hook::cb, nullptr, &st);
        double rss = rss_mb(); if (rss > peak_rss) peak_rss = rss;
        Thermal th = s.zones.empty() ? read_dumpsys_thermal() : Thermal{};
        fprintf(stderr, "%d,%d,%.0f,%d,%.0f,%.1f,%.0f,%.1f\n", r, st.prompt_tokens, st.prompt_ms, st.generated_tokens, st.gen_ms, st.gen_tps, rss, s.zones.empty() ? th.cpu_c : max_temp_c(s.zones));
        if (rc != KILN_OK) fprintf(stderr, "run %d: %s\n", r, kiln_last_error());
        s.phase = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    s.phase = 4;
    std::this_thread::sleep_for(std::chrono::seconds(cooldown_s));
    kiln_context_free(c); kiln_model_free(m);
    s.phase = 0;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    s.finish();
    fprintf(stderr, "peak rss %.0f MB, rss after free %.0f MB\n", peak_rss, rss_mb());
    if (s.out != stdout) fclose(s.out);
    return 0;
}
