// kiln-bench: resource usage over time while kiln loads a model and works with it.
// Samples RSS, process CPU%, CPU frequency, and thermal zones on a background thread.
// Writes CSV; scripts/bench_plot.py turns it into charts.
//
// What "work" is depends on the kind of model, which the tool reads from the model itself:
//   chat model       each run generates -n tokens                      speed = tokens per second
//   decision model   each run asks --calls questions (kiln_decide)     speed = ms per question
//   embedding model  each run embeds --calls texts (kiln_embed), and with --mmproj and --media
//                    each media file --media-calls times               speed = ms per input
// Runs come back to back, so the later ones show what heat does to the speed.
#include "kiln/kiln.h"

#include <algorithm>
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
    std::vector<const char*> phase_names{"idle", "load", "prompt", "generate", "cooldown", "decide", "embed", "encode"};
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
                // Android's thermal service is asked about once a second on every device: it is the
                // only source of skin temperature and throttle status, and its CPU reading is the
                // one the OS acts on. Raw sysfs zones are the fallback (some report a constant:
                // a Galaxy S26 Ultra has one stuck at 105 C).
                if (t - last_therm > 1000) { th = read_dumpsys_thermal(); last_therm = t; }
                double temp = th.cpu_c > 0 ? th.cpu_c : max_temp_c(zones);
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

// CPU temperature right now, from the thermal service where there is one, else from sysfs.
static double current_temp_c(const std::vector<ThermalZone>& zones) {
    const Thermal th = read_dumpsys_thermal();
    return th.cpu_c > 0 ? th.cpu_c : max_temp_c(zones);
}

// ---------------------------------------------------------------- per-run results
// One line per run, on stderr and (with -o x.csv) in x.runs.csv:
//   series  chat | decide | embed-text | embed-<media file>
//   unit    tok (chat: count = generated tokens) or call (count = calls in the run)
//   per_s   tokens per second, or calls per second

static FILE* g_runs = nullptr;
static const char* kRunsHeader = "series,run,unit,count,total_ms,mean_ms,min_ms,max_ms,per_s,rss_mb,temp_c,tokens_per_call,prompt_tok,prompt_ms,sample_run";

//   run         index inside the series; sample_run is the `run` value the same run has in the samples
//   min_ms / max_ms are empty for chat, where single tokens are not timed
static void report(const std::string& series_in, int run, int sample_run, const char* unit, int count, double total_ms, double mean_ms,
                   double min_ms, double max_ms, double per_s, double temp_c, int tokens_per_call, int prompt_tok, double prompt_ms) {
    std::string series = series_in;                   // comes from a file name: keep the line a valid CSV row
    for (char& ch : series) if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') ch = '_';
    char lo[32] = "", hi[32] = "";
    if (min_ms >= 0) snprintf(lo, sizeof lo, "%.2f", min_ms);
    if (max_ms >= 0) snprintf(hi, sizeof hi, "%.2f", max_ms);
    char line[640];
    snprintf(line, sizeof line, "%s,%d,%s,%d,%.0f,%.2f,%s,%s,%.1f,%.0f,%.1f,%d,%d,%.0f,%d", series.c_str(), run, unit, count, total_ms,
             mean_ms, lo, hi, per_s, rss_mb(), temp_c, tokens_per_call, prompt_tok, prompt_ms, sample_run);
    fprintf(stderr, "%s\n", line);
    if (g_runs) { fprintf(g_runs, "%s\n", line); fflush(g_runs); }
}

static std::string base_name(const std::string& path) {
    const size_t p = path.find_last_of('/');
    return p == std::string::npos ? path : path.substr(p + 1);
}

// Runs `calls` calls of fn back to back, `runs` times, and reports each run. fn returns the
// tokens the model read for that call, or a negative number on failure.
template <class Fn>
static bool burst(Sampler& s, int phase, const std::string& series, int runs, int calls, int& run_no, double& peak_rss, Fn fn) {
    for (int r = 1; r <= runs; ++r) {
        s.run = ++run_no; s.tokens = 0; s.phase = phase;
        std::vector<double> ms; ms.reserve(calls);
        long tokens = 0;
        const double t0 = now_ms();
        for (int i = 0; i < calls; ++i) {
            const double a = now_ms();
            const int tok = fn(i);
            if (tok < 0) { fprintf(stderr, "%s run %d call %d: %s\n", series.c_str(), r, i, kiln_last_error()); s.phase = 0; return false; }
            ms.push_back(now_ms() - a);
            tokens += tok;
            s.tokens++;                                   // the samples' "tokens" column counts calls here
        }
        const double total = now_ms() - t0;
        peak_rss = std::max(peak_rss, rss_mb());
        report(series, r, run_no, "call", calls, total, total / calls, *std::min_element(ms.begin(), ms.end()), *std::max_element(ms.begin(), ms.end()),
               calls * 1000.0 / total, current_temp_c(s.zones), (int)(tokens / calls), 0, 0);
        s.phase = 0; s.run = 0;                           // the gap between runs belongs to no run
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return true;
}

// A support message and three questions about it, one of each kind: what an app would ask to
// route a ticket. The same inputs every call, so run-to-run changes are the device, not the work.
static const char* kState =
    "Hi, I ordered a blender two weeks ago and it arrived broken. I have emailed twice and nobody "
    "answered. I want my money back today or I will dispute the charge with my bank.";

// Notes of different lengths, as an app would index them for search.
static const char* kTexts[] = {
    "Knead the dough and let it rise for an hour before baking.",
    "The train to Leeds leaves from platform 4 at 9:15.",
    "To get back into your account, open Settings, choose Security, then tap Forgot credentials and follow the link we send you.",
    "Water the tomatoes every second day and keep them in full sun.",
    "The dentist moved my appointment to Thursday afternoon, so the review meeting has to start an hour earlier than planned.",
    "Ahmed owes me 40 for the concert tickets.",
    "The wifi code for the flat is written under the router, and the guest network is switched off on weekdays.",
    "Replace the bike chain when it starts skipping gears; a worn chain also wears out the cassette, which costs three times as much.",
};

int main(int argc, char** argv) {
    std::string model, out_path;
    int runs = 5, n_gen = 256, cooldown_s = 5, interval_ms = 250, threads = 0, calls = 0, media_calls = 0;
    std::string mmproj;
    std::vector<std::string> media;
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
        else if (a == "--calls") calls = atoi(next());
        else if (a == "--media-calls") media_calls = atoi(next());
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--media") media.push_back(next());
        else {
            fprintf(stderr, "usage: kiln-bench -m model.gguf [-o out.csv] [-r runs] [-t threads] [--cooldown s]\n"
                            "  chat models:      [-n tokens per run]                      (default 256)\n"
                            "  decision models:  [--calls questions per run]              (default 40)\n"
                            "  embedding models: [--calls texts per run]                  (default 150)\n"
                            "                    [--mmproj projector.gguf --media file]... [--media-calls per run] (default 5)\n"
                            "  -o x.csv also writes x.runs.csv with one line per run\n");
            return 2;
        }
    }
    if (model.empty()) return 2;
    kiln_set_log_level(KILN_LOG_ERROR);

    kiln_device_info dev; kiln_probe_device(&dev);
    Sampler s; s.out = out_path.empty() ? stdout : fopen(out_path.c_str(), "w");
    if (!s.out) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    if (!out_path.empty()) {
        std::string rp = out_path;
        if (rp.size() > 4 && rp.compare(rp.size() - 4, 4, ".csv") == 0) rp.erase(rp.size() - 4);
        g_runs = fopen((rp + ".runs.csv").c_str(), "w");
        if (!g_runs) { fprintf(stderr, "cannot write %s.runs.csv\n", rp.c_str()); return 1; }
        fprintf(g_runs, "%s\n", kRunsHeader);
    }
    s.fast_mask = dev.cpu_fast_mask; s.zones = find_thermal_zones(); s.interval_ms = interval_ms;
    g_sampler = &s;
    fprintf(stderr, "device: %s, %d cores (%d fast), thermal: %s\n", dev.soc, dev.cpu_cores, dev.cpu_big_cores, read_dumpsys_thermal().ok ? "dumpsys" : !s.zones.empty() ? "sysfs" : "none");
    now_ms();
    s.start();
    std::this_thread::sleep_for(std::chrono::seconds(2));          // baseline

    s.phase = 1;
    double t_load = now_ms();
    kiln_model_params mp = kiln_model_params_default(); mp.n_threads = threads;
    kiln_model* m = nullptr;
    // from here on the sampler thread is running: every exit has to stop it first
    if (kiln_model_load(model.c_str(), &mp, &m) != KILN_OK) { fprintf(stderr, "load: %s\n", kiln_last_error()); s.finish(); return 1; }
    kiln_context_params cp = kiln_context_params_default(); cp.n_ctx = 2048;
    kiln_context* c = nullptr;
    if (kiln_context_create(m, &cp, &c) != KILN_OK) { fprintf(stderr, "ctx: %s\n", kiln_last_error()); kiln_model_free(m); s.finish(); return 1; }
    double load_ms = now_ms() - t_load;
    kiln_model_info mi; kiln_model_get_info(m, &mi);
    fprintf(stderr, "model: %s %s via %s, load %.0f ms, rss after load %.0f MB\n", mi.arch, mi.quant, mi.backend_name, load_ms, rss_mb());
    s.phase = 0;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    double peak_rss = 0;
    int run_no = 0;
    bool failed = false;                                  // a call failed: finish cleanly, then exit non-zero
    fprintf(stderr, "%s\n", kRunsHeader);
    if (mi.is_decision_model) {
        // one run = `calls` questions, cycling through the three kinds of question
        if (calls <= 0) calls = 40;
        const kiln_decision_option teams[] = {{"billing", "Payments, refunds and invoices"}, {"shipping", "Delivery status and lost parcels"},
                                              {"technical", "The product does not work as expected"}};
        const kiln_decision_option levels[] = {{nullptr, "Can wait"}, {nullptr, "Today"}, {nullptr, "Blocking the customer now"}};
        const kiln_decision_question qs[] = {
            {"wants_refund", KILN_QUESTION_NOUL, "Is the customer asking for a refund?", nullptr, 0},
            {"team", KILN_QUESTION_CHOICE, "Which team should handle this?", teams, 3},
            {"urgency", KILN_QUESTION_SCORE, "How urgent is this?", levels, 3},
        };
        failed = !burst(s, 5, "decide", runs, calls, run_no, peak_rss, [&](int i) {
            float probs[8]; kiln_decision_answer a;
            return kiln_decide(c, kState, &qs[i % 3], probs, 8, &a) == KILN_OK ? a.prompt_tokens : -1;
        });
    } else if (mi.is_embedding_model) {
        // one run = `calls` notes embedded as documents; then, with a projector, each media file
        if (calls <= 0) calls = 150;
        if (media_calls <= 0) media_calls = 5;
        std::vector<float> v(mi.n_embd > 0 ? mi.n_embd : 1);
        kiln_embed_options eo = kiln_embed_options_default(); eo.kind = KILN_EMBED_DOCUMENT;
        const int n_texts = (int)(sizeof kTexts / sizeof kTexts[0]);
        bool ok = burst(s, 6, "embed-text", runs, calls, run_no, peak_rss, [&](int i) {
            kiln_embed_stats es;
            return kiln_embed(c, kTexts[i % n_texts], &eo, v.data(), (int32_t)v.size(), &es) == KILN_OK ? es.tokens : -1;
        });
        if (ok && !mmproj.empty()) {
            s.phase = 1;
            const double tp = now_ms(), rss_before = rss_mb();
            kiln_projector_info pi;
            if (kiln_model_load_projector(m, mmproj.c_str(), nullptr, &pi) != KILN_OK) {
                fprintf(stderr, "projector: %s\n", kiln_last_error());
                ok = false;
            } else {
                fprintf(stderr, "projector: load %.0f ms, +%.0f MB rss, images=%d audio=%d\n", now_ms() - tp, rss_mb() - rss_before, pi.supports_images, pi.supports_audio);
            }
            s.phase = 0;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            for (const std::string& f : media) {
                if (!ok) break;
                ok = burst(s, 7, "embed-" + base_name(f), runs, media_calls, run_no, peak_rss, [&](int) {
                    kiln_media item = {f.c_str(), nullptr, 0};
                    kiln_embed_stats es;
                    return kiln_embed_media(c, nullptr, &item, 1, nullptr, v.data(), (int32_t)v.size(), &es) == KILN_OK ? es.tokens : -1;
                });
            }
        }
        failed = !ok;
    } else {
        kiln_chat_message msgs[] = {{"user", prompt.c_str()}};
        kiln_sampling_params sp = kiln_sampling_params_default(); sp.max_tokens = n_gen; sp.seed = 1;
        for (int r = 1; r <= runs; ++r) {
            kiln_context_reset(c);
            s.run = ++run_no; s.tokens = 0;
            // phase flips to "generate" on first piece; prompt eval is everything before that
            s.phase = 2;
            struct Hook { static bool cb(kiln_channel ch, const char* p, int32_t t, void* u) { if (g_sampler->phase.load() == 2) g_sampler->phase = 3; return on_piece(ch, p, t, u); } };
            kiln_gen_stats st;
            kiln_status rc = kiln_chat(c, msgs, 1, nullptr, &sp, Hook::cb, nullptr, &st);
            double rss = rss_mb(); if (rss > peak_rss) peak_rss = rss;
            report("chat", r, run_no, "tok", st.generated_tokens, st.gen_ms, st.generated_tokens > 0 ? st.gen_ms / st.generated_tokens : 0, -1, -1, st.gen_tps,
                   current_temp_c(s.zones), 0, st.prompt_tokens, st.prompt_ms);
            if (rc != KILN_OK) { fprintf(stderr, "run %d: %s\n", r, kiln_last_error()); failed = true; }
            s.phase = 0; s.run = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    s.phase = 4;
    std::this_thread::sleep_for(std::chrono::seconds(cooldown_s));
    kiln_context_free(c); kiln_model_free(m);
    s.phase = 0;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    s.finish();
    fprintf(stderr, "peak rss %.0f MB, rss after free %.0f MB\n", peak_rss, rss_mb());
    if (s.out != stdout) fclose(s.out);
    if (g_runs) fclose(g_runs);
    return failed ? 1 : 0;
}
