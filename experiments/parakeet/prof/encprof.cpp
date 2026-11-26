// Parakeet encoder: dump its output, time it, and time every graph node.
// Built against the same ggml as kiln (llama.cpp's), with whisper.cpp's parakeet.cpp compiled in.
//   encprof MODEL -t N [--dump DIR] [--prof] [--reps R] clip1.f32 [clip2.f32 ...]
// A clip is raw 16 kHz mono float32 samples.
#include "parakeet.cpp"

#include <chrono>
#include <cinttypes>
#include <unordered_map>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct node_stat { std::string op, name; double ms = 0; int n = 0; };

struct prof_data {
    double t_prev = 0;
    std::vector<node_stat> nodes;           // in graph order, one run
    std::unordered_map<std::string, size_t> index;
};

static std::string strip_layer(const char * name) {
    // enc_12_conv_bn -> enc_*_conv_bn
    std::string s = name;
    size_t p = s.find("enc_");
    if (p == 0) {
        size_t q = 4;
        while (q < s.size() && isdigit((unsigned char) s[q])) q++;
        if (q > 4) s = "enc_*" + s.substr(q);
    }
    return s;
}

static const char * g_dump_node = getenv("PROF_DUMP_NODE");   // also writes mel.bin
static const char * g_dump_dir  = getenv("PROF_DUMP_DIR") ? getenv("PROF_DUMP_DIR") : ".";

static bool prof_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * pd = (prof_data *) ud;
    if (ask) {
        return true;
    }
    if (g_dump_node && strcmp(t->name, g_dump_node) == 0) {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        std::ofstream f(std::string(g_dump_dir) + "/" + g_dump_node + ".bin", std::ios::binary);
        int32_t hdr[4] = { (int32_t) t->ne[0], (int32_t) t->ne[1], (int32_t) t->ne[2], (int32_t) t->ne[3] };
        f.write((char *) hdr, sizeof(hdr));
        f.write((char *) v.data(), v.size() * sizeof(float));
    }
    const double t1 = now_ms();
    std::string nm = strip_layer(t->name);
    if (nm.rfind("node_", 0) == 0 || nm.empty()) {
        char buf[160];
        const ggml_tensor * a = t->src[0];
        const ggml_tensor * b = t->src[1];
        snprintf(buf, sizeof(buf), "[%s %" PRId64 "x%" PRId64 "]x[%" PRId64 "x%" PRId64 "]",
                 a ? ggml_type_name(a->type) : "-", a ? a->ne[0] : 0, a ? a->ne[1] : 0, b ? b->ne[0] : 0, b ? b->ne[1] : 0);
        nm = buf;
    }
    const std::string key = std::string(ggml_op_desc(t)) + "|" + nm;
    auto it = pd->index.find(key);
    if (it == pd->index.end()) {
        pd->index[key] = pd->nodes.size();
        pd->nodes.push_back({ggml_op_desc(t), nm});
        it = pd->index.find(key);
    }
    pd->nodes[it->second].ms += t1 - pd->t_prev;
    pd->nodes[it->second].n  += 1;
    pd->t_prev = t1;
    return true;
}

static std::vector<float> read_f32(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    size_t n = f.tellg() / sizeof(float);
    std::vector<float> v(n);
    f.seekg(0);
    f.read((char *) v.data(), n * sizeof(float));
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: encprof MODEL -t N [--dump DIR] [--prof] [--reps R] clips...\n"); return 1; }
    const char * model = argv[1];
    int n_threads = 8, reps = 1;
    bool prof = false;
    std::string dump;
    std::vector<const char *> clips;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-t") n_threads = atoi(argv[++i]);
        else if (a == "--dump") dump = argv[++i];
        else if (a == "--prof") prof = true;
        else if (a == "--reps") reps = atoi(argv[++i]);
        else clips.push_back(argv[i]);
    }

    parakeet_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr); }, nullptr);

    auto cparams = parakeet_context_default_params();
    cparams.use_gpu = false;
    parakeet_context * ctx = parakeet_init_from_file_with_params(model, cparams);
    if (!ctx) { fprintf(stderr, "load failed\n"); return 1; }
    parakeet_state * st = ctx->state;
    const int n_state = ctx->model.hparams.n_audio_state;

    prof_data pd;
    double total_sched = 0, total_enc = 0, total_audio = 0;

    for (const char * clip : clips) {
        auto pcm = read_f32(clip);
        if (parakeet_pcm_to_mel_with_state(ctx, st, pcm.data(), (int) pcm.size(), n_threads) != 0) return 1;
        const int n_len = parakeet_n_len_from_state(st);
        if (g_dump_node) {
            std::ofstream f(std::string(g_dump_dir) + "/mel.bin", std::ios::binary);
            int32_t hdr[4] = { st->mel.n_mel, st->mel.n_len, 0, 0 };
            f.write((char *) hdr, sizeof(hdr));
            f.write((char *) st->mel.data.data(), (size_t) st->mel.n_mel * st->mel.n_len * sizeof(float));
        }

        const double ts = now_ms();
        if (!parakeet_ensure_encode_sched(*ctx, *st, n_len)) { fprintf(stderr, "sched failed\n"); return 1; }
        const double t_sched = now_ms() - ts;
        total_sched += t_sched;

        if (prof || g_dump_node) {
            ggml_backend_sched_set_eval_callback(st->sched_encode.sched, prof_cb, &pd);
        }

        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            pd.t_prev = now_ms();
            const double t0 = now_ms();
            if (!parakeet_encode_internal(*ctx, *st, 0, n_threads, nullptr, nullptr)) { fprintf(stderr, "encode failed\n"); return 1; }
            best = std::min(best, now_ms() - t0);
        }
        total_enc += best;
        total_audio += pcm.size() / 16000.0;

        const int n_frames = st->n_frames;
        fprintf(stderr, "%s: %.2f s audio, %d mel, %d frames, sched %.1f ms, encode %.1f ms (best of %d)\n",
                clip, pcm.size() / 16000.0, n_len, n_frames, t_sched, best, reps);

        if (!dump.empty()) {
            std::vector<float> out((size_t) n_state * n_frames);
            ggml_backend_tensor_get(st->enc_out, out.data(), 0, out.size() * sizeof(float));
            std::string base = clip;
            base = base.substr(base.find_last_of('/') + 1);
            std::ofstream f(dump + "/" + base + ".enc", std::ios::binary);
            int32_t hdr[2] = { n_state, n_frames };
            f.write((char *) hdr, sizeof(hdr));
            f.write((char *) out.data(), out.size() * sizeof(float));
        }
    }

    fprintf(stderr, "total: %.1f s audio, sched rebuild %.1f ms, encode %.1f ms\n", total_audio, total_sched, total_enc);

    if (prof) {
        double sum = 0;
        for (auto & n : pd.nodes) sum += n.ms;
        std::vector<node_stat> v = pd.nodes;
        std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.ms > b.ms; });
        printf("%-14s %-34s %9s %6s %6s\n", "op", "node", "ms", "share", "count");
        for (auto & n : v) {
            if (n.ms / sum < 0.002) continue;
            printf("%-14s %-34s %9.2f %5.1f%% %6d\n", n.op.c_str(), n.name.c_str(), n.ms, 100 * n.ms / sum, n.n);
        }
        printf("sum of nodes: %.1f ms\n", sum);
        std::map<std::string, double> by_op;
        for (auto & n : pd.nodes) by_op[n.op] += n.ms;
        std::vector<std::pair<std::string, double>> ops(by_op.begin(), by_op.end());
        std::sort(ops.begin(), ops.end(), [](auto & a, auto & b) { return a.second > b.second; });
        printf("\nby op:\n");
        for (auto & o : ops) if (o.second / sum >= 0.002) printf("  %-14s %8.2f ms %5.1f%%\n", o.first.c_str(), o.second, 100 * o.second / sum);
    }

    parakeet_free(ctx);
    return 0;
}
