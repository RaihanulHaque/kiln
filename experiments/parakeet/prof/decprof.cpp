// Parakeet TDT decoder: where its time goes, on the CPU or the GPU, and the text it writes.
// Built like encprof (prof/build.sh), with whisper.cpp's parakeet.cpp compiled in, so the
// state's timers can be read directly.
//   decprof MODEL -t N [--gpu] [--reps R] [--text OUT.txt] clip1.f32 [clip2.f32 ...]
// A clip is raw 16 kHz mono float32 samples. --text writes one line per clip (the transcript),
// to check that a change leaves the words alone.
#include "parakeet.cpp"

#include <chrono>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: decprof MODEL -t N [--gpu] [--reps R] [--text OUT.txt] clips...\n");
        return 2;
    }
    const char * model = argv[1];
    int n_threads = 4, reps = 1;
    bool gpu = false;
    const char * text_out = nullptr;
    std::vector<std::string> clips;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-t") n_threads = atoi(argv[++i]);
        else if (a == "--gpu") gpu = true;
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--text") text_out = argv[++i];
        else clips.push_back(a);
    }
    parakeet_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    parakeet_context_params cp = parakeet_context_default_params();
    cp.use_gpu = gpu;
    parakeet_context * ctx = parakeet_init_from_file_with_params(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to load %s\n", model);
        return 1;
    }

    // best of R runs, per clip, for each timer
    double best_total = 0, best_enc = 0, best_dec = 0, best_joint = 0, best_pred = 0, best_pbuild = 0, best_palloc = 0, best_pcomp = 0;
    int n_joint = 0, n_pred = 0;
    double audio_s = 0;
    FILE * ft = text_out ? fopen(text_out, "w") : nullptr;

    for (const auto & path : clips) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        std::vector<float> pcm(f.tellg() / sizeof(float));
        f.seekg(0);
        f.read((char *) pcm.data(), pcm.size() * sizeof(float));
        audio_s += pcm.size() / 16000.0;

        double b_total = 1e30, b_enc = 0, b_dec = 0, b_joint = 0, b_pred = 0, b_pb = 0, b_pa = 0, b_pc = 0;
        int nj = 0, np = 0;
        std::string text;
        for (int r = 0; r < reps; ++r) {
            parakeet_reset_timings(ctx);
            parakeet_state * s = ctx->state;
            s->t_predict_us = s->t_predict_build_us = s->t_predict_alloc_us = s->t_predict_compute_us = 0;
            s->n_predict = 0;
            parakeet_full_params p = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
            p.n_threads = n_threads;
            const double t0 = now_ms();
            if (parakeet_full(ctx, p, pcm.data(), (int) pcm.size()) != 0) {
                fprintf(stderr, "parakeet_full failed on %s\n", path.c_str());
                return 1;
            }
            const double t = now_ms() - t0;
            if (t < b_total) {
                b_total = t;
                b_enc   = s->t_encode_us / 1e3;
                b_joint = s->t_decode_us / 1e3;
                b_pred  = s->t_predict_us / 1e3;
                b_pb    = s->t_predict_build_us / 1e3;
                b_pa    = s->t_predict_alloc_us / 1e3;
                b_pc    = s->t_predict_compute_us / 1e3;
                nj      = s->n_decode;
                np      = s->n_predict;
                b_dec   = b_joint + b_pred;
            }
            text.clear();
            for (int i = 0; i < parakeet_full_n_segments(ctx); ++i) text += parakeet_full_get_segment_text(ctx, i);
        }
        best_total += b_total; best_enc += b_enc; best_dec += b_dec; best_joint += b_joint; best_pred += b_pred;
        best_pbuild += b_pb; best_palloc += b_pa; best_pcomp += b_pc; n_joint += nj; n_pred += np;
        if (ft) fprintf(ft, "%s\n", text.c_str());
    }
    if (ft) fclose(ft);

    printf("%s, %s, %d threads, %zu clips, %.1f s audio (best of %d per clip)\n",
           model, gpu ? "GPU" : "CPU", n_threads, clips.size(), audio_s, reps);
    printf("  total   %8.1f ms   (%.0fx real time)\n", best_total, audio_s * 1e3 / best_total);
    printf("  encode  %8.1f ms\n", best_enc);
    printf("  decode  %8.1f ms   = joint %.1f ms / %d steps (%.3f ms each) + predict %.1f ms / %d steps (%.3f ms each)\n",
           best_dec, best_joint, n_joint, best_joint / std::max(1, n_joint), best_pred, n_pred, best_pred / std::max(1, n_pred));
    printf("  predict = build %.1f + alloc %.1f + compute %.1f ms\n", best_pbuild, best_palloc, best_pcomp);
    parakeet_free(ctx);
    return 0;
}
