// kiln-cli: the truth-telling harness. Loads a GGUF, runs a prompt, prints what actually happened.
#include "kiln/kiln.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage() {
    fprintf(stderr,
        "usage: kiln-cli --probe\n"
        "       kiln-cli -m model.gguf [-p \"prompt\"] [-n max_tokens] [--cpu|--gpu] [-t threads] [-c n_ctx] [--raw] [-v]\n");
}

static bool on_token(const char* piece, int32_t, void*) { fputs(piece, stdout); fflush(stdout); return true; }

static void print_probe(const kiln_device_info& d) {
    printf("soc:        %s\n", d.soc);
    printf("abi:        %s\n", d.abi);
    printf("cores:      %d (fast: %d, mask: 0x%llx)\n", d.cpu_cores, d.cpu_big_cores, (unsigned long long)d.cpu_fast_mask);
    printf("ram:        %.1f GB total, %.1f GB available\n", d.ram_total_bytes / 1e9, d.ram_available_bytes / 1e9);
    printf("cpu feats:  fp16=%d dotprod=%d i8mm=%d sve=%d\n", d.has_fp16, d.has_dotprod, d.has_i8mm, d.has_sve);
    printf("gpu:        %s%s\n", d.has_gpu ? "yes - " : "no", d.has_gpu ? d.gpu_name : "");
    printf("npu:        %s%s\n", d.has_npu ? "yes - " : "no", d.has_npu ? d.npu_name : "");
}

int main(int argc, char** argv) {
    std::string model, prompt = "Explain in one sentence why the sky is blue.";
    int max_tokens = 128, threads = 0, n_ctx = 0;
    bool probe = false, raw = false;
    kiln_backend_kind backend = KILN_BACKEND_AUTO;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if      (a == "--probe") probe = true;
        else if (a == "-m")      model = next();
        else if (a == "-p")      prompt = next();
        else if (a == "-n")      max_tokens = atoi(next());
        else if (a == "-t")      threads = atoi(next());
        else if (a == "-c")      n_ctx = atoi(next());
        else if (a == "--cpu")   backend = KILN_BACKEND_CPU;
        else if (a == "--gpu")   backend = KILN_BACKEND_GPU;
        else if (a == "--raw")   raw = true;
        else if (a == "-v")      kiln_set_log_level(KILN_LOG_INFO);
        else if (a == "-vv")     kiln_set_log_level(KILN_LOG_DEBUG);
        else { usage(); return 2; }
    }

    kiln_device_info dev;
    kiln_probe_device(&dev);
    if (probe) { print_probe(dev); return 0; }
    if (model.empty()) { usage(); return 2; }

    kiln_model_params mp = kiln_model_params_default();
    mp.backend = backend; mp.n_threads = threads;
    kiln_model* m = nullptr;
    if (kiln_model_load(model.c_str(), &mp, &m) != KILN_OK) {
        fprintf(stderr, "load failed: %s\n", kiln_last_error()); return 1;
    }
    kiln_model_info mi; kiln_model_get_info(m, &mi);

    kiln_context_params cp = kiln_context_params_default();
    cp.n_ctx = n_ctx;
    kiln_context* c = nullptr;
    if (kiln_context_create(m, &cp, &c) != KILN_OK) {
        fprintf(stderr, "context failed: %s\n", kiln_last_error()); return 1;
    }

    std::string full = prompt;
    if (!raw) {
        kiln_chat_message msgs[] = { {"user", prompt.c_str()} };
        int need = kiln_apply_chat_template(m, msgs, 1, true, nullptr, 0);
        if (need < 0) need = -need;
        std::vector<char> buf(need + 1);
        int n = kiln_apply_chat_template(m, msgs, 1, true, buf.data(), (int)buf.size());
        if (n > 0) full.assign(buf.data(), n);
    }

    kiln_sampling_params sp = kiln_sampling_params_default();
    sp.max_tokens = max_tokens;
    kiln_gen_stats st;
    kiln_status s = kiln_generate(c, full.c_str(), &sp, on_token, nullptr, &st);
    printf("\n");
    if (s != KILN_OK && s != KILN_ERR_CANCELLED) fprintf(stderr, "generate failed: %s\n", kiln_last_error());

    fprintf(stderr, "\n--- kiln %s ---\n", kiln_version());
    fprintf(stderr, "device:   %s | %s | %d cores | %.1f GB RAM\n", dev.soc, dev.abi, dev.cpu_cores, dev.ram_total_bytes / 1e9);
    fprintf(stderr, "backend:  %s (gpu layers: %d)\n", mi.backend_name, mi.n_gpu_layers_used);
    fprintf(stderr, "model:    %s %s, %.2fB params, %.2f GB resident, ctx_train=%d\n",
            mi.arch, mi.quant, mi.n_params / 1e9, mi.model_bytes / 1e9, mi.n_ctx_train);
    fprintf(stderr, "prompt:   %d tok in %.0f ms (%.1f tok/s)\n", st.prompt_tokens, st.prompt_ms, st.prompt_tps);
    fprintf(stderr, "generate: %d tok in %.0f ms (%.1f tok/s)%s\n", st.generated_tokens, st.gen_ms, st.gen_tps,
            st.stopped_by_eos ? " [eos]" : st.stopped_by_user ? " [stopped]" : "");

    kiln_context_free(c);
    kiln_model_free(m);
    return s == KILN_OK ? 0 : 1;
}
