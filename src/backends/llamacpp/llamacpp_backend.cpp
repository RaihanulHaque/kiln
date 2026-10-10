// llama.cpp backend. This is the guaranteed tier: CPU everywhere, GPU where ggml has a device.
#include "../../core/internal.h"
#include "llama.h"
#include "ggml-backend.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

using clock_t_ = std::chrono::steady_clock;
double ms_since(clock_t_::time_point t) {
    return std::chrono::duration<double, std::milli>(clock_t_::now() - t).count();
}

void ggml_log_bridge(ggml_log_level level, const char* text, void*) {
    kiln_log_level l = KILN_LOG_DEBUG;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: l = KILN_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  l = KILN_LOG_WARN;  break;
        case GGML_LOG_LEVEL_INFO:  l = KILN_LOG_INFO;  break;
        default: break;
    }
    std::string s(text);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (!s.empty()) kiln::log(l, "llama.cpp: %s", s.c_str());
}

void ensure_backend_init() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(ggml_log_bridge, nullptr);
        llama_backend_init();
        ggml_backend_load_all();
    });
}

ggml_backend_dev_t first_gpu_device() {
    ensure_backend_init();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto* dev = ggml_backend_dev_get(i);
        auto t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) return dev;
    }
    return nullptr;
}

int default_threads(const kiln_device_info& d) {
    int n = d.cpu_big_cores > 0 ? d.cpu_big_cores : d.cpu_cores;
    if (n <= 0) n = 4;
    return n > 8 ? 8 : n;
}

// ------------------------------------------------------------------ Context

class LlamaContext final : public kiln::Context {
public:
    LlamaContext(llama_model* model, llama_context* ctx, int n_batch)
        : model_(model), ctx_(ctx), vocab_(llama_model_get_vocab(model)), n_batch_(n_batch) {}
    ~LlamaContext() override { llama_free(ctx_); }

    kiln_status reset() override {
        llama_memory_clear(llama_get_memory(ctx_), true);
        n_past_ = 0;
        return KILN_OK;
    }

    void cancel() override { cancel_.store(true); }

    static bool abort_cb(void* user) { return static_cast<LlamaContext*>(user)->cancel_.load(); }

    kiln_status generate(const char* prompt, const kiln_sampling_params& sp,
                         kiln_token_cb cb, void* user, kiln_gen_stats* stats) override {
        cancel_.store(false);
        kiln_gen_stats st{};

        // tokenize
        int n = -llama_tokenize(vocab_, prompt, (int)strlen(prompt), nullptr, 0, true, true);
        if (n <= 0) return kiln::set_error(KILN_ERR_INVALID_ARG, "empty prompt");
        std::vector<llama_token> toks(n);
        n = llama_tokenize(vocab_, prompt, (int)strlen(prompt), toks.data(), n, true, true);
        if (n < 0) return kiln::set_error(KILN_ERR_INTERNAL, "tokenize failed");
        toks.resize(n);
        st.prompt_tokens = n;

        const int n_ctx = (int)llama_n_ctx(ctx_);
        if (n_past_ + n >= n_ctx)
            return kiln::set_error(KILN_ERR_CONTEXT_FULL, "prompt does not fit in context");

        // sampler chain
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (sp.repeat_last_n > 0 && sp.repeat_penalty != 1.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab_), sp.repeat_last_n, sp.repeat_penalty, 0.0f, 0.0f));
        if (sp.temperature <= 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            if (sp.top_k > 0)      llama_sampler_chain_add(smpl, llama_sampler_init_top_k(sp.top_k));
            if (sp.top_p < 1.0f)   llama_sampler_chain_add(smpl, llama_sampler_init_top_p(sp.top_p, 1));
            if (sp.min_p > 0.0f)   llama_sampler_chain_add(smpl, llama_sampler_init_min_p(sp.min_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(sp.temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(sp.seed ? sp.seed : LLAMA_DEFAULT_SEED));
        }

        // prompt eval
        auto t0 = clock_t_::now();
        for (int i = 0; i < n; i += n_batch_) {
            int cnt = std::min(n_batch_, n - i);
            llama_batch b = llama_batch_get_one(toks.data() + i, cnt);
            if (cancel_.load()) { llama_sampler_free(smpl); st.stopped_by_user = true; if (stats) *stats = st; return KILN_ERR_CANCELLED; }
            int rc = llama_decode(ctx_, b);
            if (rc != 0) {
                llama_sampler_free(smpl);
                if (rc == 2) { st.stopped_by_user = true; if (stats) *stats = st; return KILN_ERR_CANCELLED; }
                return kiln::set_error(rc == 1 ? KILN_ERR_CONTEXT_FULL : KILN_ERR_BACKEND,
                                       "llama_decode failed during prompt eval, rc=" + std::to_string(rc));
            }
            n_past_ += cnt;
        }
        st.prompt_ms = ms_since(t0);

        // generation
        auto t1 = clock_t_::now();
        std::vector<char> piece(256);
        kiln_status result = KILN_OK;
        for (;;) {
            if (sp.max_tokens > 0 && st.generated_tokens >= sp.max_tokens) break;
            if (cancel_.load()) { st.stopped_by_user = true; result = KILN_ERR_CANCELLED; break; }
            if (n_past_ >= n_ctx) { result = KILN_ERR_CONTEXT_FULL; break; }

            llama_token tok = llama_sampler_sample(smpl, ctx_, -1);
            llama_sampler_accept(smpl, tok);
            if (llama_vocab_is_eog(vocab_, tok)) { st.stopped_by_eos = true; break; }

            int len = llama_token_to_piece(vocab_, tok, piece.data(), (int)piece.size(), 0, false);
            if (len < 0) { piece.resize(-len + 1); len = llama_token_to_piece(vocab_, tok, piece.data(), (int)piece.size(), 0, false); }
            piece[len] = 0;
            st.generated_tokens++;
            if (cb && !cb(piece.data(), tok, user)) { st.stopped_by_user = true; break; }

            llama_batch b = llama_batch_get_one(&tok, 1);
            int rc = llama_decode(ctx_, b);
            if (rc != 0) {
                if (rc == 2) { st.stopped_by_user = true; result = KILN_ERR_CANCELLED; }
                else result = kiln::set_error(rc == 1 ? KILN_ERR_CONTEXT_FULL : KILN_ERR_BACKEND, "llama_decode failed, rc=" + std::to_string(rc));
                break;
            }
            n_past_++;
        }
        st.gen_ms = ms_since(t1);
        st.prompt_tps = st.prompt_ms > 0 ? st.prompt_tokens * 1000.0 / st.prompt_ms : 0;
        st.gen_tps    = st.gen_ms > 0 ? st.generated_tokens * 1000.0 / st.gen_ms : 0;
        llama_sampler_free(smpl);
        if (stats) *stats = st;
        return result;
    }

private:
    llama_model*       model_;
    llama_context*     ctx_;
    const llama_vocab* vocab_;
    int                n_batch_;
    int                n_past_ = 0;
    std::atomic<bool>  cancel_{false};
};

// ------------------------------------------------------------------ Model

class LlamaModel final : public kiln::Model {
public:
    LlamaModel(llama_model* m, std::string backend_name, kiln_backend_kind kind, int gpu_layers, int threads)
        : model_(m), vocab_(llama_model_get_vocab(m)), backend_name_(std::move(backend_name)),
          kind_(kind), gpu_layers_(gpu_layers), threads_(threads) {}
    ~LlamaModel() override { llama_model_free(model_); }

    kiln_status info(kiln_model_info& out) const override {
        memset(&out, 0, sizeof out);
        out.backend_used = kind_;
        strncpy(out.backend_name, backend_name_.c_str(), sizeof out.backend_name - 1);
        llama_model_meta_val_str(model_, "general.architecture", out.arch, sizeof out.arch);
        char desc[128] = {0};
        llama_model_desc(model_, desc, sizeof desc);
        // desc looks like "qwen2 1.5B Q4_K - Medium": keep everything after the 2nd space
        const char* p = desc; int spaces = 0;
        while (*p && spaces < 2) { if (*p == ' ') ++spaces; ++p; }
        strncpy(out.quant, p, sizeof out.quant - 1);
        out.n_params         = (int64_t)llama_model_n_params(model_);
        out.n_ctx_train      = llama_model_n_ctx_train(model_);
        out.n_vocab          = llama_vocab_n_tokens(vocab_);
        out.n_gpu_layers_used = gpu_layers_;
        out.model_bytes      = llama_model_size(model_);
        return KILN_OK;
    }

    int32_t tokenize(const char* text, bool add_special, int32_t* out, int32_t cap) const override {
        return llama_tokenize(vocab_, text, (int)strlen(text), out, cap, add_special, true);
    }
    int32_t detokenize(const int32_t* toks, int32_t n, char* out, int32_t cap) const override {
        return llama_detokenize(vocab_, toks, n, out, cap, false, false);
    }
    int32_t apply_chat_template(const kiln_chat_message* msgs, int32_t n, bool add_ass,
                                char* out, int32_t cap) const override {
        const char* tmpl = llama_model_chat_template(model_, nullptr);
        if (!tmpl) tmpl = "chatml";
        std::vector<llama_chat_message> cm(n);
        for (int i = 0; i < n; ++i) cm[i] = { msgs[i].role, msgs[i].content };
        return llama_chat_apply_template(tmpl, cm.data(), (size_t)n, add_ass, out, cap);
    }

    kiln_status create_context(const kiln_context_params& p, std::unique_ptr<kiln::Context>& out) override {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = p.n_ctx > 0 ? (uint32_t)p.n_ctx : 4096;
        cp.n_batch         = p.n_batch > 0 ? (uint32_t)p.n_batch : 512;
        cp.n_threads       = threads_;
        cp.n_threads_batch = threads_;
        cp.flash_attn_type = p.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        auto* holder = new LlamaContext*(nullptr);
        cp.abort_callback      = [](void* u) { auto* h = *static_cast<LlamaContext**>(u); return h && LlamaContext::abort_cb(h); };
        cp.abort_callback_data = holder;
        llama_context* ctx = llama_init_from_model(model_, cp);
        if (!ctx) { delete holder; return kiln::set_error(KILN_ERR_OUT_OF_MEMORY, "llama_init_from_model failed (n_ctx=" + std::to_string(cp.n_ctx) + ")"); }
        auto* c = new LlamaContext(model_, ctx, (int)cp.n_batch);
        *holder = c;   // small leak of one pointer per context; acceptable for now, tracked in TODO
        out.reset(c);
        return KILN_OK;
    }

private:
    llama_model*       model_;
    const llama_vocab* vocab_;
    std::string        backend_name_;
    kiln_backend_kind  kind_;
    int                gpu_layers_;
    int                threads_;
};

// ------------------------------------------------------------------ Backend

class LlamaBackend final : public kiln::Backend {
public:
    const char* name() const override { return "llama.cpp"; }

    void annotate_device(kiln_device_info& d) const override {
        if (auto* dev = first_gpu_device()) {
            d.has_gpu = true;
            strncpy(d.gpu_name, ggml_backend_dev_description(dev), sizeof d.gpu_name - 1);
        }
    }

    bool supports(kiln_backend_kind kind, const kiln_device_info& d) const override {
        switch (kind) {
            case KILN_BACKEND_AUTO:
            case KILN_BACKEND_CPU: return true;
            case KILN_BACKEND_GPU: return d.has_gpu;
            default: return false;
        }
    }

    kiln_status load_model(const char* path, const kiln_model_params& p,
                           const kiln_device_info& d, std::unique_ptr<kiln::Model>& out) override {
        ensure_backend_init();

        bool want_gpu = false;
        if (p.backend == KILN_BACKEND_GPU) want_gpu = true;
        else if (p.backend == KILN_BACKEND_AUTO) {
#if defined(__APPLE__)
            want_gpu = d.has_gpu;              // Metal is reliably faster than CPU on Apple silicon
#else
            want_gpu = false;                  // mobile Vulkan/OpenCL is a bench-and-decide path, not a default
#endif
        }
        if (want_gpu && !d.has_gpu) return kiln::set_error(KILN_ERR_NO_BACKEND, "GPU requested but no ggml GPU device present");

        llama_model_params mp = llama_model_default_params();
        if (p.use_mmap && p.use_mlock) mp.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
        else if (p.use_mmap)           mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        else if (p.use_mlock)          mp.load_mode = LLAMA_LOAD_MODE_MLOCK;
        else                           mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.n_gpu_layers = want_gpu ? (p.n_gpu_layers < 0 ? 999 : p.n_gpu_layers) : 0;

        llama_model* m = llama_model_load_from_file(path, mp);
        if (!m) return kiln::set_error(KILN_ERR_MODEL_LOAD, std::string("failed to load ") + path);

        std::string bname = "llama.cpp/cpu";
        kiln_backend_kind kind = KILN_BACKEND_CPU;
        if (mp.n_gpu_layers > 0) {
            if (auto* dev = first_gpu_device()) {
                bname = std::string("llama.cpp/") + ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
                for (auto& ch : bname) ch = (char)tolower(ch);
                kind = KILN_BACKEND_GPU;
            }
        }
        int threads = p.n_threads > 0 ? p.n_threads : default_threads(d);
        kiln::log(KILN_LOG_INFO, "loaded %s via %s, threads=%d, gpu_layers=%d", path, bname.c_str(), threads, mp.n_gpu_layers);
        out.reset(new LlamaModel(m, bname, kind, mp.n_gpu_layers, threads));
        return KILN_OK;
    }
};

} // namespace

namespace kiln {
Backend* make_llamacpp_backend() { static LlamaBackend b; return &b; }
}
