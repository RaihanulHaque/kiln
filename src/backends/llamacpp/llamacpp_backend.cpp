// llama.cpp backend. This is the guaranteed tier: CPU everywhere, GPU where ggml has a device.
#include "../../core/internal.h"
#include "decision.h"
#include "embedding.h"
#include "../../probe/fast_cores.h"
#include "reasoning_splitter.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"          // llama.cpp/common: jinja templates via minja

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
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
        mtmd_helper_log_set(ggml_log_bridge, nullptr);
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
    return kiln::default_thread_count(d.cpu_cores, d.cpu_big_cores, d.cpu_fast_mask != 0);   // rules in fast_cores.h
}

std::string rtrim(std::string s) {
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    return s;
}

struct ThinkTags {
    bool supported = false;
    std::string start;
    std::vector<std::string> ends;
};

// ------------------------------------------------------------------ Context

class LlamaContext final : public kiln::Context {
public:
    LlamaContext(llama_model* model, const ThinkTags& tags)
        : model_(model), vocab_(llama_model_get_vocab(model)), tags_(tags),
          partial_state_(llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) {}
    ~LlamaContext() override {
        if (ctx_) { if (tp_) llama_detach_threadpool(ctx_); llama_free(ctx_); }
        if (tp_) ggml_threadpool_free(tp_);
    }

    // set by LlamaModel::create_context after llama_init_from_model
    llama_context*    ctx_ = nullptr;
    ggml_threadpool_t tp_  = nullptr;
    int               n_batch_ = 512;
    std::atomic<bool> cancel_{false};
    const kiln::DecisionHead* decision_ = nullptr;   // owned by the model; set for decision models
    const kiln::EmbeddingHead* embedding_ = nullptr; // owned by the model; set for embedding models

    static bool abort_cb(void* user) { return static_cast<LlamaContext*>(user)->cancel_.load(); }

    kiln_status reset() override {
        llama_memory_clear(llama_get_memory(ctx_), true);
        cache_.clear();
        ckpts_.clear();
        return KILN_OK;
    }

    void cancel() override { cancel_.store(true); }

    kiln_status decide(const char* state, const kiln_decision_question& q,
                       float* probs, int32_t probs_cap, kiln_decision_answer& out) override {
        if (!decision_ || !decision_->is_decision_model()) return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not a decision model");
        cancel_.store(false);
        return decision_->decide(ctx_, state, q, probs, probs_cap, out);
    }

    kiln_status embed(const char* text, const kiln_media* media, int32_t n_media, const kiln_embed_options& o,
                      float* out, int32_t out_cap, kiln_embed_stats& stats) override {
        if (!embedding_ || !embedding_->is_embedding_model()) return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not an embedding model");
        cancel_.store(false);
        return embedding_->embed(ctx_, text, media, n_media, o, out, out_cap, stats);
    }

    kiln_status generate(const kiln::RenderedPrompt& in, const kiln_sampling_params& sp,
                         kiln_token_cb cb, void* user, kiln_gen_stats* stats) override {
        cancel_.store(false);
        kiln_gen_stats st{};
        auto done = [&](kiln_status s) { if (stats) *stats = st; return s; };
        if (decision_ && decision_->is_decision_model())
            return done(kiln::set_error(KILN_ERR_NOT_SUPPORTED, "decision models do not generate text; use kiln_decide"));
        if (embedding_ && embedding_->is_embedding_model())
            return done(kiln::set_error(KILN_ERR_NOT_SUPPORTED, "embedding models do not generate text; use kiln_embed"));

        // ---- tokenize
        const char* text = in.prompt.c_str();
        int n = -llama_tokenize(vocab_, text, (int)in.prompt.size(), nullptr, 0, true, true);
        if (n <= 0) return done(kiln::set_error(KILN_ERR_INVALID_ARG, "empty prompt"));
        std::vector<llama_token> toks(n);
        n = llama_tokenize(vocab_, text, (int)in.prompt.size(), toks.data(), n, true, true);
        if (n < 0) return done(kiln::set_error(KILN_ERR_INTERNAL, "tokenize failed"));
        toks.resize(n);
        st.prompt_tokens = n;

        const int n_ctx = (int)llama_n_ctx(ctx_);
        if (n >= n_ctx)
            return done(kiln::set_error(KILN_ERR_CONTEXT_FULL, "prompt (" + std::to_string(n) + " tok) does not fit in context (" + std::to_string(n_ctx) + ")"));

        // ---- where to checkpoint (hybrid / recurrent models only): the end of the stable prefix
        size_t ckpt_at = 0;
        if (partial_state_ && !in.stable_prefix.empty()) {
            std::vector<llama_token> pre(in.stable_prefix.size() + 8);
            int np = llama_tokenize(vocab_, in.stable_prefix.c_str(), (int)in.stable_prefix.size(), pre.data(), (int)pre.size(), true, true);
            while (np > 0 && ckpt_at < (size_t)np && ckpt_at + 1 < toks.size() && pre[ckpt_at] == toks[ckpt_at]) ++ckpt_at;
        }

        // ---- prefix reuse: keep what the KV cache already has, drop the rest
        size_t common = 0;
        while (common < cache_.size() && common < toks.size() && cache_[common] == toks[common]) ++common;
        if (common == toks.size()) --common;            // identical prompt: re-decode last token to get logits
        if (common < cache_.size()) {
            // Recurrent / hybrid models (LFM2, Qwen3.5, Mamba) cannot drop a partial tail of their
            // state; seq_rm tells us. Then go back to the newest checkpoint inside the shared prefix,
            // or re-evaluate everything rather than generate from stale state.
            if (!llama_memory_seq_rm(llama_get_memory(ctx_), 0, (llama_pos)common, -1))
                common = restore_checkpoint(common);
        }
        while (!ckpts_.empty() && ckpts_.back().n_tokens > common) ckpts_.pop_back();
        cache_.resize(common);
        st.cached_tokens = (int32_t)common;
        int n_past = (int)common;

        // ---- sampler chain
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (sp.repeat_last_n > 0 && sp.repeat_penalty != 1.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab_), sp.repeat_last_n, sp.repeat_penalty, 0.0f, 0.0f));
        if (sp.temperature <= 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            if (sp.top_k > 0)    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(sp.top_k));
            if (sp.top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(sp.top_p, 1));
            if (sp.min_p > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(sp.min_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(sp.temperature));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(sp.seed ? sp.seed : LLAMA_DEFAULT_SEED));
        }
        for (auto t : toks) llama_sampler_accept(smpl, t);   // penalties see the prompt
        struct SmplGuard { llama_sampler* s; ~SmplGuard() { llama_sampler_free(s); } } guard{smpl};

        // ---- prompt eval
        auto t0 = clock_t_::now();
        for (int i = (int)common; i < n; ) {
            int cnt = std::min(n_batch_, n - i);
            if (i < (int)ckpt_at) cnt = std::min(cnt, (int)ckpt_at - i);   // end a batch exactly at the checkpoint
            if (cancel_.load()) { st.stopped_by_user = true; return done(KILN_ERR_CANCELLED); }
            int rc = llama_decode(ctx_, llama_batch_get_one(toks.data() + i, cnt));
            if (rc == 2) { st.stopped_by_user = true; return done(KILN_ERR_CANCELLED); }
            if (rc != 0) return done(kiln::set_error(rc == 1 ? KILN_ERR_CONTEXT_FULL : KILN_ERR_BACKEND, "llama_decode failed in prompt eval, rc=" + std::to_string(rc)));
            i += cnt;
            n_past += cnt;
            cache_.insert(cache_.end(), toks.begin() + i - cnt, toks.begin() + i);
            if (i == (int)ckpt_at) save_checkpoint(ckpt_at);
        }
        st.prompt_ms = ms_since(t0);

        // ---- generation
        kiln::ReasoningSplitter split(tags_.start, tags_.ends, in.starts_in_reasoning);
        bool user_stopped = false;
        auto emit = [&](kiln_channel ch, const std::string& s) -> bool {
            if (!cb) return true;
            if (!cb(ch, s.c_str(), -1, user)) { user_stopped = true; return false; }
            return true;
        };

        auto t1 = clock_t_::now();
        std::vector<char> piece(256);
        kiln_status result = KILN_OK;
        for (;;) {
            if (sp.max_tokens > 0 && st.generated_tokens >= sp.max_tokens) break;
            if (cancel_.load()) { st.stopped_by_user = true; result = KILN_ERR_CANCELLED; break; }
            if (n_past >= n_ctx) { result = kiln::set_error(KILN_ERR_CONTEXT_FULL, "context full after " + std::to_string(st.generated_tokens) + " tokens"); break; }

            llama_token tok = llama_sampler_sample(smpl, ctx_, -1);
            llama_sampler_accept(smpl, tok);
            if (llama_vocab_is_eog(vocab_, tok)) {
                // Feed the EOG token too: the next turn's template re-emits it, so the whole
                // conversation so far stays a pure prefix and even recurrent models reuse the cache.
                if (n_past < n_ctx && llama_decode(ctx_, llama_batch_get_one(&tok, 1)) == 0) { n_past++; cache_.push_back(tok); }
                st.stopped_by_eos = true; break;
            }

            // special=true so think tags surface as text for the splitter to strip
            int len = llama_token_to_piece(vocab_, tok, piece.data(), (int)piece.size(), 0, true);
            if (len < 0) { piece.resize(-len + 1); len = llama_token_to_piece(vocab_, tok, piece.data(), (int)piece.size(), 0, true); }
            st.generated_tokens++;
            if (split.in_reasoning()) st.reasoning_tokens++;
            if (!split.feed(std::string(piece.data(), (size_t)len), emit)) { st.stopped_by_user = true; break; }

            int rc = llama_decode(ctx_, llama_batch_get_one(&tok, 1));
            if (rc == 2) { st.stopped_by_user = true; result = KILN_ERR_CANCELLED; break; }
            if (rc != 0) { result = kiln::set_error(rc == 1 ? KILN_ERR_CONTEXT_FULL : KILN_ERR_BACKEND, "llama_decode failed, rc=" + std::to_string(rc)); break; }
            n_past++;
            cache_.push_back(tok);

            // Thinking budget: small models often never close the think block themselves. Close it
            // for them by feeding the end tag as if the model had written it, then keep generating.
            if (in.max_reasoning_tokens > 0 && split.in_reasoning() && st.reasoning_tokens >= in.max_reasoning_tokens && !tags_.ends.empty()) {
                const std::string close = "\n" + tags_.ends[0] + "\n\n";
                std::vector<llama_token> ct(close.size() + 8);
                int nc = llama_tokenize(vocab_, close.c_str(), (int)close.size(), ct.data(), (int)ct.size(), false, true);
                if (nc <= 0) { result = kiln::set_error(KILN_ERR_INTERNAL, "could not tokenize the reasoning end tag"); break; }
                if (n_past + nc >= n_ctx) { result = kiln::set_error(KILN_ERR_CONTEXT_FULL, "context full while closing the reasoning block"); break; }
                int rc2 = llama_decode(ctx_, llama_batch_get_one(ct.data(), nc));
                if (rc2 == 2) { st.stopped_by_user = true; result = KILN_ERR_CANCELLED; break; }
                if (rc2 != 0) { result = kiln::set_error(KILN_ERR_BACKEND, "llama_decode failed closing the reasoning block, rc=" + std::to_string(rc2)); break; }
                n_past += nc;
                for (int i = 0; i < nc; ++i) { cache_.push_back(ct[i]); llama_sampler_accept(smpl, ct[i]); }
                st.reasoning_truncated = true;
                if (!split.feed(tags_.ends[0], emit)) { st.stopped_by_user = true; break; }
            }
        }
        if (!user_stopped && result != KILN_ERR_CANCELLED) split.finish(emit);
        st.gen_ms = ms_since(t1);
        st.prompt_tps = st.prompt_ms > 0 ? (st.prompt_tokens - st.cached_tokens) * 1000.0 / st.prompt_ms : 0;
        st.gen_tps    = st.gen_ms > 0 ? st.generated_tokens * 1000.0 / st.gen_ms : 0;
        return done(result);
    }

private:
    // Saved recurrent state of a hybrid / recurrent model after its first n_tokens cache tokens.
    // Only the part seq_rm cannot rewind is stored (PARTIAL_ONLY); attention KV is truncated instead.
    struct Checkpoint { size_t n_tokens; std::vector<uint8_t> data; };
    static constexpr size_t kMaxCheckpoints = 2;   // latest turn, plus one back for an edited last message

    void save_checkpoint(size_t n_tokens) {
        if (!ckpts_.empty() && ckpts_.back().n_tokens == n_tokens) return;
        const size_t size = llama_state_seq_get_size_ext(ctx_, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (size == 0) return;
        Checkpoint ck{n_tokens, std::vector<uint8_t>(size)};
        if (llama_state_seq_get_data_ext(ctx_, ck.data.data(), size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size) return;
        if (ckpts_.size() >= kMaxCheckpoints) ckpts_.erase(ckpts_.begin());
        ckpts_.push_back(std::move(ck));
        kiln::log(KILN_LOG_DEBUG, "checkpoint at %zu tokens (%.1f MB)", n_tokens, size / 1048576.0);
    }

    // Rewind to the newest checkpoint at or before `common` tokens. Returns the tokens kept;
    // 0 (memory cleared) when no checkpoint fits.
    size_t restore_checkpoint(size_t common) {
        llama_memory_t mem = llama_get_memory(ctx_);
        for (auto it = ckpts_.rbegin(); it != ckpts_.rend(); ++it) {
            if (it->n_tokens > common) continue;
            if (llama_state_seq_set_data_ext(ctx_, it->data.data(), it->data.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == it->data.size()
                && llama_memory_seq_rm(mem, 0, (llama_pos)it->n_tokens, -1))
                return it->n_tokens;
            break;
        }
        llama_memory_clear(mem, true);
        ckpts_.clear();
        return 0;
    }

    llama_model*             model_;
    const llama_vocab*       vocab_;
    ThinkTags                tags_;
    bool                     partial_state_;   // recurrent or hybrid: needs checkpoints to rewind
    std::vector<llama_token> cache_;   // tokens currently in the KV cache, in order
    std::vector<Checkpoint>  ckpts_;   // ascending n_tokens
};

// ------------------------------------------------------------------ Model

class LlamaModel final : public kiln::Model {
public:
    LlamaModel(llama_model* m, std::string backend_name, kiln_backend_kind kind, int gpu_layers, int threads, uint64_t fast_mask)
        : model_(m), vocab_(llama_model_get_vocab(m)), backend_name_(std::move(backend_name)),
          kind_(kind), gpu_layers_(gpu_layers), threads_(threads), fast_mask_(fast_mask) {
        tmpls_ = common_chat_templates_init(model_, "");
        // Probe the template once: learn its reasoning tags, and whether enable_thinking actually
        // changes the prompt (common's supports_thinking is a heuristic that also fires on templates
        // that merely strip old </think> blocks, e.g. LFM2.5).
        common_chat_templates_inputs in;
        { common_chat_msg m; m.role = "user"; m.content = "hi"; in.messages.push_back(std::move(m)); }
        try {
            in.enable_thinking = true;
            auto on = common_chat_templates_apply(tmpls_.get(), in);
            in.enable_thinking = false;
            auto off = common_chat_templates_apply(tmpls_.get(), in);
            tags_.supported = on.supports_thinking && on.prompt != off.prompt;
            tags_.start = on.thinking_start_tag;
            tags_.ends  = on.thinking_end_tags;
        } catch (const std::exception& e) {
            kiln::log(KILN_LOG_WARN, "chat template probe failed: %s", e.what());
        }
        if (tags_.start.empty() && tags_.ends.empty()) { tags_.start = "<think>"; tags_.ends = {"</think>"}; }
        decision_.init(model_);
        if (!decision_.is_decision_model()) embedding_.init(model_);
    }
    ~LlamaModel() override { embedding_.free_projector(); llama_model_free(model_); }

    kiln_status info(kiln_model_info& out) const override {
        memset(&out, 0, sizeof out);
        out.backend_used = kind_;
        strncpy(out.backend_name, backend_name_.c_str(), sizeof out.backend_name - 1);
        llama_model_meta_val_str(model_, "general.architecture", out.arch, sizeof out.arch);
        char desc[128] = {0};
        llama_model_desc(model_, desc, sizeof desc);
        const char* p = desc; int spaces = 0;
        while (*p && spaces < 2) { if (*p == ' ') ++spaces; ++p; }
        strncpy(out.quant, p, sizeof out.quant - 1);
        out.n_params          = (int64_t)llama_model_n_params(model_);
        out.n_ctx_train       = llama_model_n_ctx_train(model_);
        out.n_vocab           = llama_vocab_n_tokens(vocab_);
        out.n_gpu_layers_used = gpu_layers_;
        out.model_bytes       = llama_model_size(model_);
        out.supports_thinking = tags_.supported;
        out.is_decision_model = decision_.is_decision_model();
        out.is_embedding_model = embedding_.is_embedding_model();
        out.n_embd            = embedding_.n_embd();
        return KILN_OK;
    }

    int32_t tokenize(const char* text, bool add_special, int32_t* out, int32_t cap) const override {
        return llama_tokenize(vocab_, text, (int)strlen(text), out, cap, add_special, true);
    }
    int32_t detokenize(const int32_t* toks, int32_t n, char* out, int32_t cap) const override {
        return llama_detokenize(vocab_, toks, n, out, cap, false, false);
    }

    kiln_status render_chat(const kiln_chat_message* msgs, int32_t n, const kiln_chat_options& o,
                            kiln::RenderedPrompt& out) const override {
        common_chat_templates_inputs in;
        for (int i = 0; i < n; ++i) {
            if (!msgs[i].role || !msgs[i].content) return kiln::set_error(KILN_ERR_INVALID_ARG, "message role/content is null");
            common_chat_msg m; m.role = msgs[i].role; m.content = msgs[i].content;
            in.messages.push_back(std::move(m));
        }
        in.add_generation_prompt = o.add_assistant_prefix;
        in.enable_thinking = o.enable_thinking && tags_.supported;
        in.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        try {
            auto p = common_chat_templates_apply(tmpls_.get(), in);
            out.prompt = p.prompt;
            if (in.add_generation_prompt) {
                in.add_generation_prompt = false;
                auto h = common_chat_templates_apply(tmpls_.get(), in);
                if (p.prompt.compare(0, h.prompt.size(), h.prompt) == 0) out.stable_prefix = h.prompt;
            }
            out.starts_in_reasoning = in.enable_thinking && !tags_.start.empty()
                                   && rtrim(p.prompt).size() >= tags_.start.size()
                                   && rtrim(p.prompt).compare(rtrim(p.prompt).size() - tags_.start.size(), tags_.start.size(), tags_.start) == 0;
            return KILN_OK;
        } catch (const std::exception& e) {
            return kiln::set_error(KILN_ERR_BACKEND, std::string("chat template failed: ") + e.what());
        }
    }

    kiln_status create_context(const kiln_context_params& p, std::unique_ptr<kiln::Context>& out) override {
        auto c = std::make_unique<LlamaContext>(model_, tags_);

        // Pin worker threads to the fast cores. Without this, Android schedules ggml workers onto
        // little cores and every barrier waits for the slowest one: measured 1 thread > 5 threads on Pixel 8.
        // Benchmarking escape hatches: KILN_NO_AFFINITY=1 disables pinning,
        // KILN_CPU_MASK=<hex> forces an exact core set (thread count follows the mask).
        uint64_t mask; int n_threads;
        const bool pin = affinity(mask, n_threads);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = p.n_ctx > 0 ? (uint32_t)p.n_ctx : 4096;
        cp.n_batch         = p.n_batch > 0 ? (uint32_t)p.n_batch : 512;
        // must match the threadpool size below: with OpenMP (Linux builds) ggml starts cp.n_threads
        // workers and indexes the pool's per-thread state, so a larger count reads past its end
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        if (decision_.is_decision_model()) kiln::DecisionHead::configure(cp);
        if (embedding_.is_embedding_model()) embedding_.configure(cp, p.n_ctx);
        cp.flash_attn_type = p.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.abort_callback      = LlamaContext::abort_cb;
        cp.abort_callback_data = c.get();

        c->ctx_ = llama_init_from_model(model_, cp);
        if (!c->ctx_) return kiln::set_error(KILN_ERR_OUT_OF_MEMORY, "llama_init_from_model failed (n_ctx=" + std::to_string(cp.n_ctx) + ")");
        c->n_batch_ = (int)cp.n_batch;
        c->decision_ = &decision_;
        c->embedding_ = &embedding_;

        if (pin) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
            int set = 0;
            for (int i = 63; i >= 0 && set < n_threads; --i)      // highest-index cores are usually the biggest
                if (mask & (1ull << i)) { tpp.cpumask[i] = true; ++set; }
            tpp.strict_cpu = true;
            c->tp_ = ggml_threadpool_new(&tpp);
            if (c->tp_) llama_attach_threadpool(c->ctx_, c->tp_, c->tp_);
            else kiln::log(KILN_LOG_WARN, "threadpool creation failed, running without affinity");
        }
        out = std::move(c);
        return KILN_OK;
    }

    kiln_status load_projector(const char* path, const kiln_projector_params& p, kiln_projector_info& out) override {
        if (!embedding_.is_embedding_model())
            return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "images and audio are supported for embedding models only");
        // same cores and thread count as text; see AffinityScope in embedding.h for why the mask is needed
        uint64_t mask; int n_threads;
        const bool pin = affinity(mask, n_threads);
        return embedding_.load_projector(model_, path, p, kind_ == KILN_BACKEND_GPU, n_threads, pin ? mask : 0, out);
    }

private:
    // Which cores to use and how many threads. Returns whether threads are pinned to `mask`.
    // Benchmarking escape hatches: KILN_NO_AFFINITY=1 disables pinning,
    // KILN_CPU_MASK=<hex> forces an exact core set (thread count follows the mask).
    bool affinity(uint64_t& mask, int& n_threads) const {
        const char* no_aff = getenv("KILN_NO_AFFINITY");
        const char* mask_s = getenv("KILN_CPU_MASK");
        mask      = fast_mask_;
        n_threads = threads_;
        if (mask_s && *mask_s) {
            mask = strtoull(mask_s, nullptr, 16);
            if (mask) n_threads = __builtin_popcountll(mask);
        }
        return mask && !(no_aff && *no_aff == '1');
    }

    llama_model*              model_;
    const llama_vocab*        vocab_;
    std::string               backend_name_;
    kiln_backend_kind         kind_;
    int                       gpu_layers_;
    int                       threads_;
    uint64_t                  fast_mask_;
    common_chat_templates_ptr tmpls_;
    ThinkTags                 tags_;
    kiln::DecisionHead        decision_;
    kiln::EmbeddingHead       embedding_;
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
                if (bname == "llama.cpp/mtl") bname = "llama.cpp/metal";
                kind = KILN_BACKEND_GPU;
            }
            int n_layer = llama_model_n_layer(m) + 1;   // +1 = output layer
            if (mp.n_gpu_layers > n_layer) mp.n_gpu_layers = n_layer;
        }
        int threads = p.n_threads > 0 ? p.n_threads : default_threads(d);
        kiln::log(KILN_LOG_INFO, "loaded %s via %s, threads=%d, gpu_layers=%d", path, bname.c_str(), threads, mp.n_gpu_layers);
        out.reset(new LlamaModel(m, bname, kind, mp.n_gpu_layers, threads, d.cpu_fast_mask));
        return KILN_OK;
    }
};

} // namespace

namespace kiln {
Backend* make_llamacpp_backend() { static LlamaBackend b; return &b; }
}
