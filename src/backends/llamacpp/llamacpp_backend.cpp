// llama.cpp backend. This is the guaranteed tier: CPU everywhere, GPU where ggml has a device.
#include "../../core/internal.h"
#include "decision.h"
#include "embedding.h"
#include "transcript_filter.h"
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
        if (gen_) mtmd_helper_gen_audio_free(gen_);
        if (tts_ctx_) { if (tp_) llama_detach_threadpool(tts_ctx_); llama_free(tts_ctx_); }
        if (ctx_) { if (tp_) llama_detach_threadpool(ctx_); llama_free(ctx_); }
        if (tp_) ggml_threadpool_free(tp_);
    }

    // set by LlamaModel::create_context after llama_init_from_model
    llama_context*    ctx_ = nullptr;
    ggml_threadpool_t tp_  = nullptr;
    int               n_batch_ = 512;
    llama_context*    tts_ctx_ = nullptr;   // for kiln_speak: needs embeddings on, made on first use, shares tp_
    mtmd_helper_gen_audio* gen_ = nullptr;  // kiln_speak's helper, kept between calls (making one copies the token table)
    const kiln::DecisionHead* decision_ = nullptr;   // owned by the model; set for decision models
    const kiln::EmbeddingHead* embedding_ = nullptr; // owned by the model; set for embedding models
    const kiln::MediaProjector* media_ = nullptr;    // owned by the model; holds encoders once a projector is loaded

    static bool abort_cb(void* user) { return static_cast<LlamaContext*>(user)->cancelled(); }

    kiln_status reset() override {
        llama_memory_clear(llama_get_memory(ctx_), true);
        cache_.clear();
        ckpts_.clear();
        return KILN_OK;
    }

    kiln_status decide(const char* state, const kiln_decision_question& q,
                       float* probs, int32_t probs_cap, kiln_decision_answer& out) override {
        if (!decision_ || !decision_->is_decision_model()) return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not a decision model");
        return decision_->decide(ctx_, state, q, probs, probs_cap, out);
    }

    kiln_status embed(const char* text, const kiln_media* media, int32_t n_media, const kiln_embed_options& o,
                      float* out, int32_t out_cap, kiln_embed_stats& stats) override {
        if (!embedding_ || !embedding_->is_embedding_model()) return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not an embedding model");
        return embedding_->embed(ctx_, *media_, text, media, n_media, o, out, out_cap, stats);
    }

    kiln_status generate(const kiln::RenderedPrompt& in, const kiln_sampling_params& sp,
                         kiln_token_cb cb, void* user, kiln_gen_stats* stats) override {
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

        // ---- where to checkpoint (hybrid / recurrent models only): the end of the system prompt
        // (reused when only the user message changes) and the end of the stable prefix (reused
        // by the next turn of a conversation)
        std::vector<size_t> ckpt_at;
        if (partial_state_) {
            for (const std::string* prefix : {&in.system_prefix, &in.stable_prefix}) {
                if (prefix->empty()) continue;
                std::vector<llama_token> pre(prefix->size() + 8);
                int np = llama_tokenize(vocab_, prefix->c_str(), (int)prefix->size(), pre.data(), (int)pre.size(), true, true);
                size_t k = 0;
                while (np > 0 && k < (size_t)np && k + 1 < toks.size() && pre[k] == toks[k]) ++k;
                if (k > 0 && (ckpt_at.empty() || k > ckpt_at.back())) ckpt_at.push_back(k);
            }
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
            for (size_t c : ckpt_at)   // end a batch exactly at each checkpoint
                if (i < (int)c) { cnt = std::min(cnt, (int)c - i); break; }
            if (cancelled()) { st.stopped_by_user = true; return done(KILN_ERR_CANCELLED); }
            int rc = llama_decode(ctx_, llama_batch_get_one(toks.data() + i, cnt));
            if (rc == 2) { st.stopped_by_user = true; return done(KILN_ERR_CANCELLED); }
            if (rc != 0) return done(kiln::set_error(rc == 1 ? KILN_ERR_CONTEXT_FULL : KILN_ERR_BACKEND, "llama_decode failed in prompt eval, rc=" + std::to_string(rc)));
            i += cnt;
            n_past += cnt;
            cache_.insert(cache_.end(), toks.begin() + i - cnt, toks.begin() + i);
            for (size_t c : ckpt_at)
                if (i == (int)c) save_checkpoint(c);
        }
        st.prompt_ms = ms_since(t0);

        const kiln_status result = sample_loop(smpl, in, sp, n_past, cb, user, st);
        st.prompt_tps = st.prompt_ms > 0 ? (st.prompt_tokens - st.cached_tokens) * 1000.0 / st.prompt_ms : 0;
        st.gen_tps    = st.gen_ms > 0 ? st.generated_tokens * 1000.0 / st.gen_ms : 0;
        return done(result);
    }

    // Speech to text: the instruction and the audio go in as one prompt, then the model writes
    // the words like any other answer. Follows llama.cpp's mtmd helper (mtmd_helper_eval_chunks),
    // with the encoder timed on its own and spread over the fast cores.
    kiln_status transcribe(const kiln::RenderedPrompt& in, const kiln_transcribe_options& o, const kiln_media& audio,
                           kiln_token_cb cb, void* user, kiln_transcribe_stats& st) override {
        const int32_t max_tokens = o.max_tokens;
        if ((decision_ && decision_->is_decision_model()) || (embedding_ && embedding_->is_embedding_model()))
            return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not a speech model");
        const auto t_all = clock_t_::now();

        std::unique_lock<std::mutex> lock(media_->mutex());    // the encoders are not thread-safe
        mtmd_context* mctx = media_->ctx();
        if (!mctx || !mtmd_support_audio(mctx))
            return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "no audio encoder loaded; call kiln_model_load_projector with the speech model's mmproj file");

        kiln::MediaProjector::Bitmaps clip;
        if (kiln_status s = media_->read(&audio, 1, clip); s != KILN_OK) return s;
        if (!mtmd_bitmap_is_audio(clip.v[0])) return kiln::set_error(KILN_ERR_INVALID_ARG, "kiln_transcribe takes an audio clip, not an image");
        const int rate = mtmd_get_audio_sample_rate(mctx);
        if (rate > 0) st.audio_seconds = (double)(mtmd_bitmap_get_n_bytes(clip.v[0]) / sizeof(float)) / rate;

        struct Chunks {
            mtmd_input_chunks* c = mtmd_input_chunks_init();
            ~Chunks() { mtmd_input_chunks_free(c); }
        } chunks;
        mtmd_input_text text{in.prompt.c_str(), in.prompt.size(), true, true};
        const int32_t trc = mtmd_tokenize(mctx, chunks.c, &text, const_cast<const mtmd_bitmap**>(clip.v.data()), clip.v.size());
        if (trc != 0) return kiln::set_error(KILN_ERR_BACKEND, "could not prepare the audio for the model (mtmd_tokenize rc=" + std::to_string(trc) + ")");

        const int n     = (int)mtmd_helper_get_n_tokens(chunks.c);
        const int n_ctx = (int)llama_n_ctx(ctx_);
        st.prompt_tokens = n;
        if (n >= n_ctx)
            return kiln::set_error(KILN_ERR_CONTEXT_FULL, "the clip (" + std::to_string(n) + " tok) does not fit in context (" + std::to_string(n_ctx) + "); cut it into shorter pieces or raise n_ctx");

        // Each clip stands alone: whatever conversation the context held is dropped.
        reset();
        llama_pos n_past = 0;
        const size_t n_chunks = mtmd_input_chunks_size(chunks.c);
        for (size_t ci = 0; ci < n_chunks; ++ci) {
            const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.c, ci);
            if (cancelled()) { reset(); st.stopped_by_user = true; return KILN_ERR_CANCELLED; }
            int32_t rc = 0;
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                const auto t0 = clock_t_::now();
                rc = mtmd_helper_eval_chunk_single(mctx, ctx_, chunk, n_past, 0, n_batch_, ci + 1 == n_chunks, &n_past);
                st.prompt_ms += ms_since(t0);
            } else {
                {
                    kiln::AffinityScope spread(media_->cpu_mask());   // see AffinityScope: the encoder's threads inherit this
                    const auto te = clock_t_::now();
                    rc = mtmd_encode_chunk(mctx, chunk);
                    st.encode_ms += ms_since(te);
                }
                if (rc != 0) { reset(); return kiln::set_error(KILN_ERR_BACKEND, "the audio encoder failed"); }
                st.audio_tokens += (int32_t)mtmd_input_chunk_get_n_tokens(chunk);
                const auto t0 = clock_t_::now();
                rc = mtmd_helper_decode_image_chunk(mctx, ctx_, chunk, mtmd_get_output_embd(mctx), n_past, 0, n_batch_, &n_past, nullptr, nullptr);
                st.prompt_ms += ms_since(t0);
            }
            if (rc != 0) {
                reset();
                if (rc == 2) { st.stopped_by_user = true; return KILN_ERR_CANCELLED; }
                return kiln::set_error(KILN_ERR_BACKEND, "decoding the prompt failed in transcribe, rc=" + std::to_string(rc));
            }
        }

        lock.unlock();     // the encoders are done; writing the words needs only this context

        // Always the most likely word: a transcript should be the same every time. No repeat
        // penalty either, people do say the same word twice.
        kiln_sampling_params sp{};
        sp.max_tokens = max_tokens;
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        struct SmplGuard { llama_sampler* s; ~SmplGuard() { llama_sampler_free(s); } } guard{smpl};

        kiln::TranscriptFilter filter(cb, user);
        kiln_gen_stats gs{};
        const kiln_status result = sample_loop(smpl, in, sp, (int)n_past, kiln::TranscriptFilter::on_piece, &filter, gs);
        if (result == KILN_OK && !gs.stopped_by_user) filter.finish();
        reset();     // the audio is not part of any conversation; leave nothing behind for kiln_chat

        st.generated_tokens = gs.generated_tokens;
        st.gen_ms           = gs.gen_ms;
        st.stopped_by_eos   = gs.stopped_by_eos;
        st.stopped_by_user  = gs.stopped_by_user || filter.stopped();
        strncpy(st.language, filter.language().c_str(), sizeof st.language - 1);
        st.ms = ms_since(t_all);
        return result;
    }

    // Text to speech: the text (and the voice reference) go through the backbone, then one
    // frame at a time the backbone's hidden state is turned into audio codes and the codes
    // into samples (llama.cpp's mtmd audio-generation helper, the same path as llama-tts).
    // The audio is handed to the caller in pieces of about a second, so playback can start
    // before the sentence is finished.
    kiln_status speak(const char* text, const kiln_speak_options& o, kiln_audio_cb cb, void* user,
                      kiln_speak_stats& st) override {
        if ((decision_ && decision_->is_decision_model()) || (embedding_ && embedding_->is_embedding_model()))
            return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not a text-to-speech model");
        const auto t_all = clock_t_::now();

        std::unique_lock<std::mutex> lock(media_->mutex());    // the projector is not thread-safe
        mtmd_context* mctx = media_->ctx();
        if (!mctx || !media_->generates_audio())
            return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "no text-to-speech projector loaded; call kiln_model_load_projector with the model's mmproj file");
        const mtmd_gen_audio_info gi = mtmd_gen_audio_get_info(mctx);

        // a context of its own: the backbone must return hidden states (embeddings), which
        // the chat context does not, and the conversation in ctx_ must stay untouched
        if (!tts_ctx_) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 4096; cp.n_batch = 512; cp.embeddings = true;
            cp.n_threads = llama_n_threads(ctx_); cp.n_threads_batch = llama_n_threads_batch(ctx_);
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.abort_callback = abort_cb; cp.abort_callback_data = this;
            tts_ctx_ = llama_init_from_model(model_, cp);
            if (!tts_ctx_) return kiln::set_error(KILN_ERR_OUT_OF_MEMORY, "could not create the voice context");
            if (tp_) llama_attach_threadpool(tts_ctx_, tp_, tp_);
        }
        llama_memory_clear(llama_get_memory(tts_ctx_), true);

        kiln::MediaProjector::Bitmaps voice;
        if (o.voice) {
            if (kiln_status s = media_->read(o.voice, 1, voice); s != KILN_OK) return s;
            if (!mtmd_bitmap_is_audio(voice.v[0])) return kiln::set_error(KILN_ERR_INVALID_ARG, "the voice reference must be an audio clip, not an image");
        }

        if (!gen_) gen_ = mtmd_helper_gen_audio_init(tts_ctx_, mctx);
        else mtmd_helper_gen_audio_reset(gen_);
        if (!gen_) return kiln::set_error(KILN_ERR_BACKEND, "could not start audio generation");
        struct Gen { mtmd_helper_gen_audio* g; } gen{gen_};
        mtmd_helper_gen_audio_inp inp{};
        inp.seq_id = 0; inp.prompt = text; inp.prompt_len = strlen(text);
        inp.speaker_ref = voice.v.empty() ? nullptr : voice.v[0];
        inp.lang  = o.language && *o.language ? o.language : (gi.type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS ? "en" : nullptr);
        inp.top_k = o.top_k > 0 ? o.top_k : 40;
        inp.top_p = o.top_p > 0 ? o.top_p : 0.95f;
        inp.seed  = o.seed;
        inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;

        kiln::AffinityScope spread(media_->cpu_mask());   // the audio encoder and decoder start threads of their own
        // set_input encodes the voice reference (about a second on a phone): it is part of "prompt"
        const auto t0 = clock_t_::now();
        if (mtmd_helper_gen_audio_set_input(gen.g, &inp) != 0) return kiln::set_error(KILN_ERR_BACKEND, "the model could not take this text (set_input failed)");
        for (;;) {
            if (cancelled()) return KILN_ERR_CANCELLED;
            const int32_t left = mtmd_helper_gen_audio_step_prompt(gen.g, n_batch_);
            if (left < 0) return kiln::set_error(KILN_ERR_BACKEND, "reading the text failed (step_prompt)");
            if (left == 0) break;
        }
        st.prompt_tokens = llama_memory_seq_pos_max(llama_get_memory(tts_ctx_), 0) + 1;
        st.prompt_ms = ms_since(t0);

        // the semantic codes are sampled like text tokens
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(inp.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(inp.top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(o.seed == UINT32_MAX ? LLAMA_DEFAULT_SEED : o.seed));
        struct SmplGuard { llama_sampler* s; ~SmplGuard() { llama_sampler_free(s); } } guard{smpl};
        auto sample = [&]() { llama_token t = llama_sampler_sample(smpl, tts_ctx_, -1); llama_sampler_accept(smpl, t); return t; };

        const int max_frames = o.max_frames > 0 ? o.max_frames : 512;
        const auto t1 = clock_t_::now();
        llama_token sampled = sample();
        const float* h = llama_get_embeddings_ith(tts_ctx_, -1);
        int64_t emitted = 0;
        st.sample_rate = gi.sample_rate;
        const int frames_per_piece = 12;                       // about a second of audio per callback
        auto emit = [&](bool final_piece) -> bool {           // hands the samples made since the last call to cb
            int32_t rate = 0; const char* data = nullptr; size_t len = 0; int64_t n = 0;
            if (mtmd_helper_gen_audio_get_output(gen.g, &rate, &data, &len, &n) != 0) return true;
            st.sample_rate = rate;
            if (n > emitted) {
                const float* pcm = reinterpret_cast<const float*>(data);
                if (cb && !cb(pcm + emitted, (int32_t)(n - emitted), rate, user)) { st.stopped_by_user = true; return false; }
                emitted = n;
            }
            (void)final_piece;
            return true;
        };
        bool stop = false;
        while (!stop && st.frames < max_frames) {
            if (cancelled()) { st.gen_ms = ms_since(t1); return KILN_ERR_CANCELLED; }
            const float* h_next = nullptr;
            if (mtmd_helper_gen_audio_step_gen(gen.g, sampled, h, &h_next, &stop) != 0) return kiln::set_error(KILN_ERR_BACKEND, "generating a frame failed (step_gen)");
            if (!h_next) break;
            ++st.frames;
            h = h_next;
            sampled = sample();
            if (st.frames % frames_per_piece == 0 && !emit(false)) break;
        }
        if (!st.stopped_by_user) emit(true);
        st.stopped_by_eos = stop;
        st.gen_ms = ms_since(t1);
        st.audio_seconds = st.sample_rate > 0 ? (double)emitted / st.sample_rate : 0.0;
        st.ms = ms_since(t_all);
        return KILN_OK;
    }

private:
    // The token-by-token loop shared by generate() and transcribe(): sample, hand the piece to
    // the callback, feed the token back. n_past = tokens already in the context. Fills the
    // generation fields of st.
    kiln_status sample_loop(llama_sampler* smpl, const kiln::RenderedPrompt& in, const kiln_sampling_params& sp,
                            int n_past, kiln_token_cb cb, void* user, kiln_gen_stats& st) {
        const int n_ctx = (int)llama_n_ctx(ctx_);
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
            if (cancelled()) { st.stopped_by_user = true; result = KILN_ERR_CANCELLED; break; }
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
        return result;
    }

    // Saved recurrent state of a hybrid / recurrent model after its first n_tokens cache tokens.
    // Only the part seq_rm cannot rewind is stored (PARTIAL_ONLY); attention KV is truncated instead.
    struct Checkpoint { size_t n_tokens; std::vector<uint8_t> data; };
    static constexpr size_t kMaxCheckpoints = 3;   // system prompt, latest turn, one back for an edited last message

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
    ~LlamaModel() override { media_.free(); llama_model_free(model_); }

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
        out.is_speech_model   = false;   // Qwen3-ASR is a chat-style model; its encoder comes as a projector
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
            // The system prefix: render the same system message(s) with a different user message
            // and keep what both renders share. Rendering the system message alone is not an
            // option, because some templates (Qwen3.5) refuse a conversation with no user turn.
            size_t n_sys = 0;
            while (n_sys < in.messages.size() && in.messages[n_sys].role == "system") ++n_sys;
            if (n_sys > 0 && n_sys < in.messages.size()) {
                common_chat_templates_inputs alt;
                alt.messages.assign(in.messages.begin(), in.messages.begin() + n_sys);
                common_chat_msg u; u.role = "user"; u.content = "\x01";
                alt.messages.push_back(std::move(u));
                alt.add_generation_prompt = false;
                alt.enable_thinking = in.enable_thinking;
                alt.reasoning_format = in.reasoning_format;
                try {
                    const std::string a = common_chat_templates_apply(tmpls_.get(), alt).prompt;
                    size_t k = 0;
                    while (k < a.size() && k < p.prompt.size() && a[k] == p.prompt[k]) ++k;
                    if (k < out.stable_prefix.size() || out.stable_prefix.empty()) out.system_prefix = p.prompt.substr(0, k);
                } catch (const std::exception&) {
                    // no system prefix; the stable prefix still works for multi-turn chat
                }
            }
            out.starts_in_reasoning = in.enable_thinking && !tags_.start.empty()
                                   && rtrim(p.prompt).size() >= tags_.start.size()
                                   && rtrim(p.prompt).compare(rtrim(p.prompt).size() - tags_.start.size(), tags_.start.size(), tags_.start) == 0;
            return KILN_OK;
        } catch (const std::exception& e) {
            return kiln::set_error(KILN_ERR_BACKEND, std::string("chat template failed: ") + e.what());
        }
    }

    // The instruction a speech model gets, in the model's own chat format, with the place for
    // the audio marked. Follows llama.cpp's server (convert_transcriptions_to_chatcmpl).
    kiln_status render_transcribe(const kiln_transcribe_options& o, kiln::RenderedPrompt& out) const override {
        const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls_.get());
        std::string user = o.prompt && *o.prompt ? o.prompt : preset.user;
        if (o.language && *o.language) user += std::string(" (language: ") + o.language + ")";
        user += mtmd_default_marker();
        std::vector<kiln_chat_message> msgs;
        if (!preset.system.empty()) msgs.push_back({"system", preset.system.c_str()});
        msgs.push_back({"user", user.c_str()});
        kiln_chat_options co{};
        co.add_assistant_prefix = true;
        return render_chat(msgs.data(), (int32_t)msgs.size(), co, out);
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
        c->n_batch_ = (int)llama_n_batch(c->ctx_);   // llama.cpp lowers it to n_ctx when the context is smaller
        c->decision_ = &decision_;
        c->embedding_ = &embedding_;
        c->media_ = &media_;

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
        if (decision_.is_decision_model())
            return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "images and audio are not supported for decision models");
        // same cores and thread count as text; see AffinityScope in media.h for why the mask is needed
        uint64_t mask; int n_threads;
        const bool pin = affinity(mask, n_threads);
        return media_.load(model_, path, p, kind_ == KILN_BACKEND_GPU, n_threads, pin ? mask : 0, out);
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
    kiln::MediaProjector      media_;
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

    bool can_load(const char* path) const override {
        char magic[4] = {0};
        FILE* f = fopen(path, "rb");
        if (!f) return true;                      // let the loader report the missing file
        const size_t n = fread(magic, 1, 4, f);
        fclose(f);
        return n < 4 || memcmp(magic, "GGUF", 4) == 0;
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
