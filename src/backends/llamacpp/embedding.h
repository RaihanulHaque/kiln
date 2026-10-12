// Embedding models: text in, one vector out, no generated tokens.
// The model gives one vector per token; its GGUF says how to combine them (pooling), and
// llama.cpp does that. What kiln adds: detection, the task prefix the model was trained with,
// optional shortening of the vector, and normalising to length 1.
// Images and audio go through llama.cpp's mtmd library: an encoder (the projector file) turns
// each one into vectors of the same kind the model uses for text tokens, and they are fed in
// next to the text. The way they are batched follows llama.cpp's server (add_prompt_mixed).
#pragma once
#include "../../core/internal.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace kiln {

// Lets the calling thread run on every core of `mask` for as long as it lives, then puts the
// old setting back.
// Why: kiln's pinned threadpool ties the thread that calls into kiln to one core (it is worker 0
// of the pool). The image / audio encoder starts its own threads, and new threads inherit the
// caller's setting, so all of them landed on that single core: one small image took 14 s on a
// Pixel 8 instead of about 3 s. Widening the caller first lets the encoder spread over the fast
// cores. mask == 0 does nothing. Only Linux and Android pin threads this way.
class AffinityScope {
public:
    explicit AffinityScope(uint64_t mask) {
#if defined(__linux__)
        if (!mask || sched_getaffinity(0, sizeof old_, &old_) != 0) return;
        cpu_set_t wide;
        CPU_ZERO(&wide);
        for (int i = 0; i < 64; ++i) if (mask & (1ull << i)) CPU_SET(i, &wide);
        active_ = sched_setaffinity(0, sizeof wide, &wide) == 0;
#else
        (void)mask;
#endif
    }
    ~AffinityScope() {
#if defined(__linux__)
        if (active_) sched_setaffinity(0, sizeof old_, &old_);
#endif
    }
    AffinityScope(const AffinityScope&) = delete;
    AffinityScope& operator=(const AffinityScope&) = delete;
private:
#if defined(__linux__)
    cpu_set_t old_;
    bool      active_ = false;
#endif
};

class EmbeddingHead {
public:
    // An embedding model stores a pooling type of mean, cls or last. Chat models store none,
    // rerankers store "rank" and are not handled here.
    void init(const llama_model* model) {
        char arch[64] = {0}, val[32] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof arch);
        if (llama_model_meta_val_str(model, (std::string(arch) + ".pooling_type").c_str(), val, sizeof val) < 0) return;
        const int pooling = atoi(val);
        if (pooling != LLAMA_POOLING_TYPE_MEAN && pooling != LLAMA_POOLING_TYPE_CLS && pooling != LLAMA_POOLING_TYPE_LAST) return;

        is_embedding_ = true;
        n_embd_       = llama_model_n_embd_out(model);
        n_ctx_train_  = llama_model_n_ctx_train(model);

        // Task prefixes, from each model's card. Without them the vectors are still valid, but
        // search quality drops. Unknown families get no prefix.
        const std::string a = arch;
        char name_buf[128] = {0};
        llama_model_meta_val_str(model, "general.name", name_buf, sizeof name_buf);
        std::string name = name_buf;
        for (auto& ch : name) ch = (char)tolower((unsigned char)ch);
        if (a == "gemma-embedding" || a == "gemma-embedding2") {
            query_prefix_ = "task: search result | query: ";
            doc_prefix_   = "title: none | text: ";
            doc_titled_   = true;                       // "title: {title} | text: "
        } else if (a == "nomic-bert" || a == "nomic-bert-moe") {
            query_prefix_ = "search_query: ";
            doc_prefix_   = "search_document: ";
        } else if (a == "eurobert" && name.find("omni") != std::string::npos && name.find("retrieval") != std::string::npos) {
            // jina-embeddings-v5-omni-*-retrieval; its GGUF name is "Omni Nano Retrieval Text Hf"
            query_prefix_ = "Query: ";
            doc_prefix_   = "Document: ";
        }
    }

    bool    is_embedding_model() const { return is_embedding_; }
    int32_t n_embd() const { return n_embd_; }

    ~EmbeddingHead() { free_projector(); }
    void free_projector() { if (mctx_) { mtmd_free(mctx_); mctx_ = nullptr; } }

    // cpu_mask: the cores the encoders may use (0 = leave thread placement to the system)
    kiln_status load_projector(const llama_model* model, const char* path, const kiln_projector_params& p, bool use_gpu, int n_threads, uint64_t cpu_mask, kiln_projector_info& out) {
        std::lock_guard<std::mutex> lock(media_mutex_);
        if (mctx_) return kiln::set_error(KILN_ERR_INVALID_ARG, "a projector is already loaded for this model");
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu       = use_gpu;
        mp.n_threads     = n_threads;
        mp.print_timings = false;
        mp.warmup        = false;
        // A phone photo at the model's own limit can take over a thousand tokens and many
        // seconds in the encoder; 280 is what EmbeddingGemma's card uses by default.
        mp.image_max_tokens = p.image_max_tokens == 0 ? kDefaultImageTokens : p.image_max_tokens < 0 ? -1 : p.image_max_tokens;
        mctx_ = mtmd_init_from_file(path, model, mp);
        if (!mctx_) return kiln::set_error(KILN_ERR_MODEL_LOAD, std::string("failed to load projector ") + path + " (missing, corrupt, or made for another model)");
        out.supports_images = mtmd_support_vision(mctx_);
        out.supports_audio  = mtmd_support_audio(mctx_);
        media_mask_ = cpu_mask;
        return KILN_OK;
    }

    // These models are not causal: every token looks at every other one, so a text has to go
    // through in a single batch. The context is sized to the longest text it will accept.
    void configure(llama_context_params& cp, int32_t requested_n_ctx) const {
        uint32_t n_ctx = requested_n_ctx > 0 ? (uint32_t)requested_n_ctx : kDefaultCtx;
        if (n_ctx_train_ > 0 && n_ctx > (uint32_t)n_ctx_train_) n_ctx = (uint32_t)n_ctx_train_;
        cp.n_ctx      = n_ctx;
        cp.n_batch    = n_ctx;
        cp.n_ubatch   = n_ctx;
        cp.embeddings = true;                           // pooling type stays the model's own
    }

    kiln_status embed(llama_context* ctx, const char* text, const kiln_media* media, int32_t n_media,
                      const kiln_embed_options& o, float* out, int32_t out_cap, kiln_embed_stats& st) const {
        if (!is_embedding_) return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not an embedding model");
        const auto t0 = std::chrono::steady_clock::now();

        int32_t dim = n_embd_;
        if (o.dimensions > 0) {
            if (o.dimensions > n_embd_) return kiln::set_error(KILN_ERR_INVALID_ARG, "dimensions is larger than the model's vector (" + std::to_string(n_embd_) + ")");
            dim = o.dimensions;
        }
        if (out_cap < dim) return kiln::set_error(KILN_ERR_INVALID_ARG, "out holds " + std::to_string(out_cap) + " floats, need " + std::to_string(dim));

        kiln_status s = n_media > 0 ? decode_media(ctx, text, media, n_media, st) : decode_text(ctx, text, o, st);
        if (s != KILN_OK) return s;

        const float* e = llama_get_embeddings_seq(ctx, 0);
        if (!e) return kiln::set_error(KILN_ERR_BACKEND, "the model returned no pooled embedding");

        // Shortening keeps the leading numbers (Matryoshka models are trained for this); the
        // shorter vector is normalised again so a dot product is still a cosine similarity.
        double norm = 1.0;
        if (o.normalize) {
            norm = 0.0;
            for (int i = 0; i < dim; ++i) norm += (double)e[i] * e[i];
            norm = std::sqrt(norm);
            if (norm <= 0.0) norm = 1.0;
        }
        for (int i = 0; i < dim; ++i) out[i] = (float)(e[i] / norm);

        st.dimensions = dim;
        st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return KILN_OK;
    }

private:
    static kiln_status context_full(int n, int n_ctx) {
        return kiln::set_error(KILN_ERR_CONTEXT_FULL, "input (" + std::to_string(n) + " tok) does not fit in context (" + std::to_string(n_ctx) + "); split it into smaller pieces");
    }

    kiln_status decode_text(llama_context* ctx, const char* text, const kiln_embed_options& o, kiln_embed_stats& st) const {
        std::string input;
        if (o.kind == KILN_EMBED_QUERY) {
            input = query_prefix_;
        } else if (o.kind == KILN_EMBED_DOCUMENT) {
            if (doc_titled_ && o.title && *o.title) input = std::string("title: ") + o.title + " | text: ";
            else                                    input = doc_prefix_;
        }
        st.prefixed = !input.empty();
        input += text;

        const llama_vocab* vocab = llama_model_get_vocab(llama_get_model(ctx));
        int n = -llama_tokenize(vocab, input.c_str(), (int)input.size(), nullptr, 0, true, true);
        if (n <= 0) return kiln::set_error(KILN_ERR_INVALID_ARG, "empty text");
        std::vector<llama_token> toks(n);
        n = llama_tokenize(vocab, input.c_str(), (int)input.size(), toks.data(), n, true, true);
        if (n < 0) return kiln::set_error(KILN_ERR_INTERNAL, "tokenize failed");
        st.tokens = n;
        const int n_ctx = (int)std::min(llama_n_ctx(ctx), llama_n_batch(ctx));   // one batch holds the whole input
        if (n > n_ctx) return context_full(n, n_ctx);

        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            batch.token[i] = toks[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true;
        }
        batch.n_tokens = n;
        llama_memory_clear(llama_get_memory(ctx), true);   // each text stands alone
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc == 2) return KILN_ERR_CANCELLED;
        if (rc != 0) return kiln::set_error(KILN_ERR_BACKEND, "llama_decode failed in embed, rc=" + std::to_string(rc));

        return KILN_OK;
    }

    // Text with images / audio, or media alone. mtmd splits the input into chunks: runs of
    // text tokens, and one chunk per media item. Each media chunk is run through its encoder,
    // then everything goes to the model as one batch that mixes token ids and vectors.
    kiln_status decode_media(llama_context* ctx, const char* text, const kiln_media* media, int32_t n_media, kiln_embed_stats& st) const {
        std::lock_guard<std::mutex> lock(media_mutex_);     // the encoders are not thread-safe
        if (!mctx_) return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "no projector loaded; call kiln_model_load_projector with the model's mmproj file");

        struct Bitmaps {
            std::vector<mtmd_bitmap*> v;
            ~Bitmaps() { for (auto* b : v) mtmd_bitmap_free(b); }
        } bitmaps;
        for (int32_t i = 0; i < n_media; ++i) {
            const mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
            mtmd_helper_bitmap_wrapper w = media[i].path
                ? mtmd_helper_bitmap_init_from_file(mctx_, media[i].path, false, opt)
                : mtmd_helper_bitmap_init_from_buf(mctx_, media[i].data, media[i].size, false, opt);
            if (w.video_ctx) { mtmd_helper_video_free(w.video_ctx); return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "video is not supported"); }
            if (!w.bitmap) {
                const bool both = mtmd_support_vision(mctx_) && mtmd_support_audio(mctx_);
                return kiln::set_error(KILN_ERR_INVALID_ARG, "media item " + std::to_string(i) + " could not be read as an image or audio file"
                                       + (both ? "" : mtmd_support_vision(mctx_) ? " (this projector takes images only)" : " (this projector takes audio only)"));
            }
            bitmaps.v.push_back(w.bitmap);
            const bool audio = mtmd_bitmap_is_audio(w.bitmap);
            if (audio ? !mtmd_support_audio(mctx_) : !mtmd_support_vision(mctx_))
                return kiln::set_error(KILN_ERR_NOT_SUPPORTED, std::string("the loaded projector does not take ") + (audio ? "audio" : "images"));
        }

        // markers written by the caller are used as they are; otherwise the media follow the text
        const std::string marker = mtmd_default_marker();
        std::string input = text ? text : "";
        int32_t n_markers = 0;
        for (size_t p = input.find(marker); p != std::string::npos; p = input.find(marker, p + marker.size())) ++n_markers;
        if (n_markers == 0) {
            for (int32_t i = 0; i < n_media; ++i) { if (!input.empty()) input += ' '; input += marker; }
        } else if (n_markers != n_media) {
            return kiln::set_error(KILN_ERR_INVALID_ARG, "text has " + std::to_string(n_markers) + " media markers for " + std::to_string(n_media) + " media items");
        }

        struct Chunks {
            mtmd_input_chunks* c = mtmd_input_chunks_init();
            ~Chunks() { mtmd_input_chunks_free(c); }
        } chunks;
        mtmd_input_text in{input.c_str(), input.size(), true, true};
        const int32_t trc = mtmd_tokenize(mctx_, chunks.c, &in, const_cast<const mtmd_bitmap**>(bitmaps.v.data()), bitmaps.v.size());
        if (trc != 0) return kiln::set_error(KILN_ERR_BACKEND, "could not prepare the media for the model (mtmd_tokenize rc=" + std::to_string(trc) + ")");

        const int n     = (int)mtmd_helper_get_n_tokens(chunks.c);
        const int n_ctx = (int)std::min(llama_n_ctx(ctx), llama_n_batch(ctx));   // one batch holds the whole input
        st.tokens = n;
        if (n > n_ctx) return context_full(n, n_ctx);

        struct Batch {
            llama_batch_ext* b;
            ~Batch() { llama_batch_ext_free(b); }
        } batch{llama_batch_ext_init(ctx)};
        const llama_model* model   = llama_get_model(ctx);
        const size_t n_embd_inp    = (size_t)llama_model_n_embd_inp(model);
        const bool   mrope         = mtmd_decode_use_mrope(mctx_);
        std::vector<std::vector<float>> encoded;            // must outlive llama_process
        double encoder_ms = 0.0;                            // time inside the encoders only
        llama_pos pos = 0;
        for (size_t ci = 0; ci < mtmd_input_chunks_size(chunks.c); ++ci) {
            const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.c, ci);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t nt = 0;
                const llama_token* toks = mtmd_input_chunk_get_tokens_text(chunk, &nt);
                for (size_t i = 0; i < nt; ++i) {
                    const int32_t idx = llama_batch_ext_add_token(batch.b, 0, toks[i]);
                    if (idx < 0) return kiln::set_error(KILN_ERR_BACKEND, "batch rejected a text token, rc=" + std::to_string(idx));
                    llama_batch_ext_set_pos(batch.b, idx, &pos);
                    llama_batch_ext_set_output_embd(batch.b, idx, true);
                    ++pos;
                }
                continue;
            }
            {
                AffinityScope spread(media_mask_);          // see AffinityScope: the encoder's threads inherit this
                const auto te = std::chrono::steady_clock::now();
                const int32_t erc = mtmd_encode_chunk(mctx_, chunk);
                encoder_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
                if (erc != 0) return kiln::set_error(KILN_ERR_BACKEND, "the image / audio encoder failed");
            }
            const size_t nt   = mtmd_input_chunk_get_n_tokens(chunk);
            const float* embd = mtmd_get_output_embd(mctx_);
            encoded.emplace_back(embd, embd + nt * n_embd_inp);
            const float* rows = encoded.back().data();
            const mtmd_image_tokens* image = mtmd_input_chunk_get_tokens_image(chunk);
            const llama_pos pos_0 = pos;
            for (size_t i = 0; i < nt; ++i) {
                const int32_t idx = llama_batch_ext_add_embd(batch.b, 0, llama_embd{rows + i * n_embd_inp, 1, n_embd_inp});
                if (idx < 0) return kiln::set_error(KILN_ERR_BACKEND, "batch rejected a media vector, rc=" + std::to_string(idx));
                const llama_pos p = pos_0 + (llama_pos)i;
                llama_pos pos4[4] = {p, p, p, p};
                if (mrope && image) {
                    const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
                    pos4[0] = (llama_pos)rel.t; pos4[1] = (llama_pos)rel.y; pos4[2] = (llama_pos)rel.x; pos4[3] = (llama_pos)rel.z;
                }
                llama_batch_ext_set_pos(batch.b, idx, pos4);
                llama_batch_ext_set_output_embd(batch.b, idx, true);
            }
            pos += (llama_pos)mtmd_input_chunk_get_n_pos(chunk);
            st.media_tokens += (int32_t)nt;
        }
        st.media_ms = encoder_ms;

        llama_memory_clear(llama_get_memory(ctx), true);
        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.b);
        if (rc == 2) return KILN_ERR_CANCELLED;
        if (rc != 0) return kiln::set_error(KILN_ERR_BACKEND, "llama_process failed in embed, rc=" + std::to_string(rc));
        return KILN_OK;
    }

    // Longest text accepted when the caller does not ask for more. Cost grows faster than
    // length, and some GGUFs state a training context far above what the model was built for.
    static constexpr uint32_t kDefaultCtx = 2048;
    static constexpr int kDefaultImageTokens = 280;

    bool        is_embedding_ = false;
    int32_t     n_embd_       = 0;
    int32_t     n_ctx_train_  = 0;
    std::string query_prefix_;
    std::string doc_prefix_;
    bool        doc_titled_   = false;

    mtmd_context*      mctx_ = nullptr;     // image / audio encoders, loaded on request
    uint64_t           media_mask_ = 0;     // cores the encoders may use; 0 = not restricted
    mutable std::mutex media_mutex_;
};

} // namespace kiln
