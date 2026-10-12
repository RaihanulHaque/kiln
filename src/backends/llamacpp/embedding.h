// Embedding models: text in, one vector out, no generated tokens.
// The model gives one vector per token; its GGUF says how to combine them (pooling), and
// llama.cpp does that. What kiln adds: detection, the task prefix the model was trained with,
// optional shortening of the vector, and normalising to length 1.
#pragma once
#include "../../core/internal.h"

#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace kiln {

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
        if (a == "gemma-embedding" || a == "gemma-embedding2") {
            query_prefix_ = "task: search result | query: ";
            doc_prefix_   = "title: none | text: ";
            doc_titled_   = true;                       // "title: {title} | text: "
        } else if (a == "nomic-bert" || a == "nomic-bert-moe") {
            query_prefix_ = "search_query: ";
            doc_prefix_   = "search_document: ";
        }
    }

    bool    is_embedding_model() const { return is_embedding_; }
    int32_t n_embd() const { return n_embd_; }

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

    kiln_status embed(llama_context* ctx, const char* text, const kiln_embed_options& o,
                      float* out, int32_t out_cap, kiln_embed_stats& st) const {
        if (!is_embedding_) return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not an embedding model");
        const auto t0 = std::chrono::steady_clock::now();

        int32_t dim = n_embd_;
        if (o.dimensions > 0) {
            if (o.dimensions > n_embd_) return kiln::set_error(KILN_ERR_INVALID_ARG, "dimensions is larger than the model's vector (" + std::to_string(n_embd_) + ")");
            dim = o.dimensions;
        }
        if (out_cap < dim) return kiln::set_error(KILN_ERR_INVALID_ARG, "out holds " + std::to_string(out_cap) + " floats, need " + std::to_string(dim));

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
        const int n_ctx = (int)llama_n_ctx(ctx);
        if (n > n_ctx)
            return kiln::set_error(KILN_ERR_CONTEXT_FULL, "text (" + std::to_string(n) + " tok) does not fit in context (" + std::to_string(n_ctx) + "); split it into smaller pieces");

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
    // Longest text accepted when the caller does not ask for more. Cost grows faster than
    // length, and some GGUFs state a training context far above what the model was built for.
    static constexpr uint32_t kDefaultCtx = 2048;

    bool        is_embedding_ = false;
    int32_t     n_embd_       = 0;
    int32_t     n_ctx_train_  = 0;
    std::string query_prefix_;
    std::string doc_prefix_;
    bool        doc_titled_   = false;
};

} // namespace kiln
