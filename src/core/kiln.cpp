#include "internal.h"
#include <cstring>
#include <mutex>

using namespace kiln;

static const kiln_device_info& cached_device() {
    static kiln_device_info d;
    static std::once_flag once;
    std::call_once(once, [] {
        memset(&d, 0, sizeof d);
        probe_cpu(d);
        for (auto* b : backends()) b->annotate_device(d);
    });
    return d;
}

extern "C" {

kiln_status kiln_probe_device(kiln_device_info* out) {
    if (!out) return set_error(KILN_ERR_INVALID_ARG, "out is null");
    *out = cached_device();
    return KILN_OK;
}

kiln_model_params kiln_model_params_default(void) {
    kiln_model_params p{};
    p.backend = KILN_BACKEND_AUTO; p.n_gpu_layers = -1; p.n_threads = 0;
    p.use_mmap = true; p.use_mlock = false;
    return p;
}

kiln_status kiln_model_load(const char* path, const kiln_model_params* params, kiln_model** out) {
    if (!path || !out) return set_error(KILN_ERR_INVALID_ARG, "path/out is null");
    kiln_model_params p = params ? *params : kiln_model_params_default();
    const kiln_device_info& dev = cached_device();

    for (auto* b : backends()) {
        if (!b->supports(p.backend, dev)) continue;
        std::unique_ptr<Model> m;
        kiln_status s = b->load_model(path, p, dev, m);
        if (s == KILN_OK) {
            auto* h = new kiln_model{std::move(m), dev};
            *out = h;
            return KILN_OK;
        }
        log(KILN_LOG_WARN, "backend %s failed to load %s: %s", b->name(), path, kiln_last_error());
        if (p.backend != KILN_BACKEND_AUTO) return s;   // explicit request, don't fall through
    }
    return set_error(KILN_ERR_NO_BACKEND, "no backend could load the model on this device");
}

void kiln_model_free(kiln_model* m) { delete m; }

kiln_status kiln_model_get_info(const kiln_model* m, kiln_model_info* out) {
    if (!m || !out) return set_error(KILN_ERR_INVALID_ARG, "null handle");
    return m->impl->info(*out);
}

kiln_context_params kiln_context_params_default(void) {
    kiln_context_params p{};
    p.n_ctx = 0; p.n_batch = 512; p.flash_attn = true;
    return p;
}

kiln_status kiln_context_create(kiln_model* m, const kiln_context_params* params, kiln_context** out) {
    if (!m || !out) return set_error(KILN_ERR_INVALID_ARG, "null handle");
    kiln_context_params p = params ? *params : kiln_context_params_default();
    std::unique_ptr<Context> c;
    kiln_status s = m->impl->create_context(p, c);
    if (s != KILN_OK) return s;
    *out = new kiln_context{std::move(c)};
    return KILN_OK;
}

void        kiln_context_free(kiln_context* c)  { delete c; }
kiln_status kiln_context_reset(kiln_context* c) {
    if (!c) return set_error(KILN_ERR_INVALID_ARG, "null handle");
    return c->impl->reset();
}

int32_t kiln_tokenize(const kiln_model* m, const char* text, bool add_special, int32_t* out, int32_t cap) {
    if (!m || !text) return 0;
    return m->impl->tokenize(text, add_special, out, cap);
}
int32_t kiln_detokenize(const kiln_model* m, const int32_t* toks, int32_t n, char* out, int32_t cap) {
    if (!m || !toks) return 0;
    return m->impl->detokenize(toks, n, out, cap);
}
int32_t kiln_apply_chat_template(const kiln_model* m, const kiln_chat_message* msgs, int32_t n,
                                 bool add_ass, char* out, int32_t cap) {
    if (!m || !msgs) return 0;
    return m->impl->apply_chat_template(msgs, n, add_ass, out, cap);
}

kiln_sampling_params kiln_sampling_params_default(void) {
    kiln_sampling_params p{};
    p.temperature = 0.7f; p.top_p = 0.9f; p.top_k = 40; p.min_p = 0.05f;
    p.repeat_penalty = 1.1f; p.repeat_last_n = 64; p.seed = 0; p.max_tokens = 0;
    return p;
}

kiln_status kiln_generate(kiln_context* c, const char* prompt, const kiln_sampling_params* params,
                          kiln_token_cb cb, void* user, kiln_gen_stats* stats) {
    if (!c || !prompt) return set_error(KILN_ERR_INVALID_ARG, "null handle/prompt");
    kiln_sampling_params p = params ? *params : kiln_sampling_params_default();
    return c->impl->generate(prompt, p, cb, user, stats);
}

void kiln_cancel(kiln_context* c) { if (c) c->impl->cancel(); }

}
