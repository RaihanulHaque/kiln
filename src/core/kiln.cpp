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

    kiln_status last = KILN_OK;
    std::string last_msg;
    bool format_known = false;
    for (auto* b : backends()) {
        if (!b->supports(p.backend, dev)) continue;
        if (!b->can_load(path)) continue;
        format_known = true;
        std::unique_ptr<Model> m;
        kiln_status s = b->load_model(path, p, dev, m);
        if (s == KILN_OK) {
            *out = new kiln_model{std::move(m), dev};
            return KILN_OK;
        }
        last = s; last_msg = kiln_last_error();
        log(KILN_LOG_WARN, "backend %s failed to load %s: %s", b->name(), path, last_msg.c_str());
        if (p.backend != KILN_BACKEND_AUTO) return s;   // explicit request, don't fall through
    }
    if (last != KILN_OK) return set_error(last, last_msg);   // every capable backend tried and failed
    if (!format_known) return set_error(KILN_ERR_MODEL_LOAD, std::string("failed to load ") + path + ": not a GGUF or whisper.cpp model file (missing, or another format)");
    return set_error(KILN_ERR_NO_BACKEND, "no backend supports the requested kind on this device");
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
    *out = new kiln_context{std::move(c), m->impl.get()};
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
kiln_chat_options kiln_chat_options_default(void) {
    kiln_chat_options o{}; o.add_assistant_prefix = true; o.enable_thinking = false; o.max_reasoning_tokens = 0; return o;
}

int32_t kiln_apply_chat_template(const kiln_model* m, const kiln_chat_message* msgs, int32_t n,
                                 const kiln_chat_options* opts, char* out, int32_t cap) {
    if (!m || !msgs) return 0;
    kiln_chat_options o = opts ? *opts : kiln_chat_options_default();
    RenderedPrompt r;
    if (m->impl->render_chat(msgs, n, o, r) != KILN_OK) return 0;
    int32_t need = (int32_t)r.prompt.size();
    if (!out || cap < need + 1) return -need;
    memcpy(out, r.prompt.data(), need); out[need] = 0;
    return need;
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
    CallScope call(*c->impl);
    kiln_sampling_params p = params ? *params : kiln_sampling_params_default();
    RenderedPrompt r; r.prompt = prompt;
    return c->impl->generate(r, p, cb, user, stats);
}

kiln_status kiln_chat(kiln_context* c, const kiln_chat_message* msgs, int32_t n_msgs,
                      const kiln_chat_options* chat_opts, const kiln_sampling_params* params,
                      kiln_token_cb cb, void* user, kiln_gen_stats* stats) {
    if (!c || !msgs || n_msgs <= 0) return set_error(KILN_ERR_INVALID_ARG, "null handle/messages");
    CallScope call(*c->impl);   // before rendering: a cancel during it must count
    kiln_chat_options o = chat_opts ? *chat_opts : kiln_chat_options_default();
    o.add_assistant_prefix = true;
    kiln_sampling_params p = params ? *params : kiln_sampling_params_default();
    RenderedPrompt r;
    kiln_status s = c->model->render_chat(msgs, n_msgs, o, r);
    if (s != KILN_OK) return s;
    r.max_reasoning_tokens = o.max_reasoning_tokens;
    return c->impl->generate(r, p, cb, user, stats);
}

void kiln_cancel(kiln_context* c) { if (c) c->impl->cancel(); }

kiln_status kiln_decide(kiln_context* c, const char* state, const kiln_decision_question* q,
                        float* probs, int32_t probs_cap, kiln_decision_answer* answer) {
    if (!c || !q || !probs) return set_error(KILN_ERR_INVALID_ARG, "null handle/question/probs");
    if (!q->instructions || !q->id) return set_error(KILN_ERR_INVALID_ARG, "question id/instructions is null");
    if (q->n_options < 0 || (q->n_options > 0 && !q->options)) return set_error(KILN_ERR_INVALID_ARG, "bad options");
    kiln_decision_answer a{};
    CallScope call(*c->impl);
    kiln_status s = c->impl->decide(state, *q, probs, probs_cap, a);
    if (answer) *answer = a;
    return s;
}

kiln_embed_options kiln_embed_options_default(void) {
    kiln_embed_options o{}; o.kind = KILN_EMBED_RAW; o.title = nullptr; o.dimensions = 0; o.normalize = true; return o;
}

kiln_status kiln_embed(kiln_context* c, const char* text, const kiln_embed_options* opts,
                       float* out, int32_t out_cap, kiln_embed_stats* stats) {
    if (!c || !text || !out) return set_error(KILN_ERR_INVALID_ARG, "null handle/text/out");
    kiln_embed_options o = opts ? *opts : kiln_embed_options_default();
    kiln_embed_stats st{};
    CallScope call(*c->impl);
    kiln_status s = c->impl->embed(text, nullptr, 0, o, out, out_cap, st);
    if (stats) *stats = st;
    return s;
}

kiln_projector_params kiln_projector_params_default(void) {
    kiln_projector_params p{}; p.image_max_tokens = 0; return p;
}

kiln_status kiln_model_load_projector(kiln_model* m, const char* path, const kiln_projector_params* params, kiln_projector_info* out) {
    if (!m || !path) return set_error(KILN_ERR_INVALID_ARG, "null handle/path");
    kiln_projector_params p = params ? *params : kiln_projector_params_default();
    kiln_projector_info info{};
    kiln_status s = m->impl->load_projector(path, p, info);
    if (out) *out = info;
    return s;
}

kiln_status kiln_embed_media(kiln_context* c, const char* text, const kiln_media* media, int32_t n_media,
                             const kiln_embed_options* opts, float* out, int32_t out_cap, kiln_embed_stats* stats) {
    if (!c || !out) return set_error(KILN_ERR_INVALID_ARG, "null handle/out");
    if (n_media < 0 || (n_media > 0 && !media)) return set_error(KILN_ERR_INVALID_ARG, "bad media");
    if (n_media == 0 && !text) return set_error(KILN_ERR_INVALID_ARG, "no text and no media");
    for (int32_t i = 0; i < n_media; ++i)
        if (!media[i].path && (!media[i].data || media[i].size == 0)) return set_error(KILN_ERR_INVALID_ARG, "media item without path or data");
    kiln_embed_options o = opts ? *opts : kiln_embed_options_default();
    kiln_embed_stats st{};
    CallScope call(*c->impl);
    kiln_status s = c->impl->embed(text, media, n_media, o, out, out_cap, st);
    if (stats) *stats = st;
    return s;
}

kiln_speak_options kiln_speak_options_default(void) {
    kiln_speak_options o{}; o.voice = nullptr; o.language = nullptr; o.max_frames = 0; o.top_k = 0; o.top_p = 0; o.seed = UINT32_MAX; return o;
}

kiln_status kiln_speak(kiln_context* c, const char* text, const kiln_speak_options* opts,
                       kiln_audio_cb cb, void* user, kiln_speak_stats* stats) {
    kiln_speak_stats st{};
    if (stats) *stats = st;
    if (!c || !text) return set_error(KILN_ERR_INVALID_ARG, "null handle/text");
    if (!*text) return set_error(KILN_ERR_INVALID_ARG, "text is empty");
    kiln_speak_options o = opts ? *opts : kiln_speak_options_default();
    if (o.voice && !o.voice->path && (!o.voice->data || o.voice->size == 0)) return set_error(KILN_ERR_INVALID_ARG, "voice without path or data");
    CallScope call(*c->impl);
    kiln_status s = c->impl->speak(text, o, cb, user, st);
    if (stats) *stats = st;
    return s;
}

kiln_transcribe_options kiln_transcribe_options_default(void) {
    kiln_transcribe_options o{}; o.language = nullptr; o.prompt = nullptr; o.max_tokens = 0; return o;
}

kiln_status kiln_transcribe(kiln_context* c, const kiln_media* audio, const kiln_transcribe_options* opts,
                            kiln_token_cb cb, void* user, kiln_transcribe_stats* stats) {
    kiln_transcribe_stats st{};
    if (stats) *stats = st;
    if (!c || !audio) return set_error(KILN_ERR_INVALID_ARG, "null handle/audio");
    if (!audio->path && (!audio->data || audio->size == 0)) return set_error(KILN_ERR_INVALID_ARG, "audio without path or data");
    kiln_transcribe_options o = opts ? *opts : kiln_transcribe_options_default();
    CallScope call(*c->impl);
    RenderedPrompt r;
    kiln_status s = c->model->render_transcribe(o, r);
    if (s == KILN_OK) s = c->impl->transcribe(r, o, *audio, cb, user, st);
    if (stats) *stats = st;
    return s;
}

}
