#pragma once
#include "kiln/kiln.h"
#include <memory>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace kiln {

// ---- error / log plumbing (core/error.cpp) ----
kiln_status set_error(kiln_status s, const std::string& msg);
void        log(kiln_log_level lvl, const char* fmt, ...);

// ---- backend interface (backends/backend.h) ----
// What a rendered chat turn looks like to the generator.
struct RenderedPrompt {
    std::string prompt;
    bool        starts_in_reasoning = false;   // template already opened the think block
    // Leading part of `prompt` that the next turn's prompt will also start with: the history
    // without the assistant prefix. Empty if unknown (raw kiln_generate). Hybrid models save
    // their state here, because templates often rewrite the assistant reply that follows.
    std::string stable_prefix;
    // Leading part of `prompt` that stays the same when only the last user message changes:
    // the system message(s) and the opening of the user turn. Lets an app that sends the same
    // system prompt with a new user message each time (extraction, classification) reuse it on
    // hybrid models too. Empty when there is no leading system message.
    std::string system_prefix;
    int32_t     max_reasoning_tokens = 0;      // thinking budget; <= 0 = unlimited
};

class Context {
public:
    virtual ~Context() = default;
    virtual kiln_status reset() = 0;
    virtual kiln_status generate(const RenderedPrompt& in, const kiln_sampling_params& sp,
                                 kiln_token_cb cb, void* user, kiln_gen_stats* stats) = 0;

    // kiln_cancel. A public call (kiln.cpp) numbers itself with begin_call() before any work,
    // rendering the prompt included, and a cancel marks the call that is running at that moment.
    // So a cancel that arrives while kiln_chat renders its template is kept, one that arrives
    // while no call runs is dropped, and one aimed at a call that just ended cannot stop the next.
    void cancel() {
        const uint64_t c = call_.load();
        if (c) cancelled_call_.store(c);
    }
    bool cancelled() const {
        const uint64_t c = call_.load();
        return c && cancelled_call_.load() == c;
    }
    void begin_call() { call_.store(++calls_); }
    void end_call() { call_.store(0); }
    virtual kiln_status decide(const char* state, const kiln_decision_question& q,
                               float* probs, int32_t probs_cap, kiln_decision_answer& out) = 0;
    // text may be null when media are given
    virtual kiln_status embed(const char* text, const kiln_media* media, int32_t n_media, const kiln_embed_options& o,
                              float* out, int32_t out_cap, kiln_embed_stats& stats) = 0;
    // in: the instruction from Model::render_transcribe, with the media marker where the audio goes
    // (empty for a model that takes no instruction); o: the caller's options, for the fields a
    // backend reads directly (language, max_tokens)
    virtual kiln_status transcribe(const RenderedPrompt& in, const kiln_transcribe_options& o, const kiln_media& audio,
                                   kiln_token_cb cb, void* user, kiln_transcribe_stats& stats) = 0;
    virtual kiln_status speak(const char* text, const kiln_speak_options& o, kiln_audio_cb cb, void* user,
                              kiln_speak_stats& stats) = 0;

private:
    std::atomic<uint64_t> call_{0};            // number of the running call, 0 = none
    std::atomic<uint64_t> cancelled_call_{0};  // the call kiln_cancel stopped
    std::atomic<uint64_t> calls_{0};           // calls made so far (they run one at a time)
};

// Marks a public call on a context as running for its whole length (see Context::cancel).
class CallScope {
public:
    explicit CallScope(Context& c) : c_(c) { c_.begin_call(); }
    ~CallScope() { c_.end_call(); }
    CallScope(const CallScope&) = delete;
    CallScope& operator=(const CallScope&) = delete;
private:
    Context& c_;
};

class Model {
public:
    virtual ~Model() = default;
    virtual kiln_status info(kiln_model_info& out) const = 0;
    virtual int32_t tokenize(const char* text, bool add_special, int32_t* out, int32_t cap) const = 0;
    virtual int32_t detokenize(const int32_t* toks, int32_t n, char* out, int32_t cap) const = 0;
    virtual kiln_status render_chat(const kiln_chat_message* msgs, int32_t n, const kiln_chat_options& o,
                                    RenderedPrompt& out) const = 0;
    virtual kiln_status render_transcribe(const kiln_transcribe_options& o, RenderedPrompt& out) const = 0;
    virtual kiln_status create_context(const kiln_context_params& p, std::unique_ptr<Context>& out) = 0;
    virtual kiln_status load_projector(const char* path, const kiln_projector_params& p, kiln_projector_info& out) = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char*       name() const = 0;
    // Fill in gpu/npu fields of the probe result if this backend can see them.
    virtual void              annotate_device(kiln_device_info& d) const = 0;
    // Can this backend serve the requested kind on this device?
    virtual bool              supports(kiln_backend_kind kind, const kiln_device_info& d) const = 0;
    // Is this a file format the backend reads? Looks at the first bytes only (GGUF for
    // llama.cpp, ggml's older "lmgg" header for whisper.cpp); a false means the file is not
    // even tried, so an explicit backend request fails with a clear message.
    virtual bool              can_load(const char* path) const = 0;
    virtual kiln_status       load_model(const char* path, const kiln_model_params& p,
                                         const kiln_device_info& d, std::unique_ptr<Model>& out) = 0;
};

// Registry (core/registry.cpp). Order = priority.
const std::vector<Backend*>& backends();

// Platform probe (probe/probe.cpp). Fills CPU/RAM/ABI fields only.
void probe_cpu(kiln_device_info& d);

} // namespace kiln

struct kiln_model   { std::unique_ptr<kiln::Model>   impl; kiln_device_info device; };
struct kiln_context { std::unique_ptr<kiln::Context> impl; kiln::Model* model; };
