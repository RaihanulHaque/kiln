#pragma once
#include "kiln/kiln.h"
#include <memory>
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
};

class Context {
public:
    virtual ~Context() = default;
    virtual kiln_status reset() = 0;
    virtual kiln_status generate(const RenderedPrompt& in, const kiln_sampling_params& sp,
                                 kiln_token_cb cb, void* user, kiln_gen_stats* stats) = 0;
    virtual void cancel() = 0;
};

class Model {
public:
    virtual ~Model() = default;
    virtual kiln_status info(kiln_model_info& out) const = 0;
    virtual int32_t tokenize(const char* text, bool add_special, int32_t* out, int32_t cap) const = 0;
    virtual int32_t detokenize(const int32_t* toks, int32_t n, char* out, int32_t cap) const = 0;
    virtual kiln_status render_chat(const kiln_chat_message* msgs, int32_t n, const kiln_chat_options& o,
                                    RenderedPrompt& out) const = 0;
    virtual kiln_status create_context(const kiln_context_params& p, std::unique_ptr<Context>& out) = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char*       name() const = 0;
    // Fill in gpu/npu fields of the probe result if this backend can see them.
    virtual void              annotate_device(kiln_device_info& d) const = 0;
    // Can this backend serve the requested kind on this device?
    virtual bool              supports(kiln_backend_kind kind, const kiln_device_info& d) const = 0;
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
