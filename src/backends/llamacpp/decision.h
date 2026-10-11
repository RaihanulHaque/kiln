// Decision models: one typed question answered in a single forward pass, no generated tokens.
// Port of the lfm2-d1-omni path of llama.cpp's tools/server/server-decision.cpp, text state only.
// The model's "systemone" template lays out state, question and options; each option ends in a
// mask token, and the score of an option is read from the embeddings output at that token.
#pragma once
#include "../../core/internal.h"

#include "chat.h"
#include "common.h"
#include "llama.h"

#include "json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace kiln {

class DecisionHead {
public:
    using json = common_json;

    // Reads "<arch>.decision.*". Leaves supported() false for chat models and for decision
    // families that are not ported.
    void init(const llama_model* model) {
        const common_decision_type type = common_get_decision_type(model);
        is_decision_ = type != COMMON_DECISION_TYPE_NONE;
        if (type != COMMON_DECISION_TYPE_LFM2_D1_OMNI) return;

        vocab_ = llama_model_get_vocab(model);
        const char* src = llama_model_chat_template(model, "systemone");
        marker_ = llama_vocab_mask(vocab_);
        if (!src || marker_ == LLAMA_TOKEN_NULL) { kiln::log(KILN_LOG_WARN, "decision model without a systemone template or mask token"); return; }
        try {
            tmpl_ = std::make_shared<const common_chat_template>(src, "", "");
        } catch (const std::exception& e) {
            kiln::log(KILN_LOG_WARN, "decision template failed to parse: %s", e.what());
            return;
        }

        // calibration temperatures: "<type>" or "<type>.<n_options bucket>"
        char arch[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof arch);
        const std::string prefix = std::string(arch) + ".decision.temperature.";
        for (int32_t i = 0; i < llama_model_meta_count(model); ++i) {
            char key[256], val[64];
            if (llama_model_meta_key_by_index(model, i, key, sizeof key) < 0 || strncmp(key, prefix.c_str(), prefix.size()) != 0) continue;
            if (llama_model_meta_val_str_by_index(model, i, val, sizeof val) < 0) continue;
            const float t = std::strtof(val, nullptr);
            if (t > 0.0f) temperatures_[key + prefix.size()] = t;
        }
        n_embd_out_ = llama_model_n_embd_out(model);
        supported_ = true;
    }

    bool is_decision_model() const { return is_decision_; }
    bool supported() const { return supported_; }

    // Context settings the head needs: scores come out of the embeddings output, and the model
    // is not causal, so the whole prompt has to go through in one batch.
    static void configure(llama_context_params& cp) {
        cp.embeddings            = true;
        cp.pooling_type          = LLAMA_POOLING_TYPE_NONE;
        cp.n_batch               = cp.n_ctx;
        cp.n_ubatch              = cp.n_ctx;
        cp.n_outputs_max         = cp.n_batch;
        cp.n_outputs_max_per_seq = 1;
    }

    kiln_status decide(llama_context* ctx, const char* state, const kiln_decision_question& q,
                       float* probs, int32_t probs_cap, kiln_decision_answer& out) const {
        if (!supported_) return kiln::set_error(KILN_ERR_NOT_SUPPORTED, is_decision_ ? "this decision model family is not supported" : "not a decision model");
        const auto t0 = std::chrono::steady_clock::now();

        // ---- options, in the order of the model outputs
        struct Option { std::string key; json description; };
        std::vector<Option> options;
        const char* type_name = "";
        int column = 0;                                   // output column: choice, score, noul
        switch (q.type) {
            case KILN_QUESTION_NOUL: {
                type_name = "noul"; column = 2;
                options = {{"false", json()}, {"true", json()}};
                for (int32_t i = 0; i < q.n_options; ++i) {
                    const std::string k = q.options[i].key ? q.options[i].key : "";
                    if (!q.options[i].description) continue;
                    if (k == "false" || k == "no")       options[0].description = q.options[i].description;
                    else if (k == "true" || k == "yes")  options[1].description = q.options[i].description;
                    else return kiln::set_error(KILN_ERR_INVALID_ARG, "noul option keys must be true/false or yes/no");
                }
                break;
            }
            case KILN_QUESTION_CHOICE:
                type_name = "choice"; column = 0;
                if (q.n_options < 1) return kiln::set_error(KILN_ERR_INVALID_ARG, "a choice question needs at least one option");
                for (int32_t i = 0; i < q.n_options; ++i) {
                    if (!q.options[i].key) return kiln::set_error(KILN_ERR_INVALID_ARG, "choice option key is null");
                    options.push_back({q.options[i].key, q.options[i].description ? json(q.options[i].description) : json()});
                }
                break;
            case KILN_QUESTION_SCORE:
                type_name = "score"; column = 1;
                if (q.n_options < 2 || q.n_options > 10) return kiln::set_error(KILN_ERR_INVALID_ARG, "a score question needs 2 to 10 levels");
                for (int32_t i = 0; i < q.n_options; ++i) {
                    const char* d = q.options[i].description ? q.options[i].description : q.options[i].key;
                    if (!d) return kiln::set_error(KILN_ERR_INVALID_ARG, "score level has no description");
                    options.push_back({std::to_string(i), json(d)});
                }
                break;
            default:
                return kiln::set_error(KILN_ERR_INVALID_ARG, "unknown question type");
        }
        const size_t n = options.size();
        if (n > kMaxOptions) return kiln::set_error(KILN_ERR_INVALID_ARG, "too many options (max 255)");
        if (probs_cap < (int32_t)n) return kiln::set_error(KILN_ERR_INVALID_ARG, "probs is too small: need " + std::to_string(n));
        if (column >= n_embd_out_) return kiln::set_error(KILN_ERR_BACKEND, "decision head has no output for this question type");

        // ---- render the systemone template
        std::string prompt;
        try {
            json opts = json::array();
            for (auto& o : options) opts.push_back(json{{"key", o.key}, {"description", o.description}});
            json inp = json{
                {"id",           q.id},
                {"type",         type_name},
                {"instructions", q.instructions},
                {"state",        state ? json(state) : json()},
                {"options",      opts},
            };
            // user text must not contain special tokens or the piece markers
            inp = replace_text(escape_special(inp), kMarker, "<<d1omni ");
            inp["audio"]         = false;
            inp["sep"]           = kSep;
            inp["mark_state"]    = kMarkState;
            inp["mark_question"] = kMarkQuestion;
            inp["mark_option"]   = kMarkOption;
            inp["images"]        = json::array();

            jinja::context jctx(tmpl_->source());
            jinja::global_from_json(jctx, inp, false);
            jinja::runtime runtime(jctx);
            const jinja::value results = runtime.execute(tmpl_->prog);
            prompt = jinja::runtime::gather_string_parts(results)->as_string().str();
        } catch (const std::exception& e) {
            return kiln::set_error(KILN_ERR_BACKEND, std::string("decision template failed: ") + e.what());
        }

        // ---- tokenize piece by piece: question and options are cut to a budget, the state takes the rest
        const kiln_status bad_layout = KILN_ERR_BACKEND;
        std::vector<std::string> pieces = string_split(prompt, kSep);
        if (pieces.empty() || !pieces[0].empty()) return kiln::set_error(bad_layout, "unexpected layout of the decision prompt");

        const size_t  n_max          = std::min<size_t>(kMaxTokens, llama_n_batch(ctx));
        const int64_t n_options      = (int64_t)n;
        const int64_t n_budget       = std::max<int64_t>(96, std::min<int64_t>(24 * n_options + 32, (int64_t)n_max / 2));
        const int64_t n_option_max   = std::max<int64_t>(2, (n_budget - 3 * n_options) / n_options);
        const int64_t n_question_max = std::max<int64_t>(16, n_budget);

        std::vector<llama_token> head, body, tail;        // before the state, the state, after it
        bool has_state = false;
        for (size_t i = 1; i < pieces.size(); ++i) {
            std::string piece = pieces[i];
            int64_t piece_max = -1;
            bool is_state = false;
            if (piece.rfind(kMarkState, 0) == 0)         { piece = piece.substr(strlen(kMarkState)); is_state = true; }
            else if (piece.rfind(kMarkQuestion, 0) == 0) { piece = piece.substr(strlen(kMarkQuestion)); piece_max = n_question_max; }
            else if (piece.rfind(kMarkOption, 0) == 0)   { piece = piece.substr(strlen(kMarkOption)); piece_max = n_option_max; }

            std::vector<llama_token> toks = common_tokenize(vocab_, piece, false, true);
            if (piece_max >= 0 && (int64_t)toks.size() > piece_max) toks.resize(piece_max);
            if (is_state) {
                if (has_state) return kiln::set_error(bad_layout, "unexpected layout of the decision prompt");
                body = std::move(toks); has_state = true;
            } else {
                auto& dst = has_state ? tail : head;
                dst.insert(dst.end(), toks.begin(), toks.end());
            }
        }
        if (!has_state) return kiln::set_error(bad_layout, "unexpected layout of the decision prompt");

        const size_t n_room = n_max - std::min(n_max, head.size() + tail.size());
        body.resize(std::min(body.size(), n_room));
        std::vector<llama_token> tokens = std::move(head);
        tokens.insert(tokens.end(), body.begin(), body.end());
        tokens.insert(tokens.end(), tail.begin(), tail.end());
        tokens.resize(std::min(tokens.size(), n_max));

        std::vector<int32_t> markers;                     // one mask token per option
        for (size_t i = 0; i < tokens.size(); ++i) if (tokens[i] == marker_) markers.push_back((int32_t)i);
        if (markers.size() != n) return kiln::set_error(KILN_ERR_CONTEXT_FULL, "the options do not fit in the context");

        // ---- one forward pass, every token an output
        if (llama_memory_t mem = llama_get_memory(ctx)) llama_memory_clear(mem, true);
        llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = (llama_pos)i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
        }
        batch.n_tokens = (int32_t)tokens.size();
        const int rc = llama_decode(ctx, batch);
        std::vector<float> scores(n);
        bool ok = rc == 0;
        for (size_t i = 0; ok && i < n; ++i) {
            const float* e = llama_get_embeddings_ith(ctx, markers[i]);
            if (!e || std::isnan(e[column])) ok = false; else scores[i] = e[column];
        }
        llama_batch_free(batch);
        if (rc == 2) return KILN_ERR_CANCELLED;
        if (!ok) return kiln::set_error(KILN_ERR_BACKEND, "decision pass failed, rc=" + std::to_string(rc));

        // ---- calibrated softmax over the options
        const float temperature = temperature_for(type_name, n);
        const float smax = *std::max_element(scores.begin(), scores.end());
        std::vector<double> p(n);
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) { p[i] = std::exp((double)(scores[i] - smax) / temperature); sum += p[i]; }
        for (size_t i = 0; i < n; ++i) { p[i] /= sum; probs[i] = (float)p[i]; }

        const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        out.n_probs = (int32_t)n;
        out.best = (int32_t)best;
        out.prompt_tokens = (int32_t)tokens.size();
        if (q.type == KILN_QUESTION_NOUL) {
            out.value = (float)p[1];
            out.confidence = (float)std::fabs(2.0 * p[1] - 1.0);
        } else if (q.type == KILN_QUESTION_CHOICE) {
            out.value = (float)p[best];
            const double uniform = 1.0 / n;
            out.confidence = n < 2 ? 1.0f : (float)std::max(0.0, (p[best] - uniform) / (1.0 - uniform));
        } else {
            // expected level, and how tightly the distribution sits around its mode
            double expected = 0.0, dist = 0.0, dist_uniform = 0.0;
            for (size_t i = 0; i < n; ++i) {
                expected     += i * p[i];
                dist         += p[i] * std::fabs((double)i - (double)best);
                dist_uniform += std::fabs((double)i - (n - 1) / 2.0) / n;
            }
            out.value = (float)expected;
            out.confidence = (float)std::max(0.0, 1.0 - dist / dist_uniform);
        }
        out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return KILN_OK;
    }

private:
    // text given to the template: between the pieces of the prompt, and at the start of the
    // pieces that are cut to a token budget
    static constexpr const char* kMarker       = "<<d1omni:";
    static constexpr const char* kSep          = "<<d1omni:sep>>";
    static constexpr const char* kMarkState    = "<<d1omni:state>>";
    static constexpr const char* kMarkQuestion = "<<d1omni:question>>";
    static constexpr const char* kMarkOption   = "<<d1omni:option>>";
    static constexpr size_t      kMaxTokens    = 16384;   // max_length of the model config
    static constexpr size_t      kMaxOptions   = 255;

    // <|name|> in user text would tokenize as a special token
    static json escape_special(const json& val) {
        static const std::regex re("<\\|([A-Za-z0-9_]+)\\|>");
        const char* fmt = "<\xC2\xA6$1\xC2\xA6>";
        if (val.is_string()) return std::regex_replace(val.get<std::string>(), re, fmt);
        if (val.is_array())  { json o = json::array();  for (const auto& v : val) o.push_back(escape_special(v)); return o; }
        if (val.is_object()) { json o = json::object(); for (const auto& [k, v] : val.items()) o[std::regex_replace(k, re, fmt)] = escape_special(v); return o; }
        return val;
    }

    static json replace_text(const json& val, const std::string& search, const std::string& replace) {
        if (val.is_string()) { std::string s = val.get<std::string>(); string_replace_all(s, search, replace); return s; }
        if (val.is_array())  { json o = json::array();  for (const auto& v : val) o.push_back(replace_text(v, search, replace)); return o; }
        if (val.is_object()) { json o = json::object(); for (const auto& [k, v] : val.items()) o[k] = replace_text(v, search, replace); return o; }
        return val;
    }

    // the temperature can depend on the number of options; the buckets are the ones used to fit it
    float temperature_for(const std::string& type_name, size_t n) const {
        const std::string bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
        for (const auto& name : {type_name + "." + bucket, type_name}) {
            const auto it = temperatures_.find(name);
            if (it != temperatures_.end()) return it->second;
        }
        return 1.0f;
    }

    bool                                        is_decision_ = false;
    bool                                        supported_   = false;
    const llama_vocab*                          vocab_       = nullptr;
    std::shared_ptr<const common_chat_template> tmpl_;
    llama_token                                 marker_      = LLAMA_TOKEN_NULL;
    int32_t                                     n_embd_out_  = 0;
    std::map<std::string, float>                temperatures_;
};

} // namespace kiln
