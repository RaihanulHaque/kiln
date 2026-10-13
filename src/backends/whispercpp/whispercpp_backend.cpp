// whisper.cpp backend: the Whisper and Parakeet speech models, through the two libraries that
// ship in the whisper.cpp repository (libwhisper and libparakeet). Both run on the same ggml
// that llama.cpp uses, so libkiln holds one copy of the maths.
//
// These models are not "text model plus audio encoder" like Qwen3-ASR; each is a complete
// speech model with its own decoder, which llama.cpp has no code for. They have no chat, no
// decisions and no embeddings: every call but kiln_transcribe answers KILN_ERR_NOT_SUPPORTED.
// The file format is ggml's older one ("lmgg" magic, not GGUF); can_load() tells them apart.
#include "../../core/internal.h"
#include "../../probe/fast_cores.h"
#include "../llamacpp/media.h"   // AffinityScope

#include "whisper.h"
#include "parakeet.h"
#include "ggml.h"

// Reads wav / mp3 / flac files into 16 kHz mono samples. The same header llama.cpp's mtmd
// compiles for kiln_embed_media; every symbol is static here, so the two copies do not meet.
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
// the two symbols MA_API does not cover (the second exists on Android only)
#define ma_atomic_global_lock  kiln_whispercpp_ma_atomic_global_lock
#define ma_android_sdk_version kiln_whispercpp_ma_android_sdk_version
#include "miniaudio/miniaudio.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

using clock_t_ = std::chrono::steady_clock;
double ms_since(clock_t_::time_point t0) { return std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count(); }

constexpr int      kSampleRate = 16000;          // both engines want 16 kHz mono
constexpr uint32_t kGgmlMagic  = 0x67676d6c;     // "lmgg": ggml's pre-GGUF file header

enum class Family { whisper, parakeet };

// What the first twelve bytes of the file say: ggml magic, then n_vocab. Whisper's vocabulary
// has 51864 to 51866 entries; Parakeet's has 1024 or 8192.
bool sniff(const char* path, bool& is_ggml, Family& fam) {
    uint32_t head[3] = {0, 0, 0};
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    const size_t n = fread(head, sizeof(uint32_t), 3, f);
    fclose(f);
    is_ggml = n >= 1 && head[0] == kGgmlMagic;
    fam = (n >= 2 && head[1] >= 50000) ? Family::whisper : Family::parakeet;
    return true;
}

bool read_file(const char* path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return false; }
    out.resize((size_t)n);
    const bool ok = fread(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok;
}

bool decode_audio(const uint8_t* buf, size_t len, std::vector<float>& pcm) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, kSampleRate);
    ma_decoder dec;
    if (ma_decoder_init_memory(buf, len, &cfg, &dec) != MA_SUCCESS) return false;
    ma_uint64 frames = 0;
    if (ma_decoder_get_length_in_pcm_frames(&dec, &frames) != MA_SUCCESS || frames == 0) { ma_decoder_uninit(&dec); return false; }
    pcm.resize((size_t)frames);
    ma_uint64 got = 0;
    ma_decoder_read_pcm_frames(&dec, pcm.data(), frames, &got);
    ma_decoder_uninit(&dec);
    pcm.resize((size_t)got);
    return got > 0;
}

const char* ftype_name(int ftype) {
    switch (ftype) {
        case GGML_FTYPE_ALL_F32:     return "F32";
        case GGML_FTYPE_MOSTLY_F16:  return "F16";
        case GGML_FTYPE_MOSTLY_BF16: return "BF16";
        case GGML_FTYPE_MOSTLY_Q4_0: return "Q4_0";
        case GGML_FTYPE_MOSTLY_Q4_1: return "Q4_1";
        case GGML_FTYPE_MOSTLY_Q5_0: return "Q5_0";
        case GGML_FTYPE_MOSTLY_Q5_1: return "Q5_1";
        case GGML_FTYPE_MOSTLY_Q8_0: return "Q8_0";
        case GGML_FTYPE_MOSTLY_Q2_K: return "Q2_K";
        case GGML_FTYPE_MOSTLY_Q3_K: return "Q3_K";
        case GGML_FTYPE_MOSTLY_Q4_K: return "Q4_K";
        case GGML_FTYPE_MOSTLY_Q5_K: return "Q5_K";
        case GGML_FTYPE_MOSTLY_Q6_K: return "Q6_K";
        default: return "?";
    }
}

void ggml_log_bridge(ggml_log_level level, const char* text, void*) {
    kiln_log_level l = KILN_LOG_DEBUG;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: l = KILN_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  l = KILN_LOG_WARN;  break;
        default: break;                                  // the engines narrate every load at INFO; keep that at DEBUG
    }
    std::string s = text ? text : "";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (!s.empty()) kiln::log(l, "whisper.cpp: %s", s.c_str());
}

// Hands each finished piece of text to the caller's callback, in order, and remembers whether
// the caller asked to stop. The first piece loses its leading spaces: Whisper writes " The".
struct Streamer {
    kiln_token_cb cb; void* user;
    bool started = false, stopped = false;
    void emit(const char* text) {
        if (!text || stopped) return;
        if (!started) {
            while (*text == ' ' || *text == '\n') ++text;
            if (!*text) return;
            started = true;
        }
        if (cb && !cb(KILN_CHANNEL_ANSWER, text, -1, user)) stopped = true;
    }
};

class SpeechModel;

class SpeechContext final : public kiln::Context {
public:
    explicit SpeechContext(SpeechModel& m) : m_(m) {}
    kiln_status reset() override { return KILN_OK; }
    void cancel() override { cancel_.store(true); }
    kiln_status generate(const kiln::RenderedPrompt&, const kiln_sampling_params&, kiln_token_cb, void*, kiln_gen_stats*) override {
        return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "speech models do not generate text; use kiln_transcribe");
    }
    kiln_status decide(const char*, const kiln_decision_question&, float*, int32_t, kiln_decision_answer&) override {
        return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not a decision model");
    }
    kiln_status embed(const char*, const kiln_media*, int32_t, const kiln_embed_options&, float*, int32_t, kiln_embed_stats&) override {
        return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "not an embedding model");
    }
    kiln_status transcribe(const kiln::RenderedPrompt& in, const kiln_transcribe_options& o, const kiln_media& audio,
                           kiln_token_cb cb, void* user, kiln_transcribe_stats& st) override;

    // Checked by the engine before every piece of work: kiln_cancel, or the caller's callback
    // having answered "stop", ends the run at the next check.
    static bool abort_cb(void* self) {
        auto* c = static_cast<SpeechContext*>(self);
        return c->cancel_.load() || (c->out_ && c->out_->stopped);
    }

private:
    SpeechModel&      m_;
    std::atomic<bool> cancel_{false};
    Streamer*         out_ = nullptr;      // the streamer of the call in progress
};

class SpeechModel final : public kiln::Model {
public:
    SpeechModel(Family fam, whisper_context* w, parakeet_context* p, std::string backend_name, kiln_backend_kind kind,
                int threads, uint64_t cpu_mask, uint64_t file_bytes)
        : fam_(fam), w_(w), p_(p), backend_name_(std::move(backend_name)), kind_(kind), threads_(threads), cpu_mask_(cpu_mask), file_bytes_(file_bytes) {}
    ~SpeechModel() override {
        if (w_) whisper_free(w_);
        if (p_) parakeet_free(p_);
    }

    kiln_status info(kiln_model_info& out) const override {
        memset(&out, 0, sizeof out);
        out.backend_used = kind_;
        strncpy(out.backend_name, backend_name_.c_str(), sizeof out.backend_name - 1);
        strncpy(out.arch, fam_ == Family::whisper ? "whisper" : "parakeet", sizeof out.arch - 1);
        // Parakeet reports the raw header value, which carries ggml's quantisation version in the thousands
        strncpy(out.quant, ftype_name((fam_ == Family::whisper ? whisper_model_ftype(w_) : parakeet_model_ftype(p_)) % 1000), sizeof out.quant - 1);
        out.n_vocab         = fam_ == Family::whisper ? whisper_model_n_vocab(w_) : parakeet_model_n_vocab(p_);
        out.n_ctx_train     = fam_ == Family::whisper ? whisper_model_n_audio_ctx(w_) : parakeet_model_n_audio_ctx(p_);
        out.model_bytes     = file_bytes_;   // the engines keep the whole file in memory, mapped or copied
        out.n_gpu_layers_used = kind_ == KILN_BACKEND_GPU ? -1 : 0;
        out.is_speech_model = true;
        return KILN_OK;
    }

    int32_t tokenize(const char* text, bool, int32_t* out, int32_t cap) const override {
        if (fam_ == Family::whisper) return whisper_tokenize(w_, text, out, cap);
        return parakeet_tokenize(p_, text, out, cap);
    }
    int32_t detokenize(const int32_t* toks, int32_t n, char* out, int32_t cap) const override {
        std::string s;
        for (int32_t i = 0; i < n; ++i) {
            const char* piece = fam_ == Family::whisper ? whisper_token_to_str(w_, toks[i]) : parakeet_token_to_str(p_, toks[i]);
            if (piece) s += piece;
        }
        if ((int32_t)s.size() + 1 > cap) return -(int32_t)s.size() - 1;
        memcpy(out, s.c_str(), s.size() + 1);
        return (int32_t)s.size();
    }
    kiln_status render_chat(const kiln_chat_message*, int32_t, const kiln_chat_options&, kiln::RenderedPrompt&) const override {
        return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "speech models do not generate text; use kiln_transcribe");
    }
    kiln_status render_transcribe(const kiln_transcribe_options& o, kiln::RenderedPrompt& out) const override {
        out.prompt = o.prompt ? o.prompt : "";       // Whisper reads it as text the words follow on from
        return KILN_OK;
    }
    kiln_status create_context(const kiln_context_params&, std::unique_ptr<kiln::Context>& out) override {
        out.reset(new SpeechContext(*this));
        return KILN_OK;
    }
    kiln_status load_projector(const char*, const kiln_projector_params&, kiln_projector_info&) override {
        return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "this speech model has its audio encoder built in; there is no projector to load");
    }

    // The engine keeps one working state per model, so calls from different contexts take turns.
    std::mutex&       mutex() { return mutex_; }
    Family            family() const { return fam_; }
    whisper_context*  whisper() const { return w_; }
    parakeet_context* parakeet() const { return p_; }
    int               threads() const { return threads_; }
    uint64_t          cpu_mask() const { return cpu_mask_; }   // the cores the engine's threads may use; 0 = any

private:
    Family            fam_;
    whisper_context*  w_;
    parakeet_context* p_;
    std::string       backend_name_;
    kiln_backend_kind kind_;
    int               threads_;
    uint64_t          cpu_mask_;
    uint64_t          file_bytes_;
    std::mutex        mutex_;
};

kiln_status SpeechContext::transcribe(const kiln::RenderedPrompt& in, const kiln_transcribe_options& o, const kiln_media& audio,
                                      kiln_token_cb cb, void* user, kiln_transcribe_stats& st) {
    cancel_.store(false);
    const auto t_all = clock_t_::now();

    std::vector<uint8_t> file;
    const uint8_t* bytes = audio.data;
    size_t len = audio.size;
    if (audio.path) {
        if (!read_file(audio.path, file)) return kiln::set_error(KILN_ERR_INVALID_ARG, std::string("could not read ") + audio.path);
        bytes = file.data(); len = file.size();
    }
    std::vector<float> pcm;
    if (!decode_audio(bytes, len, pcm))
        return kiln::set_error(KILN_ERR_INVALID_ARG, "media item 0 could not be read as an audio file (wav, mp3 or flac)");
    st.audio_seconds = (double)pcm.size() / kSampleRate;

    Streamer out{cb, user};
    std::lock_guard<std::mutex> lock(m_.mutex());
    out_ = &out;
    struct Clear { Streamer*& p; ~Clear() { p = nullptr; } } clear{out_};
    // The engine starts its own threads for every graph it computes, and new threads inherit
    // this thread's core set. Left alone they land on any core, little ones included, and
    // every operation then waits for the slowest thread: Whisper base took 3.8 s for a
    // 3-second clip on a Pixel 8. Widened to the fast cores it is several times quicker.
    kiln::AffinityScope on_fast_cores(m_.cpu_mask());
    int rc = 0;
    double encode_ms = 0;

    if (m_.family() == Family::whisper) {
        whisper_context* w = m_.whisper();
        whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        p.n_threads       = m_.threads();
        p.no_timestamps   = true;
        p.print_progress  = false;
        p.print_realtime  = false;
        p.print_timestamps = false;
        p.print_special   = false;
        p.max_tokens      = o.max_tokens > 0 ? o.max_tokens : 0;
        // No temperature fallback: whisper.cpp otherwise retries a window it is unsure about up
        // to five times with sampled (random) candidates, which is slow on a phone (a pure tone
        // took minutes on a Pixel 8) and would make the same clip give different texts.
        p.temperature_inc = 0.0f;
        p.greedy.best_of  = 1;
        p.initial_prompt  = in.prompt.empty() ? nullptr : in.prompt.c_str();
        std::string lang;
        if (o.language && *o.language) {
            lang = o.language;
            for (char& ch : lang) ch = (char)tolower((unsigned char)ch);
            if (whisper_lang_id(lang.c_str()) < 0) return kiln::set_error(KILN_ERR_INVALID_ARG, std::string("Whisper does not know the language '") + o.language + "'");
            if (!whisper_is_multilingual(w) && lang != "en" && lang != "english")
                return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "this Whisper file is English only");
            p.language = lang.c_str();
        } else {
            p.language = whisper_is_multilingual(w) ? "auto" : "en";
        }
        p.new_segment_callback = [](whisper_context* c, whisper_state* s, int n_new, void* self) {
            auto* o_ = static_cast<Streamer*>(self);
            const int n = whisper_full_n_segments_from_state(s);
            for (int i = n - n_new; i < n; ++i) o_->emit(whisper_full_get_segment_text_from_state(s, i));
            (void)c;
        };
        p.new_segment_callback_user_data = &out;
        p.abort_callback = abort_cb;
        p.abort_callback_user_data = this;

        whisper_reset_timings(w);
        rc = whisper_full(w, p, pcm.data(), (int)pcm.size());
        if (rc == 0) {
            const int n_seg = whisper_full_n_segments(w);
            for (int i = 0; i < n_seg; ++i) st.generated_tokens += whisper_full_n_tokens(w, i);
            const int lang_id = whisper_full_lang_id(w);
            if (lang_id >= 0) {
                std::string name = whisper_lang_str_full(lang_id);
                if (!name.empty()) name[0] = (char)toupper((unsigned char)name[0]);
                strncpy(st.language, name.c_str(), sizeof st.language - 1);
            }
            if (whisper_timings* t = whisper_get_timings(w)) {
                encode_ms = t->encode_ms * std::ceil(st.audio_seconds / 30.0);   // the figure is per 30-second window
                delete t;
            }
        }
        st.audio_tokens = (int32_t)std::lround(st.audio_seconds * 50);   // one encoder position per 20 ms
    } else {
        parakeet_context* pk = m_.parakeet();
        parakeet_full_params p = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
        p.n_threads = m_.threads();
        p.new_segment_callback = [](parakeet_context*, parakeet_state* s, int n_new, void* self) {
            auto* o_ = static_cast<Streamer*>(self);
            const int n = parakeet_full_n_segments_from_state(s);
            for (int i = n - n_new; i < n; ++i) o_->emit(parakeet_full_get_segment_text_from_state(s, i));
        };
        p.new_segment_callback_user_data = &out;
        p.abort_callback = abort_cb;
        p.abort_callback_user_data = this;
        parakeet_reset_timings(pk);
        rc = parakeet_full(pk, p, pcm.data(), (int)pcm.size());
        if (rc == 0) {
            const int n_seg = parakeet_full_n_segments(pk);
            for (int i = 0; i < n_seg; ++i) st.generated_tokens += parakeet_full_n_tokens(pk, i);
            if (parakeet_timings* t = parakeet_get_timings(pk)) { encode_ms = t->encode_ms; delete t; }
        }
        st.audio_tokens = (int32_t)std::lround(st.audio_seconds * 12.5);  // one encoder frame per 80 ms
    }

    st.prompt_tokens   = st.audio_tokens;        // no instruction goes in; the audio is the whole prompt
    st.ms              = ms_since(t_all);
    st.encode_ms       = encode_ms;
    st.gen_ms          = st.ms > encode_ms ? st.ms - encode_ms : 0;   // reading the file and the spectrogram are in here too
    st.stopped_by_user = out.stopped;
    if (out.stopped) { st.stopped_by_eos = false; return KILN_OK; }
    if (cancel_.load()) return KILN_ERR_CANCELLED;
    if (rc != 0) return kiln::set_error(KILN_ERR_BACKEND, std::string(m_.family() == Family::whisper ? "whisper_full" : "parakeet_full") + " failed, rc=" + std::to_string(rc));
    st.stopped_by_eos = true;
    return KILN_OK;
}

class WhisperBackend final : public kiln::Backend {
public:
    const char* name() const override { return "whisper.cpp"; }
    void annotate_device(kiln_device_info&) const override {}   // the llama.cpp backend reports the GPU; it is the same ggml

    bool supports(kiln_backend_kind kind, const kiln_device_info& d) const override {
        switch (kind) {
            case KILN_BACKEND_AUTO:
            case KILN_BACKEND_CPU: return true;
            case KILN_BACKEND_GPU: return d.has_gpu;
            default: return false;
        }
    }

    bool can_load(const char* path) const override {
        bool is_ggml = false; Family fam;
        return sniff(path, is_ggml, fam) && is_ggml;
    }

    kiln_status load_model(const char* path, const kiln_model_params& p,
                           const kiln_device_info& d, std::unique_ptr<kiln::Model>& out) override {
        static std::once_flag once;
        std::call_once(once, [] { whisper_log_set(ggml_log_bridge, nullptr); parakeet_log_set(ggml_log_bridge, nullptr); });

        bool is_ggml = false; Family fam = Family::whisper;
        if (!sniff(path, is_ggml, fam) || !is_ggml) return kiln::set_error(KILN_ERR_MODEL_LOAD, std::string("failed to load ") + path + " (not a whisper.cpp model file)");

        bool want_gpu = false;
        if (p.backend == KILN_BACKEND_GPU) want_gpu = true;
        else if (p.backend == KILN_BACKEND_AUTO) {
#if defined(__APPLE__)
            want_gpu = d.has_gpu;              // Metal, as for the llama.cpp backend
#endif
        }
        if (want_gpu && !d.has_gpu) return kiln::set_error(KILN_ERR_NO_BACKEND, "GPU requested but no ggml GPU device present");

        uint64_t file_bytes = 0;
        if (FILE* f = fopen(path, "rb")) { fseek(f, 0, SEEK_END); file_bytes = (uint64_t)ftell(f); fclose(f); }

        whisper_context* w = nullptr; parakeet_context* pk = nullptr;
        if (fam == Family::whisper) {
            whisper_context_params cp = whisper_context_default_params();
            cp.use_gpu = want_gpu;
            w = whisper_init_from_file_with_params(path, cp);
            if (!w) return kiln::set_error(KILN_ERR_MODEL_LOAD, std::string("failed to load ") + path + " as a Whisper model (missing, corrupt, or another format)");
        } else {
            parakeet_context_params cp = parakeet_context_default_params();
            cp.use_gpu = want_gpu;
            pk = parakeet_init_from_file_with_params(path, cp);
            if (!pk) return kiln::set_error(KILN_ERR_MODEL_LOAD, std::string("failed to load ") + path + " as a Parakeet model (missing, corrupt, or another format)");
        }
        const kiln_backend_kind kind = want_gpu ? KILN_BACKEND_GPU : KILN_BACKEND_CPU;
        const std::string bname = std::string("whisper.cpp/") + (want_gpu ? "gpu" : "cpu");
        // same core set as the llama.cpp backend: the probe's fast cores, KILN_CPU_MASK to force
        // one, KILN_NO_AFFINITY=1 for none (then the thread count follows the core count)
        uint64_t mask = d.cpu_fast_mask;
        if (const char* m = getenv("KILN_CPU_MASK"); m && *m) mask = strtoull(m, nullptr, 16);
        if (const char* no = getenv("KILN_NO_AFFINITY"); no && *no && *no != '0') mask = 0;
        int threads = p.n_threads > 0 ? p.n_threads
                    : mask ? __builtin_popcountll(mask)
                    : kiln::default_thread_count(d.cpu_cores, d.cpu_big_cores, false);
        if (threads > 8) threads = 8;
        kiln::log(KILN_LOG_INFO, "loaded %s via %s (%s), threads=%d, cores=0x%llx", path, bname.c_str(), fam == Family::whisper ? "whisper" : "parakeet", threads, (unsigned long long)mask);
        out.reset(new SpeechModel(fam, w, pk, bname, kind, threads, mask, file_bytes));
        return KILN_OK;
    }
};

} // namespace

namespace kiln {
Backend* make_whispercpp_backend() { static WhisperBackend b; return &b; }
}
