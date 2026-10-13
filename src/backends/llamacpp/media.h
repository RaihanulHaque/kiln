// Images and audio, shared by embedding models (kiln_embed_media) and speech models
// (kiln_transcribe). Both go through llama.cpp's mtmd library: an encoder (the projector file,
// an "mmproj" GGUF) turns an image or an audio clip into vectors of the same kind the model
// uses for text tokens, and they are fed to the model next to the text.
#pragma once
#include "../../core/internal.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

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

// The image / audio encoders of one model, loaded on request and owned by the model.
class MediaProjector {
public:
    ~MediaProjector() { free(); }
    void free() { if (mctx_) { mtmd_free(mctx_); mctx_ = nullptr; } }

    // Hold this around every use of ctx(): the encoders are not thread-safe.
    std::mutex&   mutex() const { return mutex_; }
    mtmd_context* ctx() const { return mctx_; }
    uint64_t      cpu_mask() const { return cpu_mask_; }   // cores the encoders may use; 0 = not restricted

    // cpu_mask: the cores the encoders may use (0 = leave thread placement to the system)
    kiln_status load(const llama_model* model, const char* path, const kiln_projector_params& p, bool use_gpu, int n_threads, uint64_t cpu_mask, kiln_projector_info& out) {
        std::lock_guard<std::mutex> lock(mutex_);
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
        cpu_mask_ = cpu_mask;
        return KILN_OK;
    }

    // Decoded images (pixels) and audio clips (samples); mtmd calls both "bitmaps".
    struct Bitmaps {
        std::vector<mtmd_bitmap*> v;
        ~Bitmaps() { for (auto* b : v) mtmd_bitmap_free(b); }
    };

    // Reads and decodes each item. The caller holds mutex() and has checked that ctx() is set.
    kiln_status read(const kiln_media* media, int32_t n_media, Bitmaps& out) const {
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
            out.v.push_back(w.bitmap);
            const bool audio = mtmd_bitmap_is_audio(w.bitmap);
            if (audio ? !mtmd_support_audio(mctx_) : !mtmd_support_vision(mctx_))
                return kiln::set_error(KILN_ERR_NOT_SUPPORTED, std::string("the loaded projector does not take ") + (audio ? "audio" : "images"));
        }
        return KILN_OK;
    }

private:
    static constexpr int kDefaultImageTokens = 280;

    mtmd_context*      mctx_ = nullptr;
    uint64_t           cpu_mask_ = 0;
    mutable std::mutex mutex_;
};

} // namespace kiln
