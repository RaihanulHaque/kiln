// OCR backend: OCRia-nano-1 (LlamaIndex) on ggml. Reads the printed text in an image.
//
// The model is a pipeline of small networks, converted from ONNX by scripts/convert_ocria.py
// into one GGUF ("ocria" architecture), which this backend alone reads:
//   1. the detector turns the page into a map of "text here" probabilities; connected regions
//      of that map become line boxes (rotated rectangles, grown back to the full line height);
//   2. each line is cut out and straightened (a perspective warp), and the router guesses its
//      script (Latin, CJK, right-to-left, other) and whether the page is upside down;
//   3. the recognizer of that script reads the line: for every 4 (or 8) pixels of width it
//      scores each character, and the best character per step, with repeats and blanks
//      removed, is the text (CTC decoding).
// The steps around the networks are a port of LiteParse's ocria engine (crates/liteparse/src/
// ocr/ocria, Copyright LlamaIndex, Apache License 2.0), its benchmarked settings included.
// Not ported yet: the second detection pass for sideways pages, column-gutter splitting, the
// re-read of low-confidence Latin pages with other recognizers ("rescue"), faint single
// characters, look-alike letter folding and right-to-left reordering.
#include "../../core/internal.h"
#include "../../probe/fast_cores.h"
#include "../llamacpp/media.h"   // AffinityScope
#include "ocr_graph.h"
#include "ocr_image.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb/stb_image.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string>
#include <vector>

namespace {

using namespace kiln::ocr;
using clock_t_ = std::chrono::steady_clock;
double ms_since(clock_t_::time_point t0) { return std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count(); }

constexpr const char* kArch = "ocria";
constexpr const char* kGroups[4] = {"latin", "cjk", "rtl", "world"};

// detector
constexpr float kShrink   = 0.3f;     // kernel shrink ratio the detector's targets were built with
constexpr float kMinScore = 0.6f;     // mean probability a region needs to be a line
// recognizer
constexpr size_t kH       = 32;       // line height the recognizers and the router read
constexpr size_t kMaxW    = 1280;
constexpr size_t kBucketW = 128;      // batch widths are padded to a multiple of this
constexpr size_t kBatch   = 32;
constexpr size_t kRouterW = 512;
constexpr size_t kRouterBatch = 64;

float norm(Pt v) { return std::sqrt(v[0] * v[0] + v[1] * v[1]); }
float min_x(const Quad& q) { float m = q[0][0]; for (const auto& p : q) m = std::min(m, p[0]); return m; }
float min_y(const Quad& q) { float m = q[0][1]; for (const auto& p : q) m = std::min(m, p[1]); return m; }

// corners as [top-left, top-right, bottom-right, bottom-left] along the line's long axis
Quad order_quad(const Quad& q) {
    const Pt c = {(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.f, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.f};
    const Pt e1 = {q[1][0] - q[0][0], q[1][1] - q[0][1]};
    const Pt e2 = {q[2][0] - q[1][0], q[2][1] - q[1][1]};
    Pt u = norm(e1) >= norm(e2) ? e1 : e2;
    const float n = norm(u) + 1e-6f;
    u = {u[0] / n, u[1] / n};
    if (u[0] < 0.f) u = {-u[0], -u[1]};
    const Pt v = {-u[1], u[0]};
    float a[4], b[4];
    for (int i = 0; i < 4; ++i) {
        a[i] = (q[i][0] - c[0]) * u[0] + (q[i][1] - c[1]) * u[1];
        b[i] = (q[i][0] - c[0]) * v[0] + (q[i][1] - c[1]) * v[1];
    }
    auto pick = [&](auto f, bool max) {
        int best = 0;
        for (int i = 1; i < 4; ++i)
            if ((max && f(i) > f(best)) || (!max && f(i) < f(best))) best = i;
        return best;
    };
    const int tl = pick([&](int i) { return a[i] + b[i]; }, false);
    const int tr = pick([&](int i) { return a[i] - b[i]; }, true);
    const int br = pick([&](int i) { return a[i] + b[i]; }, true);
    const int bl = pick([&](int i) { return a[i] - b[i]; }, false);
    return {q[tl], q[tr], q[br], q[bl]};
}

// Join near-horizontal line boxes that continue one another: each box links to its nearest
// qualifying right neighbour when that neighbour's nearest left neighbour is the same box, and
// each chain becomes one rectangle along its longest member's axis.
std::vector<Quad> merge_collinear(const std::vector<Quad>& boxes, float gap) {
    constexpr float kMaxTilt = 0.35f, kMaxPerp = 0.35f, kMinHRatio = 0.6f, kMaxDAngle = 0.07f;
    const size_t n = boxes.size();
    if (n < 2) return boxes;
    std::vector<Pt> c(n), u(n), v(n);
    std::vector<float> len(n), hgt(n);
    std::vector<bool> flat(n);
    for (size_t i = 0; i < n; ++i) {
        const Quad& q = boxes[i];
        c[i] = {(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.f, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.f};
        const Pt d = {q[1][0] - q[0][0], q[1][1] - q[0][1]};
        const float l = std::max(norm(d), 1e-3f);
        u[i] = {d[0] / l, d[1] / l};
        len[i] = l;
        hgt[i] = std::max(norm({q[3][0] - q[0][0], q[3][1] - q[0][1]}), 1e-3f);
        v[i] = {-u[i][1], u[i][0]};
        flat[i] = std::fabs(u[i][1]) <= kMaxTilt && len[i] >= 0.5f * hgt[i];
    }
    const float cos_min = 1.f - kMaxDAngle * kMaxDAngle / 2.f;
    std::vector<long> right(n, -1);
    std::vector<std::pair<float, long>> left_best(n, {INFINITY, -1});
    for (size_t a = 0; a < n; ++a) {
        std::pair<float, long> best = {INFINITY, -1};
        if (flat[a]) {
            for (size_t b = 0; b < n; ++b) {
                if (a == b || !flat[b]) continue;
                const Pt d = {c[b][0] - c[a][0], c[b][1] - c[a][1]};
                const float along = d[0] * u[a][0] + d[1] * u[a][1];
                const float perp = std::fabs(d[0] * v[a][0] + d[1] * v[a][1]);
                const float hm = (hgt[a] + hgt[b]) / 2.f;
                const float g = along - (len[a] + len[b]) / 2.f;
                const bool ok = along > 0.f
                    && std::min(hgt[a], hgt[b]) / std::max(hgt[a], hgt[b]) >= kMinHRatio
                    && std::fabs(u[a][0] * u[b][0] + u[a][1] * u[b][1]) >= cos_min
                    && perp <= kMaxPerp * hm
                    && g <= gap * hm
                    && g >= -0.5f * std::min(len[a], len[b]);
                if (ok) {
                    if (g < best.first) best = {g, (long)b};
                    if (g < left_best[b].first) left_best[b] = {g, (long)a};
                }
            }
        }
        right[a] = best.second;
    }
    std::vector<long> nxt(n, -1);
    for (size_t a = 0; a < n; ++a) {
        const long b = right[a];
        if (b >= 0 && left_best[(size_t)b].second == (long)a) nxt[a] = b;
    }
    std::vector<bool> has_prev(n, false), seen(n, false);
    for (long b : nxt) if (b >= 0) has_prev[(size_t)b] = true;
    std::vector<Quad> out;
    for (size_t a = 0; a < n; ++a) {
        if (has_prev[a] || seen[a]) continue;
        std::vector<size_t> chain = {a};
        seen[a] = true;
        for (;;) {
            const long nx = nxt[chain.back()];
            if (nx < 0 || seen[(size_t)nx]) break;
            chain.push_back((size_t)nx);
            seen[(size_t)nx] = true;
        }
        if (chain.size() == 1) { out.push_back(boxes[a]); continue; }
        std::vector<Pt> pts;
        for (size_t i : chain) for (const auto& p : boxes[i]) pts.push_back(p);
        size_t k = chain[0];
        for (size_t j = 1; j < chain.size(); ++j) if (len[chain[j]] > len[k]) k = chain[j];
        const float np = (float)pts.size();
        Pt ca = {0.f, 0.f};
        for (const auto& p : pts) { ca[0] += p[0]; ca[1] += p[1]; }
        ca = {ca[0] / np, ca[1] / np};
        const Pt ua = u[k], va = v[k];
        float a0 = INFINITY, a1 = -INFINITY, b0 = INFINITY, b1 = -INFINITY;
        for (const auto& p : pts) {
            const float dx = p[0] - ca[0], dy = p[1] - ca[1];
            const float pa = dx * ua[0] + dy * ua[1], pb = dx * va[0] + dy * va[1];
            a0 = std::min(a0, pa); a1 = std::max(a1, pa);
            b0 = std::min(b0, pb); b1 = std::max(b1, pb);
        }
        auto at = [&](float s, float t) { return Pt{ca[0] + ua[0] * s + va[0] * t, ca[1] + ua[1] * s + va[1] * t}; };
        out.push_back(order_quad({at(a0, b0), at(a1, b0), at(a1, b1), at(a0, b1)}));
    }
    for (size_t i = 0; i < n; ++i) if (!seen[i]) out.push_back(boxes[i]);
    return out;
}

// a line image at the recognizers' height, width scaled with the aspect ratio
Img prep_line(const Img& crop, size_t min_w, size_t max_w) {
    const Img g = crop.to_gray();
    const size_t nw = std::clamp((size_t)std::nearbyint((double)g.w * (double)kH / (double)std::max<size_t>(g.h, 1)), min_w, max_w);
    return g.h > kH ? resize_area(g, nw, kH) : resize_linear(g, nw, kH);
}

bool utf8_is_alnum(uint32_t cp) {
    // ASCII letters and digits, and every letter / digit outside ASCII except punctuation blocks:
    // enough for the junk filter, which only counts "word" characters
    if (cp < 0x80) return (cp >= '0' && cp <= '9') || ((cp | 0x20) >= 'a' && (cp | 0x20) <= 'z');
    if (cp >= 0x2000 && cp <= 0x2BFF) return false;   // punctuation, symbols, arrows, boxes
    if (cp >= 0x3000 && cp <= 0x303F) return false;   // CJK punctuation
    if (cp >= 0xFF00 && cp <= 0xFF0F) return false;
    if (cp == 0xA0 || (cp >= 0xA1 && cp <= 0xBF) || cp == 0xD7 || cp == 0xF7) return false;
    return true;
}

std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        uint32_t cp = n == 1 ? c : c & (0x7F >> n);
        for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += (size_t)n;
    }
    return out;
}

// a detector false positive on a picture: no letter or digit, or one or two unsure characters
bool is_junk(const std::string& text, float conf) {
    int word = 0;
    for (uint32_t cp : utf8_decode(text)) if (utf8_is_alnum(cp)) ++word;
    return word == 0 || (word <= 2 && conf < 0.8f);
}

// characters that show a reading really is in `group`'s script
size_t script_chars(int group, const std::string& text) {
    auto in = [](uint32_t u, std::initializer_list<std::pair<uint32_t, uint32_t>> r) {
        for (const auto& [a, b] : r) if (u >= a && u <= b) return true;
        return false;
    };
    size_t n = 0;
    for (uint32_t u : utf8_decode(text)) {
        bool yes;
        switch (group) {
        case 0: yes = (u < 0x80 && ((u | 0x20) >= 'a' && (u | 0x20) <= 'z')) || in(u, {{0x00c0, 0x024f}, {0x0370, 0x04ff}}); break;
        case 1: yes = in(u, {{0x3040, 0x30ff}, {0x3400, 0x9fff}, {0xac00, 0xd7af}, {0xf900, 0xfaff}}); break;
        case 2: yes = in(u, {{0x0590, 0x08ff}, {0xfb1d, 0xfdff}, {0xfe70, 0xfeff}}); break;
        default: yes = in(u, {{0x0900, 0x0dff}, {0x0e00, 0x0eff}, {0x0f00, 0x109f}, {0x10a0, 0x10ff},
                              {0x0530, 0x058f}, {0x1200, 0x139f}, {0x1780, 0x17ff}});
        }
        if (yes) ++n;
    }
    return n;
}

// Latin letters (ASCII and Latin-1 / Extended-A/B; not Greek or Cyrillic)
size_t latin_letters(const std::string& text) {
    size_t n = 0;
    for (uint32_t u : utf8_decode(text))
        if ((u < 0x80 && (u | 0x20) >= 'a' && (u | 0x20) <= 'z') || (u >= 0xc0 && u <= 0x24f && u != 0xd7 && u != 0xf7)) ++n;
    return n;
}

bool is_blank(const std::string& s) {
    for (uint32_t cp : utf8_decode(s))
        if (!(cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0xA0 || cp == 0x3000)) return false;
    return true;
}

class OcrModel;

class OcrContext : public kiln::Context {
public:
    OcrContext(OcrModel& m, ggml_backend_t cpu) : m_(m), cpu_(cpu) {
        ggml_backend_cpu_set_abort_callback(cpu_, [](void* p) { return static_cast<OcrContext*>(p)->cancelled(); }, this);
    }
    ~OcrContext() override { ggml_backend_free(cpu_); }

    kiln_status reset() override { return KILN_OK; }
    kiln_status generate(const kiln::RenderedPrompt&, const kiln_sampling_params&, kiln_token_cb, void*, kiln_gen_stats*) override { return not_supported(); }
    kiln_status decide(const char*, const kiln_decision_question&, float*, int32_t, kiln_decision_answer&) override { return not_supported(); }
    kiln_status embed(const char*, const kiln_media*, int32_t, const kiln_embed_options&, float*, int32_t, kiln_embed_stats&) override { return not_supported(); }
    kiln_status transcribe(const kiln::RenderedPrompt&, const kiln_transcribe_options&, const kiln_media&, kiln_token_cb, void*, kiln_transcribe_stats&) override { return not_supported(); }
    kiln_status speak(const char*, const kiln_speak_options&, kiln_audio_cb, void*, kiln_speak_stats&) override { return not_supported(); }
    kiln_status ocr(const kiln_media& image, const kiln_ocr_options& o, kiln_ocr_cb cb, void* user, kiln_ocr_stats& st) override;

private:
    static kiln_status not_supported() { return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "an OCR model only reads images (kiln_ocr)"); }
    kiln_status fail(const std::string& err) {
        if (cancelled()) return KILN_ERR_CANCELLED;
        return kiln::set_error(KILN_ERR_BACKEND, err);
    }
    bool detect(const Img& img, int long_side, std::vector<Quad>& boxes, std::string& err);
    bool route(const std::vector<Img>& crops, std::vector<int>& groups, bool& flip, double share[4], std::string& err);
    bool read_lines(const std::vector<Img>& crops, std::vector<std::pair<std::string, float>>& out,
                    std::vector<int>& groups, bool& flip, std::string& err);
    bool recognize(int group, const std::vector<const Img*>& crops, std::vector<std::pair<std::string, float>>& out, std::string& err);

    OcrModel& m_;
    ggml_backend_t cpu_;
};

class OcrModel : public kiln::Model {
public:
    OcrModel(gguf_context* g, ggml_context* w, int threads, uint64_t mask, uint64_t bytes, int64_t n_params)
        : g_(g), w_(w), threads_(threads), mask_(mask), bytes_(bytes), n_params_(n_params) {}
    ~OcrModel() override { graphs_.clear(); if (g_) gguf_free(g_); if (w_) ggml_free(w_); }

    bool init(std::string& err) {
        const char* names[6] = {"det", "router", "rec_latin", "rec_cjk", "rec_rtl", "rec_world"};
        for (int i = 0; i < 6; ++i) {
            Graph gr;
            if (!gr.load(g_, w_, names[i], err)) return false;
            graphs_.push_back(std::move(gr));
        }
        for (int i = 0; i < 4; ++i) {
            const std::string key = std::string("ocria.rec_") + kGroups[i] + ".charset";
            const int64_t id = gguf_find_key(g_, key.c_str());
            if (id < 0) { err = "missing " + key; return false; }
            const size_t n = gguf_get_arr_n(g_, id);
            charsets_[i].reserve(n);
            for (size_t k = 0; k < n; ++k) charsets_[i].emplace_back(gguf_get_arr_str(g_, id, k));
            const std::string rk = std::string("ocria.rec_") + kGroups[i] + ".visual_rtl";
            const int64_t rid = gguf_find_key(g_, rk.c_str());
            visual_rtl_[i] = rid >= 0 && gguf_get_val_bool(g_, rid);
        }
        if (int64_t id = gguf_find_key(g_, "ocria.det.thresh"); id >= 0) det_thresh_ = gguf_get_val_f32(g_, id);
        if (int64_t id = gguf_find_key(g_, "ocria.det.merge"); id >= 0) det_merge_ = gguf_get_val_f32(g_, id);
        // input pixels per output step of each recognizer: one probe run on a 64-pixel line
        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) { err = "no CPU backend"; return false; }
        std::vector<float> probe(kH * 64, 1.f);
        for (int i = 0; i < 4; ++i) {
            std::vector<Tensor> out;
            if (!rec(i).run(cpu, {1, 1, (int64_t)kH, 64}, probe.data(), out, err)) { ggml_backend_free(cpu); return false; }
            stride_[i] = (size_t)(64 / std::max<int64_t>(out[0].shape[2], 1));
        }
        ggml_backend_free(cpu);
        return true;
    }

    kiln_status info(kiln_model_info& o) const override {
        o = {};
        o.backend_used = KILN_BACKEND_CPU;
        snprintf(o.backend_name, sizeof o.backend_name, "ocr/cpu");
        snprintf(o.arch, sizeof o.arch, "%s", kArch);
        snprintf(o.quant, sizeof o.quant, "F32");
        o.n_params = n_params_;
        o.model_bytes = bytes_;
        o.is_ocr_model = true;
        return KILN_OK;
    }
    int32_t tokenize(const char*, bool, int32_t*, int32_t) const override { return 0; }
    int32_t detokenize(const int32_t*, int32_t, char*, int32_t) const override { return 0; }
    kiln_status render_chat(const kiln_chat_message*, int32_t, const kiln_chat_options&, kiln::RenderedPrompt&) const override {
        return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "an OCR model has no chat");
    }
    kiln_status render_transcribe(const kiln_transcribe_options&, kiln::RenderedPrompt&) const override {
        return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "an OCR model does not transcribe audio");
    }
    kiln_status create_context(const kiln_context_params&, std::unique_ptr<kiln::Context>& out) override {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (!cpu) return kiln::set_error(KILN_ERR_BACKEND, "no CPU backend");
        ggml_backend_cpu_set_n_threads(cpu, threads_);
        out.reset(new OcrContext(*this, cpu));
        return KILN_OK;
    }
    kiln_status load_projector(const char*, const kiln_projector_params&, kiln_projector_info&) override {
        return kiln::set_error(KILN_ERR_NOT_SUPPORTED, "an OCR model takes no projector");
    }

    const Graph& det() const { return graphs_[0]; }
    const Graph& router() const { return graphs_[1]; }
    const Graph& rec(int group) const { return graphs_[2 + group]; }
    const std::vector<std::string>& charset(int group) const { return charsets_[group]; }
    size_t stride(int group) const { return stride_[group]; }
    float det_thresh() const { return det_thresh_; }
    float det_merge() const { return det_merge_; }
    uint64_t cpu_mask() const { return mask_; }
    std::mutex& mutex() { return mu_; }

private:
    gguf_context* g_;
    ggml_context* w_;
    int threads_;
    uint64_t mask_, bytes_;
    int64_t n_params_;
    std::vector<Graph> graphs_;
    std::vector<std::string> charsets_[4];
    bool visual_rtl_[4] = {};
    size_t stride_[4] = {4, 4, 4, 4};
    float det_thresh_ = 0.5f, det_merge_ = 0.6f;
    std::mutex mu_;
};

bool OcrContext::detect(const Img& img, int long_side, std::vector<Quad>& boxes, std::string& err) {
    const double big_h = (double)img.h, big_w = (double)img.w;
    const double s = std::min((double)long_side / std::max(big_h, big_w), 1.0);
    const size_t h = std::max<size_t>(32, (size_t)std::nearbyint(big_h * s / 32.0) * 32);
    const size_t w = std::max<size_t>(32, (size_t)std::nearbyint(big_w * s / 32.0) * 32);
    const double sx = (double)w / big_w, sy = (double)h / big_h;
    const Img small = resize_area(img, w, h);
    std::vector<float> x(3 * h * w);
    for (size_t c = 0; c < 3; ++c)
        for (size_t i = 0; i < h * w; ++i) x[c * h * w + i] = (float)small.data[i * 3 + c] / 127.5f - 1.f;
    std::vector<Tensor> out;
    if (!m_.det().run(cpu_, {1, 3, (int64_t)h, (int64_t)w}, x.data(), out, err)) return false;
    const std::vector<float>& prob = out[0].data;
    const size_t ph = (size_t)out[0].shape[2], pw = (size_t)out[0].shape[3];
    const float stride = (float)(h / ph);
    const float thresh = m_.det_thresh();
    std::vector<uint8_t> mask(prob.size());
    for (size_t i = 0; i < prob.size(); ++i) mask[i] = prob[i] > thresh;
    boxes.clear();
    for (const auto& comp : components4(mask, pw, ph)) {
        if (comp.pixels.size() < 2) continue;
        double mean = 0;
        for (const auto& [px, py] : comp.pixels) mean += prob[py * pw + px];
        mean /= (double)comp.pixels.size();
        if ((float)mean < kMinScore) continue;
        // the hull of the component's cells: each row's leftmost and rightmost cell
        const size_t rows = comp.y1 - comp.y0 + 1;
        std::vector<std::pair<size_t, size_t>> span(rows, {SIZE_MAX, 0});
        for (const auto& [px, py] : comp.pixels) {
            auto& r = span[py - comp.y0];
            r.first = std::min<size_t>(r.first, px);
            r.second = std::max<size_t>(r.second, px);
        }
        std::vector<std::pair<float, float>> pts;
        pts.reserve(rows * 4);
        for (size_t i = 0; i < rows; ++i) {
            if (span[i].first == SIZE_MAX) continue;
            const float y = (float)(comp.y0 + i), x0 = (float)span[i].first, x1 = (float)span[i].second + 1.f;
            for (auto [px, py] : {std::pair{x0, y}, {x0, y + 1.f}, {x1, y}, {x1, y + 1.f}})
                pts.emplace_back(px * stride, py * stride);
        }
        RotRect r = min_area_rect(pts);
        // the line's height is the side nearer vertical (a single digit's kernel is taller than wide)
        const float ang = std::fmod(std::fmod(r.angle + 90.f, 180.f) + 180.f, 180.f) - 90.f;
        const bool w_is_horizontal = std::fabs(ang) < 45.f;
        const float kernel_h = w_is_horizontal ? r.h : r.w;
        const float line_h = kernel_h / (1.f - 2.f * kShrink);
        const float d = kShrink * line_h;
        r.w += 2.f * d;
        r.h += 2.f * d;
        Quad q = box_points(r);
        for (auto& p : q) {
            p[0] = std::clamp((float)((double)p[0] / sx), 0.f, (float)img.w);
            p[1] = std::clamp((float)((double)p[1] / sy), 0.f, (float)img.h);
        }
        boxes.push_back(order_quad(q));
    }
    if (m_.det_merge() > 0.f) boxes = merge_collinear(boxes, m_.det_merge());
    // reading order: rows of 8 pixels, then left edge (a stable sort, as Python's)
    std::stable_sort(boxes.begin(), boxes.end(), [](const Quad& a, const Quad& b) {
        const double ka = std::nearbyint((double)min_y(a) / 8.0), kb = std::nearbyint((double)min_y(b) / 8.0);
        if (ka != kb) return ka < kb;
        return min_x(a) < min_x(b);
    });
    return true;
}

// Page-level routing: every line takes the page's main script unless it is long enough to
// judge and confidently another; Latin unless another script holds a clear majority of the
// line width. The page counts as upside down when most of the width votes so.
bool OcrContext::route(const std::vector<Img>& crops, std::vector<int>& groups, bool& flip, double share[4], std::string& err) {
    constexpr float kOverride = 0.9f;
    constexpr double kMinAspect = 4.0, kFlipVote = 0.7, kOtherShare = 0.65;
    constexpr size_t kFlipMinLines = 3;
    groups.assign(crops.size(), 0);
    flip = false;
    for (int k = 0; k < 4; ++k) share[k] = 0;
    if (crops.empty()) return true;
    std::vector<std::array<float, 4>> gp;
    std::vector<float> fp;
    auto softmax = [](const float* z, int n, float* out) {
        float m = -INFINITY;
        for (int i = 0; i < n; ++i) m = std::max(m, z[i]);
        float s = 0.f;
        for (int i = 0; i < n; ++i) { out[i] = std::exp(z[i] - m); s += out[i]; }
        for (int i = 0; i < n; ++i) out[i] /= s;
    };
    for (size_t start = 0; start < crops.size(); start += kRouterBatch) {
        const size_t n = std::min(kRouterBatch, crops.size() - start);
        std::vector<float> x(n * kH * kRouterW);
        for (size_t j = 0; j < n; ++j) {
            const Img g = prep_line(crops[start + j], 8, SIZE_MAX);
            // short lines are tiled across the router's width, then cut to it
            for (size_t y = 0; y < kH; ++y)
                for (size_t xx = 0; xx < kRouterW; ++xx)
                    x[(j * kH + y) * kRouterW + xx] = (float)g.data[y * g.w + xx % g.w] / 127.5f - 1.f;
        }
        std::vector<Tensor> out;
        if (!m_.router().run(cpu_, {(int64_t)n, 1, (int64_t)kH, (int64_t)kRouterW}, x.data(), out, err)) return false;
        for (size_t j = 0; j < n; ++j) {
            std::array<float, 4> p;
            softmax(&out[0].data[j * 4], 4, p.data());
            gp.push_back(p);
            float f[2];
            softmax(&out[1].data[j * 2], 2, f);
            fp.push_back(f[1]);
        }
    }
    double wsum = 0;
    for (size_t i = 0; i < crops.size(); ++i) {
        const double w = (double)crops[i].w;
        wsum += w;
        for (int k = 0; k < 4; ++k) share[k] += (double)gp[i][k] * w;
    }
    for (int k = 0; k < 4; ++k) share[k] /= wsum;
    int dom = 0;
    for (int k = 1; k < 4; ++k) if (share[k] > share[dom]) dom = k;
    if (dom != 0 && share[dom] <= kOtherShare) dom = 0;
    double vote = 0;
    for (size_t i = 0; i < crops.size(); ++i) {
        int own = 0;
        for (int k = 1; k < 4; ++k) if (gp[i][k] > gp[i][own]) own = k;
        const float pmax = *std::max_element(gp[i].begin(), gp[i].end());
        groups[i] = (own != dom && pmax > kOverride && (double)crops[i].w >= kMinAspect * (double)crops[i].h) ? own : dom;
        vote += (double)fp[i] * (double)crops[i].w;
    }
    flip = crops.size() >= kFlipMinLines && vote > kFlipVote * wsum;
    return true;
}

// Readings per line (LiteParse's `read`): route, read each script's lines with its recognizer,
// then the two second looks. "Rescue": a page of another script that the router sent to Latin
// reads as confident-looking Latin junk at 0.5-0.7, while the right recognizer scores 0.83-0.90.
// "Odd lines": a line routed away from the page's script is also read by the page's recognizer,
// and the routed reading must win clearly (the receipt's "#####" rows read as CJK "井井井" at
// 0.55 and as "# # #" at 0.95).
bool OcrContext::read_lines(const std::vector<Img>& crops_in, std::vector<std::pair<std::string, float>>& out,
                            std::vector<int>& groups, bool& flip, std::string& err) {
    constexpr float kKeepConf = 0.75f;
    double share[4];
    if (!route(crops_in, groups, flip, share, err)) return false;
    std::vector<Img> turned;
    if (flip) { turned.reserve(crops_in.size()); for (const auto& c : crops_in) turned.push_back(c.rot90(2)); }
    const std::vector<Img>& crops = flip ? turned : crops_in;
    const size_t n = crops.size();
    out.assign(n, {std::string(), 0.f});
    auto read = [&](int g, const std::vector<size_t>& idx, std::vector<std::pair<std::string, float>>& res) {
        std::vector<const Img*> sub;
        for (size_t i : idx) sub.push_back(&crops[i]);
        return recognize(g, sub, res, err);
    };
    for (int g = 0; g < 4; ++g) {
        std::vector<size_t> idx;
        for (size_t i = 0; i < n; ++i) if (groups[i] == g) idx.push_back(i);
        if (idx.empty()) continue;
        std::vector<std::pair<std::string, float>> res;
        if (!read(g, idx, res)) return false;
        for (size_t k = 0; k < idx.size(); ++k) out[idx[k]] = std::move(res[k]);
    }
    int dom = 0;
    {
        size_t cnt[4] = {0, 0, 0, 0};
        for (int g : groups) ++cnt[g];
        for (int k = 1; k < 4; ++k) if (cnt[k] > cnt[dom]) dom = k;
    }
    if (dom == 0) {
        constexpr double kRescueConf = 0.75, kRescueMargin = 0.15, kRescueShare = 0.2;
        std::vector<size_t> lat;
        for (size_t i = 0; i < n; ++i) if (groups[i] == dom) lat.push_back(i);
        bool other = false;
        for (int k = 1; k < 4; ++k) if (share[k] >= kRescueShare) other = true;
        if (!lat.empty() && other) {
            double lat_conf = 0;
            for (size_t i : lat) lat_conf += (double)out[i].second;
            lat_conf /= (double)lat.size();
            if (lat_conf < kRescueConf) {
                std::vector<size_t> sample(lat.begin(), lat.begin() + (long)std::min<size_t>(lat.size(), 40));
                int best = -1;
                double best_conf = lat_conf + kRescueMargin;
                for (int gi = 1; gi < 4; ++gi) {
                    if (share[gi] < kRescueShare) continue;
                    std::vector<std::pair<std::string, float>> res;
                    if (!read(gi, sample, res)) return false;
                    double c = 0;
                    for (const auto& r : res) c += (double)r.second;
                    c /= (double)res.size();
                    if (c >= best_conf) { best = gi; best_conf = c; }
                }
                if (best >= 0) {
                    std::vector<std::pair<std::string, float>> res;
                    if (!read(best, lat, res)) return false;
                    for (size_t k = 0; k < lat.size(); ++k) {
                        const size_t i = lat[k];
                        // per line: a bilingual page keeps its confident Latin lines
                        if (!(out[i].second >= kKeepConf && out[i].second >= res[k].second)) {
                            out[i] = res[k];
                            groups[i] = best;
                        }
                    }
                    dom = best;
                }
            }
        }
        // per line, when the page stays Latin: lines of another script among English labels
        if (dom == 0) {
            constexpr float kLineMargin = 0.1f, kLineConf = 0.8f;
            constexpr double kLineShare = 0.15;
            for (int gi = 1; gi < 4; ++gi) {
                if (share[gi] < kLineShare) continue;
                std::vector<size_t> weak;
                for (size_t i = 0; i < n; ++i) if (groups[i] == 0) weak.push_back(i);
                if (weak.empty()) break;
                std::vector<std::pair<std::string, float>> res;
                if (!read(gi, weak, res)) return false;
                for (size_t k = 0; k < weak.size(); ++k) {
                    const size_t i = weak[k];
                    const size_t sc = script_chars(gi, res[k].first);
                    if (res[k].second >= out[i].second + kLineMargin && res[k].second >= kLineConf &&
                        sc >= 3 && sc > latin_letters(res[k].first)) {
                        out[i] = res[k];
                        groups[i] = gi;
                    }
                }
            }
        }
    }
    // lines routed away from the page's script, read with the page's recognizer too
    std::vector<size_t> odd;
    for (size_t i = 0; i < n; ++i) if (groups[i] != dom) odd.push_back(i);
    if (!odd.empty()) {
        std::vector<std::pair<std::string, float>> res;
        if (!read(dom, odd, res)) return false;
        for (size_t k = 0; k < odd.size(); ++k) {
            const size_t i = odd[k];
            const size_t sc = script_chars(groups[i], out[i].first);
            // a confident reading that is mostly the routed script stands
            const bool routed_ok = out[i].second >= kKeepConf && sc >= 2 && sc > latin_letters(out[i].first);
            if (!routed_ok && (out[i].second < res[k].second + 0.1f || sc < 3)) out[i] = res[k];
        }
    }
    return true;
}

// Lines are batched by width (sorted, 32 per batch, padded to a multiple of 128 pixels) as the
// reference does: a line's reading depends on the padded width it was run at.
bool OcrContext::recognize(int group, const std::vector<const Img*>& crops,
                           std::vector<std::pair<std::string, float>>& out, std::string& err) {
    std::vector<Img> lines;
    lines.reserve(crops.size());
    for (const Img* c : crops) lines.push_back(prep_line(*c, 8, kMaxW));
    std::vector<size_t> order(lines.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lines[a].w < lines[b].w; });
    out.assign(lines.size(), {std::string(), 0.f});
    const auto& chars = m_.charset(group);
    const size_t stride = m_.stride(group);
    for (size_t start = 0; start < order.size(); start += kBatch) {
        const size_t n = std::min(kBatch, order.size() - start);
        size_t maxw = 1;
        for (size_t j = 0; j < n; ++j) maxw = std::max(maxw, lines[order[start + j]].w);
        const size_t w = (maxw + kBucketW - 1) / kBucketW * kBucketW;
        std::vector<float> x(n * kH * w, 1.f);
        for (size_t j = 0; j < n; ++j) {
            const Img& l = lines[order[start + j]];
            for (size_t y = 0; y < kH; ++y)
                for (size_t xx = 0; xx < l.w; ++xx)
                    x[(j * kH + y) * w + xx] = (float)((double)l.data[y * l.w + xx] / 127.5 - 1.0);
        }
        std::vector<Tensor> res;
        if (!m_.rec(group).run(cpu_, {(int64_t)n, 1, (int64_t)kH, (int64_t)w}, x.data(), res, err, true)) return false;
        // logits (N, K, T), the K scores of one step side by side: per step, the best class
        // (0 = blank) and its softmax probability
        const size_t K = (size_t)res[0].shape[1], T = (size_t)res[0].shape[2];
        const float* lg = res[0].data.data();
        const size_t ks = res[0].channels_first ? 1 : T;                         // stride between classes
        auto step = [&](size_t j, size_t t) { return res[0].channels_first ? lg + (j * T + t) * K : lg + j * K * T + t; };
        for (size_t j = 0; j < n; ++j) {
            const size_t i = order[start + j];
            const size_t t_n = std::min(lines[i].w / stride, T);
            std::string text;
            double conf_sum = 0;
            int conf_n = 0;
            size_t prev = 0;
            for (size_t t = 0; t < t_n; ++t) {
                const float* col = step(j, t);
                size_t best = 0;
                float m = col[0];
                for (size_t k = 1; k < K; ++k) if (col[k * ks] > m) { m = col[k * ks]; best = k; }
                if (best != 0 && best != prev) {
                    double s = 0;
                    for (size_t k = 0; k < K; ++k) s += std::exp((double)(col[k * ks] - m));
                    conf_sum += (double)(float)std::exp(-std::log(s));
                    ++conf_n;
                    if (best - 1 < chars.size()) text += chars[best - 1];
                }
                prev = best;
            }
            out[i] = {text, conf_n ? (float)(conf_sum / conf_n) : 0.f};
        }
    }
    return true;
}

kiln_status OcrContext::ocr(const kiln_media& image, const kiln_ocr_options& o, kiln_ocr_cb cb, void* user, kiln_ocr_stats& st) {
    const auto t_all = clock_t_::now();
    int w = 0, h = 0, n = 0;
    unsigned char* px = image.path ? stbi_load(image.path, &w, &h, &n, 3)
                                   : stbi_load_from_memory(image.data, (int)image.size, &w, &h, &n, 3);
    if (!px) return kiln::set_error(KILN_ERR_INVALID_ARG, std::string("could not read the image (jpg, png, bmp or gif): ") + stbi_failure_reason());
    Img img((size_t)w, (size_t)h, 3);
    memcpy(img.data.data(), px, img.data.size());
    stbi_image_free(px);
    st.width = w;
    st.height = h;

    std::lock_guard<std::mutex> lock(m_.mutex());     // the model's graphs are shared between contexts
    kiln::AffinityScope on_fast_cores(m_.cpu_mask());
    std::string err;
    const auto t_det = clock_t_::now();
    std::vector<Quad> boxes;
    if (!detect(img, o.long_side, boxes, err)) return fail(err);
    st.detect_ms = ms_since(t_det);
    st.lines_found = (int32_t)boxes.size();

    const auto t_read = clock_t_::now();
    std::vector<std::pair<std::string, float>> reads(boxes.size());
    std::vector<int> groups;
    bool flip = false;
    if (!boxes.empty()) {
        std::vector<Img> crops;
        crops.reserve(boxes.size());
        for (const auto& q : boxes) crops.push_back(crop(img, q));
        if (!read_lines(crops, reads, groups, flip, err)) return fail(err);
    }
    st.read_ms = ms_since(t_read);
    st.upside_down = flip;

    // the boxes are sorted top to bottom in the image; on a page read upside down that is the
    // reading order backwards
    for (size_t kk = 0; kk < boxes.size(); ++kk) {
        const size_t k = flip ? boxes.size() - 1 - kk : kk;
        const auto& [text, conf] = reads[k];
        if (is_blank(text) || is_junk(text, conf)) continue;
        if (o.min_confidence > 0.f && conf < o.min_confidence) continue;
        Quad q = boxes[k];
        if (flip) q = {q[2], q[3], q[0], q[1]};     // corners in reading order on an upside-down page
        kiln_ocr_line line{};
        line.text = text.c_str();
        line.x0 = INFINITY; line.y0 = INFINITY; line.x1 = -INFINITY; line.y1 = -INFINITY;
        for (int i = 0; i < 4; ++i) {
            line.quad[2 * i] = q[i][0];
            line.quad[2 * i + 1] = q[i][1];
            line.x0 = std::min(line.x0, q[i][0]); line.y0 = std::min(line.y0, q[i][1]);
            line.x1 = std::max(line.x1, q[i][0]); line.y1 = std::max(line.y1, q[i][1]);
        }
        line.x0 = std::trunc(line.x0); line.y0 = std::trunc(line.y0);
        line.x1 = std::trunc(line.x1); line.y1 = std::trunc(line.y1);
        line.confidence = conf;
        line.script = groups.empty() ? 0 : groups[k];
        ++st.lines;
        if (cb && !cb(&line, user)) { st.stopped_by_user = true; break; }
        if (cancelled()) { st.stopped_by_user = true; st.ms = ms_since(t_all); return KILN_ERR_CANCELLED; }
    }
    st.ms = ms_since(t_all);
    return KILN_OK;
}

bool is_ocria_gguf(const char* path) {
    uint32_t magic = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    const bool gguf = fread(&magic, 4, 1, f) == 1 && magic == 0x46554747;   // "GGUF"
    fclose(f);
    if (!gguf) return false;
    gguf_init_params gp = {true, nullptr};
    gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) return false;
    const int64_t id = gguf_find_key(g, "general.architecture");
    const bool yes = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING && strcmp(gguf_get_val_str(g, id), kArch) == 0;
    gguf_free(g);
    return yes;
}

class OcrBackend : public kiln::Backend {
public:
    const char* name() const override { return "ocr"; }
    void annotate_device(kiln_device_info&) const override {}
    bool supports(kiln_backend_kind kind, const kiln_device_info&) const override {
        return kind == KILN_BACKEND_AUTO || kind == KILN_BACKEND_CPU;
    }
    bool can_load(const char* path) const override { return is_ocria_gguf(path); }
    kiln_status load_model(const char* path, const kiln_model_params& p, const kiln_device_info& d,
                           std::unique_ptr<kiln::Model>& out) override {
        ggml_context* w = nullptr;
        gguf_init_params gp = {false, &w};
        gguf_context* g = gguf_init_from_file(path, gp);
        if (!g) return kiln::set_error(KILN_ERR_MODEL_LOAD, std::string("failed to load ") + path + " (missing or corrupt)");
        // the model's own tensors (gguf also puts the whole data section in one extra tensor)
        uint64_t bytes = 0;
        int64_t n_params = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            if (ggml_tensor* t = ggml_get_tensor(w, gguf_get_tensor_name(g, i))) { bytes += ggml_nbytes(t); n_params += ggml_nelements(t); }
        }
        // the same core set as the other backends: the probe's fast cores, KILN_CPU_MASK to
        // force one, KILN_NO_AFFINITY=1 for none
        uint64_t mask = d.cpu_fast_mask;
        if (const char* m = getenv("KILN_CPU_MASK"); m && *m) mask = strtoull(m, nullptr, 16);
        if (const char* no = getenv("KILN_NO_AFFINITY"); no && *no && *no != '0') mask = 0;
        int threads = p.n_threads > 0 ? p.n_threads
                    : mask ? __builtin_popcountll(mask)
                    : kiln::default_thread_count(d.cpu_cores, d.cpu_big_cores, false);
        if (threads > 8) threads = 8;
        auto m = std::make_unique<OcrModel>(g, w, threads, mask, bytes, n_params);
        std::string err;
        if (!m->init(err)) return kiln::set_error(KILN_ERR_MODEL_LOAD, std::string(path) + ": " + err);
        kiln::log(KILN_LOG_INFO, "loaded %s via ocr/cpu (OCRia-nano), threads=%d, cores=0x%llx", path, threads, (unsigned long long)mask);
        out = std::move(m);
        return KILN_OK;
    }
};

} // namespace

namespace kiln {
Backend* make_ocr_backend() { static OcrBackend b; return &b; }
}
