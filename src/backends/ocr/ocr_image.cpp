// See ocr_image.h. Ported from LiteParse's img.rs and page.rs (Apache License 2.0).
#include "ocr_image.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

namespace kiln::ocr {

namespace {

constexpr int RESIZE_BITS = 11;              // OpenCV's INTER_RESIZE_COEF_BITS
constexpr int RESIZE_ONE  = 1 << RESIZE_BITS;

// cvRound: half to even, as saturate_cast does in the default rounding mode
inline int64_t cv_round(double v) { return (int64_t)std::nearbyint(v); }
inline int32_t sat_short(float v) {
    return (int32_t)std::clamp<int64_t>(cv_round((double)v), INT16_MIN, INT16_MAX);
}

struct Tap { size_t s; int32_t a0, a1; };

// two-tap source index and fixed-point weights per output index (cv2.resize's linear path);
// `area` selects INTER_AREA's upscaling coefficients
std::vector<Tap> linear_taps(size_t src, size_t dst, bool area) {
    const double inv_scale = (double)dst / (double)src;
    const double scale = 1.0 / inv_scale;
    std::vector<Tap> out(dst);
    for (size_t d = 0; d < dst; ++d) {
        int64_t sx; float f;
        if (area) {
            sx = (int64_t)std::floor((double)d * scale);
            f = (float)((double)(d + 1) - (double)(sx + 1) * inv_scale);
            f = f <= 0.f ? 0.f : f - std::floor(f);
        } else {
            const float fx = (float)(((double)d + 0.5) * scale - 0.5);
            const float fl = std::floor(fx);
            sx = (int64_t)fl;
            f = fx - fl;
        }
        if (sx < 0) { sx = 0; f = 0.f; }
        if (sx >= (int64_t)src - 1) { sx = (int64_t)src - 1; f = 0.f; }
        out[d] = {(size_t)sx, sat_short((1.f - f) * (float)RESIZE_ONE), sat_short(f * (float)RESIZE_ONE)};
    }
    return out;
}

Img resize_linear_taps(const Img& img, size_t w, size_t h, bool area) {
    const auto xt = linear_taps(img.w, w, area);
    const auto yt = linear_taps(img.h, h, area);
    const size_t ch = img.ch;
    std::vector<int32_t> rows(img.h * w * ch);     // horizontal pass, one row per source row
    for (size_t y = 0; y < img.h; ++y) {
        int32_t* r = &rows[y * w * ch];
        for (size_t dx = 0; dx < w; ++dx) {
            const Tap t = xt[dx];
            for (size_t c = 0; c < ch; ++c) {
                const int32_t s0 = img.at(t.s, y, c);
                r[dx * ch + c] = t.s + 1 < img.w ? s0 * t.a0 + (int32_t)img.at(t.s + 1, y, c) * t.a1
                                                 : s0 * RESIZE_ONE;
            }
        }
    }
    Img out(w, h, ch);
    for (size_t dy = 0; dy < h; ++dy) {
        const Tap t = yt[dy];
        const int32_t* r0 = &rows[t.s * w * ch];
        const int32_t* r1 = &rows[std::min(t.s + 1, img.h - 1) * w * ch];
        uint8_t* o = &out.data[dy * w * ch];
        for (size_t i = 0; i < w * ch; ++i) {
            // FixedPtCast<int, uchar, 22>
            const int64_t v = ((int64_t)r0[i] * t.a0 + (int64_t)r1[i] * t.a1 + (1 << 21)) >> 22;
            o[i] = (uint8_t)std::clamp<int64_t>(v, 0, 255);
        }
    }
    return out;
}

// OpenCV computeResizeAreaTab: (source index, weight) runs per output index
std::vector<std::vector<std::pair<size_t, float>>> area_tab(size_t src, size_t dst, double scale) {
    std::vector<std::vector<std::pair<size_t, float>>> out(dst);
    for (size_t d = 0; d < dst; ++d) {
        const double fsx1 = (double)d * scale;
        const double fsx2 = fsx1 + scale;
        const double cell = std::min(scale, (double)src - fsx1);
        int64_t sx2 = std::min((int64_t)std::floor(fsx2), (int64_t)src - 1);
        const int64_t sx1 = std::min((int64_t)std::ceil(fsx1), sx2);
        auto& t = out[d];
        if ((double)sx1 - fsx1 > 1e-3) t.emplace_back((size_t)(sx1 - 1), (float)(((double)sx1 - fsx1) / cell));
        for (int64_t sx = sx1; sx < sx2; ++sx) t.emplace_back((size_t)sx, (float)(1.0 / cell));
        if (fsx2 - (double)sx2 > 1e-3)
            t.emplace_back((size_t)sx2, (float)(std::min(std::min(fsx2 - (double)sx2, 1.0), cell) / cell));
    }
    return out;
}

double dist(const Pt& a, const Pt& b) {
    return std::sqrt(std::pow(a[0] - b[0], 2.f) + std::pow(a[1] - b[1], 2.f));
}

std::array<double, 9> invert3(const std::array<double, 9>& m) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
                     + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (det == 0.0) return {};
    const double d = 1.0 / det;
    return {(m[4] * m[8] - m[5] * m[7]) * d, (m[2] * m[7] - m[1] * m[8]) * d, (m[1] * m[5] - m[2] * m[4]) * d,
            (m[5] * m[6] - m[3] * m[8]) * d, (m[0] * m[8] - m[2] * m[6]) * d, (m[2] * m[3] - m[0] * m[5]) * d,
            (m[3] * m[7] - m[4] * m[6]) * d, (m[1] * m[6] - m[0] * m[7]) * d, (m[0] * m[4] - m[1] * m[3]) * d};
}

constexpr int64_t INTER_BITS = 5;
constexpr int64_t INTER_TAB  = 1 << INTER_BITS;
constexpr int32_t REMAP_SCALE = 1 << 15;

// OpenCV interpolateCubic (A = -0.75)
std::array<float, 4> cubic_coeffs(float x) {
    const float A = -0.75f;
    const float c0 = ((A * (x + 1.f) - 5.f * A) * (x + 1.f) + 8.f * A) * (x + 1.f) - 4.f * A;
    const float c1 = ((A + 2.f) * x - (A + 3.f)) * x * x + 1.f;
    const float c2 = ((A + 2.f) * (1.f - x) - (A + 3.f)) * (1.f - x) * (1.f - x) + 1.f;
    return {c0, c1, c2, 1.f - c0 - c1 - c2};
}

// OpenCV initInterTab2D(INTER_CUBIC): 32 x 32 sub-pixel positions, 16 weights summing to 2^15
const std::vector<std::array<int32_t, 16>>& bicubic_tab() {
    static const std::vector<std::array<int32_t, 16>> tab = [] {
        std::vector<std::array<float, 4>> one(INTER_TAB);
        for (int64_t i = 0; i < INTER_TAB; ++i) one[i] = cubic_coeffs((float)i / (float)INTER_TAB);
        std::vector<std::array<int32_t, 16>> t((size_t)(INTER_TAB * INTER_TAB));
        for (size_t iy = 0; iy < (size_t)INTER_TAB; ++iy)
            for (size_t ix = 0; ix < (size_t)INTER_TAB; ++ix) {
                auto& e = t[iy * INTER_TAB + ix];
                int32_t isum = 0;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j) {
                        e[i * 4 + j] = sat_short(one[iy][i] * one[ix][j] * (float)REMAP_SCALE);
                        isum += e[i * 4 + j];
                    }
                if (isum != REMAP_SCALE) {
                    const int32_t diff = isum - REMAP_SCALE;
                    int mk0 = 2, mk1 = 2, bg0 = 2, bg1 = 2;
                    for (int k1 = 2; k1 < 4; ++k1)
                        for (int k2 = 2; k2 < 4; ++k2) {
                            if (e[k1 * 4 + k2] < e[mk0 * 4 + mk1]) { mk0 = k1; mk1 = k2; }
                            else if (e[k1 * 4 + k2] > e[bg0 * 4 + bg1]) { bg0 = k1; bg1 = k2; }
                        }
                    if (diff < 0) e[bg0 * 4 + bg1] -= diff;
                    else          e[mk0 * 4 + mk1] -= diff;
                }
            }
        return t;
    }();
    return tab;
}

double cross(std::pair<double, double> o, std::pair<double, double> a, std::pair<double, double> b) {
    return (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first);
}

} // namespace

Img Img::to_gray() const {
    if (ch == 1) return *this;
    Img out(w, h, 1);
    for (size_t i = 0; i < w * h; ++i) {
        const uint8_t* p = &data[i * ch];
        const uint32_t v = (uint32_t)p[0] * 4899 + (uint32_t)p[1] * 9617 + (uint32_t)p[2] * 1868 + (1u << 13);
        out.data[i] = (uint8_t)(v >> 14);
    }
    return out;
}

Img Img::rot90(unsigned k) const {
    switch (k % 4) {
    case 0: return *this;
    case 1: {
        Img out(h, w, ch);
        for (size_t i = 0; i < out.h; ++i)
            for (size_t j = 0; j < out.w; ++j)
                for (size_t c = 0; c < ch; ++c)
                    out.data[(i * out.w + j) * ch + c] = at(w - 1 - i, j, c);
        return out;
    }
    case 2: {
        Img out(w, h, ch);
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                for (size_t c = 0; c < ch; ++c)
                    out.data[(y * w + x) * ch + c] = at(w - 1 - x, h - 1 - y, c);
        return out;
    }
    default: return rot90(1).rot90(2);
    }
}

Img resize_linear(const Img& img, size_t w, size_t h) {
    if (w == img.w && h == img.h) return img;
    return resize_linear_taps(img, w, h, false);
}

Img resize_area(const Img& img, size_t w, size_t h) {
    if (w == img.w && h == img.h) return img;
    const double scale_x = (double)img.w / (double)w, scale_y = (double)img.h / (double)h;
    if (scale_x < 1.0 || scale_y < 1.0) return resize_linear_taps(img, w, h, true);
    const size_t ch = img.ch;
    const double isx = std::round(scale_x), isy = std::round(scale_y);
    if (std::fabs(scale_x - isx) < std::numeric_limits<double>::epsilon() &&
        std::fabs(scale_y - isy) < std::numeric_limits<double>::epsilon()) {
        // integer factors: plain block mean (resizeAreaFast)
        const size_t kx = (size_t)isx, ky = (size_t)isy;
        const uint32_t area = (uint32_t)(kx * ky);
        Img out(w, h, ch);
        for (size_t dy = 0; dy < h; ++dy)
            for (size_t dx = 0; dx < w; ++dx)
                for (size_t c = 0; c < ch; ++c) {
                    uint32_t s = 0;
                    for (size_t yy = 0; yy < ky; ++yy)
                        for (size_t xx = 0; xx < kx; ++xx) s += img.at(dx * kx + xx, dy * ky + yy, c);
                    const uint32_t v = area == 4 ? (s + 2) >> 2
                                                 : (uint32_t)cv_round((double)s * (double)(1.0f / (float)area));
                    out.data[(dy * w + dx) * ch + c] = (uint8_t)std::min<uint32_t>(v, 255);
                }
        return out;
    }
    // flat horizontal taps (output offset, source offset, weight), in OpenCV's order
    struct XT { size_t d, s; float a; };
    std::vector<XT> xt;
    {
        const auto tab = area_tab(img.w, w, scale_x);
        for (size_t dx = 0; dx < tab.size(); ++dx)
            for (const auto& [sx, a] : tab[dx]) xt.push_back({dx * ch, sx * ch, a});
    }
    const auto yt = area_tab(img.h, h, scale_y);
    Img out(w, h, ch);
    const size_t row_len = img.w * ch;
    std::vector<float> buf(w * ch), sum(w * ch);
    size_t buf_row = SIZE_MAX;
    for (size_t dy = 0; dy < yt.size(); ++dy) {
        std::fill(sum.begin(), sum.end(), 0.f);
        for (const auto& [sy, beta] : yt[dy]) {
            if (buf_row != sy) {       // adjacent output rows share boundary rows: keep the last sums
                std::fill(buf.begin(), buf.end(), 0.f);
                const uint8_t* row = &img.data[sy * row_len];
                for (const auto& t : xt)
                    for (size_t c = 0; c < ch; ++c) buf[t.d + c] += (float)row[t.s + c] * t.a;
                buf_row = sy;
            }
            for (size_t i = 0; i < sum.size(); ++i) sum[i] += buf[i] * beta;
        }
        uint8_t* o = &out.data[dy * w * ch];
        for (size_t i = 0; i < sum.size(); ++i) o[i] = (uint8_t)std::clamp<int64_t>(cv_round((double)sum[i]), 0, 255);
    }
    return out;
}

std::array<double, 9> perspective_transform(const Quad& src, const Quad& dst) {
    double a[8][9] = {};
    for (int i = 0; i < 4; ++i) {
        const double x = src[i][0], y = src[i][1], u = dst[i][0], v = dst[i][1];
        const double r0[9] = {x, y, 1, 0, 0, 0, -x * u, -y * u, u};
        const double r1[9] = {0, 0, 0, x, y, 1, -x * v, -y * v, v};
        std::copy(r0, r0 + 9, a[i]);
        std::copy(r1, r1 + 9, a[i + 4]);
    }
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 8; ++col) {
        int piv = col;
        for (int p = col + 1; p < 8; ++p)
            if (std::fabs(a[p][col]) >= std::fabs(a[piv][col])) piv = p;   // max_by keeps the last maximum
        for (int k = 0; k < 9; ++k) std::swap(a[col][k], a[piv][k]);
        const double d = a[col][col];
        if (std::fabs(d) < 1e-300) continue;
        for (int r = 0; r < 8; ++r) {
            if (r == col) continue;
            const double f = a[r][col] / d;
            if (f != 0.0)
                for (int k = col; k < 9; ++k) a[r][k] -= f * a[col][k];
        }
    }
    std::array<double, 9> m{};
    for (int i = 0; i < 8; ++i) m[i] = a[i][i] != 0.0 ? a[i][8] / a[i][i] : 0.0;
    m[8] = 1.0;
    return m;
}

Img warp_perspective_cubic(const Img& img, const std::array<double, 9>& m, size_t w, size_t h) {
    const auto inv = invert3(m);
    const auto& tab = bicubic_tab();
    const size_t ch = img.ch;
    Img out(w, h, ch);
    const int64_t sw = (int64_t)img.w, sh = (int64_t)img.h;
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
            const double xf = (double)x, yf = (double)y;
            const double x0 = inv[0] * xf + inv[1] * yf + inv[2];
            const double y0 = inv[3] * xf + inv[4] * yf + inv[5];
            double ww = inv[6] * xf + inv[7] * yf + inv[8];
            ww = ww != 0.0 ? (double)INTER_TAB / ww : 0.0;
            const double fx = std::clamp(x0 * ww, (double)INT32_MIN, (double)INT32_MAX);
            const double fy = std::clamp(y0 * ww, (double)INT32_MIN, (double)INT32_MAX);
            const int64_t xi = cv_round(fx), yi = cv_round(fy);
            const int64_t sx = xi >> INTER_BITS, sy = yi >> INTER_BITS;
            const auto& wt = tab[(size_t)((yi & (INTER_TAB - 1)) * INTER_TAB + (xi & (INTER_TAB - 1)))];
            size_t cols[4];
            for (int j = 0; j < 4; ++j) cols[j] = (size_t)std::clamp<int64_t>(sx - 1 + j, 0, sw - 1);   // BORDER_REPLICATE
            int64_t sum[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4; ++i) {
                const size_t yy = (size_t)std::clamp<int64_t>(sy - 1 + i, 0, sh - 1);
                const uint8_t* row = &img.data[yy * img.w * ch];
                for (int j = 0; j < 4; ++j) {
                    const int64_t wv = wt[i * 4 + j];
                    for (size_t c = 0; c < ch; ++c) sum[c] += (int64_t)row[cols[j] * ch + c] * wv;
                }
            }
            uint8_t* o = &out.data[(y * w + x) * ch];
            for (size_t c = 0; c < ch; ++c) o[c] = (uint8_t)std::clamp<int64_t>((sum[c] + (1 << 14)) >> 15, 0, 255);
        }
    return out;
}

std::vector<Component> components4(const std::vector<uint8_t>& mask, size_t w, size_t h) {
    std::vector<uint8_t> seen(w * h, 0);
    std::vector<Component> out;
    std::vector<size_t> stack;
    for (size_t start = 0; start < w * h; ++start) {
        if (!mask[start] || seen[start]) continue;
        seen[start] = 1;
        stack.push_back(start);
        Component comp{SIZE_MAX, SIZE_MAX, 0, 0, {}};
        while (!stack.empty()) {
            const size_t p = stack.back(); stack.pop_back();
            const size_t x = p % w, y = p / w;
            comp.x0 = std::min(comp.x0, x); comp.y0 = std::min(comp.y0, y);
            comp.x1 = std::max(comp.x1, x); comp.y1 = std::max(comp.y1, y);
            comp.pixels.emplace_back((uint32_t)x, (uint32_t)y);
            auto visit = [&](size_t q) { if (mask[q] && !seen[q]) { seen[q] = 1; stack.push_back(q); } };
            if (x > 0) visit(p - 1);
            if (x + 1 < w) visit(p + 1);
            if (y > 0) visit(p - w);
            if (y + 1 < h) visit(p + w);
        }
        out.push_back(std::move(comp));
    }
    return out;
}

RotRect min_area_rect(const std::vector<std::pair<float, float>>& pts) {
    std::vector<std::pair<double, double>> p;
    p.reserve(pts.size());
    for (const auto& [x, y] : pts) p.emplace_back(x, y);
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    if (p.empty()) return {0, 0, 0, 0, 0};
    if (p.size() == 1) return {(float)p[0].first, (float)p[0].second, 0, 0, 0};
    std::vector<std::pair<double, double>> hull;     // monotone chain
    for (const auto& q : p) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), q) <= 0.0) hull.pop_back();
        hull.push_back(q);
    }
    const size_t lower = hull.size() + 1;
    for (auto it = p.rbegin() + 1; it != p.rend(); ++it) {
        while (hull.size() >= lower && cross(hull[hull.size() - 2], hull.back(), *it) <= 0.0) hull.pop_back();
        hull.push_back(*it);
    }
    hull.pop_back();
    bool have = false;
    double best_area = 0;
    RotRect best{(float)p[0].first, (float)p[0].second, 0, 0, 0};
    for (size_t i = 0; i < hull.size(); ++i) {
        const auto a = hull[i], b = hull[(i + 1) % hull.size()];
        const double dx = b.first - a.first, dy = b.second - a.second;
        const double len = std::hypot(dx, dy);
        if (len == 0.0) continue;
        const double ux = dx / len, uy = dy / len, vx = -uy, vy = ux;
        double u0 = INFINITY, u1 = -INFINITY, v0 = INFINITY, v1 = -INFINITY;
        for (const auto& q : hull) {
            const double pu = q.first * ux + q.second * uy, pv = q.first * vx + q.second * vy;
            u0 = std::min(u0, pu); u1 = std::max(u1, pu);
            v0 = std::min(v0, pv); v1 = std::max(v1, pv);
        }
        const double area = (u1 - u0) * (v1 - v0);
        if (!have || area < best_area) {
            have = true;
            best_area = area;
            const double cu = (u0 + u1) / 2.0, cv = (v0 + v1) / 2.0;
            best = {(float)(cu * ux + cv * vx), (float)(cu * uy + cv * vy), (float)(u1 - u0), (float)(v1 - v0),
                    (float)(std::atan2(uy, ux) * 180.0 / M_PI)};
        }
    }
    return best;
}

Quad box_points(const RotRect& r) {
    const double a = (double)r.angle * M_PI / 180.0;
    const float b = (float)(std::cos(a) * 0.5), c = (float)(std::sin(a) * 0.5);
    const Pt p0 = {r.cx - c * r.h - b * r.w, r.cy + b * r.h - c * r.w};
    const Pt p1 = {r.cx + c * r.h - b * r.w, r.cy - b * r.h - c * r.w};
    return {p0, p1, Pt{2.f * r.cx - p0[0], 2.f * r.cy - p0[1]}, Pt{2.f * r.cx - p1[0], 2.f * r.cy - p1[1]}};
}

Img crop(const Img& img, const Quad& q) {
    const size_t w = std::max<size_t>((size_t)(float)std::max(dist(q[0], q[1]), dist(q[2], q[3])), 2);
    const size_t h = std::max<size_t>((size_t)(float)std::max(dist(q[0], q[3]), dist(q[1], q[2])), 2);
    const Quad dst = {Pt{0.f, 0.f}, Pt{(float)w, 0.f}, Pt{(float)w, (float)h}, Pt{0.f, (float)h}};
    Img c = warp_perspective_cubic(img, perspective_transform(q, dst), w, h);
    if ((double)h / (double)w >= 1.5) return c.rot90(1);    // vertical box: turn it flat
    return c;
}

} // namespace kiln::ocr
