// Image steps of the OCRia-nano pipeline: resizing, grey, rotation, perspective crop, connected
// components and rotated rectangles. Ported from LiteParse (crates/liteparse/src/ocr/ocria/img.rs,
// Copyright LlamaIndex, Apache License 2.0), which writes them to OpenCV's definitions (pixel
// centres, fixed-point weights, border rules) so the models see the inputs they were trained on.
// A crop that differs by one grey level can change a reading, so this follows it line by line.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace kiln::ocr {

// An 8-bit image, `ch` interleaved channels per pixel (1 = grey, 3 = RGB), row-major.
struct Img {
    size_t w = 0, h = 0, ch = 0;
    std::vector<uint8_t> data;
    Img() = default;
    Img(size_t w_, size_t h_, size_t ch_) : w(w_), h(h_), ch(ch_), data(w_ * h_ * ch_, 0) {}
    uint8_t at(size_t x, size_t y, size_t c) const { return data[(y * w + x) * ch + c]; }
    Img to_gray() const;            // cv2.cvtColor(RGB2GRAY), 14-bit fixed point
    Img rot90(unsigned k) const;    // np.rot90: k quarter turns counter-clockwise
};

using Pt   = std::array<float, 2>;
using Quad = std::array<Pt, 4>;     // [top-left, top-right, bottom-right, bottom-left] along the line

Img resize_linear(const Img& img, size_t w, size_t h);   // cv2.resize INTER_LINEAR
Img resize_area(const Img& img, size_t w, size_t h);     // cv2.resize INTER_AREA

// cv2.getPerspectiveTransform(src, dst), then cv2.warpPerspective(INTER_CUBIC, BORDER_REPLICATE)
std::array<double, 9> perspective_transform(const Quad& src, const Quad& dst);
Img warp_perspective_cubic(const Img& img, const std::array<double, 9>& m, size_t w, size_t h);

struct Component {
    size_t x0, y0, x1, y1;
    std::vector<std::pair<uint32_t, uint32_t>> pixels;
};
// 4-connected components, numbered in raster order of their first pixel (as OpenCV does)
std::vector<Component> components4(const std::vector<uint8_t>& mask, size_t w, size_t h);

struct RotRect { float cx, cy, w, h, angle; };   // angle of the w side, in degrees
RotRect min_area_rect(const std::vector<std::pair<float, float>>& pts);   // cv2.minAreaRect
// cv2.boxPoints order (bottom-left, top-left, top-right, bottom-right), not Quad order: pass the
// result through order_quad before crop
Quad box_points(const RotRect& r);

// Perspective-crop a (possibly rotated) line quad; a box clearly taller than wide is turned flat
Img crop(const Img& img, const Quad& q);

} // namespace kiln::ocr
