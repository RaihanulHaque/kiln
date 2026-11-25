// See ocr_graph.h.
//
// Image tensors are kept channels-first in memory: a ggml tensor of ne = (C, W, H, N), the
// channels of one pixel next to each other (rank-3 (N, C, W) tensors use H = 1). That layout
// makes the two convolutions these networks are built from fast in ggml: a 1x1 convolution is
// one matrix multiply over all pixels, with no copies, and a depthwise convolution takes ggml's
// SIMD path (conv_2d_dw_cwhn) instead of its one-multiply-at-a-time loop. On a 64-channel map of
// the detector's size, a depthwise 3x3 plus a 1x1 took 61.5 ms the plain way and 15.0 ms this way
// (Ryzen 5 8500G, 8 threads). Other tensors (the router's pooled features, Gemm) use ggml's plain
// order: ONNX shape reversed.
#include "ocr_graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace kiln::ocr {

namespace {

using json = nlohmann::json;

struct Val {
    ggml_tensor* t = nullptr;
    int rank = 0;                  // ONNX rank
    bool cf = false;               // channels-first image tensor (C, W, H, N); else ONNX shape reversed
};

std::vector<int64_t> ints(const json& a, const char* k) {
    std::vector<int64_t> v;
    if (a.contains(k)) for (const auto& x : a[k]) v.push_back(x.get<int64_t>());
    return v;
}

// shape of a nested JSON list ([[[[0.299]], [[0.587]], [[0.114]]]] is 1 x 3 x 1 x 1) and its values
void flatten(const json& j, std::vector<int64_t>& shape, std::vector<float>& vals, size_t depth = 0) {
    if (j.is_array()) {
        if (shape.size() == depth) shape.push_back((int64_t)j.size());
        for (const auto& x : j) flatten(x, shape, vals, depth + 1);
    } else {
        vals.push_back(j.get<float>());
    }
}

// ggml dimension of an ONNX axis: plain order, and channels-first (C, W, H, N)
int gdim(int rank, int64_t axis) { return rank - 1 - (int)(axis < 0 ? axis + rank : axis); }
int cfdim(int rank, int64_t axis) {
    const int a = (int)(axis < 0 ? axis + rank : axis);
    if (rank == 4) { const int m[4] = {3, 0, 2, 1}; return m[a]; }   // N C H W
    const int m[3] = {3, 0, 1};                                       // N C W
    return m[a];
}

// a channels-first tensor as a ggml view of logical (W, H, C, N) whose channels are contiguous,
// the form conv_2d_dw_direct takes for its fast path; and back
ggml_tensor* cf_to_whcn_view(ggml_context* ctx, ggml_tensor* t) { return ggml_permute(ctx, t, 2, 0, 1, 3); }
ggml_tensor* whcn_view_to_cf(ggml_context* ctx, ggml_tensor* t) { return ggml_permute(ctx, t, 1, 2, 0, 3); }

// numpy broadcasting for the shapes these graphs use: the smaller operand repeats into the larger
bool broadcast(ggml_tensor*& a, ggml_tensor*& b, bool commutative) {
    if (ggml_can_repeat(b, a)) return true;
    if (commutative && ggml_can_repeat(a, b)) { std::swap(a, b); return true; }
    return false;
}

ggml_tensor* reshape(ggml_context* ctx, ggml_tensor* t, const std::vector<int64_t>& ne) {
    t = ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
    switch (ne.size()) {
    case 1: return ggml_reshape_1d(ctx, t, ne[0]);
    case 2: return ggml_reshape_2d(ctx, t, ne[0], ne[1]);
    case 3: return ggml_reshape_3d(ctx, t, ne[0], ne[1], ne[2]);
    default: return ggml_reshape_4d(ctx, t, ne[0], ne[1], ne[2], ne[3]);
    }
}

// The fused epilogue: x + bias (per channel, channels-first), ReLU, FakeQuant; one pass, split
// over the threads. The arithmetic is the separate ops' own, in the same order, so the results
// do not change: ggml_add, then (x > 0 ? x : 0) as ggml_relu, then x / s rounded half to even.
// Pixel by pixel, channel by channel (channels are contiguous), so the inner loop has no
// division to find the channel and no branch: the compiler vectorises it. Rounding half to even
// uses the float trick (q + 1.5 * 2^23) - 1.5 * 2^23, which vectorises on any CPU (nearbyint
// needs SSE4.1 on x86) and is exact for |q| < 2^22; anything larger is clamped either way.
// kiln is built without -ffast-math, so the compiler may not cancel the two additions.
template <bool BIAS, bool RELU, bool FQ>
void epilogue_rows(const float* x, const float* bias, float* y, int64_t r0, int64_t r1, int64_t c, const Node::Epilogue& e) {
    constexpr float kMagic = 12582912.f;     // 1.5 * 2^23
    const float s = e.scale, lo = e.qmin - e.zero_point, hi = e.qmax - e.zero_point;
    for (int64_t r = r0; r < r1; ++r) {
        const float* xr = x + r * c;
        float* yr = y + r * c;
        for (int64_t i = 0; i < c; ++i) {
            float v = xr[i];
            if (BIAS) v += bias[i];
            if (RELU) v = v > 0.f ? v : 0.f;
            if (FQ) {
                float q = (v / s + kMagic) - kMagic;
                q = q < lo ? lo : q;
                q = q > hi ? hi : q;
                v = q * s;
            }
            yr[i] = v;
        }
    }
}

void epilogue_op(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b, int ith, int nth, void* ud) {
    const auto& e = *static_cast<const Node::Epilogue*>(ud);
    const int64_t c = dst->ne[0], rows = ggml_nelements(dst) / c;
    const int64_t per = (rows + nth - 1) / nth, r0 = ith * per, r1 = std::min(rows, r0 + per);
    if (r0 >= r1) return;
    const float* x = (const float*)a->data;
    const float* bias = b ? (const float*)b->data : nullptr;
    float* y = (float*)dst->data;
    const int k = (bias ? 4 : 0) | (e.relu ? 2 : 0) | (e.fq ? 1 : 0);
    switch (k) {
    case 0: epilogue_rows<false, false, false>(x, bias, y, r0, r1, c, e); break;
    case 1: epilogue_rows<false, false, true >(x, bias, y, r0, r1, c, e); break;
    case 2: epilogue_rows<false, true,  false>(x, bias, y, r0, r1, c, e); break;
    case 3: epilogue_rows<false, true,  true >(x, bias, y, r0, r1, c, e); break;
    case 4: epilogue_rows<true,  false, false>(x, bias, y, r0, r1, c, e); break;
    case 5: epilogue_rows<true,  false, true >(x, bias, y, r0, r1, c, e); break;
    case 6: epilogue_rows<true,  true,  false>(x, bias, y, r0, r1, c, e); break;
    default: epilogue_rows<true, true,  true >(x, bias, y, r0, r1, c, e); break;
    }
}
void epilogue_op1(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void* ud) { epilogue_op(dst, a, nullptr, ith, nth, ud); }

// A convolution over a one-channel image, written channels-first: each network's first layer
// (3x3, stride 2, 1 -> 16 channels). src[0] is the image (W, H, 1, N), src[1] the kernel
// rearranged with the output channels contiguous (Cout, 1, KW, KH). For each output pixel the
// inner loop runs over the output channels, which the compiler vectorises. As im2col plus a
// matrix multiply this layer took a quarter of the detector's time: 9 inputs per pixel is too
// thin a matrix for the multiply to pay off.
void conv1ch_op(ggml_tensor* dst, int ith, int nth, void* ud) {
    const auto& g = *static_cast<const Node::ConvGeom*>(ud);
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const int64_t cout = dst->ne[0], ow = dst->ne[1], oh = dst->ne[2], nb = dst->ne[3];
    const int64_t iw = x->ne[0], ih = x->ne[1], kw = k->ne[2], kh = k->ne[3];
    const float* xd = (const float*)x->data;
    const float* kd = (const float*)k->data;
    float* yd = (float*)dst->data;
    const int64_t rows = oh * nb, per = (rows + nth - 1) / nth, r0 = ith * per, r1 = std::min(rows, r0 + per);
    std::vector<float> acc((size_t)cout);
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t n = r / oh, oy = r % oh;
        const float* img = xd + n * iw * ih;
        for (int64_t ox = 0; ox < ow; ++ox) {
            std::fill(acc.begin(), acc.end(), 0.f);
            for (int64_t ky = 0; ky < kh; ++ky) {
                const int64_t iy = oy * g.s1 + ky - g.p1;
                if (iy < 0 || iy >= ih) continue;
                for (int64_t kx = 0; kx < kw; ++kx) {
                    const int64_t ix = ox * g.s0 + kx - g.p0;
                    if (ix < 0 || ix >= iw) continue;
                    const float v = img[iy * iw + ix];
                    const float* kc = kd + (ky * kw + kx) * cout;
                    for (int64_t c = 0; c < cout; ++c) acc[(size_t)c] += v * kc[c];
                }
            }
            std::copy(acc.begin(), acc.end(), yd + ((n * oh + oy) * ow + ox) * cout);
        }
    }
}

} // namespace

bool Graph::load(const gguf_context* g, ggml_context* weights, const std::string& name, std::string& err) {
    name_ = name;
    const std::string key = "ocria." + name + ".graph";
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) { err = "missing " + key; return false; }
    json gj;
    try { gj = json::parse(gguf_get_val_str(g, id)); }
    catch (const std::exception& e) { err = key + ": " + e.what(); return false; }
    input_ = gj["input"].get<std::string>();
    for (const auto& o : gj["outputs"]) outputs_.push_back(o.get<std::string>());

    // constants that are real tensors (the detector's grey weights), and the depthwise kernels
    // rearranged channels-first, live in a context of their own
    std::vector<std::pair<std::string, std::pair<std::vector<int64_t>, std::vector<float>>>> consts;
    std::vector<std::string> dw_kernels;

    for (const auto& nj : gj["nodes"]) {
        Node n;
        n.op = nj["op"].get<std::string>();
        for (const auto& x : nj["in"]) n.in.push_back(x.get<std::string>());
        for (const auto& x : nj["out"]) n.out.push_back(x.get<std::string>());
        const json& a = nj["attrs"];
        n.ints_kernel  = ints(a, "kernel_shape");
        n.ints_strides = ints(a, "strides");
        n.ints_pads    = ints(a, "pads");
        n.ints_axes    = ints(a, "axes");
        if (a.contains("group"))    n.group    = a["group"].get<int64_t>();
        if (a.contains("axis"))     n.axis     = a["axis"].get<int64_t>();
        if (a.contains("keepdims")) n.keepdims = a["keepdims"].get<int64_t>();
        if (a.contains("transB"))   n.trans_b  = a["transB"].get<int64_t>();
        if (n.op == "FakeQuant") {
            n.scale = a["scale"].get<float>();
            n.zero_point = a["zero_point"].get<float>();
            n.qmin = a["qmin"].get<float>();
            n.qmax = a["qmax"].get<float>();
        }
        for (auto& in : n.in) {
            if (in.rfind("const:", 0) != 0) continue;
            const json& cj = a["const"][in.substr(6)];
            std::vector<int64_t> shape; std::vector<float> vals;
            flatten(cj, shape, vals);
            if (n.op == "Resize")       { n.resize_scales = vals; in.clear(); }
            else if (n.op == "Squeeze" || n.op.rfind("Reduce", 0) == 0) {
                for (float v : vals) n.ints_axes.push_back((int64_t)v);
                in.clear();
            } else {
                consts.push_back({in, {shape, vals}});
            }
        }
        for (const auto& in : n.in) {
            if (in.rfind("w:", 0) != 0) continue;
            ggml_tensor* t = ggml_get_tensor(weights, in.substr(2).c_str());
            if (!t) { err = name + ": missing tensor " + in; return false; }
            weights_[in] = t;
        }
        if (n.op == "Conv") {
            const auto& d = a.contains("dilations") ? a["dilations"] : json::array();
            for (const auto& x : d) if (x.get<int64_t>() != 1) { err = name + ": dilated convolution"; return false; }
            const ggml_tensor* w = weights_[n.in[1]];
            if (n.group > 1) {
                if (w->ne[2] != 1 || w->ne[3] != n.group) { err = name + ": grouped convolution that is not depthwise"; return false; }
                dw_kernels.push_back(n.in[1]);
            } else if (w->ne[2] == 1 && !(w->ne[0] == 1 && w->ne[1] == 1)) {
                dw_kernels.push_back(n.in[1]);      // one input channel: the same rearrangement
            }
            if (n.ints_kernel.size() == 1) {
                n.geom.s0 = n.ints_strides.empty() ? 1 : (int)n.ints_strides[0];
                n.geom.p0 = n.ints_pads.empty() ? 0 : (int)n.ints_pads[0];
            } else {
                n.geom.s1 = (int)n.ints_strides[0]; n.geom.s0 = (int)n.ints_strides[1];
                n.geom.p1 = (int)n.ints_pads[0];    n.geom.p0 = (int)n.ints_pads[1];
            }
        }
        nodes_.push_back(std::move(n));
    }

    size_t bytes = 1024;
    for (const auto& c : consts) bytes += c.second.second.size() * sizeof(float) + ggml_tensor_overhead();
    for (const auto& k : dw_kernels) bytes += ggml_nbytes(weights_[k]) + ggml_tensor_overhead();
    ggml_init_params ip = {bytes, nullptr, false};
    const_ctx_ = ggml_init(ip);
    for (const auto& [nm, sv] : consts) {
        const auto& [shape, vals] = sv;
        const int r = (int)shape.size();
        ggml_tensor* t;
        if (r == 4) {
            // an image-shaped constant (1, C, 1, 1) goes channels-first: (C, 1, 1, 1)
            if (shape[0] != 1 || shape[2] != 1 || shape[3] != 1) { err = name + ": a 4-D constant other than (1, C, 1, 1)"; return false; }
            t = ggml_new_tensor_4d(const_ctx_, GGML_TYPE_F32, shape[1], 1, 1, 1);
            const_cf_[nm] = true;
        } else {
            int64_t ne[4] = {1, 1, 1, 1};
            for (int d = 0; d < r; ++d) ne[d] = shape[r - 1 - d];
            t = ggml_new_tensor(const_ctx_, GGML_TYPE_F32, std::max(r, 1), ne);
        }
        memcpy(t->data, vals.data(), vals.size() * sizeof(float));
        weights_[nm] = t;
        const_rank_[nm] = r;
    }
    // depthwise kernel (KW, KH, 1, C) -> memory (C fastest, then KW, then KH): what
    // conv_2d_dw_direct's channels-first path reads, as ggml's own test arranges it
    for (const auto& k : dw_kernels) {
        if (dw_.count(k)) continue;
        const ggml_tensor* w = weights_[k];
        const int64_t kw = w->ne[0], kh = w->ne[1], c = w->ne[3];
        ggml_tensor* t = ggml_new_tensor_4d(const_ctx_, GGML_TYPE_F32, c, 1, kw, kh);
        const float* src = (const float*)w->data;
        float* dst = (float*)t->data;
        for (int64_t ci = 0; ci < c; ++ci)
            for (int64_t y = 0; y < kh; ++y)
                for (int64_t x = 0; x < kw; ++x) dst[(y * kw + x) * c + ci] = src[(ci * kh + y) * kw + x];
        dw_[k] = t;
    }
    if (!nodes_.empty() && nodes_[0].op == "FakeQuant" && nodes_[0].in[0] == input_) input_quant_ = &nodes_[0];

    // fusion: who reads each value (graph outputs count as a reader)
    std::map<std::string, int> readers;
    std::map<std::string, size_t> reader_of;
    for (size_t i = 0; i < nodes_.size(); ++i)
        for (const auto& in : nodes_[i].in) if (!in.empty()) { ++readers[in]; reader_of[in] = i; }
    for (const auto& o : outputs_) readers[o] += 2;
    auto only_reader = [&](const std::string& v, const char* op) -> Node* {
        if (readers[v] != 1) return nullptr;
        Node& r = nodes_[reader_of[v]];
        return r.op == op ? &r : nullptr;
    };
    for (Node& n : nodes_) {
        if (n.op != "Conv") continue;
        n.epi.bias = n.in.size() > 2 && !n.in[2].empty();
        std::string cur = n.out[0];
        if (Node* r = only_reader(cur, "Relu")) { n.epi.relu = true; r->skip = true; cur = r->out[0]; }
        if (Node* q = only_reader(cur, "FakeQuant")) {
            n.epi.fq = true; n.epi.scale = q->scale; n.epi.zero_point = q->zero_point; n.epi.qmin = q->qmin; n.epi.qmax = q->qmax;
            q->skip = true; cur = q->out[0];
        }
        n.fused_out = cur;
    }
    for (Node& n : nodes_)
        if (n.op == "FakeQuant" && !n.skip && &n != input_quant_) {
            n.epi.fq = true; n.epi.scale = n.scale; n.epi.zero_point = n.zero_point; n.epi.qmin = n.qmin; n.epi.qmax = n.qmax;
        }
    // the detector's grey conversion: Mul(input, (1, 3, 1, 1) weights), then ReduceSum over the channels
    if (nodes_.size() >= 2 && nodes_[0].op == "Mul" && nodes_[0].in[0] == input_ && const_cf_.count(nodes_[0].in[1]) &&
        nodes_[1].op == "ReduceSum" && nodes_[1].in[0] == nodes_[0].out[0] && nodes_[1].keepdims &&
        nodes_[1].ints_axes.size() == 1 && nodes_[1].ints_axes[0] == 1 && readers[nodes_[0].out[0]] == 1) {
        const ggml_tensor* w = weights_[nodes_[0].in[1]];
        grey_w_.assign((const float*)w->data, (const float*)w->data + w->ne[0]);
        grey_out_ = nodes_[1].out[0];
        nodes_[0].skip = nodes_[1].skip = true;
    }
    return true;
}

Graph::~Graph() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (const_ctx_) ggml_free(const_ctx_);
}

Graph::Graph(Graph&& o) noexcept { *this = std::move(o); }
Graph& Graph::operator=(Graph&& o) noexcept {
    if (this == &o) return *this;
    if (galloc_) ggml_gallocr_free(galloc_);
    if (const_ctx_) ggml_free(const_ctx_);
    galloc_ = o.galloc_; o.galloc_ = nullptr;
    name_ = std::move(o.name_); input_ = std::move(o.input_); outputs_ = std::move(o.outputs_);
    nodes_ = std::move(o.nodes_); weights_ = std::move(o.weights_); const_rank_ = std::move(o.const_rank_);
    const_cf_ = std::move(o.const_cf_); dw_ = std::move(o.dw_);
    grey_w_ = std::move(o.grey_w_); grey_out_ = std::move(o.grey_out_);
    const_ctx_ = o.const_ctx_; o.const_ctx_ = nullptr;
    input_quant_ = (!nodes_.empty() && nodes_[0].op == "FakeQuant" && nodes_[0].in[0] == input_) ? &nodes_[0] : nullptr;
    return *this;
}

bool Graph::run(ggml_backend_t cpu, const std::vector<int64_t>& shape, const float* input,
                std::vector<Tensor>& outputs, std::string& err, bool channels_first) const {
    const size_t n_nodes = nodes_.size() * 12 + 64;
    ggml_init_params ip = {ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    struct Guard { ggml_context* c; ~Guard() { ggml_free(c); } } guard{ctx};

    std::map<std::string, Val> vals;
    const int in_rank = (int)shape.size();
    if (in_rank != 4) { err = name_ + ": the input must be (N, C, H, W)"; return false; }
    const int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    const bool grey = !grey_out_.empty();
    if (grey && (int64_t)grey_w_.size() != C) { err = name_ + ": the grey weights do not match the input's channels"; return false; }
    const int64_t C_in = grey ? 1 : C;
    ggml_tensor* in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, C_in, W, H, N);     // channels-first
    ggml_set_input(in);
    vals[grey ? grey_out_ : input_] = {in, 4, true};

    auto get = [&](const std::string& n, Val& v) -> bool {
        if (auto it = vals.find(n); it != vals.end()) { v = it->second; return true; }
        if (auto it = weights_.find(n); it != weights_.end()) {
            auto r = const_rank_.find(n);
            v = {it->second, r != const_rank_.end() ? r->second : ggml_n_dims(it->second), const_cf_.count(n) > 0};
            return true;
        }
        err = name_ + ": no value " + n;
        return false;
    };

    for (const Node& n : nodes_) {
        if (n.skip) continue;               // folded into an earlier node
        if (&n == input_quant_) {           // done on the host below
            vals[n.out[0]] = vals[input_];
            continue;
        }
        Val x;
        if (!get(n.in[0], x)) return false;
        Val y = x;
        const std::string& op = n.op;
        if (op == "Conv") {
            if (!x.cf) { err = name_ + ": convolution on a tensor that is not an image"; return false; }
            Val w, b;
            if (!get(n.in[1], w)) return false;
            int s0, s1, p0, p1;     // ggml order: 0 = width, 1 = height
            if (x.rank == 3) {
                s0 = n.ints_strides.empty() ? 1 : (int)n.ints_strides[0]; s1 = 1;
                p0 = n.ints_pads.empty() ? 0 : (int)n.ints_pads[0]; p1 = 0;
                if (n.ints_pads.size() == 2 && n.ints_pads[0] != n.ints_pads[1]) { err = name_ + ": uneven padding"; return false; }
            } else {
                s1 = (int)n.ints_strides[0]; s0 = (int)n.ints_strides[1];
                p1 = (int)n.ints_pads[0];    p0 = (int)n.ints_pads[1];
                if (n.ints_pads[0] != n.ints_pads[2] || n.ints_pads[1] != n.ints_pads[3]) { err = name_ + ": uneven padding"; return false; }
            }
            const int64_t kw = w.t->ne[0], kh = w.t->ne[1], cout = w.t->ne[3];
            ggml_tensor* r;
            if (n.group > 1) {
                // depthwise: the channels-first SIMD path, with the kernel rearranged at load
                ggml_tensor* k = ggml_permute(ctx, dw_.at(n.in[1]), 3, 2, 0, 1);
                r = ggml_conv_2d_dw_direct(ctx, k, cf_to_whcn_view(ctx, x.t), s0, s1, p0, p1, 1, 1);
                r = whcn_view_to_cf(ctx, r);
            } else if (kw == 1 && kh == 1 && s0 == 1 && s1 == 1 && p0 == 0 && p1 == 0) {
                // 1x1: one matrix multiply over every pixel
                const int64_t cin = x.t->ne[0], px = x.t->ne[1] * x.t->ne[2] * x.t->ne[3];
                ggml_tensor* m = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w.t, cin, cout), reshape(ctx, x.t, {cin, px}));
                r = ggml_reshape_4d(ctx, m, cout, x.t->ne[1], x.t->ne[2], x.t->ne[3]);
            } else if (x.t->ne[0] == 1 && dw_.count(n.in[1])) {
                // a one-channel image (each network's first layer): the image's bytes are the same
                // in both layouts, and conv1ch_op writes channels-first directly
                ggml_tensor* xw = ggml_reshape_4d(ctx, x.t, x.t->ne[1], x.t->ne[2], 1, x.t->ne[3]);
                const int64_t ow = (xw->ne[0] + 2 * p0 - kw) / s0 + 1, oh = (xw->ne[1] + 2 * p1 - kh) / s1 + 1;
                ggml_tensor* args[2] = {xw, dw_.at(n.in[1])};
                r = ggml_custom_4d(ctx, GGML_TYPE_F32, cout, ow, oh, xw->ne[3], args, 2, conv1ch_op, GGML_N_TASKS_MAX, (void*)&n.geom);
            } else {
                // anything else: lay every patch out as a row (im2col, F32) and multiply once,
                // which gives the channels-first result directly
                ggml_tensor* xw = ggml_cont(ctx, cf_to_whcn_view(ctx, x.t));
                ggml_tensor* cols = ggml_im2col(ctx, w.t, xw, s0, s1, p0, p1, 1, 1, true, GGML_TYPE_F32);   // (K, OW, OH, N)
                const int64_t k = cols->ne[0], ow = cols->ne[1], oh = cols->ne[2], nb = cols->ne[3];
                ggml_tensor* m = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w.t, k, cout), ggml_reshape_2d(ctx, cols, k, ow * oh * nb));
                r = ggml_reshape_4d(ctx, m, cout, ow, oh, nb);
            }
            if (!ggml_is_contiguous(r)) r = ggml_cont(ctx, r);
            if (n.epi.bias) {
                if (!get(n.in[2], b)) return false;
                r = ggml_map_custom2(ctx, r, b.t, epilogue_op, GGML_N_TASKS_MAX, (void*)&n.epi);
            } else if (n.epi.relu || n.epi.fq) {
                r = ggml_map_custom1(ctx, r, epilogue_op1, GGML_N_TASKS_MAX, (void*)&n.epi);
            }
            vals[n.fused_out] = {r, x.rank, true};
            continue;
        } else if (op == "Relu") {
            y.t = ggml_relu(ctx, x.t);
        } else if (op == "Sigmoid") {
            y.t = ggml_sigmoid(ctx, x.t);
        } else if (op == "Exp") {
            y.t = ggml_exp(ctx, x.t);
        } else if (op == "Add" || op == "Sub" || op == "Mul") {
            Val z;
            if (!get(n.in[1], z)) return false;
            if (x.cf != z.cf) { err = name_ + ": " + op + " of an image and a non-image"; return false; }
            ggml_tensor *a = x.t, *b = z.t;
            if (!broadcast(a, b, op != "Sub")) { err = name_ + ": " + op + " shapes do not broadcast"; return false; }
            y.t = op == "Add" ? ggml_add(ctx, a, b) : op == "Mul" ? ggml_mul(ctx, a, b) : ggml_sub(ctx, a, b);
            y.rank = std::max(x.rank, z.rank);
        } else if (op == "FakeQuant") {
            // round to the 8-bit grid and back: clamp(round(x / s) + zp, qmin, qmax) - zp, times s.
            // As ONNX: a true division by s, and rounding half to even (std::nearbyint), in one
            // pass. As five ggml ops (div, two adds of 1.5 * 2^23 for the rounding, clamp, scale)
            // it took two thirds of the Latin recognizer's time.
            ggml_tensor* t = ggml_is_contiguous(x.t) ? x.t : ggml_cont(ctx, x.t);
            y.t = ggml_map_custom1(ctx, t, epilogue_op1, GGML_N_TASKS_MAX, (void*)&n.epi);
        } else if (op == "Resize") {
            const auto& s = n.resize_scales;
            if (!x.cf || x.rank != 4 || s.size() != 4 || s[0] != 1.f || s[1] != 1.f || s[2] != s[3] || s[2] != std::floor(s[2])) {
                err = name_ + ": only an integer 2-D nearest resize is supported";
                return false;
            }
            const int f = (int)s[2];
            y.t = ggml_interpolate(ctx, x.t, x.t->ne[0], x.t->ne[1] * f, x.t->ne[2] * f, x.t->ne[3], GGML_SCALE_MODE_NEAREST);
        } else if (op == "Concat") {
            ggml_tensor* t = x.t;
            for (size_t i = 1; i < n.in.size(); ++i) {
                Val z;
                if (!get(n.in[i], z)) return false;
                t = ggml_concat(ctx, t, z.t, x.cf ? cfdim(x.rank, n.axis) : gdim(x.rank, n.axis));
            }
            y.t = t;
        } else if (op == "ReduceSum" || op == "ReduceMean" || op == "ReduceMax") {
            if (n.ints_axes.size() != 1) { err = name_ + ": reduce over several axes"; return false; }
            if (!x.cf) { err = name_ + ": reduce on a tensor that is not an image"; return false; }
            const int d = cfdim(x.rank, n.ints_axes[0]);
            // bring the axis to dimension 0, reduce it there
            ggml_tensor* t = x.t;
            if (d != 0) {
                int perm[4] = {0, 1, 2, 3};
                std::swap(perm[0], perm[d]);
                t = ggml_cont(ctx, ggml_permute(ctx, t, perm[0], perm[1], perm[2], perm[3]));
            }
            if (op == "ReduceSum")       t = ggml_sum_rows(ctx, t);
            else if (op == "ReduceMean") t = ggml_mean(ctx, t);
            else                         t = ggml_pool_1d(ctx, t, GGML_OP_POOL_MAX, (int)t->ne[0], (int)t->ne[0], 0);
            if (n.keepdims) {
                if (d != 0) {
                    int perm[4] = {0, 1, 2, 3};
                    std::swap(perm[0], perm[d]);
                    t = ggml_cont(ctx, ggml_permute(ctx, t, perm[0], perm[1], perm[2], perm[3]));
                }
                y.t = t;
            } else {
                // what remains, in plain order: these graphs reduce the last spatial axis of
                // (N, C, W) and keep (N, C)
                if (x.rank != 3 || d != 1) { err = name_ + ": reduce without keepdims other than over W of (N, C, W)"; return false; }
                y = {reshape(ctx, t, {x.t->ne[0], x.t->ne[3]}), 2, false};
            }
        } else if (op == "Squeeze") {
            // (N, C, 1, W) -> (N, C, W): channels-first keeps H = 1, nothing moves
            if (!x.cf || x.rank != 4 || n.ints_axes.size() != 1 || (n.ints_axes[0] != 2 && n.ints_axes[0] != -2) || x.t->ne[2] != 1) {
                err = name_ + ": only the height axis of size 1 can be squeezed";
                return false;
            }
            y.rank = 3;
        } else if (op == "Gemm") {
            Val w, b;
            if (!get(n.in[1], w)) return false;
            if (!n.trans_b || x.cf) { err = name_ + ": Gemm without transB, or on an image"; return false; }
            ggml_tensor* t = ggml_mul_mat(ctx, w.t, x.t);       // (N, K) x (M, K)^T -> (N, M)
            if (n.in.size() > 2 && !n.in[2].empty()) {
                if (!get(n.in[2], b)) return false;
                t = ggml_add(ctx, t, b.t);
            }
            y = {t, 2, false};
        } else {
            err = name_ + ": op " + op + " is not supported";
            return false;
        }
        vals[n.out[0]] = y;
    }

    // outputs in ONNX order: an image tensor goes back to (W, H, C, N) in ggml, i.e. ONNX's
    // (N, C, H, W) with W fastest
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, n_nodes, false);
    struct Out { ggml_tensor* t; std::vector<int64_t> shape; bool cf; };
    std::vector<Out> outs;
    for (const auto& o : outputs_) {
        Val v;
        if (!get(o, v)) return false;
        ggml_tensor* t = v.t;
        std::vector<int64_t> sh;
        bool kept = false;
        if (v.cf && v.rank == 3 && channels_first) {
            // as stored: no copy, and the reader finds one position's values side by side
            t = ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
            sh = {t->ne[3], t->ne[0], t->ne[1]};
            kept = true;
        } else if (v.cf) {
            if (v.rank == 4) { t = ggml_cont(ctx, cf_to_whcn_view(ctx, t)); sh = {t->ne[3], t->ne[2], t->ne[1], t->ne[0]}; }
            else             { t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 0, 2, 3)); sh = {t->ne[3], t->ne[1], t->ne[0]}; }
        } else {
            t = ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
            for (int d = v.rank - 1; d >= 0; --d) sh.push_back(t->ne[d]);
        }
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
        outs.push_back({t, sh, kept});
    }
    if (!galloc_) galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    if (!ggml_gallocr_alloc_graph(galloc_, gf)) { err = name_ + ": out of memory for the graph"; return false; }

    // the input arrives as ONNX (N, C, H, W); write it channels-first, with the input FakeQuant
    // (QuantizeLinear rounds half to even) applied on the way
    const size_t n_in = (size_t)(N * C_in * H * W);
    std::vector<float> xs(n_in);
    const Node* q = input_quant_;
    if (grey) {
        // as the graph's Mul then ReduceSum: products in float, summed in double (ggml_vec_sum_f32)
        const int64_t plane = H * W;
        for (int64_t ni = 0; ni < N; ++ni)
            for (int64_t i = 0; i < plane; ++i) {
                double s = 0;
                for (int64_t ci = 0; ci < C; ++ci) s += (double)(input[(ni * C + ci) * plane + i] * grey_w_[ci]);
                xs[ni * plane + i] = (float)s;
            }
    } else {
        for (int64_t ni = 0; ni < N; ++ni)
            for (int64_t ci = 0; ci < C; ++ci)
                for (int64_t hi = 0; hi < H; ++hi) {
                    const float* src = input + ((ni * C + ci) * H + hi) * W;
                    float* dst = xs.data() + ((ni * H + hi) * W) * C + ci;
                    for (int64_t wi = 0; wi < W; ++wi) {
                        float v = src[wi];
                        if (q) v = (std::clamp(std::nearbyint(v / q->scale) + q->zero_point, q->qmin, q->qmax) - q->zero_point) * q->scale;
                        dst[wi * C] = v;
                    }
                }
    }
    ggml_backend_tensor_set(in, xs.data(), 0, n_in * sizeof(float));
    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) { err = name_ + ": compute failed or was stopped"; return false; }

    outputs.clear();
    for (const auto& [t, sh, kept] : outs) {
        Tensor o;
        o.shape = sh;
        o.channels_first = kept;
        o.data.resize((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, o.data.data(), 0, o.data.size() * sizeof(float));
        outputs.push_back(std::move(o));
    }
    return true;
}

} // namespace kiln::ocr
