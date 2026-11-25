// Runs one OCRia-nano model (detector, router or a recognizer) on ggml. The graph comes from the
// GGUF written by scripts/convert_ocria.py: a JSON list of ONNX-style nodes and the weights they
// read. Each run builds the ggml graph for the input's size (the detector's input changes with
// every page, a recognizer's with every batch width), computes it on the CPU and copies the
// outputs out.
//
// Shapes follow ONNX (N, C, H, W); a ggml tensor holds them in reverse (ne = W, H, C, N), so an
// ONNX axis a of a rank-r tensor is ggml dimension r - 1 - a.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;
struct ggml_gallocr;
typedef struct ggml_backend* ggml_backend_t;

namespace kiln::ocr {

struct Tensor {                    // an output, shape in ONNX order
    std::vector<int64_t> shape;
    std::vector<float>   data;
    // run(..., channels_first = true): a rank-3 (N, C, W) output comes as stored, the C values of
    // one position next to each other ((n * W + w) * C + c); else ONNX's order, W fastest
    bool channels_first = false;
};

struct Node {
    std::string op;
    std::vector<std::string> in, out;
    // attributes the ops read, already parsed
    std::vector<int64_t> ints_kernel, ints_strides, ints_pads, ints_axes;
    int64_t group = 1, axis = 0, keepdims = 1, trans_b = 0;
    float   scale = 1.f, zero_point = 0.f, qmin = 0.f, qmax = 0.f;
    std::vector<float> resize_scales;
    // Fusion, decided at load: a convolution's bias, the ReLU and the FakeQuant that follow it
    // (when each is the only reader of the one before) run as one pass over memory ("epilogue");
    // the nodes folded in are skipped and the chain's last output names the result.
    struct Epilogue {
        bool  bias = false, relu = false, fq = false;
        float scale = 1.f, zero_point = 0.f, qmin = 0.f, qmax = 0.f;
    } epi;
    // stride and padding of a convolution in ggml order (0 = width, 1 = height), for the
    // one-channel first-layer kernel
    struct ConvGeom { int s0 = 1, s1 = 1, p0 = 0, p1 = 0; } geom;
    std::string fused_out;      // set on a fused convolution: the name its result goes under
    bool skip = false;          // folded into an earlier node
};

class Graph {
public:
    Graph() = default;
    ~Graph();
    Graph(Graph&& o) noexcept;
    Graph& operator=(Graph&& o) noexcept;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    // name: "det", "router", "rec_latin", ...; weights: the tensors of the whole file
    bool load(const gguf_context* g, ggml_context* weights, const std::string& name, std::string& err);
    const std::string& name() const { return name_; }

    // Runs the graph on `input` (ONNX shape `shape`) and returns its outputs in order.
    // The backend's abort callback (kiln_cancel) makes the run fail.
    bool run(ggml_backend_t cpu, const std::vector<int64_t>& shape, const float* input,
             std::vector<Tensor>& outputs, std::string& err, bool channels_first = false) const;

private:
    std::string name_, input_;
    std::vector<std::string> outputs_;
    std::vector<Node> nodes_;
    std::map<std::string, ggml_tensor*> weights_;       // "w:<name>" and "const:<name>"
    std::map<std::string, int> const_rank_;             // ONNX rank of each constant
    std::map<std::string, bool> const_cf_;              // constants stored channels-first
    std::map<std::string, ggml_tensor*> dw_;            // depthwise kernels rearranged channels-first
    ggml_context* const_ctx_ = nullptr;                 // holds the constants' and rearranged kernels' data
    // the activations' memory, kept from one run to the next: fresh memory costs a page fault per
    // 4 KB on first touch, which on the detector was a fifth of a run. Calls on one model are
    // serialised (OcrModel::mutex), so one allocator per graph is enough.
    mutable ggml_gallocr* galloc_ = nullptr;
    // a FakeQuant on the graph input is done on the host, with round-half-even as ONNX does
    // (the input is x / 127.5 - 1, so half the pixels sit exactly between two levels)
    const Node* input_quant_ = nullptr;
    // the detector starts by turning RGB to grey (Mul by three weights, ReduceSum over the
    // channels); that is done on the host while the page is copied in, and the graph starts at
    // the grey image
    std::vector<float> grey_w_;
    std::string grey_out_;
};

} // namespace kiln::ocr
