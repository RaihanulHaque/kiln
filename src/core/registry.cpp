#include "internal.h"

namespace kiln {
Backend* make_llamacpp_backend();   // backends/llamacpp

const std::vector<Backend*>& backends() {
    static std::vector<Backend*> list = [] {
        std::vector<Backend*> v;
        // Priority order. NPU backends go first once they exist; llama.cpp is the floor.
        v.push_back(make_llamacpp_backend());
        return v;
    }();
    return list;
}
}
