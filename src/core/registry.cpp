#include "internal.h"

namespace kiln {
Backend* make_llamacpp_backend();   // backends/llamacpp
Backend* make_whispercpp_backend(); // backends/whispercpp: Whisper and Parakeet speech models

const std::vector<Backend*>& backends() {
    static std::vector<Backend*> list = [] {
        std::vector<Backend*> v;
        // Priority order. NPU backends go first once they exist; llama.cpp is the floor.
        v.push_back(make_llamacpp_backend());
        v.push_back(make_whispercpp_backend());   // reads only its own file format, so order does not matter
        return v;
    }();
    return list;
}
}
