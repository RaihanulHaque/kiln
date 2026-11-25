#include "internal.h"

namespace kiln {
Backend* make_llamacpp_backend();   // backends/llamacpp
Backend* make_whispercpp_backend(); // backends/whispercpp: Whisper and Parakeet speech models
Backend* make_ocr_backend();        // backends/ocr: OCRia-nano text recognition (a GGUF llama.cpp cannot read)

const std::vector<Backend*>& backends() {
    static std::vector<Backend*> list = [] {
        std::vector<Backend*> v;
        // Priority order. NPU backends go first once they exist; llama.cpp is the floor.
        // The OCR backend takes only GGUFs whose architecture is "ocria", so it goes before
        // llama.cpp, which would take any GGUF.
        v.push_back(make_ocr_backend());
        v.push_back(make_llamacpp_backend());
        v.push_back(make_whispercpp_backend());   // reads only its own file format, so order does not matter
        return v;
    }();
    return list;
}
}
