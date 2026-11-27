#!/bin/sh
# Builds encprof against kiln's host build of ggml (run `cmake --build --preset host` first).
# whisper.cpp's parakeet.cpp is compiled in, so the patch in ../parakeet-bias-frontend.patch must be applied.
# Linux: OpenMP, as kiln. macOS: no OpenMP (kiln's Mac build has none); the ggml registry also
# pulls in the Metal and BLAS backends, so those libraries and frameworks are linked too. The
# profiler itself runs the CPU backend.
set -e
K=$(cd "$(dirname "$0")/../../.." && pwd)
G=$K/build/host/third_party/llama.cpp/ggml/src
INC="-I$K/third_party/whisper.cpp/include -I$K/third_party/whisper.cpp/src -I$K/third_party/llama.cpp/ggml/include"
if [ "$(uname)" = Darwin ]; then
  c++ -O2 -std=c++17 -mcpu=native -DPARAKEET_VERSION='"prof"' $INC \
    "$K/experiments/parakeet/prof/encprof.cpp" -o "$K/build/asr/prof/encprof" \
    "$G/libggml.a" "$G/libggml-cpu.a" "$G/ggml-metal/libggml-metal.a" "$G/ggml-blas/libggml-blas.a" "$G/libggml-base.a" \
    -framework Accelerate -framework Metal -framework MetalKit -framework Foundation
else
  g++ -O2 -std=c++17 -march=native -fopenmp -DPARAKEET_VERSION='"prof"' $INC \
    "$K/experiments/parakeet/prof/encprof.cpp" -o "$K/build/asr/prof/encprof" \
    "$G/libggml.a" "$G/libggml-cpu.a" "$G/libggml-base.a" -lpthread
fi
