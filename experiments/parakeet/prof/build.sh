#!/bin/sh
# Builds encprof against kiln's host build of ggml (run `cmake --build --preset host` first).
# whisper.cpp's parakeet.cpp is compiled in, so the patch in ../parakeet-bias-frontend.patch must be applied.
set -e
K=$(cd "$(dirname "$0")/../../.." && pwd)
G=$K/build/host/third_party/llama.cpp/ggml/src
g++ -O2 -std=c++17 -march=native -fopenmp -DPARAKEET_VERSION='"prof"' \
  -I"$K/third_party/whisper.cpp/include" -I"$K/third_party/whisper.cpp/src" -I"$K/third_party/llama.cpp/ggml/include" \
  "$K/experiments/parakeet/prof/encprof.cpp" -o "$K/build/asr/prof/encprof" \
  "$G/libggml.a" "$G/libggml-cpu.a" "$G/libggml-base.a" -lpthread
