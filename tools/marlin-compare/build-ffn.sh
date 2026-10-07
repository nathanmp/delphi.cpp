#!/usr/bin/env bash
# Builds the ggml-ffn harness against the backend of one build directory. The harness reads private backend structures (CUDA graphs
# of the backend context): the headers and the libraries must come from the same source and build flags.
#   tools/marlin-compare/build-ffn.sh <build-dir>
# Then run it with DELPHI_COMPARE_BINDIR=<build-dir>/bin LD_LIBRARY_PATH=<build-dir>/bin:/opt/rocm/lib.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${1:?usage: build-ffn.sh <build-dir>}"
cd "$root"
nvcc -O3 -std=c++17 -arch=sm_80 -DGGML_CUDA_USE_GRAPHS -Iggml/include -Iggml/src -Iinclude \
    tools/marlin-compare/ggml-ffn.cu -L"$build/bin" -lllama -lggml -lggml-base -lggml-cuda -lcublas -lcuda \
    -o "$build/ggml-ffn"
