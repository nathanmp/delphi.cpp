#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
probe_build="$(realpath "${1:?usage: build.sh BUILD_DIR}")"
cd "$root"
g++ -O3 -std=c++17 -Iggml/include -Iinclude tools/cpu-g128/ffn.cpp \
    -L"$probe_build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$probe_build/bin" -o "$probe_build/cpu-g128-ffn"
g++ -O2 -std=c++17 -Iggml/include tests/test-quantize-fns.cpp \
    -L"$probe_build/bin" -lggml-cpu -lggml-base -Wl,-rpath,"$probe_build/bin" -o "$probe_build/test-quantize-fns"
