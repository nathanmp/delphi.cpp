#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
probe_build="$(realpath "${1:?usage: build.sh BUILD_DIR}")"
cd "$root"
g++ -O2 -std=c++17 -Icommon -Iinclude -Iggml/include tools/rocm-int4/capture.cpp \
    -L"$probe_build/bin" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$probe_build/bin" -o "$probe_build/rocm-int4-capture"
/opt/rocm/bin/hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DGGML_USE_HIP -DGGML_CUDA_USE_GRAPHS \
    -Iggml/include -Iggml/src -Iinclude -x hip tools/rocm-int4/ffn.hip \
    -L"$probe_build/bin" -lllama -lggml -lggml-base -lggml-hip -o "$probe_build/rocm-int4-ffn"
