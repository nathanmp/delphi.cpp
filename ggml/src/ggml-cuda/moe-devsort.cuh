#pragma once

#include "common.cuh"

#include <vector>

// Device-side expert grouping for the synchronized sorted MUL_MAT_ID fallback in ggml-cuda.cu.
//
// The upstream fallback copies the whole ids tensor to the host, groups the (token, slot)
// assignments by expert in a CPU loop, and uploads the result. Both transfers are followed by a
// stream synchronize, so the GPU is idle for the entire grouping. The loop is
// O(n_experts * n_tokens * n_expert_used), which grows with the ubatch while the per-expert GEMM
// it feeds becomes more efficient, so it takes a larger share of prefill as the ubatch grows.
//
// ggml_cuda_moe_build_routing replaces that with an on-device counting sort (histogram ->
// exclusive scan -> scatter) and returns only tokens_per_expert to the host, which the caller
// still needs to size the per-expert GEMMs. The original host loop is kept as an A/B arm.
//
// The sort is stable, so it reproduces the host loop's ordering exactly. That is a requirement, not
// a convenience: measured on gfx1201 the per-expert GEMM is not invariant to the order of its rows,
// so an unstable sort - the obvious atomicAdd cursor - changes the model's output and makes it
// differ from run to run.
//
// Provenance: adapted from TheTom/llama-cpp-turboquant PR #322 (commits 8c1a0d3b4, 676dc97d5).
//
// Environment controls:
//   GGML_CUDA_MOE_ROUTE=auto|device|host   routing arm; auto uses the device sort above the
//                                          crossover below (default auto)
//   GGML_CUDA_MOE_ROUTE_MIN_WORK=N         crossover for auto, in routed slots
//                                          (n_tokens * n_expert_used); 0 (the default) always
//                                          routes on the device, 4096 skips the shapes measured
//                                          flat-to-slightly-negative on gfx1201
//   GGML_CUDA_MOE_ROUTE_DEBUG=1            log the selected arm per shape and a summary at exit
//   GGML_CUDA_MOE_ROUTE_VERIFY=1           compare the device routing against the host loop's,
//                                          element by element, on every call
void ggml_cuda_moe_build_routing(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor *         dst,
    int32_t *                   ids_buf_dev,
    std::vector<int32_t> &      tokens_per_expert,
    cudaStream_t                stream);
