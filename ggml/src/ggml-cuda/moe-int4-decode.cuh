#pragma once

#include "common.cuh"

// The kernels use NVIDIA PTX (mma.sync bf16, cp.async). Elsewhere the INT4 path is never selected and the old path runs.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_MOE_INT4_AVAILABLE
#endif

// delphi.cpp: one complete GLM-shaped routed expert FFN (gate/up -> clipped SiLU * up -> down * router weight -> slot sum) with INT4
// weights and BF16 tensor-core MMA, for the decode and speculative-verification widths of 1..GGML_CUDA_MOE_INT4_MAX_TOKENS rows.
// Port of the W4A16 decode kernels of vllm-cmp170hx (ampere_marlin/decode_orig.cu); see moe-int4-decode.cu for provenance.
// Selected by GGML_CUDA_MOE_INT4_DECODE=1 (default 0: the old path). Only for a region that ggml_cuda_match_moe_int4_ffn
// (ggml-cuda.cu) accepts and whose Q4_0 expert tensors satisfy the group-128 scale contract:
//   - Q4_0 gate/up [4096, 2048, E] and down [2048, 4096, E], 8 experts per row;
//   - every four successive Q4_0 block scales along K are equal, and they are exactly representable in BF16.
// Other types, shapes and scales take the old path. The weights are rearranged once into a second device copy (4.125 bpw)
// before graph capture (ggml_cuda_moe_int4_prepare) and freed with the buffer of the Q4_0 tensors. Whatever the INT4 path does
// not support stays on the old path, so the switch only changes graphs that it can run completely.
// The work is five kernels on the compute stream: input bridge + aligned routing, W13, clipped SwiGLU, W2 with router weights,
// slot sum + output bridge. Scratch is per backend context, allocated before capture, and has no state that survives a launch.
// Arithmetic (rounding points follow the reference): F32 -> BF16 input; INT4 -> BF16 weights, one BF16 scale multiply, FP32 MMA
// sums; W13 partials of four K slices added in a fixed order; BF16 gate/up; clamped SiLU in FP32 -> BF16; BF16 down sum times the
// BF16 router weight -> BF16; FP32 sum of the eight slots in slot order -> BF16 -> F32.

#define GGML_CUDA_MOE_INT4_MAX_TOKENS 16
#define GGML_CUDA_MOE_INT4_TOPK       8

struct ggml_cuda_moe_int4_ffn {
    const ggml_tensor * gate    = nullptr; // Q4_0 [hidden, ff, E]
    const ggml_tensor * up      = nullptr; // Q4_0 [hidden, ff, E]
    const ggml_tensor * down    = nullptr; // Q4_0 [ff, hidden, E]
    const ggml_tensor * x       = nullptr; // F32  [hidden, 1, M]
    const ggml_tensor * ids     = nullptr; // I32  [8, M]
    const ggml_tensor * weights = nullptr; // F32  [1, 8, M], contiguous: the router weights by slot
    ggml_tensor *       dst     = nullptr; // F32  [hidden, M], contiguous
    float               limit   = 0.0f;    // SwiGLU clamp of the model; the kernel is built for GLM's 10
};

// GGML_CUDA_MOE_INT4_DECODE, or the override of ggml_backend_cuda_moe_int4_set_mode.
bool ggml_cuda_moe_int4_enabled();

// Checks the tensors of ffn and, on the first call for these weights, checks the scale contract and builds the INT4 copy. Returns
// whether ggml_cuda_moe_int4_run can execute ffn. Must not be called during stream capture. A negative answer is remembered
// per weight tensor set, so a later call costs a lookup.
bool ggml_cuda_moe_int4_prepare(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn);

// True if prepare answered yes for ffn and the scratch of ctx exists (never allocates: safe during capture).
bool ggml_cuda_moe_int4_ready(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn);

// Runs the region. Only after ggml_cuda_moe_int4_ready.
void ggml_cuda_moe_int4_run(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn);

// Forgets the INT4 copies of the weights in [base, base + size) of the device and frees them. Called when a buffer is freed:
// the copies are keyed by the tensor address, which a later buffer can reuse. New data written into a live buffer is not detected.
void ggml_cuda_moe_int4_forget(int device, const void * base, size_t size);

// Frees the scratch of a backend context (called by its destructor).
void ggml_cuda_moe_int4_free_scratch(const ggml_backend_cuda_context * ctx);

// Test entry points, returned by get_proc_address.
// mode: -1 uses GGML_CUDA_MOE_INT4_DECODE, else 0 or 1.
void ggml_backend_cuda_moe_int4_set_mode(int mode);
// Frees every INT4 copy.
void ggml_backend_cuda_moe_int4_clear(void);
// out[0..7]: copies in the cache, of them usable, incompatible, builds, build microseconds, bytes of INT4 copies, region runs recorded
// (captures count once), scratch bytes.
void ggml_backend_cuda_moe_int4_get_stats(int64_t * out);
// The device copy of the weights whose gate tensor has the data address gate_data: which = 0 W13 codes, 1 W13 scales, 2 W2 codes,
// 3 W2 scales. Returns false if there is none.
bool ggml_backend_cuda_moe_int4_get_pack(const void * gate_data, int which, const void ** ptr, size_t * bytes);
