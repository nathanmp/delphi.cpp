#pragma once

#include "common.cuh"

// NVIDIA uses BF16 PTX MMA. HIP has an independent RDNA4 W4A8 implementation in moe-int4-hip.cu.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_MOE_INT4_AVAILABLE
#endif
#if defined(GGML_USE_HIP)
#define GGML_HIP_MOE_INT4_AVAILABLE
#endif

// HIP: a complete ordinary/clipped SwiGLU FFN with 2..16 experts per token is enabled by default for eligible RDNA4 G128 weights.
// GGML_HIP_MOE_INT4_DECODE=0 disables this fusion. K and N must be
// multiples of 128. Batches of at least 64 rows group routes across up to 1024 tokens and use I8 WMMA; smaller batches use the
// decode path in chunks of at most 16 tokens. GGML_HIP_MOE_INT4_BATCH=0 selects the old chunked path for A/B tests.
// Input and hidden activations use symmetric Q8 groups of 128;
// dots, scales, SwiGLU and ordered weighted reduction have F32 boundaries. Unmatched nodes read G128 directly, without a shadow bank.
// Weights are packed in place per tensor. Readback or writes first restore its canonical bytes; stable device layout flags let
// captured fallback kernels read either layout. Scratch is owned by backend context, stream and matrix shape, and lives until
// context destruction. Mode overrides are test-only; use a fresh backend when changing mode after graph capture.
// GGML_HIP_MOE_INT4_PATH=auto (default) uses wide vector loads for one-row H=2560/FF=640 work and I8 WMMA for its wider batches;
// other shapes use K-split vector dots. Explicit vector/wide/wmma/mixed modes are experimental comparators. Vector and WMMA sum
// K groups in different F32 orders, so the mode/width selection does not promise bitwise batch invariance.
// Weight-bank views keep the CPU fallback. Partial root-bank nodes support masked routes; fused partial FFNs require at most 16 rows.

// delphi.cpp: one complete GLM-shaped routed expert FFN (gate/up -> clipped SiLU * up -> down * router weight -> slot sum) with INT4
// weights and BF16 tensor-core MMA, for any number of rows: the decode kernels for 1..GGML_CUDA_MOE_INT4_MAX_TOKENS rows (and wider batches
// in chunks of that many rows), the prefill kernels (moe-int4-prefill.cuh) from GGML_CUDA_MOE_INT4_PREFILL_MIN_ROWS rows (default 64).
// Every row has the same bits on every path and in any batch.
// Port of the W4A16 decode kernels of vllm-cmp170hx (ampere_marlin/decode_orig.cu); see moe-int4-decode.cu for provenance.
//
// Storage: the routed expert tensors are GGML_TYPE_Q4_G128 (4.125 bpw: symmetric INT4, one BF16 scale per 128 values). A weight
// set (gate, up, down of one layer) is in one of two states, and the tensors keep the type in both:
//   ROWMAJOR  the layout of the GGUF file;
//   PACKED    rewritten in place (each expert inside its own bytes) into the layout of the kernels, once, by
//             ggml_cuda_moe_int4_prepare, before graph capture. The registry of this file knows the packed sets.
// Only code of moe-int4-decode.cu reads PACKED bytes. Every other consumer goes through ggml_cuda_moe_int4_shadow, which converts
// the weights of a MUL_MAT_ID node (from either state) into a temporary Q4_0 tensor that the ordinary kernels read. A buffer that is
// freed or written forgets its packed sets.
//
// Selected by GGML_CUDA_MOE_INT4_DECODE=1 (default 0: the weights stay ROWMAJOR and every node takes the shadow path, which is
// correct and slow: use Q4_0 weights of the same values for a baseline). A region that the matcher of ggml-cuda.cu accepts runs as
// five kernels on the compute stream (decode) or about nine (prefill): input bridge + routing, W13, clipped SwiGLU, W2 with router weights,
// slot sum + output bridge. A region that does not fit (other shapes, unaligned activations) takes the shadow path per node.
// Arithmetic (rounding points follow the reference): F32 -> BF16 input; INT4 -> BF16 weights, one BF16 scale multiply, FP32 MMA
// sums; W13 partials of four K slices added in a fixed order; BF16 gate/up; clamped SiLU in FP32 -> BF16; BF16 down sum times the
// BF16 router weight -> BF16; FP32 sum of the eight slots in slot order -> BF16 -> F32.
//
// Scratch of the decode kernels (routing tables, partial sums, intermediates; about 10 MB) is per backend context and per stream of that
// context, allocated before capture; the prefill kernels take theirs (about 100 MiB per 1000 rows, at most 2048 rows per pass) from the pool of
// the context for the duration of the region. The regions of one context are ordered by data dependence (a layer reads the output of the previous
// one, also across the join of a concurrent stream group), so they never overlap on one scratch; the per-stream split keeps the
// regions of different streams apart even if a future graph optimizer ordered them differently.

#define GGML_CUDA_MOE_INT4_MAX_TOKENS 16
#define GGML_CUDA_MOE_INT4_TOPK       8

struct ggml_cuda_moe_int4_ffn {
    const ggml_tensor * gate    = nullptr; // Q4_G128 [hidden, ff, E]
    const ggml_tensor * up      = nullptr; // Q4_G128 [hidden, ff, E]
    const ggml_tensor * down    = nullptr; // Q4_G128 [ff, hidden, E]
    const ggml_tensor * x       = nullptr; // F32  [hidden, 1, M]
    const ggml_tensor * ids     = nullptr; // I32  [topk, M]
    const ggml_tensor * weights = nullptr; // F32  [1, topk, M], contiguous: the router weights by slot
    ggml_tensor *       dst     = nullptr; // F32  [hidden, M], contiguous
    bool                partial = false; // HIP: negative or out-of-range IDs have zero contribution
    float               limit   = 0.0f;    // 0 for ordinary SwiGLU; NVIDIA requires GLM's clamp
};

// GGML_CUDA_MOE_INT4_DECODE (opt-in), GGML_HIP_MOE_INT4_DECODE (opt-out), or the test mode override.
bool ggml_cuda_moe_int4_enabled();

// Device eligibility for G128 routed weights: NVIDIA (ordinary shadow path) or RDNA4 (direct G128 path).
bool ggml_cuda_moe_int4_type_supported(int device);

// The type that the ordinary dispatch logic must assume for the weights of a MUL_MAT_ID (Q4_0 for Q4_G128 weights, else the type).
ggml_type ggml_cuda_moe_int4_dispatch_type(const ggml_tensor * src0);

// Checks the tensors of ffn and, on the first call for these weights, packs them in place (after the device is idle) and primes the
// per-device launch tables and the scratch. Returns whether ggml_cuda_moe_int4_run can execute ffn. Must not be called during stream
// capture. The state change (ROWMAJOR -> PACKED) happens at most once per weight set.
bool ggml_cuda_moe_int4_prepare(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn);

// True if the weights of ffn are PACKED and the scratch of the current stream of ctx exists (allocated here when no capture is
// active; counts the reason when false: never allocates during capture).
bool ggml_cuda_moe_int4_ready(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn);

// Runs the region. Only after ggml_cuda_moe_int4_ready.
void ggml_cuda_moe_int4_run(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn);

// The ordinary path for one MUL_MAT_ID node whose weights are Q4_G128: converts all experts into a Q4_0 tensor in storage (from the
// state they are in) and fills src0_q (Q4_0 weights) and dst_q (a copy of the node that reads src0_q). The caller runs the ordinary
// MUL_MAT_ID on dst_q and keeps storage alive until then.
void ggml_cuda_moe_int4_shadow(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, ggml_cuda_pool_alloc<char> & storage,
        ggml_tensor & src0_q, ggml_tensor & dst_q);

// Forget metadata when the owning buffer is freed.
void ggml_cuda_moe_int4_forget(int device, const void * base, size_t size);

// Preserve untouched packed bytes before a partial write. HIP retains stable layout metadata for captured fallbacks.
void ggml_cuda_moe_int4_before_write(int device, const void * base, size_t size);

#ifdef GGML_HIP_MOE_INT4_AVAILABLE
// Register stable layout flags before capture; unmatched nodes use the direct G128 implementation.
void ggml_hip_moe_int4_prepare_graph(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph);
void ggml_hip_moe_int4_note_tensor(const ggml_tensor * tensor);
void ggml_hip_moe_int4_mul_mat_id(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
#endif

// NVIDIA refuses packed readback; HIP restores the canonical layout before readback or a generic copy.
void ggml_cuda_moe_int4_check_not_packed(int device, const void * data, size_t size, const char * what);

// Frees the scratch of a backend context (called by its destructor).
void ggml_cuda_moe_int4_free_scratch(const ggml_backend_cuda_context * ctx);

// Test entry points, returned by get_proc_address.
// mode: -1 uses GGML_CUDA_MOE_INT4_DECODE, else 0 or 1.
void ggml_backend_cuda_moe_int4_set_mode(int mode);
// out[0..15]: packed weight sets, sets packed so far, packing microseconds, bytes of packed weights, regions built (a capture counts
// once), scratch bytes, nodes on the shadow path, of them with more than 16 rows, regions refused for: no rows, activation
// layout, weights, missing state (not packed / not primed / no scratch), the weight sets that were refused at
// prepare, regions run on the prefill kernels, their rows, regions of more than 16 rows run as decode chunks.
// HIP counts packed tensors in [0..1], direct fallback nodes in [6], batch FFNs/rows in [13..14], and chunked FFNs in [15].
void ggml_backend_cuda_moe_int4_get_stats(int64_t * out);
// The packed layout of the tensor with the data address data: which = 0 codes (int32 [K/16][2N], Marlin tiles), 1 scales
// (bf16 [K/128][N]). Expert e is at ptr + e * stride. Returns false if the tensor is not PACKED.
// HIP layout: codes uint32 [N/16][K/16][lane32], scales bf16 [N][K/128], in the same expert-sized slot.
bool ggml_backend_cuda_moe_int4_get_pack(const void * data, int which, const void ** ptr, size_t * bytes, size_t * stride);
#ifdef GGML_HIP_MOE_INT4_AVAILABLE
// Copy expert slab `expert` of a G128 bank from device memory on the stream of ctx. A packed bank stays packed.
void ggml_cuda_moe_int4_set_expert(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int64_t expert, const void * src);
// Forget the layout of weights that lie inside t (staging slots lent from t), but not that of t.
void ggml_cuda_moe_int4_forget_inside(int device, const ggml_tensor * t);
#endif
