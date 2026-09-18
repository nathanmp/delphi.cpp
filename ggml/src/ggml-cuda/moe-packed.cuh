#pragma once

#include "common.cuh"

// The packed kernels use NVIDIA PTX (cp.async, ldmatrix, mma.sync, prmt). Elsewhere the packed path is never selected.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_MOE_PACKED_AVAILABLE
#endif

// delphi.cpp: packed-weight execution of the grouped tile lists of moe-grouped.cu. The routed expert weights are rearranged once
// into a layout that mma.sync can read directly, and a pipelined cp.async + mma.sync kernel runs instead of the MMQ tile body.
// Selected by GGML_CUDA_MOE_GROUPED=packed, packed-f16 or packed-i8.
//   packed:     Q8_0, Q5_0, IQ4_NL, MXFP4, Q5_K, IQ3_S, IQ4_XS, Q4_K, Q6_K, Q4_1, Q5_1, Q3_K, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S,
//               IQ2_K, IQ3_K, IQ4_K and IQ5_K (other types use grouped MMQ). A lossless rearrangement of the GGUF bits with the same
//               size as the GGUF tensor (moe-packed-q.cuh). The kernel decodes the int8 codes in registers and uses the MMQ arithmetic.
//               GGML_CUDA_MOE_PACKED_TYPES=q8_0,q5_0,iq4_nl,mxfp4,q5_K,iq3_s,iq4_xs,q4_K,q6_K,q4_1,q5_1,q3_k,iq3_xxs,iq2_xxs,
//               iq2_xs,iq2_s,iq2_k,iq3_k,iq4_k,iq5_k,iq2_ks,iq3_ks,iq4_ks,iq5_ks sets the types that it packs (default: all).
//   packed-f16: fp16 weights (the GGUF dequantization rounded to fp16), fp16 activations, fp32 accumulation (Phase 3A0 experiment).
//   packed-i8:  the int8 codes and fp32 per-32 scales that MMQ builds in shared memory, and the q8_1 activations of MMQ (Phase 3A0
//               experiment). packed-f16 and packed-i8 are 2.1-4.7x the GGUF size and are for measurement only.
// The packed weights of the MUL_MAT_ID nodes of a graph are built before the graph runs (ggml_cuda_moe_packed_prepare). They are
// cached per src0 tensor in device memory that is freed by ggml_backend_cuda_moe_packed_clear or with the buffer of src0.
// GGML_CUDA_MOE_PACKED_INPLACE=1 (packed only) overwrites the GGUF data of src0 with the packed layout instead of allocating a copy.
// After that, the grouped packed kernel and MMVQ read src0 in the packed layout (MMVQ with the same results as on the GGUF layout).
// Every other reader of src0 (MMQ, the MUL_MAT_ID fallback, a later ggml_backend_tensor_get) sees the packed bytes: MMQ and the
// fallback abort, and new data written to src0 after the pack is not detected.
// GGML_CUDA_MOE_PACKED_VERIFY=1 unpacks every tensor that is newly packed in a lossless format on the device, and aborts if a byte
// differs from the source.
// GGML_CUDA_MOE_PACKED_ALIAS=1 reads the weights of expert 0 for every expert, =2 also the first row tile for every row tile
// (weights stay in L2; the results are wrong). For measurement only.
// GGML_CUDA_MOE_PACKED_ROWTILE_OUTER=1 orders the units of a class launch by row tile first (activations are read once per row tile).
// GGML_CUDA_MOE_PACKED_I8_MAGIC=0 converts the int32 sums of packed-i8 with I2F instead of an integer and an fp32 add (same results).
// GGML_CUDA_MOE_PACKED_VARIANT=0..3 selects a fixed warp count and stage count for all J (moe-packed.cu), for comparison.

enum ggml_cuda_moe_packed_fmt {
    GGML_CUDA_MOE_PACKED_NONE = 0,
    GGML_CUDA_MOE_PACKED_F16  = 1,
    GGML_CUDA_MOE_PACKED_I8   = 2,
    GGML_CUDA_MOE_PACKED_IQ3S = 3,
    GGML_CUDA_MOE_PACKED_IQ4XS = 4,
    GGML_CUDA_MOE_PACKED_Q4K  = 5,
    GGML_CUDA_MOE_PACKED_Q6K  = 6,
    GGML_CUDA_MOE_PACKED_Q80  = 7,
    GGML_CUDA_MOE_PACKED_Q50  = 8,
    GGML_CUDA_MOE_PACKED_IQ4NL = 9,
    GGML_CUDA_MOE_PACKED_MXFP4 = 10,
    GGML_CUDA_MOE_PACKED_Q5K  = 11,
    GGML_CUDA_MOE_PACKED_Q41   = 12,
    GGML_CUDA_MOE_PACKED_Q51   = 13,
    GGML_CUDA_MOE_PACKED_Q3K   = 14,
    GGML_CUDA_MOE_PACKED_IQ3XXS = 15,
    GGML_CUDA_MOE_PACKED_IQ2XXS = 16,
    GGML_CUDA_MOE_PACKED_IQ2XS  = 17,
    GGML_CUDA_MOE_PACKED_IQ2S   = 18,
    GGML_CUDA_MOE_PACKED_IQ2K   = 19,
    GGML_CUDA_MOE_PACKED_IQ3K   = 20,
    GGML_CUDA_MOE_PACKED_IQ4K   = 21,
    GGML_CUDA_MOE_PACKED_IQ5K   = 22,
    GGML_CUDA_MOE_PACKED_IQ2KS  = 23,
    GGML_CUDA_MOE_PACKED_IQ3KS  = 24,
    GGML_CUDA_MOE_PACKED_IQ4KS  = 25,
    GGML_CUDA_MOE_PACKED_IQ5KS  = 26,
};

// F16 and I8 layout per expert: [row tile of 128][k chunk][8 x 16 rows][4 x k step][32 lanes][16 B], one mma.sync A fragment
// per lane and k step. A k chunk is 64 values (f16) or 128 values (i8); both are 16 KiB per row tile.
// I8 also has fp32 scales per 32 values: [row tile][k chunk][8 x 16 rows][4 x k step][16 rows].
//
// Lossless layout per expert: [row tile of 128][QK_K block] superblocks with the size of 128 GGUF blocks (moe-packed-q.cuh):
//   2 k chunks of 128 values with the codes in m16n8k32 A fragment order for 8 x 16 rows and 32 lanes, then the per-row metadata
//   IQ3S:  14080 B, chunk 6656 B (grid indices, signs, grid index bit 8), metadata 768 B
//   IQ4XS: 17408 B, chunk 8192 B (the qs words, nibbles in value order), metadata 1024 B (d, 6-bit scales as 48 bits)
//   Q4K:   18432 B, chunk 8192 B (the qs words), metadata 2048 B (d and dmin, 6-bit scales and mins as 96 bits)
//   Q6K:   26880 B, chunk 12288 B (the ql words, the qh bits), metadata 2304 B (d, the 16 int8 scales)
//   Q8_0:  34816 B, chunk 16384 B (the qs words), metadata 2048 B (d)
//   Q5_0:  22528 B, chunk 10240 B (the qs words and qh bits), metadata 2048 B (d)
//   IQ4NL: 18432 B, chunk 8192 B (the qs words), metadata 2048 B (d)
//   MXFP4: 17408 B, chunk 8192 B (the qs words), metadata 1024 B (e8m0 exponents)
//   Q5K:   22528 B, chunk 10240 B (the qs words and qh bits), metadata 2048 B (d and dmin, 6-bit scales and mins as 96 bits)
//   Q4_1:  20480 B, chunk 8192 B (the qs bytes), metadata 4096 B (d and m)
//   Q5_1:  24576 B, chunk 10240 B (the qs bytes and qh bytes), metadata 4096 B (d and m)
//   Q3_K:  14080 B, chunk 6144 B (the qs and hmask bytes), metadata 1792 B (d and scales)
//   IQ3_XXS: 12544 B, chunk 6144 B (qs and scale/sign bytes), metadata 256 B (d)
//   IQ2_XXS: 8448 B, chunk 4096 B (qs bytes), metadata 256 B (d)
//   IQ2_XS:  9472 B, chunk 4096 B (qs bytes), metadata 1280 B (d and scales)
//   IQ2_S:  10496 B, chunk 4096 B (qs and sign bytes), metadata 2304 B (d, qh and scales)
//   IQ2_K:  9728 B, chunk 4096 B (qs bytes), metadata 1536 B (d, extra and scales)
//   IQ3_K: 14080 B, chunk 4096 B (qs bytes), metadata 5888 B (d, extra, scales and qh)
//   IQ4_K: 18432 B, chunk 8192 B (qs bytes), metadata 2048 B (d, extra and scales)
//   IQ5_K: 22528 B, chunk 8192 B (qs bytes), metadata 6144 B (d, extra, scales and qh)
struct moe_packed_weights {
    const char  * w;
    const float * d;
    int64_t       stride_w; // bytes per expert
    int64_t       stride_d; // floats per expert
};

// The packed format that the current GGML_CUDA_MOE_GROUPED mode uses for src0 on the grouped path, GGML_CUDA_MOE_PACKED_NONE if
// src0 stays on grouped MMQ. warn: log when a packed-f16 or packed-i8 experiment cannot take src0.
int ggml_cuda_moe_packed_select(const ggml_tensor * src0, bool warn);

// Forgets the packed weights of every src0 in [base, base + size) of the device, and frees their copies. Called when a buffer is freed.
void ggml_cuda_moe_packed_forget(int device, const void * base, size_t size);

// The packed format of src0 if its data was packed in place (GGML_CUDA_MOE_PACKED_INPLACE), else GGML_CUDA_MOE_PACKED_NONE.
int ggml_cuda_moe_packed_inplace_fmt(const ggml_tensor * src0);

// Aborts if src0 was packed in place: called by the paths that read only the GGUF layout.
void ggml_cuda_moe_packed_check_not_inplace(const ggml_tensor * src0);

// Builds the packed weights of the src0 of every MUL_MAT_ID in cgraph that the current mode packs. Called before a CUDA graph is
// captured: a tensor packed in place after a capture would be read in the GGUF layout when the graph is replayed.
void ggml_cuda_moe_packed_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);

// Returns the packed weights of src0. They are cached per src0->data and built on the first call, which must not be during stream capture.
// The cache does not see new data at an old address, and a captured CUDA graph keeps the packed address: call ggml_backend_cuda_moe_packed_clear
// before src0 is freed, and do not replay such a graph afterwards. GGML_CUDA_MOE_PACKED_NOCACHE=1 builds the weights on every call into
// storage instead (for tests that reuse buffer addresses).
moe_packed_weights ggml_cuda_moe_packed_get(ggml_cuda_pool_alloc<char> & storage, int fmt, const ggml_tensor * src0, cudaStream_t stream);

// fp16 activations in grouped order: scatter (broadcast src1) writes token t to the rows ids[t*n_expert_used + k],
// gather writes row r from source row ids[r]. y has n_rows + GGML_CUDA_MOE_GROUPED_Y_PAD rows of ne10 values.
void ggml_cuda_moe_packed_y_f16(const float * x, const int32_t * ids, half * y, int64_t ne10, int64_t stride_row,
        int64_t n_src_rows, int n_expert_used, bool scatter, cudaStream_t stream);

struct moe_packed_launch_args {
    moe_packed_weights w;
    const char    * y;        // f16: rows of ncols_k halves; else block_q8_1_mmq [ncols_k/128][ncols_y]
    int64_t         ncols_y;
    const int32_t * ids_dst;
    const int32_t * expert_bounds;
    const int2    * tiles;
    const int32_t * n_tiles;
    float         * dst;
    int64_t         nrows;
    int64_t         ncols_k;
    int64_t         stride_col_dst;
};

// One launch for the tile list of one class with tile width J.
void ggml_cuda_moe_packed_launch(int fmt, int J, const moe_packed_launch_args & args, cudaStream_t stream);

// Frees all packed weights. Entry point for tests and benchmarks, returned by get_proc_address.
void ggml_backend_cuda_moe_packed_clear(void);

// -1: use GGML_CUDA_MOE_PACKED_INPLACE, else 0 or 1. Entry point for tests, returned by get_proc_address.
void ggml_backend_cuda_moe_packed_set_inplace(int inplace);
