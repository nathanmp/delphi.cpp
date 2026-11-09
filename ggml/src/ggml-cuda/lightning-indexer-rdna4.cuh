#pragma once

// delphi: RDNA4 WMMA lightning indexer for F16 keys, 32 or 64 heads of 128 values.
// The vector kernel spends most of its time in a warp reduction for each (key, head, query) dot product.
// Here each warp keeps the WMMA fragments of its key rows in registers and reuses them for every query of
// the block. Each query's Q is staged in LDS as F16 (as the NVIDIA WMMA kernel does).
// Heads are the rows of matrix A and keys the rows of matrix B. A lane of the F32 result then holds 8 heads of
// one key, so the ReLU and head weights reduce in registers plus one shuffle.

#include "common.cuh"
#include "mma.cuh"

template <int N_HEAD, int N_WARPS, int KV_TILES, int BATCH_TILE>
static __global__ void __launch_bounds__(N_WARPS*WARP_SIZE) lightning_indexer_kernel_rdna4(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        const int n_batch, const int n_kv,
        const size_t nb1, const size_t nb3,
        const size_t nbq1, const size_t nbq2, const size_t nbq3,
        const size_t nbk2, const size_t nbk3,
        const size_t nbw1, const size_t nbw3,
        const size_t nbm1, const size_t nbm3,
        const int nem3, const bool m_counts) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    using namespace ggml_cuda_mma;
    typedef tile<16,  8, half2>                      tile_AB; // 16 rows of 16 halves
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR> tile_C;  // x[l]: row get_i(l) of A, row get_j(l) of B

    constexpr int N_EMBD   = 128;
    constexpr int K_STEPS  = N_EMBD/16;
    constexpr int Q_STRIDE = N_EMBD/2 + 4; // half2; the padding avoids LDS bank conflicts in the fragment loads
    constexpr int KV_BLOCK = N_WARPS*KV_TILES*16;

    const int lane     = threadIdx.x;
    const int warp     = threadIdx.y;
    const int tid      = warp*WARP_SIZE + lane;
    const int i_stream = blockIdx.z;
    const int b0       = blockIdx.y*BATCH_TILE;
    const int kv_warp  = blockIdx.x*KV_BLOCK + warp*KV_TILES*16;

    __shared__ half2 q_s[N_HEAD*Q_STRIDE];
    __shared__ float w_s[N_HEAD];

    tile_AB k_frag[KV_TILES][K_STEPS];
#pragma unroll
    for (int t = 0; t < KV_TILES; ++t) {
        const int i_kv = kv_warp + 16*t + tile_AB::get_i(0);
        if (i_kv < n_kv) {
            const half2 * k_row = (const half2 *) (K + i_kv*nbk2 + i_stream*nbk3);
#pragma unroll
            for (int ks = 0; ks < K_STEPS; ++ks) {
                ggml_cuda_memcpy_1<sizeof(k_frag[t][ks].x)>(k_frag[t][ks].x, k_row + ks*(16/2) + tile_AB::get_j(0));
            }
        }
    }

    const int n_b = min(BATCH_TILE, n_batch - b0);
    for (int ib = 0; ib < n_b; ++ib) {
        const int i_batch = b0 + ib;

        __syncthreads(); // the previous query's Q and weights are no longer read
        const char * q_base = (const char *) Q + i_batch*nbq2 + i_stream*nbq3;
#pragma unroll
        for (int i = tid; i < N_HEAD*N_EMBD/4; i += N_WARPS*WARP_SIZE) {
            const int h = i / (N_EMBD/4);
            const int e = i % (N_EMBD/4);
            const float4 q = *(const float4 *) (q_base + h*nbq1 + e*sizeof(float4));
            q_s[h*Q_STRIDE + 2*e + 0] = make_half2(q.x, q.y);
            q_s[h*Q_STRIDE + 2*e + 1] = make_half2(q.z, q.w);
        }
        if (tid < N_HEAD) {
            w_s[tid] = ((const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3))[tid];
        }
        __syncthreads();

        float score[KV_TILES] = {0.0f};
#pragma unroll
        for (int ht = 0; ht < N_HEAD/16; ++ht) {
            tile_C qk[KV_TILES];
#pragma unroll
            for (int ks = 0; ks < K_STEPS; ++ks) {
                tile_AB q_frag;
                load_ldmatrix(q_frag, q_s + ht*16*Q_STRIDE + ks*(16/2), Q_STRIDE);
#pragma unroll
                for (int t = 0; t < KV_TILES; ++t) {
                    mma(qk[t], q_frag, k_frag[t][ks]);
                }
            }
#pragma unroll
            for (int t = 0; t < KV_TILES; ++t) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    score[t] += fmaxf(qk[t].x[l], 0.0f)*w_s[ht*16 + tile_C::get_i(l)];
                }
            }
        }

        const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream % nem3)*nbm3);
        float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
#pragma unroll
        for (int t = 0; t < KV_TILES; ++t) {
            // the two half-waves hold the two groups of 8 heads of the same key
            score[t] += __shfl_xor(score[t], 16, WARP_SIZE);
            const int i_kv = kv_warp + 16*t + tile_C::get_j(0);
            if (lane < 16 && i_kv < n_kv) {
                dst_base[i_kv] = score[t] + lid_mask_value(m_base, m_counts, i_kv);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, W, M, dst, n_batch, n_kv, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3, m_counts);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
}
