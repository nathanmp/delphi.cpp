#include "mmvq-dedup.cuh"
#include "moe-packed.cuh"
#include "moe-packed-q.cuh"
#include "unary.cuh"
#include "vecdotq-iqk.cuh"

#include <cstdlib>
#include <type_traits>

#ifdef GGML_CUDA_MOE_PACKED_AVAILABLE

#define MMVQ_DEDUP_WARPS     8 // as MMVQ_PACKED_WARPS: one warp per 8th of the upstream threads of a row
#define MMVQ_DEDUP_MAX_TOK   4
#define MMVQ_DEDUP_MAX_PAIRS 32

static __device__ __forceinline__ float mmvq_dedup_glu(const float x, const float gate, const ggml_glu_op op, const float limit) {
    switch (op) {
        case GGML_GLU_OP_SWIGLU:       return x * ggml_cuda_op_silu_single(gate);
        case GGML_GLU_OP_GEGLU:        return x * ggml_cuda_op_gelu_single(gate);
        case GGML_GLU_OP_SWIGLU_OAI:   return ggml_cuda_op_swiglu_oai_single(gate, x);
        case GGML_GLU_OP_SWIGLU_CLAMP: return ggml_cuda_op_swiglu_clamp_single(gate, x, limit);
        default:                       return x * gate;
    }
}

// vec_dot2_row_meta of moe_packed_q<IQ4KS> for up to MMVQ_DEDUP_MAX_TOK activation columns: the codes are decoded once,
// and the sum of every column is formed with the same operations in the same order.
template <int N>
static __device__ __forceinline__ void mmvq_dedup_iq4ks_dot(const char * __restrict__ sb, const float * __restrict__ row_d,
        const int mg, const int g, const int ib, const block_q8_1 * const * y, float (&acc)[N][2]) {
    using pq = moe_packed_q<GGML_CUDA_MOE_PACKED_IQ4KS>;
    const int c  = ib/4;
    const int ks = ib%4;
    const uint8_t * local = (const uint8_t *) (sb + 2*pq::chunk_bytes);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = 16*mg + g + 8*h;
        const uint4 qv = *((const uint4 *) ((const uint32_t *) (sb + c*pq::chunk_bytes) + (mg*4 + ks)*2*WARP_SIZE + h*WARP_SIZE + 4*g));
        const uint32_t q[4] = { qv.x, qv.y, qv.z, qv.w };
        const uint8_t scale = local[r*(QK_K/32) + ib];
        const uint32_t sh = (scale & 1)*0x04040404u;
        int2 v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = moe_iq4_decode(q[i], iq4k_values);
            v[i].x += sh;
            v[i].y += sh;
        }
        const float d  = row_d[r];
        const int   sc = int(scale & 254) - 127;
#pragma unroll
        for (int m = 0; m < N; ++m) {
            const block_q8_1 * yb = y[m] + ib;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sumi = ggml_cuda_dp4a(v[i].x, get_int_b4(yb->qs, i),     sumi);
                sumi = ggml_cuda_dp4a(v[i].y, get_int_b4(yb->qs, i + 4), sumi);
            }
            acc[m][h] += d*sc*__low2float(yb->ds)*sumi;
        }
    }
}

// blockDim: (32, MMVQ_DEDUP_WARPS), gridDim: (nrows/16, MMVQ_DEDUP_BLOCKS_Y). The distinct experts of the (token, slot) pairs are
// numbered in pair order; block y takes experts y, y + MMVQ_DEDUP_BLOCKS_Y, ... and every warp one work item of each.
#define MMVQ_DEDUP_BLOCKS_Y 8
#define MMVQ_DEDUP_PART     8 // (token, slot) results held in shared memory before a reduction
template <bool has_gate>
__launch_bounds__(MMVQ_DEDUP_WARPS*WARP_SIZE, has_gate ? 1 : 3) // without the gate: 80 registers, 3 blocks per SM
static __global__ void mmvq_dedup_iq4ks(
        const char * __restrict__ vx, const char * __restrict__ vgate, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst, const ggml_glu_op glu_op, const float glu_limit,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t nchannels_y, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const int64_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t nchannels_dst, const uint32_t ids_stride) {
    using pq = moe_packed_q<GGML_CUDA_MOE_PACKED_IQ4KS>;
    constexpr int T   = 8; // upstream MMVQ threads per QK_K block of IQ4_KS (mmvq_packed_threads)
    constexpr int nvq = WARP_SIZE/T;
    static_assert(QK_K/ggml_cuda_type_traits<GGML_TYPE_IQ4_KS>::qk*ggml_cuda_type_traits<GGML_TYPE_IQ4_KS>::qi/VDR_IQ4_KS_Q8_1_MMVQ == T,
        "IQ4_KS upstream threads per QK_K block");
    static_assert(MMVQ_DEDUP_WARPS == WARP_SIZE/4, "one work item per warp and expert");

    // every warp groups the pairs itself: lane i holds the expert of pair i, its members are the lanes with the same expert
    const int npairs = ncols_dst*nchannels_dst;
    const int lane_e = threadIdx.x < (unsigned) npairs ?
        ids[threadIdx.x % nchannels_dst + (threadIdx.x / nchannels_dst)*ids_stride] : -1 - (int) threadIdx.x;
    const unsigned lane_members = __match_any_sync(0xffffffff, lane_e);
    const unsigned lm = __ballot_sync(0xffffffff, threadIdx.x < (unsigned) npairs && (lane_members & ((1u << threadIdx.x) - 1)) == 0);
    const int ng = __popc(lm);

    const int mg = blockIdx.x % (MOE_PACKED_I/16);
    const int row_tile = blockIdx.x / (MOE_PACKED_I/16);
    const int ks = threadIdx.x / 8;
    const int g  = threadIdx.x % 8;
    const int blocks_per_row_x = ncols_x / QK_K;
    const int64_t off_tile = (int64_t) row_tile*blocks_per_row_x*pq::sb_bytes;
    const int64_t off_meta = (stride_channel_x - (int64_t) nrows_x*sizeof(float)) + (int64_t) row_tile*MOE_PACKED_I*sizeof(float);

    // one work item of mul_mat_vec_q_packed_moe per warp: upstream thread v = T*vq + p, QK_K blocks vq, vq + nvq, ...
    const int a  = threadIdx.y;
    const int p  = 4*(a % (T/4)) + ks;
    const int v  = T*(a/(T/4)) + p;
    const int vq = a/(T/4);

    __shared__ float part[MMVQ_DEDUP_PART][16][WARP_SIZE + 4];
    [[maybe_unused]] __shared__ float part_gate[has_gate ? MMVQ_DEDUP_PART : 1][16][WARP_SIZE + 4];
    __shared__ int s_dst[MMVQ_DEDUP_PART];

    auto flush = [&](const int n_entries) {
        __syncthreads();
        for (int it = threadIdx.y; it < n_entries*16; it += MMVQ_DEDUP_WARPS) {
            const int m = it / 16;
            const int i = it % 16;
            float result = warp_reduce_sum<WARP_SIZE>(part[m][i][threadIdx.x]);
            if constexpr (has_gate) {
                const float gate = warp_reduce_sum<WARP_SIZE>(part_gate[m][i][threadIdx.x]);
                result = mmvq_dedup_glu(result, gate, glu_op, glu_limit);
            }
            if (threadIdx.x == 0) {
                dst[s_dst[m] + 16*blockIdx.x + i] = result;
            }
        }
        __syncthreads();
    };

    int n_entries = 0;
    for (int gi = blockIdx.y; gi < ng; gi += gridDim.y) {
        const int leader = __fns(lm, 0, gi + 1);
        const int expert = __shfl_sync(0xffffffff, lane_e, leader);
        unsigned mem = __shfl_sync(0xffffffff, lane_members, leader);
        const int n = __popc(mem);
        if (n_entries + n > MMVQ_DEDUP_PART) {
            flush(n_entries);
            n_entries = 0;
        }
        auto member_y = [&](const int pi, const int m) {
            const int t = pi / nchannels_dst;
            const int k = pi % nchannels_dst;
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                s_dst[n_entries + m] = k*stride_channel_dst + t*stride_col_dst;
            }
            return (const block_q8_1 *) vy + (k % nchannels_y)*stride_channel_y + t*stride_col_y;
        };
        const char  * x      = vx + expert*stride_channel_x + off_tile;
        const float * row_d  = (const float *) (vx + expert*stride_channel_x + off_meta);
        const char  * xg     = has_gate ? vgate + expert*stride_channel_x + off_tile : nullptr;
        const float * row_dg = has_gate ? (const float *) (vgate + expert*stride_channel_x + off_meta) : nullptr;
        auto run = [&](auto n_const, const block_q8_1 * const * y, const int e0) {
            constexpr int N = decltype(n_const)::value;
            float acc[N][2]  = {};
            float accg[N][2] = {};
            for (int kbx = vq; kbx < blocks_per_row_x; kbx += nvq) {
                const block_q8_1 * yk[N];
#pragma unroll
                for (int m = 0; m < N; ++m) {
                    yk[m] = y[m] + kbx*(QK_K/QK8_1);
                }
                mmvq_dedup_iq4ks_dot<N>(x + (int64_t) kbx*pq::sb_bytes, row_d, mg, g, p, yk, acc);
                if constexpr (has_gate) {
                    mmvq_dedup_iq4ks_dot<N>(xg + (int64_t) kbx*pq::sb_bytes, row_dg, mg, g, p, yk, accg);
                }
            }
#pragma unroll
            for (int m = 0; m < N; ++m) {
                part[e0 + m][g][v]     = acc[m][0];
                part[e0 + m][g + 8][v] = acc[m][1];
                if constexpr (has_gate) {
                    part_gate[e0 + m][g][v]     = accg[m][0];
                    part_gate[e0 + m][g + 8][v] = accg[m][1];
                }
            }
        };
        // members in chunks of up to 2 columns: groups of 3 or 4 tokens are rare, and 4 columns would double the registers
        for (int m = 0; m < n; m += 2) {
            const block_q8_1 * y[2];
            y[0] = member_y(__ffs(mem) - 1, m);
            mem &= mem - 1;
            if (mem) {
                y[1] = member_y(__ffs(mem) - 1, m + 1);
                mem &= mem - 1;
                run(std::integral_constant<int, 2>(), y, n_entries + m);
            } else {
                run(std::integral_constant<int, 1>(), y, n_entries + m);
            }
        }
        n_entries += n;
    }
    if (n_entries > 0) {
        flush(n_entries);
    }
}

static bool mmvq_dedup_enabled() {
    static const bool on = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_DEDUP");
        return s == nullptr || atoi(s) != 0;
    }();
    return on;
}

bool ggml_cuda_mmvq_dedup_launch(int fmt, const char * vx, const void * vy, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device & fusion, float * dst, int ncols_x, int nrows_x, int ncols_dst, int stride_col_y,
        int stride_col_dst, int nchannels_y, int nchannels_dst, int64_t stride_channel_x, int stride_channel_y,
        int stride_channel_dst, int ids_stride, cudaStream_t stream) {
    if (fmt != GGML_CUDA_MOE_PACKED_IQ4KS || !mmvq_dedup_enabled()) {
        return false;
    }
    // only the fused gate/up product: for down (a column per pair, short K) and for 2 tokens it was slower than
    // mul_mat_vec_q_packed_moe on CMP 170HX (GLM shapes, 64 of 288 experts: gate/up n=4 -13 %, n=2 +2 %, down n=4 +5 %)
    if (!fusion.gate || ncols_dst < 3 || ncols_dst > MMVQ_DEDUP_MAX_TOK || ncols_dst*nchannels_dst > MMVQ_DEDUP_MAX_PAIRS) {
        return false;
    }
    if (fusion.x_bias || fusion.gate_bias || fusion.x_scale || fusion.gate_scale || nrows_x % MOE_PACKED_I != 0 || ncols_x % QK_K != 0) {
        return false;
    }
    const dim3 block_nums(nrows_x/16, MMVQ_DEDUP_BLOCKS_Y);
    const dim3 block_dims(WARP_SIZE, MMVQ_DEDUP_WARPS);
    mmvq_dedup_iq4ks<true><<<block_nums, block_dims, 0, stream>>>(vx, (const char *) fusion.gate, vy, ids, dst, fusion.glu_op,
        fusion.glu_limit, ncols_x, nrows_x, nchannels_y, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, ncols_dst, nchannels_dst, ids_stride);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmvq_dedup_launch(int, const char *, const void *, const int32_t *, const ggml_cuda_mm_fusion_args_device &, float *,
        int, int, int, int, int, int, int, int64_t, int, int, int, cudaStream_t) {
    return false;
}

#endif // GGML_CUDA_MOE_PACKED_AVAILABLE
