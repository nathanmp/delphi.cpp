#include "mmvq-q8-tc.cuh"
#include "quantize.cuh"
#include "cp-async.cuh"

#include <cstdlib>

static constexpr int Q8_TC_MAX_COLS = 16;
static constexpr int Q8_TC_KBS      = 64; // Q8_0 blocks per stage: 2048 values of K, 2176 B per row

// A block computes row tiles of 16 rows x (8*ntiles) columns with int8 mma.m16n8k32. K is cut into stages of
// Q8_TC_KBS blocks. A block does all its tiles for the first stage of K, then all its tiles for the next stage, and so on,
// so each warp works on the same part of K for many tiles: it loads the activations of that part into registers once.
// Measured on the CMP: activations read from L1/L2 for every tile cost up to 40% at 8 columns, and activations staged in
// shared memory cost occupancy. The threads copy the weights of nbuf stages at a time into shared memory with 16-byte
// cp.async, 2176 B of each row in one run; a buffer is filled again as soon as its stage is multiplied.
// The warps are reduced at the end of each stage; dst holds the sum of the stages done.
// Fragment layout of mma.m16n8k32 s8 (PTX ISA): g = lane/4, t = lane%4;
// A: a0 = row g, k 4t..4t+3; a1 = row g+8, k 4t..; a2 = row g, k 16+4t..; a3 = row g+8, k 16+4t..
// B: b0 = col g, k 4t..4t+3; b1 = col g, k 16+4t..
// C: c0, c1 = row g, cols 2t, 2t+1; c2, c3 = row g+8, cols 2t, 2t+1
static constexpr int Q8_TC_ROW_STRIDE = Q8_TC_KBS*34 + 16;    // 16 B of padding: conflict-free fragment loads
static constexpr int Q8_TC_STAGE      = 16*Q8_TC_ROW_STRIDE;

// int32 -> float for |x| < 2^22 without the quarter-rate I2F: the products of one 32-value block are below 2^19.
static __device__ __forceinline__ float q8_tc_i2f(const int x) {
    return __int_as_float(x + 0x4B400000) - 12582912.0f;
}

static __device__ __forceinline__ void q8_tc_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template <int n>
static __device__ __forceinline__ void q8_tc_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}

template <int ntiles, int nwarps, int nbuf>
static __global__ void __launch_bounds__(nwarps*WARP_SIZE) mul_mat_vec_q8_0_tc(
        const char * __restrict__ x, const char * __restrict__ y, float * __restrict__ dst,
        const int nstage_k, const int nrows, const int ncols,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    constexpr int kbw = Q8_TC_KBS/nwarps; // blocks of one warp per stage, even
    static_assert(kbw % 2 == 0, "bad nwarps");

    extern __shared__ __align__(16) char q8_tc_smem[];
    float * part = (float *) (q8_tc_smem + nbuf*Q8_TC_STAGE); // [nwarps - 1][ntiles][4][WARP_SIZE]

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;
    const int g = lane >> 2;
    const int t = lane & 3;

    // persistent blocks: block b takes row tiles b, b + gridDim.x, ...
    // item n is tile n % my_tiles of the block and stage n / my_tiles of K
    constexpr int pieces_row = Q8_TC_KBS*34/16;
    const int ntiles_row = (nrows + 15)/16;
    const int my_tiles   = (ntiles_row - 1 - blockIdx.x)/gridDim.x + 1;
    const int nitems     = my_tiles*nstage_k;

    const auto issue = [&](const int n) {
        if (n < nitems) {
            const int tile = blockIdx.x + (n % my_tiles)*gridDim.x;
            const char * xs = x + (int64_t) (n / my_tiles)*(Q8_TC_KBS*34);
            char * buf = q8_tc_smem + (n % nbuf)*Q8_TC_STAGE;
#pragma unroll 2
            for (int p = tid; p < 16*pieces_row; p += nwarps*WARP_SIZE) {
                const int r = p / pieces_row;
                const int q = p % pieces_row;
                const int row = min(tile*16 + r, nrows - 1);
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(buf + r*Q8_TC_ROW_STRIDE + 16*q), xs + row*stride_row_x + 16*q);
            }
        }
        q8_tc_commit(); // empty groups keep the count of pending groups fixed
    };

#pragma unroll
    for (int s = 0; s < nbuf; ++s) {
        issue(s);
    }

    // B fragments and scales of this warp's part of K
    int   b0[ntiles][kbw], b1[ntiles][kbw];
    float s0[ntiles][kbw], s1[ntiles][kbw];

    for (int n = 0; n < nitems; ++n) {
        const int ks = n / my_tiles;
        const int ti = n % my_tiles;

        if (ti == 0) { // a new stage of K: load the activations of this warp
            const int64_t kg0 = (int64_t) ks*Q8_TC_KBS + warp*kbw;
#pragma unroll
            for (int j = 0; j < ntiles; ++j) {
                const int cb = j*8 + g;
                const int c0 = j*8 + 2*t;
                const char * yb  = y + (int64_t) min(cb,     ncols - 1)*stride_col_y + kg0*sizeof(block_q8_1);
                const char * ys0 = y + (int64_t) min(c0,     ncols - 1)*stride_col_y + kg0*sizeof(block_q8_1);
                const char * ys1 = y + (int64_t) min(c0 + 1, ncols - 1)*stride_col_y + kg0*sizeof(block_q8_1);
#pragma unroll
                for (int i = 0; i < kbw; ++i) {
                    const int o = i*sizeof(block_q8_1);
                    b0[j][i] = cb     < ncols ? __ldg((const int *) (yb + o + 4 + 4*t))      : 0;
                    b1[j][i] = cb     < ncols ? __ldg((const int *) (yb + o + 4 + 16 + 4*t)) : 0;
                    s0[j][i] = c0     < ncols ? __half2float(__ldg((const half *) (ys0 + o))) : 0.0f;
                    s1[j][i] = c0 + 1 < ncols ? __half2float(__ldg((const half *) (ys1 + o))) : 0.0f;
                }
            }
        }

        const int tile = blockIdx.x + ti*gridDim.x;
        const int row0 = tile*16 + g;
        const int row1 = row0 + 8;

        // warp 0 starts from the sum of the stages of K done; it loads it now so that the load latency is hidden
        float acc[ntiles][4];
#pragma unroll
        for (int j = 0; j < ntiles; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = i < 2 ? row0 : row1;
                const int col = j*8 + 2*t + i % 2;
                acc[j][i] = warp == 0 && ks > 0 && row < nrows && col < ncols ? dst[(int64_t) col*stride_col_dst + row] : 0.0f;
            }
        }

        q8_tc_wait<nbuf - 1>(); // stage n is in shared memory
        __syncthreads();        // ... for all threads

        const char * buf = q8_tc_smem + (n % nbuf)*Q8_TC_STAGE;
        const char * a0p = buf + g*Q8_TC_ROW_STRIDE       + warp*kbw*34;
        const char * a1p = buf + (g + 8)*Q8_TC_ROW_STRIDE + warp*kbw*34;
#pragma unroll
        for (int i = 0; i < kbw; ++i) {
            // even blocks: qs at 2 mod 4, realign from aligned words; odd blocks: aligned
            const char * r0 = a0p + i*34;
            const char * r1 = a1p + i*34;
            int a[4];
            if (i % 2 == 0) {
                const int * w0 = (const int *) r0;
                const int * w1 = (const int *) r1;
                a[0] = __byte_perm(w0[t],     w0[t + 1], 0x5432);
                a[1] = __byte_perm(w1[t],     w1[t + 1], 0x5432);
                a[2] = __byte_perm(w0[t + 4], w0[t + 5], 0x5432);
                a[3] = __byte_perm(w1[t + 4], w1[t + 5], 0x5432);
            } else {
                const int * q0 = (const int *) (r0 + 2);
                const int * q1 = (const int *) (r1 + 2);
                a[0] = q0[t];
                a[1] = q1[t];
                a[2] = q0[t + 4];
                a[3] = q1[t + 4];
            }
            const float d0 = __half2float(*(const half *) r0);
            const float d1 = __half2float(*(const half *) r1);
#pragma unroll
            for (int j = 0; j < ntiles; ++j) {
                int c[4] = {0, 0, 0, 0};
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0[j][i]), "r"(b1[j][i]));
                acc[j][0] += d0*s0[j][i]*q8_tc_i2f(c[0]);
                acc[j][1] += d0*s1[j][i]*q8_tc_i2f(c[1]);
                acc[j][2] += d1*s0[j][i]*q8_tc_i2f(c[2]);
                acc[j][3] += d1*s1[j][i]*q8_tc_i2f(c[3]);
            }
        }

        if (warp > 0) {
#pragma unroll
            for (int j = 0; j < ntiles; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    part[(((warp - 1)*ntiles + j)*4 + i)*WARP_SIZE + lane] = acc[j][i];
                }
            }
        }
        __syncthreads(); // stage n is no longer read: fill its buffer again
        issue(n + nbuf);

        // reduce the warps into dst
        if (warp == 0) {
#pragma unroll
            for (int w = 0; w < nwarps - 1; ++w) {
#pragma unroll
                for (int j = 0; j < ntiles; ++j) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[j][i] += part[((w*ntiles + j)*4 + i)*WARP_SIZE + lane];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ntiles; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = i < 2 ? row0 : row1;
                    const int col = j*8 + 2*t + i % 2;
                    if (row < nrows && col < ncols) {
                        dst[(int64_t) col*stride_col_dst + row] = acc[j][i];
                    }
                }
            }
        }
        // part[] is written again only after the barrier of the next item
    }
#else
    GGML_UNUSED_VARS(x, y, dst, nstage_k, nrows, ncols, stride_row_x, stride_col_y, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

static bool q8_tc_enabled() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_Q8_TC");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

bool ggml_cuda_mmvq_q8_tc_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (!q8_tc_enabled() || ggml_cuda_spec_invariant()) { // MMVQ keeps width-1 numerics for the invariant mode
        return false;
    }
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ampere_mma_available(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // where it is faster than MMVQ (up to 8 columns) and MMQ (9 to 16 columns) on the CMP 170HX, GLM-5.3 shapes:
    // 4096 rows and more: from 3 columns for 2^25 values and more, else from 4 columns; 2048 rows and more: from 5 columns.
    // Fewer rows give too few row tiles for the SMs (MMVQ cuts K into slices instead). Above 8 columns only for K <= 4096.
    const int64_t ncols = src1->ne[1];
    const int64_t ne00  = src0->ne[0];
    const int64_t ne01  = src0->ne[1];
    if (ne01 < 2048) {
        return false;
    }
    const int min_cols = ne01 < 4096 ? 5 : ne01*ne00 >= (int64_t(1) << 25) ? 3 : 4;
    const int max_cols = ne00 <= 4096 ? Q8_TC_MAX_COLS : 8;
    if (ncols < min_cols || ncols > max_cols) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    // 16-byte copies of whole stages: K is a multiple of a stage, the rows and the tensor start are 16-byte aligned
    return src0->ne[1] <= INT_MAX && src0->ne[0] % (Q8_TC_KBS*QK8_0) == 0 && src0->nb[1] % 16 == 0 && (uintptr_t) src0->data % 16 == 0;
}

template <int ntiles, int nwarps, int nbuf>
static void q8_tc_launch(const char * x, const char * y, float * dst, const int64_t ne00, const int nrows, const int ncols,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    const size_t smem = nbuf*Q8_TC_STAGE + (nwarps - 1)*ntiles*4*WARP_SIZE*sizeof(float);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_vec_q8_0_tc<ntiles, nwarps, nbuf>), smem);
    const int id = ggml_cuda_get_device();
    int occupancy = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, mul_mat_vec_q8_0_tc<ntiles, nwarps, nbuf>, nwarps*WARP_SIZE, smem));
    // persistent grid: equal row tiles spread over all SMs within one tile of each other
    const int ntiles_row = (nrows + 15)/16;
    const dim3 grid(std::min(ntiles_row, ggml_cuda_info().devices[id].nsm*std::max(occupancy, 1)), 1, 1);
    const dim3 block(WARP_SIZE, nwarps, 1);
    mul_mat_vec_q8_0_tc<ntiles, nwarps, nbuf><<<grid, block, smem, stream>>>(
        x, y, dst, ne00/(Q8_TC_KBS*QK8_0), nrows, ncols, stride_row_x, stride_col_y, stride_col_dst);
    CUDA_CHECK(cudaGetLastError());
}

// 8 warps and 2 buffers (70 KB): 2 blocks per SM. Measured on the CMP: 4 or 16 warps, or 3 or 4 buffers, are slower.
template <int ntiles>
static void q8_tc_launch(const char * x, const char * y, float * dst, const int64_t ne00, const int nrows, const int ncols,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    q8_tc_launch<ntiles, 8, 2>(x, y, dst, ne00, nrows, ncols, stride_row_x, stride_col_y, stride_col_dst, stream);
}

void ggml_cuda_mul_mat_vec_q8_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    // same q8_1 layout and cache as MMVQ, so matrices that share an input share its quantization
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_1_bytes  = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    char * src1_q8_1_ptr = nullptr;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    if (!ctx.q8_1_cache_get(src1, q8_1_bytes, &src1_q8_1_ptr)) {
        if (src1_q8_1_ptr == nullptr) {
            src1_q8_1_ptr = src1_q8_1.alloc(q8_1_bytes);
        }
        const int64_t s11 = nb11 / sizeof(float);
        const int64_t s12 = nb12 / sizeof(float);
        const int64_t s13 = nb13 / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1_ptr, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t stride_col_y   = ne10_padded / QK8_1 * sizeof(block_q8_1);
    const int64_t stride_col_dst = nb1 / sizeof(float);

    const char * x = (const char *) src0->data;
    const char * y = (const char *) src1_q8_1_ptr;
    float * d = (float *) dst->data;

    if (ne11 <= 8) {
        q8_tc_launch<1>(x, y, d, ne00, (int) ne01, (int) ne11, nb01, stride_col_y, stride_col_dst, stream);
    } else {
        q8_tc_launch<2>(x, y, d, ne00, (int) ne01, (int) ne11, nb01, stride_col_y, stride_col_dst, stream);
    }
}
