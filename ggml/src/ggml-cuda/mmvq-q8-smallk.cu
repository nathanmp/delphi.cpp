#include "mmvq-q8-smallk.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <cstdlib>

static constexpr int Q8_SMALLK_NWARPS = 4;

// LPR lanes per row, 16 values per lane (half of a Q8_0 block); the lanes of a row add their sums with shuffles.
template <int LPR, int ncols>
static __global__ void __launch_bounds__(Q8_SMALLK_NWARPS*WARP_SIZE) mul_mat_vec_q8_0_smallk(
        const char * __restrict__ x, const char * __restrict__ y, float * __restrict__ dst, const int nrows,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst) {
    constexpr int rows_per_warp = WARP_SIZE/LPR;

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int sub  = lane % LPR;
    const int row  = (blockIdx.x*Q8_SMALLK_NWARPS + warp)*rows_per_warp + lane/LPR;

    // rows past the end read the last row: all lanes take part in the shuffles
    const block_q8_0 * bx = (const block_q8_0 *) (x + min(row, nrows - 1)*stride_row_x) + sub/2;
    const int i0 = (sub % 2)*4; // first int of the 16 values in the block
    const float dx = __half2float(bx->d);
    int qx[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        qx[k] = get_int_b2(bx->qs, i0 + k); // Q8_0 blocks are 34 bytes: 2-byte aligned
    }

    float sum[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const block_q8_1 * by = (const block_q8_1 *) (y + c*stride_col_y) + sub/2;
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            s = ggml_cuda_dp4a(qx[k], get_int_b4(by->qs, i0 + k), s);
        }
        sum[c] = dx*__low2float(by->ds)*s;
    }

#pragma unroll
    for (int c = 0; c < ncols; ++c) {
#pragma unroll
        for (int m = LPR/2; m > 0; m >>= 1) {
            sum[c] += __shfl_xor_sync(0xFFFFFFFF, sum[c], m, WARP_SIZE);
        }
    }

    if (sub == 0 && row < nrows) {
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            dst[c*stride_col_dst + row] = sum[c];
        }
    }
}

static bool q8_smallk_enabled() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_Q8_SMALLK");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

bool ggml_cuda_mmvq_q8_smallk_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (!q8_smallk_enabled() || ggml_cuda_spec_invariant()) { // MMVQ keeps width-1 numerics for the invariant mode
        return false;
    }
    if (!GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t ne00 = src0->ne[0];
    if (ne00 != 64 && ne00 != 128 && ne00 != 256) {
        return false;
    }
    if (src1->ne[1] < 1 || src1->ne[1] > 8) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    return src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        src0->ne[1] <= INT_MAX;
}

template <int LPR, int ncols>
static void q8_smallk_launch(const char * x, const char * y, float * dst, const int nrows,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    constexpr int rows_per_block = Q8_SMALLK_NWARPS*WARP_SIZE/LPR;
    const dim3 grid((nrows + rows_per_block - 1)/rows_per_block, 1, 1);
    mul_mat_vec_q8_0_smallk<LPR, ncols><<<grid, Q8_SMALLK_NWARPS*WARP_SIZE, 0, stream>>>(
        x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst);
    CUDA_CHECK(cudaGetLastError());
}

template <int LPR>
static void q8_smallk_launch(const char * x, const char * y, float * dst, const int nrows, const int ncols,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    switch (ncols) {
        case 1: q8_smallk_launch<LPR, 1>(x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 2: q8_smallk_launch<LPR, 2>(x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 3: q8_smallk_launch<LPR, 3>(x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 4: q8_smallk_launch<LPR, 4>(x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 5: q8_smallk_launch<LPR, 5>(x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 6: q8_smallk_launch<LPR, 6>(x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 7: q8_smallk_launch<LPR, 7>(x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 8: q8_smallk_launch<LPR, 8>(x, y, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("bad ncols %d", ncols);
    }
}

void ggml_cuda_mul_mat_vec_q8_smallk(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

    switch (ne00) {
        case  64: q8_smallk_launch< 4>(x, y, d, (int) ne01, (int) ne11, nb01, stride_col_y, stride_col_dst, stream); break;
        case 128: q8_smallk_launch< 8>(x, y, d, (int) ne01, (int) ne11, nb01, stride_col_y, stride_col_dst, stream); break;
        case 256: q8_smallk_launch<16>(x, y, d, (int) ne01, (int) ne11, nb01, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("bad K %d", (int) ne00);
    }
}
