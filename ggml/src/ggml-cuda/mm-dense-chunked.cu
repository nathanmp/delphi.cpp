#include "mm-dense-chunked.cuh"
#include "convert.cuh"

#include <cstdlib>
#include <cstring>
#include <type_traits>

#ifdef GGML_USE_HIP
#define DENSE_CHUNKED_COMPUTE_32F_PEDANTIC CUBLAS_COMPUTE_32F
#else
#define DENSE_CHUNKED_COMPUTE_32F_PEDANTIC CUBLAS_COMPUTE_32F_PEDANTIC
#endif

namespace {

struct dense_chunked_config {
    int64_t min_tokens;
    size_t  chunk_bytes;
    int     type; // 0 f16, 1 bf16, 2 f32 (pedantic, a precision reference)
};

const dense_chunked_config & dense_chunked_cfg() {
    static const dense_chunked_config cfg = [] {
        dense_chunked_config c;
        const char * s = getenv("GGML_CUDA_DENSE_CHUNKED_MIN_TOKENS");
        c.min_tokens = s != nullptr ? (int64_t) atoll(s) : (int64_t) -2; // -2: auto
        s = getenv("GGML_CUDA_DENSE_CHUNKED_MIB");
        c.chunk_bytes = (size_t) (s != nullptr ? atoll(s) : 64) << 20;
        s = getenv("GGML_CUDA_DENSE_CHUNKED_TYPE");
        c.type = s == nullptr ? 0 : strcmp(s, "bf16") == 0 ? 1 : strcmp(s, "f32") == 0 ? 2 : 0;
        return c;
    }();
    return cfg;
}

template <typename T>
void dense_chunked_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                        cudaDataType_t cu_type, cublasComputeType_t cu_compute, to_t_cuda_t<T> to_t_w, to_t_cuda_t<T> to_t_x) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    ggml_cuda_pool_alloc<T> x(ctx.pool());
    const T * xp = (const T *) src1->data;
    if constexpr (!std::is_same_v<T, float>) {
        x.alloc(ne00*ne11);
        to_t_x(src1->data, x.get(), 1, ne00*ne11, stream);
        xp = x.get();
    }

    const size_t chunk_bytes = dense_chunked_cfg().chunk_bytes;
    int64_t rows = chunk_bytes == 0 ? ne01 : (int64_t) (chunk_bytes / (ne00*sizeof(T)));
    rows = std::max<int64_t>(128, rows & ~(int64_t) 127);
    rows = std::min(rows, ne01);

    ggml_cuda_pool_alloc<T> w(ctx.pool(), rows*ne00);

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    float * dst_d = (float *) dst->data;

    // one GEMM per row chunk, into rows [r0, r0 + nr) of every dst column
    for (int64_t r0 = 0; r0 < ne01; r0 += rows) {
        const int64_t nr = std::min(rows, ne01 - r0);
        to_t_w((const char *) src0->data + r0*src0->nb[1], w.get(), 1, nr*ne00, stream);
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                nr, ne11, ne00,
                &alpha, w.get(), cu_type, ne00,
                        xp,      cu_type, ne00,
                &beta,  dst_d + r0, CUDA_R_32F, ne01,
                cu_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

} // namespace

bool ggml_cuda_dense_chunked_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    const dense_chunked_config & cfg = dense_chunked_cfg();
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    // auto: on GA100, whose FP16 tensor cores run F32 accumulation at full rate (CMP 170HX: about 3x MMQ's int8
    // throughput on Q8_0); the crossover with MMQ is near 128 columns. Other GPUs only when asked for.
    const int64_t min_tokens = cfg.min_tokens != -2 ? cfg.min_tokens : cc == GGML_CUDA_CC_AMPERE ? 128 : -1;
    if (min_tokens < 0 || src1->ne[1] < min_tokens) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
}

void ggml_cuda_mul_mat_dense_chunked(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    switch (dense_chunked_cfg().type) {
        case 1:
            dense_chunked_impl<nv_bfloat16>(ctx, src0, src1, dst, CUDA_R_16BF, CUBLAS_COMPUTE_32F,
                    ggml_get_to_bf16_cuda(src0->type), ggml_get_to_bf16_cuda(GGML_TYPE_F32));
            break;
        case 2:
            dense_chunked_impl<float>(ctx, src0, src1, dst, CUDA_R_32F, DENSE_CHUNKED_COMPUTE_32F_PEDANTIC,
                    ggml_get_to_fp32_cuda(src0->type), nullptr);
            break;
        default:
            dense_chunked_impl<half>(ctx, src0, src1, dst, CUDA_R_16F, CUBLAS_COMPUTE_32F,
                    ggml_get_to_fp16_cuda(src0->type), ggml_get_to_fp16_cuda(GGML_TYPE_F32));
            break;
    }
}
