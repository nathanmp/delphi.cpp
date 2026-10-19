#include "mmid-partial.cuh"
#include "mmvq.cuh"

#include <cstdlib>

// The MMVQ kernels skip negative ids (mmvq.cu). A partial MUL_MAT_ID skips every id outside [0, n_as): set those to -1.
static __global__ void mmid_partial_ids(const int32_t * src, int32_t * dst, const int64_t n, const int32_t n_as) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n) {
        const int32_t id = src[i];
        dst[i] = id >= 0 && id < n_as ? id : -1;
    }
}

bool ggml_cuda_mul_mat_id_partial_mmvq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_MMID_PARTIAL_MMVQ");
        return e == nullptr || atoi(e) != 0;
    }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!enabled || !ggml_is_quantized(src0->type) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            dst->ne[2] > MMVQ_MAX_BATCH_SIZE || dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc) ||
            !ggml_is_contiguous(ids) || ids->ne[2] != 1 || ids->ne[3] != 1) {
        return false;
    }
    const int64_t n = ggml_nelements(ids);
    ggml_cuda_pool_alloc<int32_t> mapped(ctx.pool(), n);
    const int block = 256;
    mmid_partial_ids<<<(n + block - 1)/block, block, 0, ctx.stream()>>>((const int32_t *) ids->data, mapped.get(), n, (int32_t) src0->ne[2]);
    CUDA_CHECK(cudaGetLastError());
    ggml_tensor ids_mapped = *ids;
    ids_mapped.data = mapped.get();
    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, &ids_mapped, dst);
    return true;
}
