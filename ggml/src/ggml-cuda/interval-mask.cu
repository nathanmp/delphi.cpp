#include "interval-mask.cuh"
#include "convert.cuh"

template <typename T>
static __global__ void interval_mask_kernel(
        const char * __restrict__ cells,
        const char * __restrict__ queries,
              char * __restrict__ dst,
        const int n_kv,
        const size_t c_nb1, const size_t c_nb3,
        const size_t q_nb1, const size_t q_nb3,
        const size_t d_nb1, const size_t d_nb3) {
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_kv) {
        return;
    }
    const int i = blockIdx.y;
    const int s = blockIdx.z;

    const char * q  = queries + s*q_nb3;
    const float  hi = ((const float *) q)[i];
    const float  lo = ((const float *) (q + q_nb1))[i];

    const char * c = cells + s*c_nb3;
    const float  a = ((const float *) c)[j];
    const float  b = ((const float *) (c + c_nb1))[j];

    T * out = (T *) (dst + i*d_nb1 + s*d_nb3);
    out[j] = ggml_cuda_cast<T>(a <= hi && b >= lo ? 0.0f : -INFINITY);
}

bool ggml_cuda_interval_mask_supported(const ggml_tensor * dst) {
    const ggml_tensor * cells   = dst->src[0];
    const ggml_tensor * queries = dst->src[1];

    return (dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32) &&
        cells->type == GGML_TYPE_F32 && queries->type == GGML_TYPE_F32 &&
        cells->nb[0] == sizeof(float) && queries->nb[0] == sizeof(float) &&
        dst->nb[0] == ggml_type_size(dst->type) &&
        dst->ne[1] <= 65535 && dst->ne[3] <= 65535;
}

void ggml_cuda_interval_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_interval_mask_supported(dst));

    const ggml_tensor * cells   = dst->src[0];
    const ggml_tensor * queries = dst->src[1];

    constexpr int threads = 256;
    const dim3 blocks((dst->ne[0] + threads - 1)/threads, dst->ne[1], dst->ne[3]);

    if (dst->type == GGML_TYPE_F16) {
        interval_mask_kernel<half><<<blocks, threads, 0, ctx.stream()>>>(
                (const char *) cells->data, (const char *) queries->data, (char *) dst->data, dst->ne[0],
                cells->nb[1], cells->nb[3], queries->nb[1], queries->nb[3], dst->nb[1], dst->nb[3]);
    } else {
        interval_mask_kernel<float><<<blocks, threads, 0, ctx.stream()>>>(
                (const char *) cells->data, (const char *) queries->data, (char *) dst->data, dst->ne[0],
                cells->nb[1], cells->nb[3], queries->nb[1], queries->nb[3], dst->nb[1], dst->nb[3]);
    }
    CUDA_CHECK(cudaGetLastError());
}
