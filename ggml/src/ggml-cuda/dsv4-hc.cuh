#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// the chain that produces the pre/post weights, SCALE -> SIGMOID -> SCALE (add == nullptr) or MUL -> ADD -> SIGMOID -> SCALE,
// fused into the kernel that reads the weights, or on its own (MUL chain only)
void ggml_cuda_op_dsv4_hc_wchain(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, const ggml_tensor * add,
        ggml_tensor * last);
void ggml_cuda_op_dsv4_hc_pre_fused_w(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * add,
        const ggml_tensor * last, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post_fused_w(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * add,
        const ggml_tensor * last, ggml_tensor * dst);
