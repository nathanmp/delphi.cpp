#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// SCALE -> SIGMOID -> SCALE producing the post weights, fused into DSV4_HC_POST
void ggml_cuda_op_dsv4_hc_post_fused_w(ggml_backend_cuda_context & ctx, const ggml_tensor * scale0,
        const ggml_tensor * scale1, ggml_tensor * dst);
