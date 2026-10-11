#pragma once

#include "common.cuh"

// delphi: Q8_0 x F32 for matrices with a short K (64, 128 or 256 values per row) and up to 8 columns.
// MMVQ gives each row a warp and cuts K in steps of 8 Q8_0 blocks per warp, so with K = 128 (GLM-5.3 KDA f_b and g_b,
// 8192 x 128) most lanes have no work. This kernel gives each row K/16 lanes. GGML_CUDA_Q8_SMALLK=0 disables it.

bool ggml_cuda_mmvq_q8_smallk_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_q8_smallk(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
