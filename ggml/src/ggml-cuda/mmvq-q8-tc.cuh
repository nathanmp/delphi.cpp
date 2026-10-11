#pragma once

#include "common.cuh"

// delphi: Q8_0 x F32 for few columns (speculative verify widths) with int8 tensor-core MMA.
// MMVQ redoes its dp4a chain for every column, so on GPUs with a low ALU/bandwidth ratio (CMP 170HX) it becomes
// ALU-bound above about 3 columns. This kernel reads the weights once and costs one mma per extra 8 columns.
// GGML_CUDA_Q8_TC=0 disables it.

bool ggml_cuda_mmvq_q8_tc_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_q8_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
