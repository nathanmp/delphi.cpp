#pragma once

#include "common.cuh"

void ggml_cuda_interval_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_interval_mask_supported(const ggml_tensor * dst);
