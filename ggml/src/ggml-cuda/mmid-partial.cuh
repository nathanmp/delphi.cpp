#pragma once

#include "common.cuh"

// delphi: small batches of a partial MUL_MAT_ID (ggml_mul_mat_id_partial, the expert cache parts) on the MMVQ kernels.
// Returns false if the op must take the MMQ path: the batch is too large for MMVQ, or the types or layouts do not fit.
// GGML_CUDA_MMID_PARTIAL_MMVQ=0 always takes the MMQ path.
bool ggml_cuda_mul_mat_id_partial_mmvq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst);
