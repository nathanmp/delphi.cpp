#pragma once

#include "common.cuh"

// delphi: wide dense Q8_0 products through cuBLAS without a dequantized copy of the whole weight.
// The weight is dequantized to F16 a few thousand rows at a time into one pool buffer of a fixed size, and each row
// chunk is one GEMM (F32 accumulation and output) into its slice of dst. On GA100 the FP16 tensor-core GEMM is about 3x
// MMQ's int8 throughput on Q8_0, so for prefill widths this wins while the extra VRAM stays at the chunk buffer
// (64 MiB) plus the F16 copy of src1. GLM-5.3 Flash KLD against an exact F32 reference is at the MMQ level.
//   GGML_CUDA_DENSE_CHUNKED_MIN_TOKENS  smallest src1 width that takes this path; -1 disables it; default 128 on GA100
//                                       (cc 8.0) and off elsewhere, setting it enables the path on any NVIDIA GPU >= cc 8.0
//   GGML_CUDA_DENSE_CHUNKED_MIB         chunk buffer size in MiB, default 64 (0 = the whole weight in one chunk)
//   GGML_CUDA_DENSE_CHUNKED_TYPE        f16 (default), bf16, or f32 (slow, a precision reference)

bool ggml_cuda_dense_chunked_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_dense_chunked(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
