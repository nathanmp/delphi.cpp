#pragma once

#include "common.cuh"

// delphi: dense Q8_0 weights in an mma fragment layout, packed in place before graph capture (NVIDIA, cc >= 8.0).
// A tile of 16 rows x 32 values keeps its 544 bytes in the bytes of its 16 rows: the 512 quant bytes in lane order
// of mma.m16n8k32 (one 16-byte load per lane), then the scales of 4 consecutive tiles as half2 {row g, row g + 8}
// per lane group. A product of up to 64 columns streams the weights once per 16 columns on int8 tensor cores without
// shared memory or barriers; wider products dequantize row chunks to F16 for cuBLAS. Every other reader of a packed
// weight aborts.
//   GGML_CUDA_Q8_FRAG  1: on for any NVIDIA GPU >= cc 8.0, 0: off; default on for GA100 (cc 8.0)

bool ggml_cuda_q8f_enabled(int cc);

// true if the data of t lies in packed weights of the current device (aborts if it is only partly packed)
bool ggml_cuda_q8f_packed(const ggml_tensor * t);

// aborts if [data, data + size) overlaps packed weights of device
void ggml_cuda_q8f_check_not_packed(int device, const void * data, size_t size, const char * what);

// packs the eligible Q8_0 weights of the MUL_MAT nodes of cgraph; call it before a graph capture
void ggml_cuda_q8f_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);

// dst = src0 x src1 for packed src0
void ggml_cuda_q8f_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// true if ggml_cuda_q8f_mul_mat uses cuBLAS for a product of ncols columns
bool ggml_cuda_q8f_uses_cublas(int64_t ncols);

void ggml_cuda_q8f_forget(int device, const void * base, size_t size);
