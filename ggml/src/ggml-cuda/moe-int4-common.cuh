// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// delphi.cpp: device helpers shared by the INT4 routed-FFN kernels (decode: moe-int4-decode.cu, prefill: moe-int4-prefill.cuh). They are
// the helpers of vllm-cmp170hx csrc/libtorch_stable/moe/ampere_marlin/decode_orig.cu (see moe-int4-decode.cu for the provenance); both
// kernel families dequantize and multiply with exactly this code, which is what makes their results bitwise equal.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(r)
      : "r"(a), "r"(b), "r"(c), "n"((0xf0 & 0xcc) | 0xaa));
  return r;
}

// nibbles at bits [0,4) and [16,20) of q -> bf16x2 (nib - 8), exact.
__device__ __forceinline__ __nv_bfloat162 deq2(uint32_t q) {
  uint32_t v = lop3_and_or(q, 0x000f000fu, 0x43004300u);
  const uint32_t sub = 0x43084308u;
  return __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v),
                 *reinterpret_cast<const __nv_bfloat162*>(&sub));
}

__device__ __forceinline__ uint32_t scl(__nv_bfloat162 v, __nv_bfloat162 s) {
  __nv_bfloat162 r = __hmul2(v, s);
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
               "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

} // namespace
