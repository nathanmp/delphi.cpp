// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// delphi.cpp: complete routed GLM FFN with INT4 weights and BF16 tensor cores (moe-int4-decode.cuh).
//
// Provenance: the GEMM and the clipped-SwiGLU kernels below are ported from vllm-cmp170hx (read-only reference checkout),
// csrc/libtorch_stable/moe/ampere_marlin/decode_orig.cu at c1ce6491efe53934119d306d0a0501b475458e9b, the "orig" variant that the
// replay measurements used. ATen, CUDAGuard and the Torch registration are removed and the launches take raw pointers on the ggml
// stream. The kernel text, its arithmetic and its launch parameters (cfg 0: W13 R=64 CR=8 TW=1, W2 R=128 CR=8 TW=1) are unchanged,
// except for the silu in the SwiGLU kernel (precise expf and division: ggml builds with -use_fast_math, the reference does not),
// and for the weight addressing of the GEMM (below).
// Weight addressing (deployment): the reference reads one concatenated W13 array and separate scale arrays. Here every weight tensor
// (gate, up, down) is packed in place, one matrix per tensor, and each expert slot holds its own codes followed by its scales:
//   slot e = tensor base + e * estride:  [codes int32 [K/16][2 NM] as below][scales bf16 [K/128][NM]]     (NM = columns of the tensor)
// The W13 GEMM takes the gate and up bases and picks the matrix of a 64-column tile (tiles 0..31 gate, 32..63 up). Only the address
// computation of a warp-item differs from the reference; the order and the arithmetic of every sum are the same.
// New in this file: the Q4_G128 -> packed layout packer, the aligned-routing kernel (replaces the Triton moe_align_block_size of the
// reference), the fused input bridge, the slot sum with output bridge, the conversion to Q4_0 for the ordinary path, and the
// weight/scratch ownership. The kernel layout comment of the reference follows (with E, K, N as in the reference; see above for
// where the arrays live).
/*
 * Weight-streaming W4A16 MoE decode kernels, original reduction order.
 *
 * Registered as decode_gemm_orig / decode_act_orig and selected at runtime by
 * VLLM_GLM5_MARLIN_DECODE_VARIANT=orig (the default).  The w13 projection is
 * split four ways along K, so its fp32 sums are formed in a different order
 * from the released Marlin kernels; decode.cu holds the exact-order variant.
 *
 * Reads vLLM's Marlin-packed uint4b8 / group-128 / bf16-scale layout directly:
 *   packed int32 [E][K/16][2N]: k16 row r, 64-col tile t, lane l, int32 j at
 *   ((t*32 + l)*4 + j); with c = l%4, g = l/4 the nibbles hold
 *     nib0 (k 2c,   n g)   nib4 (2c+1, g)   nib1 (2c+8, g)   nib5 (2c+9, g)
 *     nib2 (2c,   g+8)     nib6 (2c+1, g+8) nib3 (2c+8, g+8) nib7 (2c+9, g+8)
 *   (n relative to 64t + 16j) - exactly the mma.m16n8k16 A fragment of a
 *   16(n) x 16(k) block, so each lane's 16-byte load is its own fragment.
 *   scales bf16 [E][K/128][N]: inside each 64-col chunk, col c sits at
 *   8*(c%8) + c/8, so lane (g) needs the 8 contiguous scales 8g..8g+7.
 *
 * One GEMM kernel serves both projections.  Tokens (<= 8 per aligned block) are the
 * mma N dimension.  Every warp is independent: it takes warp-items (aligned block b,
 * TW adjacent 64-col tiles, k-slice ks of R k16 rows) from an integer work queue and
 * streams them as one continuous run of CR-row chunks.  Weights go through a per-lane
 * two-slot cp.async ring in shared memory: each lane copies exactly the 16-byte
 * fragments it later consumes, so no barrier or warp sync is needed and the next chunk
 * stays in flight during compute.  Scales and activation fragments (small,
 * L2-resident) are loaded with the same chunk into ping-pong registers.  Per-item
 * base pointers are computed once per item.
 * Dequant as Marlin does it: (q-8) exactly in bf16 (magic-number LOP3), times the
 * group scale in bf16 (one rounding, __hmul2), fp32 mma accumulation - the same
 * weight values and fp32 sums as the incumbent, so the result stays allclose to it.
 *
 *   w13: fp32 split-K partials part[ks][slot][2N] (deterministic, no atomics)
 *   act: fixed-order split-K sum -> bf16 (R1) -> clamp-silu-mul (R2, R3) -> h
 *   w2 : one k-slice; bf16(acc) (R4) * bf16(router w) -> bf16 (R5) -> c3[slot][K]
 */

#include "moe-int4-decode.cuh"

#ifdef GGML_CUDA_MOE_INT4_AVAILABLE

#include "ggml-impl.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <map>
#include <vector>

extern "C" __device__ float __nv_expf(float); // libdevice: the precise expf, whatever -use_fast_math says

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

constexpr int WARPS = 4;  // per CTA (warps are independent; the CTA is only a container)

template <int CR, int TW>
struct Frags {            // one chunk's small operands, prefetched in registers
  uint32_t x[CR][2];      // activation B fragments (token g; k 2c..2c+1 and 2c+8..2c+9)
  int4 s[TW];             // per tile: 8 bf16 scales of lane group g (cols g, g+8 of each j)
};

// R: k16 rows per warp-item; CR: k16 rows per chunk (2-slot per-lane cp.async ring);
// TW: adjacent 64-col tiles per warp (they share the activation fragments).
// tgroups_log2 / ksplit_log2: log2 of N/64/TW and K/16/R (both powers of two).
// Items are handed out by an integer work counter (`ctr`, zero on entry): fast warps take
// more items, so per-SM speed differences and item-count quantisation leave no tail.  The
// result of an item does not depend on which warp computes it (deterministic).  Each GEMM
// zeroes the other GEMM's counter (`ctr_other`), which is idle while this one runs.
template <int R, int CR, int TW, int K, int N, bool W13>
__global__ void __launch_bounds__(WARPS * 32)
    moe_dec_gemm(const __nv_bfloat16* __restrict__ A, int64_t a_stride,
                 const int4* __restrict__ W0, const int4* __restrict__ W1, const int4* __restrict__ S0,
                 const int4* __restrict__ S1, int64_t estride,
                 const int32_t* __restrict__ sorted, const int32_t* __restrict__ eids,
                 const int32_t* __restrict__ ntpp, const float* __restrict__ topk_w,
                 int topk_log2, int n_slots, int tgroups_log2, int ksplit_log2,
                 int* __restrict__ ctr, int* __restrict__ ctr_other,
                 float* __restrict__ part, __nv_bfloat16* __restrict__ out) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
  static_assert(R % CR == 0 && 8 % CR == 0, "chunks inside one scale group");
  constexpr int CH = R / CR;
  constexpr int SLOT = CR * TW * 32;  // int4 per ring slot (all lanes)
  extern __shared__ int4 smem_raw[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, c = lane & 3;
  int4* const ring0 = smem_raw + warp * (2 * SLOT) + lane;
  int4* const ring1 = ring0 + SLOT;
  if (blockIdx.x == 0 && threadIdx.x == 0) *ctr_other = 0;

  constexpr int krows = K >> 4;
  constexpr int NM = W13 ? N / 2 : N;          // columns of one weight tensor (W13: gate and up are separate tensors)
  constexpr int tiles_m = NM / 64;
  static_assert(W13 ? (tiles_m % TW == 0) : true, "a warp-item stays inside one tensor");
  constexpr int wrow = NM >> 1, srow = NM >> 3;  // compile-time: immediate address offsets
  const int items = (ntpp[0] >> 3) << (tgroups_log2 + ksplit_log2);

  auto split = [&](int item, int& b, int& ks, int& t0) {
    t0 = (item & ((1 << tgroups_log2) - 1)) * TW;
    const int rest = item >> tgroups_log2;
    ks = rest & ((1 << ksplit_log2) - 1);
    b = rest >> ksplit_log2;
  };

  // Load cursor, one chunk ahead of compute.  Per-item bases are computed once per grab.
  int l_item = 0, l_ch = 0;
  const int4* l_w = W0;
  const int4* l_s = S0;
  const uint32_t* l_x = reinterpret_cast<const uint32_t*>(A);
  auto grab = [&]() {
    int v = 0;
    if (lane == 0) v = atomicAdd(ctr, 1);
    l_item = __shfl_sync(0xffffffffu, v, 0);
    l_ch = 0;
    if (l_item < items) {
      int b, ks, t0;
      split(l_item, b, ks, t0);
      const int e0 = eids[b];
      const int e = e0 < 0 ? 0 : e0;
      const int r0 = ks * R;
      const int mi = W13 ? t0 / tiles_m : 0;  // the weight tensor of this tile (W13: 0 gate, 1 up)
      const int tl = t0 - mi * tiles_m;
      l_w = (mi ? W1 : W0) + (int64_t)e * estride + (int64_t)r0 * wrow + tl * 32 + lane;
      l_s = (mi ? S1 : S0) + (int64_t)e * estride + (int64_t)(r0 >> 3) * srow + tl * 8 + g;
      // Invalid tokens read row 0: their mma columns are independent and never stored.
      const int sid = sorted[b * 8 + g];
      const int arow = e0 >= 0 && sid < n_slots ? (W13 ? sid >> topk_log2 : sid) : 0;
      l_x = reinterpret_cast<const uint32_t*>(A + (int64_t)arow * a_stride + r0 * 16 + 2 * c);
    }
  };
  // chunk (l_item, l_ch): weights -> ring slot (cp.async), scales + activations -> f
  auto load = [&](int4* dst, Frags<CR, TW>& f) {
    const int4* src = l_w + l_ch * (CR * wrow);
#pragma unroll
    for (int i = 0; i < CR; i++)
#pragma unroll
      for (int u = 0; u < TW; u++)
        cp_async16(dst + (i * TW + u) * 32, src + i * wrow + u * 32, true);
    cp_async_commit();
    const int4* sp = l_s + ((l_ch * CR) >> 3) * srow;
#pragma unroll
    for (int u = 0; u < TW; u++) f.s[u] = __ldg(sp + u * 8);
    const uint32_t* ap = l_x + l_ch * CR * 8;
#pragma unroll
    for (int i = 0; i < CR; i++) {
      f.x[i][0] = __ldg(ap + i * 8);
      f.x[i][1] = __ldg(ap + i * 8 + 4);
    }
  };

  float acc[TW][4][4];
#pragma unroll
  for (int u = 0; u < TW; u++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[u][j][v] = 0.f;

  auto epilogue = [&](int item) {  // this warp's tiles of a finished item -> global
    int b, ks, t0;
    split(item, b, ks, t0);
    const bool ok = eids[b] >= 0;
    const int s0 = sorted[b * 8 + 2 * c];
    const int s1 = sorted[b * 8 + 2 * c + 1];
    const bool v0 = ok && s0 < n_slots, v1 = ok && s1 < n_slots;
    if constexpr (W13) {
      float* p0 = part + ((int64_t)ks * n_slots + s0) * N + t0 * 64 + g;
      float* p1 = part + ((int64_t)ks * n_slots + s1) * N + t0 * 64 + g;
#pragma unroll
      for (int u = 0; u < TW; u++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            if (v0) p0[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h];
            if (v1) p1[64 * u + 16 * j + 8 * h] = acc[u][j][2 * h + 1];
          }
    } else {
      const float w0 = v0 ? __bfloat162float(__float2bfloat16(topk_w[s0])) : 0.f;
      const float w1 = v1 ? __bfloat162float(__float2bfloat16(topk_w[s1])) : 0.f;
      __nv_bfloat16* o0 = out + (int64_t)s0 * N + t0 * 64 + g;
      __nv_bfloat16* o1 = out + (int64_t)s1 * N + t0 * 64 + g;
#pragma unroll
      for (int u = 0; u < TW; u++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            if (v0)
              o0[64 * u + 16 * j + 8 * h] = __float2bfloat16(
                  __bfloat162float(__float2bfloat16(acc[u][j][2 * h])) * w0);
            if (v1)
              o1[64 * u + 16 * j + 8 * h] = __float2bfloat16(
                  __bfloat162float(__float2bfloat16(acc[u][j][2 * h + 1])) * w1);
          }
    }
#pragma unroll
    for (int u = 0; u < TW; u++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int v = 0; v < 4; v++) acc[u][j][v] = 0.f;
  };

  // One loop body (compact code): slot parity picks the ring slot at run time and the
  // prefetched fragments are copied, instead of a two-way unrolled ping-pong.
  Frags<CR, TW> cur, nxt;
  grab();
  int c_item = l_item, c_ch = 0;  // compute cursor
  if (l_item < items) load(ring0, cur);
  int par = 0;
  while (c_item < items) {
    cp_async_wait<0>();  // own lane's copies of the current chunk have landed
    if (++l_ch == CH) grab();
    const int4* cs = par ? ring1 : ring0;
    if (l_item < items) load(par ? ring0 : ring1, nxt);
#pragma unroll
    for (int i = 0; i < CR; i++) {
#pragma unroll
      for (int u = 0; u < TW; u++) {
        const int4 wv = cs[(i * TW + u) * 32];
        const uint32_t* qv = reinterpret_cast<const uint32_t*>(&wv);
        const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&cur.s[u]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t qj = qv[j];
          const __nv_bfloat162 sl = __low2bfloat162(s2[j]);   // col g   (pos 2j)
          const __nv_bfloat162 sh = __high2bfloat162(s2[j]);  // col g+8 (pos 2j+1)
          mma_bf16(acc[u][j], scl(deq2(qj), sl), scl(deq2(qj >> 8), sh),
                   scl(deq2(qj >> 4), sl), scl(deq2(qj >> 12), sh), cur.x[i][0], cur.x[i][1]);
        }
      }
    }
    if (++c_ch == CH) {
      epilogue(c_item);
      c_ch = 0;
      c_item = l_item;  // the load cursor wrapped to this item one chunk ago
    }
    cur = nxt;
    par ^= 1;
  }
  cp_async_wait<0>();
#else
  NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// delphi: no `ids` argument (the routing kernel clamps every id, so every slot is valid) and __nv_expf/__fdiv_rn: they are
// bit-identical to expf and `/` of a build without -use_fast_math, which the reference uses and ggml-cuda does not.
// h[slot][n] = bf16(bf16(silu(min(G, L))) * clamp(U, -L, L)), G/U = bf16(sum_ks part).
// One block per slot, one thread per 4 columns: all 2 * KS float4 partial loads are issued
// before the fixed-order (ks = 0..KS-1) sums, so the kernel pays one L2 round trip.
template <int KS>
__global__ void moe_dec_act(const float* __restrict__ part,
                            int n_slots, int Nh, float limit, __nv_bfloat16* __restrict__ h) {
  const int slot = blockIdx.x;
  const int n = threadIdx.x * 4;
  if (n >= Nh) return;
  const int64_t row = (int64_t)n_slots * 2 * Nh;
  const float* p = part + (int64_t)slot * 2 * Nh + n;
  float4 gv[KS], uv[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ks++) {
    gv[ks] = __ldg(reinterpret_cast<const float4*>(p + ks * row));
    uv[ks] = __ldg(reinterpret_cast<const float4*>(p + ks * row + Nh));
  }
  float g[4] = {0.f, 0.f, 0.f, 0.f}, u[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int ks = 0; ks < KS; ks++) {
    g[0] += gv[ks].x; g[1] += gv[ks].y; g[2] += gv[ks].z; g[3] += gv[ks].w;
    u[0] += uv[ks].x; u[1] += uv[ks].y; u[2] += uv[ks].z; u[3] += uv[ks].w;
  }
  __nv_bfloat16 r[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float gf = fminf(__bfloat162float(__float2bfloat16(g[i])), limit);
    const float uf = fmaxf(fminf(__bfloat162float(__float2bfloat16(u[i])), limit), -limit);
    const float sl = __bfloat162float(__float2bfloat16(__fdiv_rn(gf, 1.0f + __nv_expf(-gf))));
    r[i] = __float2bfloat16(sl * uf);
  }
  *reinterpret_cast<uint2*>(h + (int64_t)slot * Nh + n) = *reinterpret_cast<const uint2*>(r);
}

// ---- delphi.cpp additions --------------------------------------------------------------------------------------------------------

constexpr int HIDDEN    = 4096;                      // W13: K, W2: N
constexpr int FF        = 2048;                      // gate/up N, W2: K
constexpr int TOPK      = GGML_CUDA_MOE_INT4_TOPK;
constexpr int MAX_M     = GGML_CUDA_MOE_INT4_MAX_TOKENS;
constexpr int MAX_SLOTS = MAX_M * TOPK;
constexpr int KSPLIT13  = 4;                         // W13 K slices (HIDDEN/16/64), summed in a fixed order by moe_dec_act

constexpr int ilog2_ce(int v) { return v <= 1 ? 0 : 1 + ilog2_ce(v >> 1); }

// Block 0: aligned routing of the n_slots = rows * 8 assignment slots. The other blocks: F32 -> BF16 input rows.
// Routing contract of the GEMM kernels: blocks of 8 slots of one expert in ascending expert order (padding = n_slots), the expert
// of each block, and 8 * the number of blocks. Slot s = row * 8 + k reads router weight s and owns row s of the intermediate
// buffers. The order inside a block does not change a result: every slot is computed on its own.
// This kernel also zeroes both work counters, so no launch depends on the state that an earlier launch left behind.
__global__ void __launch_bounds__(256) moe_int4_prep(
        const float * __restrict__ x, const int64_t x_stride, const int32_t * __restrict__ ids, const int64_t ids_stride,
        const int n_experts, const int n_rows, __nv_bfloat16 * __restrict__ xb, int32_t * __restrict__ sorted,
        int32_t * __restrict__ eids, int32_t * __restrict__ ntpp, int32_t * __restrict__ ctr) {
    if (blockIdx.x != 0) {
        const int64_t i = ((int64_t) (blockIdx.x - 1) * blockDim.x + threadIdx.x) * 4;
        if (i < (int64_t) n_rows * HIDDEN) {
            const int64_t row = i / HIDDEN, col = i % HIDDEN;
            const float4 v = *reinterpret_cast<const float4 *>(x + row * x_stride + col);
            __nv_bfloat16 r[4] = { __float2bfloat16(v.x), __float2bfloat16(v.y), __float2bfloat16(v.z), __float2bfloat16(v.w) };
            *reinterpret_cast<uint2 *>(xb + i) = *reinterpret_cast<const uint2 *>(r);
        }
        return;
    }

    const int n_slots = n_rows * TOPK;
    __shared__ int s_expert[MAX_SLOTS];
    __shared__ int s_blocks[MAX_SLOTS];
    const int s = threadIdx.x;
    for (int i = s; i < TOPK * n_slots; i += blockDim.x) {
        sorted[i] = n_slots;
    }
    for (int i = s; i < n_slots; i += blockDim.x) {
        eids[i] = -1;
    }
    if (s == 0) {
        ctr[0] = 0;
        ctr[1] = 0;
    }
    int e = -1;
    if (s < n_slots) {
        e = ids[(int64_t) (s / TOPK) * ids_stride + s % TOPK];
        e = min(max(e, 0), n_experts - 1);
        s_expert[s] = e;
    }
    __syncthreads();
    int rank = 0, count = 0;
    if (s < n_slots) {
        for (int j = 0; j < n_slots; ++j) {
            const bool same = s_expert[j] == e;
            count += same;
            rank  += same && j < s;
        }
        s_blocks[s] = rank == 0 ? (count + TOPK - 1) / TOPK : 0; // the first slot of an expert owns its blocks
    }
    __syncthreads();
    if (s < n_slots) {
        int first = 0;
        for (int j = 0; j < n_slots; ++j) {
            first += s_expert[j] < e ? s_blocks[j] : 0;
        }
        const int block = first + rank / TOPK;
        sorted[block * TOPK + rank % TOPK] = s;
        if (rank % TOPK == 0) {
            eids[block] = e;
        }
    }
    if (s == 0) {
        int total = 0;
        for (int j = 0; j < n_slots; ++j) {
            total += s_blocks[j];
        }
        ntpp[0] = total * TOPK;
    }
}

// dst[t][n] = bf16(sum over k = 0..7 in slot order of c3[t*8+k][n]), as F32 (moe_sum_vec_kernel of the reference).
__global__ void __launch_bounds__(128) moe_int4_sum(const __nv_bfloat16 * __restrict__ c3, float * __restrict__ dst) {
    const int64_t n   = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int64_t row = blockIdx.y;
    float acc[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        const uint4 v = *reinterpret_cast<const uint4 *>(c3 + (row * TOPK + k) * HIDDEN + n);
        const __nv_bfloat16 * p = reinterpret_cast<const __nv_bfloat16 *>(&v);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[i] += __bfloat162float(p[i]);
        }
    }
    float r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = __bfloat162float(__float2bfloat16(acc[i]));
    }
    float4 * o = reinterpret_cast<float4 *>(dst + row * HIDDEN + n);
    o[0] = make_float4(r[0], r[1], r[2], r[3]);
    o[1] = make_float4(r[4], r[5], r[6], r[7]);
}


// ---- weight layout -----------------------------------------------------------------------------------------------------------

// Q4_G128 (row-major, the layout of the GGUF) -> the packed int32 [K/16][2N] layout of the header comment, one matrix of N columns.
// src holds the experts of a chunk (expert stride nb2, row stride nb1); expert e is written to dst + e * dst_stride. Value (k, n) is in
// block k/128 of row n, code group (k%128)/32, byte (k%32)%16, low nibble for k%32 < 16 and high nibble otherwise; the nibble is the
// stored code of the layout (code - 8 is the weight). Row r of 16 k values is block r/8, group (r/2)%4, nibble r%2.
__global__ void moe_int4_pack_codes(const char * __restrict__ src, const int64_t nb1, const int64_t nb2, const int64_t n,
        const int64_t krows, const int64_t n_experts, char * __restrict__ dst, const int64_t dst_stride) {
    const int64_t row_words = 2 * n;
    const int64_t total     = n_experts * krows * row_words;
    for (int64_t w = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; w < total; w += (int64_t) gridDim.x * blockDim.x) {
        const int64_t word = w % row_words, rest = w / row_words;
        const int64_t r = rest % krows, e = rest / krows;
        const int t = word >> 7, l = (word >> 2) & 31, j = word & 3;
        const int g = l >> 2, c = l & 3;
        const int64_t n0 = (int64_t) t * 64 + j * 16 + g;
        const char * p0 = src + e * nb2 + n0 * nb1 + (r >> 3) * sizeof(block_q4_g128) + offsetof(block_q4_g128, qs) + 16 * ((r >> 1) & 3);
        const char * p1 = p0 + 8 * nb1;
        // bytes 2c, 2c+1 and 2c+8, 2c+9 of the 16 code bytes of rows n0 and n0 + 8
        const uint32_t a0 = *reinterpret_cast<const uint16_t *>(p0 + 2 * c), a1 = *reinterpret_cast<const uint16_t *>(p0 + 2 * c + 8);
        const uint32_t b0 = *reinterpret_cast<const uint16_t *>(p1 + 2 * c), b1 = *reinterpret_cast<const uint16_t *>(p1 + 2 * c + 8);
        const int sh = (r & 1) * 4;
        // nibble 0..7: (k 2c, n g) (2c+8, g) (2c, g+8) (2c+8, g+8) (2c+1, g) (2c+9, g) (2c+1, g+8) (2c+9, g+8)
        const uint32_t v = ((a0 >> sh) & 15) | (((a1 >> sh) & 15) << 4) | (((b0 >> sh) & 15) << 8) | (((b1 >> sh) & 15) << 12) |
            (((a0 >> (8 + sh)) & 15) << 16) | (((a1 >> (8 + sh)) & 15) << 20) | (((b0 >> (8 + sh)) & 15) << 24) |
            (((b1 >> (8 + sh)) & 15) << 28);
        reinterpret_cast<int32_t *>(dst + e * dst_stride)[r * row_words + word] = (int32_t) v;
    }
}

// Group scales (BF16 bits) -> bf16 [K/128][N] at dst + e * dst_stride + scales_offset; inside each 64-column chunk, column c is stored
// at 8 * (c % 8) + c / 8.
__global__ void moe_int4_pack_scales(const char * __restrict__ src, const int64_t nb1, const int64_t nb2, const int64_t n,
        const int64_t groups, const int64_t n_experts, char * __restrict__ dst, const int64_t dst_stride, const int64_t scales_offset) {
    const int64_t total = n_experts * groups * n;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t col = i % n, rest = i / n;
        const int64_t g = rest % groups, e = rest / groups;
        const uint16_t d = reinterpret_cast<const block_q4_g128 *>(src + e * nb2 + col * nb1)[g].d;
        reinterpret_cast<uint16_t *>(dst + e * dst_stride + scales_offset)[g * n + (col / 64) * 64 + 8 * (col % 8) + (col % 64) / 8] = d;
    }
}

__device__ __forceinline__ void moe_int4_write_q4_0(char * p, const uint16_t bf16_bits, const uint8_t * q) {
    // the scale is a BF16 value of at least 2^-14: exact in FP16
    *reinterpret_cast<__half *>(p) = __float2half(__uint_as_float((uint32_t) bf16_bits << 16));
#pragma unroll
    for (int j = 0; j < 16; j += 2) {
        *reinterpret_cast<uint16_t *>(p + 2 + j) = (uint16_t) (q[j] | (q[j + 1] << 8));
    }
}

// Packed weights of one matrix [K][N][E] -> Q4_0 blocks [E][N][K/32] (nb1 = K/32 * 18, nb2 = N * nb1): the inverse of the two packers
// above (the scale of a group becomes the scale of four blocks). One CUDA block per (64-column tile, expert) walks the K groups of
// 128: the 4 KB of packed words of the tile and group are loaded coalesced into shared memory, every thread builds one Q4_0 block
// (column, block of the group) there, and the 72 bytes of the four blocks of a column leave as 8-byte pieces. Both K and N must be
// multiples of 128 and 64 (the shapes that weights_supported accepts).
#define MOE_INT4_CVT_THREADS 256

__global__ void __launch_bounds__(MOE_INT4_CVT_THREADS) moe_int4_packed_to_q4_0(const char * __restrict__ packed, const int64_t stride,
        const int64_t k, const int64_t n, char * __restrict__ dst, const int64_t dnb1, const int64_t dnb2) {
    // one word of padding per 16 words: the 16 (g, jj) of a warp read different banks
    __shared__ uint32_t words[8][128 + 8];
    __shared__ uint16_t scales[64];
    __shared__ __align__(16) char out[64][72];
    const int t = blockIdx.x, tid = threadIdx.x;
    const char *     slot  = packed + (int64_t) blockIdx.y * stride;
    const uint32_t * codes = reinterpret_cast<const uint32_t *>(slot);
    const uint16_t * sc    = reinterpret_cast<const uint16_t *>(slot + (size_t) (k * n / 2));
    char *           dsta  = dst + (int64_t) blockIdx.y * dnb2;
    const int col = tid & 63, blk = tid >> 6;
    const int jj = (col >> 4) & 3, hi = (col >> 3) & 1, g8 = col & 7;
    const int pos = 17 * g8 + jj; // the word of (g8, c = 0, jj) of a row, with the padding

    for (int64_t g = 0; g < k / 128; ++g) {
        {
            const int rr = tid >> 5, i = (tid & 31) * 4;
            const uint4 v = *reinterpret_cast<const uint4 *>(codes + (g * 8 + rr) * (2 * n) + t * 128 + i);
            uint32_t * w = &words[rr][i + (i >> 4)];
            w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
        }
        if (tid < 64) {
            scales[tid] = sc[g * n + t * 64 + tid];
        }
        __syncthreads();

        uint32_t w0[4], w1[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            w0[c] = words[2 * blk][pos + 4 * c];
            w1[c] = words[2 * blk + 1][pos + 4 * c];
        }
        uint8_t q[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int c = (j & 7) >> 1, idx = ((j & 1) << 2) | ((j >> 3) & 1) | (hi << 1);
            q[j] = (uint8_t) (((w0[c] >> (4 * idx)) & 15) | (((w1[c] >> (4 * idx)) & 15) << 4));
        }
        moe_int4_write_q4_0(&out[col][blk * 18], scales[8 * g8 + (col >> 3)], q);
        __syncthreads();

        for (int i = tid; i < 64 * 9; i += MOE_INT4_CVT_THREADS) {
            const int c = i / 9, piece = i - c * 9;
            *reinterpret_cast<uint2 *>(dsta + (int64_t) (t * 64 + c) * dnb1 + g * 72 + piece * 8) = *reinterpret_cast<const uint2 *>(&out[c][piece * 8]);
        }
    }
}

// The same conversion from the row-major layout of the file: one thread per (expert, column, group of 128), so that consecutive
// threads read and write consecutive bytes.
__global__ void moe_int4_rowmajor_to_q4_0(const char * __restrict__ src, const int64_t nb1, const int64_t nb2, const int64_t k,
        const int64_t n, const int64_t n_experts, char * __restrict__ dst, const int64_t dnb1, const int64_t dnb2) {
    const int64_t groups = k / 128;
    const int64_t total  = n_experts * n * groups;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t g = i % groups, rest = i / groups;
        const int64_t col = rest % n, e = rest / n;
        const block_q4_g128 * blk = reinterpret_cast<const block_q4_g128 *>(src + e * nb2 + col * nb1) + g;
        const uint16_t d = blk->d;
        for (int b = 0; b < 4; ++b) {
            const char * qs = reinterpret_cast<const char *>(blk->qs) + 16 * b;
            uint8_t q[16];
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                const uint16_t v = *reinterpret_cast<const uint16_t *>(qs + j);
                q[j] = (uint8_t) (v & 0xff);
                q[j + 1] = (uint8_t) (v >> 8);
            }
            moe_int4_write_q4_0(dst + e * dnb2 + col * dnb1 + (g * 4 + b) * sizeof(block_q4_0), d, q);
        }
    }
}

// ---- launch ------------------------------------------------------------------------------------------------------------------

template <int R, int CR, int TW, int KK, int NN, bool W13>
struct moe_int4_gemm_launch {
    static constexpr int T        = WARPS * 32;
    static constexpr int SMEM     = WARPS * 2 * CR * TW * 32 * (int) sizeof(int4);
    static constexpr int ksplit   = KK / 16 / R;
    static constexpr int tgroups  = NN / 64 / TW;
    static_assert(ksplit * R * 16 == KK && (NN / 64) % TW == 0 && (ksplit & (ksplit - 1)) == 0 && (tgroups & (tgroups - 1)) == 0,
        "shape not built");
    static_assert(W13 ? true : ksplit == 1, "W2 has one K slice");

    // resident CTAs per SM, as in the reference; set by prime (before capture)
    static std::atomic<int> occupancy[GGML_CUDA_MAX_DEVICES];

    static void prime(const int device) {
        auto kern = moe_dec_gemm<R, CR, TW, KK, NN, W13>;
        CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, kern)); // loads the module of the kernel now
        int n = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, T, SMEM));
        occupancy[device].store(n > 0 ? n : 1, std::memory_order_relaxed);
    }

    static void run(const int device, cudaStream_t stream, const __nv_bfloat16 * a, const int64_t a_stride, const int4 * w0, const int4 * w1,
            const int4 * s0, const int4 * s1, const int64_t estride, const int32_t * sorted, const int32_t * eids, const int32_t * ntpp,
            const float * topk_w, const int n_slots, int * ctr, float * part, __nv_bfloat16 * out) {
        const int occ = occupancy[device].load(std::memory_order_relaxed);
        GGML_ASSERT(occ > 0 && "the INT4 kernels must be primed before they are launched");
        // one block per slot at most: every slot of the region may sit in its own aligned block
        const int64_t max_items = (int64_t) n_slots * ksplit * tgroups;
        const int64_t grid = std::min<int64_t>((max_items + WARPS - 1) / WARPS, (int64_t) ggml_cuda_info().devices[device].nsm * occ);
        moe_dec_gemm<R, CR, TW, KK, NN, W13><<<(unsigned int) grid, T, SMEM, stream>>>(a, a_stride, w0, w1, s0, s1, estride, sorted, eids, ntpp,
            topk_w, ilog2_ce(TOPK), n_slots, ilog2_ce(tgroups), ilog2_ce(ksplit), ctr + (W13 ? 0 : 1), ctr + (W13 ? 1 : 0),
            W13 ? part : nullptr, W13 ? nullptr : out);
        CUDA_CHECK(cudaGetLastError());
    }
};
template <int R, int CR, int TW, int KK, int NN, bool W13>
std::atomic<int> moe_int4_gemm_launch<R, CR, TW, KK, NN, W13>::occupancy[GGML_CUDA_MAX_DEVICES];

using moe_int4_w13_launch = moe_int4_gemm_launch<64, 8, 1, HIDDEN, 2 * FF, true>;
using moe_int4_w2_launch  = moe_int4_gemm_launch<128, 8, 1, FF, HIDDEN, false>;

} // namespace

// ---- ownership ---------------------------------------------------------------------------------------------------------------

namespace {

struct moe_int4_scratch_buffers {
    int             device = -1;
    char *          base   = nullptr;
    size_t          bytes  = 0;
    int32_t *       sorted = nullptr; // [8 * MAX_SLOTS] slot ids, padded with n_slots
    int32_t *       eids   = nullptr; // [MAX_SLOTS] expert of each block of 8
    int32_t *       ntpp   = nullptr; // [1] 8 * blocks
    int32_t *       ctr    = nullptr; // [2] work counters of W13 and W2
    float *         part   = nullptr; // [KSPLIT13][MAX_SLOTS][HIDDEN] W13 partial sums
    __nv_bfloat16 * xb     = nullptr; // [MAX_M][HIDDEN]
    __nv_bfloat16 * h      = nullptr; // [MAX_SLOTS][FF]
    __nv_bfloat16 * c3     = nullptr; // [MAX_SLOTS][HIDDEN]
};

// A weight set that was packed in place. The tensors keep their data pointers; the registry is the only record of the state.
struct moe_int4_set {
    int          device;
    const void * gate;
    const void * up;
    const void * down;
    int64_t      n_experts;
    size_t       slot;        // bytes of one expert of each tensor (codes, then scales)
};

enum moe_int4_refusal { MOE_INT4_REFUSE_ROWS, MOE_INT4_REFUSE_LAYOUT, MOE_INT4_REFUSE_WEIGHTS, MOE_INT4_REFUSE_STATE, MOE_INT4_REFUSE_COUNT };

std::mutex                    moe_int4_mutex;
std::vector<moe_int4_set>     moe_int4_sets;
std::mutex                    moe_int4_pack_mutex; // one packing at a time; a second caller finds the set
// The scratch is per backend context and stream (kept here, not in the context, to leave the shared header alone).
std::mutex                    moe_int4_scratch_mutex;
std::map<std::pair<const ggml_backend_cuda_context *, int>, moe_int4_scratch_buffers *> moe_int4_scratch_map;
std::atomic<int>              moe_int4_override{-1};
std::atomic<bool>             moe_int4_primed[GGML_CUDA_MAX_DEVICES];
std::atomic<int64_t>          moe_int4_packs{0}, moe_int4_pack_us{0}, moe_int4_runs{0}, moe_int4_scratch_bytes{0}, moe_int4_shadows{0},
                              moe_int4_shadows_wide{0}, moe_int4_prepare_refused{0};
std::atomic<int64_t>          moe_int4_refused[MOE_INT4_REFUSE_COUNT];
std::atomic<bool>             moe_int4_logged[MOE_INT4_REFUSE_COUNT + 2];

constexpr size_t MOE_INT4_CODES = (size_t) HIDDEN * FF / 2; // bytes of the codes of one expert of any of the three tensors

// One INFO line per reason and process: counters carry the totals, the log shows that a reason occurred (never per token).
static void moe_int4_log_once(const int which, const char * what) {
    if (!moe_int4_logged[which].exchange(true)) {
        GGML_LOG_INFO("moe_int4: %s (first occurrence; the counters of ggml_backend_cuda_moe_int4_get_stats have the totals)\n", what);
    }
}

static const moe_int4_set * moe_int4_find_set(int device, const void * data) { // call with moe_int4_mutex held
    for (const moe_int4_set & c : moe_int4_sets) {
        if (c.device == device && (c.gate == data || c.up == data || c.down == data)) {
            return &c;
        }
    }
    return nullptr;
}

static bool moe_int4_is_packed(int device, const void * data) {
    std::lock_guard<std::mutex> lock(moe_int4_mutex);
    return moe_int4_find_set(device, data) != nullptr;
}

static bool moe_int4_find_ffn(int device, const ggml_cuda_moe_int4_ffn & f, moe_int4_set & out) {
    std::lock_guard<std::mutex> lock(moe_int4_mutex);
    for (const moe_int4_set & c : moe_int4_sets) {
        if (c.device == device && c.gate == f.gate->data && c.up == f.up->data && c.down == f.down->data && c.n_experts == f.gate->ne[2]) {
            out = c;
            return true;
        }
    }
    return false;
}

// Checks of the three weight tensors of a region: no device work.
static bool moe_int4_weights_supported(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    if (f.gate == nullptr || f.up == nullptr || f.down == nullptr || !(f.limit > 0.0f)) {
        return false;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(ctx.device);
    const ggml_tensor * w[3] = { f.gate, f.up, f.down };
    for (int i = 0; i < 3; ++i) {
        const ggml_tensor * t = w[i];
        // weights only: a tensor that a graph computes, or a view, has no final data yet
        if (t->type != GGML_TYPE_Q4_G128 || t->op != GGML_OP_NONE || t->view_src != nullptr || t->buffer == nullptr ||
                ggml_backend_buffer_get_type(t->buffer) != buft || ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
                t->ne[3] != 1 || t->ne[2] != f.gate->ne[2] || t->nb[1] != ggml_row_size(GGML_TYPE_Q4_G128, t->ne[0]) ||
                t->nb[2] != t->nb[1] * t->ne[1] || t->nb[2] % 16 != 0 || (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    return f.gate->ne[0] == HIDDEN && f.gate->ne[1] == FF && f.up->ne[0] == HIDDEN && f.up->ne[1] == FF &&
        f.down->ne[0] == FF && f.down->ne[1] == HIDDEN && f.gate->ne[2] >= 1 && f.gate->nb[2] == f.up->nb[2] && f.gate->nb[2] == f.down->nb[2];
}

static bool moe_int4_rows_supported(const ggml_cuda_moe_int4_ffn & f) {
    if (f.x == nullptr || f.ids == nullptr) {
        return false;
    }
    const int64_t m = f.x->ne[2];
    return m >= 1 && m <= MAX_M;
}

static bool moe_int4_layout_supported(const ggml_cuda_moe_int4_ffn & f) {
    if (f.x == nullptr || f.ids == nullptr || f.weights == nullptr || f.dst == nullptr) {
        return false;
    }
    const int64_t m = f.x->ne[2];
    return f.x->type == GGML_TYPE_F32 && f.x->ne[0] == HIDDEN && f.x->ne[1] == 1 && f.x->ne[3] == 1 && f.x->nb[0] == sizeof(float) &&
        f.x->nb[2] % 16 == 0 && (uintptr_t) f.x->data % 16 == 0 &&
        f.ids->type == GGML_TYPE_I32 && f.ids->ne[0] == TOPK && f.ids->ne[1] == m && f.ids->ne[2] == 1 && f.ids->ne[3] == 1 &&
        f.ids->nb[0] == sizeof(int32_t) && f.ids->nb[1] % sizeof(int32_t) == 0 &&
        f.weights->type == GGML_TYPE_F32 && ggml_nelements(f.weights) == m * TOPK && ggml_is_contiguous(f.weights) &&
        f.dst->type == GGML_TYPE_F32 && f.dst->ne[0] == HIDDEN && f.dst->ne[1] == m && f.dst->ne[2] == 1 && f.dst->ne[3] == 1 &&
        ggml_is_contiguous(f.dst) && (uintptr_t) f.dst->data % 16 == 0;
}

static void moe_int4_prime(const int device) {
    if (moe_int4_primed[device].load(std::memory_order_acquire)) {
        return;
    }
    ggml_cuda_set_device(device);
    moe_int4_w13_launch::prime(device);
    moe_int4_w2_launch::prime(device);
    // the other kernels: their modules load now, not at their first launch (which may be inside a capture)
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, moe_int4_prep));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, moe_dec_act<KSPLIT13>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, moe_int4_sum));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, moe_int4_packed_to_q4_0));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, moe_int4_rowmajor_to_q4_0));
    moe_int4_primed[device].store(true, std::memory_order_release);
}

static moe_int4_scratch_buffers * moe_int4_scratch_find(const ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(moe_int4_scratch_mutex);
    auto it = moe_int4_scratch_map.find({ &ctx, ctx.curr_stream_no });
    return it == moe_int4_scratch_map.end() ? nullptr : it->second;
}

// Scratch of the current stream of a backend context, sized for MAX_M rows. Allocated before capture; the contents never survive a launch.
static moe_int4_scratch_buffers * moe_int4_scratch(ggml_backend_cuda_context & ctx) {
    if (moe_int4_scratch_buffers * found = moe_int4_scratch_find(ctx)) {
        return found;
    }
    auto * s = new moe_int4_scratch_buffers();
    s->device = ctx.device;
    const size_t sizes[] = {
        TOPK * MAX_SLOTS * sizeof(int32_t), MAX_SLOTS * sizeof(int32_t), 64, 64,
        (size_t) KSPLIT13 * MAX_SLOTS * HIDDEN * sizeof(float), (size_t) MAX_M * HIDDEN * sizeof(__nv_bfloat16),
        (size_t) MAX_SLOTS * FF * sizeof(__nv_bfloat16), (size_t) MAX_SLOTS * HIDDEN * sizeof(__nv_bfloat16),
    };
    size_t total = 0;
    for (size_t v : sizes) {
        total += (v + 255) / 256 * 256;
    }
    CUDA_CHECK(cudaMalloc(&s->base, total));
    CUDA_CHECK(cudaMemsetAsync(s->base, 0, total, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    s->bytes = total;
    char * p = s->base;
    auto take = [&](size_t v) { char * r = p; p += (v + 255) / 256 * 256; return r; };
    s->sorted = (int32_t *) take(sizes[0]);
    s->eids   = (int32_t *) take(sizes[1]);
    s->ntpp   = (int32_t *) take(sizes[2]);
    s->ctr    = (int32_t *) take(sizes[3]);
    s->part   = (float *) take(sizes[4]);
    s->xb     = (__nv_bfloat16 *) take(sizes[5]);
    s->h      = (__nv_bfloat16 *) take(sizes[6]);
    s->c3     = (__nv_bfloat16 *) take(sizes[7]);
    {
        std::lock_guard<std::mutex> lock(moe_int4_scratch_mutex);
        moe_int4_scratch_map[{ &ctx, ctx.curr_stream_no }] = s;
    }
    moe_int4_scratch_bytes.fetch_add(total);
    return s;
}

static bool moe_int4_capturing(ggml_backend_cuda_context & ctx) {
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));
    return status != cudaStreamCaptureStatusNone;
}

// Packs one weight tensor in place, expert by expert: the packed expert (codes, then scales) has the size of the expert in the file, so
// each expert is rewritten inside its own bytes from a temporary copy of a chunk of experts.
static void moe_int4_pack_tensor(cudaStream_t stream, const ggml_tensor * t, char * tmp, const int64_t chunk) {
    const int64_t k = t->ne[0], n = t->ne[1], e = t->ne[2];
    const size_t  slot = t->nb[2];
    char * base = (char *) t->data;
    auto grid_for = [](int64_t total) { return (unsigned int) std::min<int64_t>((total + 255) / 256, 1 << 20); };
    for (int64_t e0 = 0; e0 < e; e0 += chunk) {
        const int64_t ce = std::min(chunk, e - e0);
        CUDA_CHECK(cudaMemcpyAsync(tmp, base + e0 * slot, ce * slot, cudaMemcpyDeviceToDevice, stream));
        moe_int4_pack_codes<<<grid_for(ce * (k / 16) * 2 * n), 256, 0, stream>>>(tmp, t->nb[1], slot, n, k / 16, ce, base + e0 * slot, slot);
        moe_int4_pack_scales<<<grid_for(ce * (k / 128) * n), 256, 0, stream>>>(tmp, t->nb[1], slot, n, k / 128, ce, base + e0 * slot, slot,
            (int64_t) (k * n / 2));
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

bool ggml_cuda_moe_int4_enabled() {
    const int o = moe_int4_override.load(std::memory_order_relaxed);
    if (o >= 0) {
        return o != 0;
    }
    static const bool env = [] {
        const char * v = getenv("GGML_CUDA_MOE_INT4_DECODE");
        return v != nullptr && atoi(v) != 0;
    }();
    return env;
}

bool ggml_cuda_moe_int4_type_supported(int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc);
}

ggml_type ggml_cuda_moe_int4_dispatch_type(const ggml_tensor * src0) {
    return src0->type == GGML_TYPE_Q4_G128 ? GGML_TYPE_Q4_0 : src0->type;
}

bool ggml_cuda_moe_int4_prepare(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f) {
    if (!moe_int4_weights_supported(ctx, f)) {
        moe_int4_prepare_refused.fetch_add(1);
        moe_int4_log_once(MOE_INT4_REFUSE_COUNT, "weights of a region are not supported (type, shape, buffer or alignment): ordinary path");
        return false;
    }
    moe_int4_set found;
    if (!moe_int4_find_ffn(ctx.device, f, found)) {
        GGML_ASSERT(!moe_int4_capturing(ctx) && "INT4 MoE weights must be prepared before graph capture");
        std::lock_guard<std::mutex> pack_lock(moe_int4_pack_mutex);
        if (!moe_int4_find_ffn(ctx.device, f, found)) {
            cudaStream_t stream = ctx.stream();
            const int64_t t0 = ggml_time_us();
            // other contexts of the device may still read the row-major bytes: nothing may be in flight while they change
            CUDA_CHECK(cudaDeviceSynchronize());
            const int64_t chunk = 16; // experts per temporary copy
            const size_t  slot  = f.gate->nb[2];
            char * tmp = nullptr;
            CUDA_CHECK(cudaMalloc(&tmp, std::min<int64_t>(chunk, f.gate->ne[2]) * slot));
            for (const ggml_tensor * t : { f.gate, f.up, f.down }) {
                moe_int4_pack_tensor(stream, t, tmp, chunk);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(tmp));
            const moe_int4_set c = { ctx.device, f.gate->data, f.up->data, f.down->data, f.gate->ne[2], slot };
            {
                std::lock_guard<std::mutex> lock(moe_int4_mutex);
                moe_int4_sets.push_back(c);
            }
            const int64_t us = ggml_time_us() - t0;
            moe_int4_packs.fetch_add(1);
            moe_int4_pack_us.fetch_add(us);
            GGML_LOG_INFO("%s: CUDA%d packed %s in place, %" PRId64 " experts, 3 x %.1f MiB in %.1f ms\n", __func__, ctx.device, f.gate->name,
                c.n_experts, c.n_experts * slot / 1048576.0, us / 1000.0);
        }
    }
    // prime the per-device launch tables and the scratch of the current stream outside capture
    moe_int4_prime(ctx.device);
    moe_int4_scratch(ctx);
    return true;
}

bool ggml_cuda_moe_int4_ready(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f) {
    auto refuse = [](moe_int4_refusal why, const char * what) {
        moe_int4_refused[why].fetch_add(1);
        moe_int4_log_once(why, what);
        return false;
    };
    if (!moe_int4_weights_supported(ctx, f)) {
        return refuse(MOE_INT4_REFUSE_WEIGHTS, "region refused: weights not supported (type, shape, buffer or alignment): ordinary path");
    }
    if (!moe_int4_rows_supported(f)) {
        return refuse(MOE_INT4_REFUSE_ROWS, "region refused: more than 16 rows (prefill or wide catch-up): ordinary path on Q4_0 conversions");
    }
    if (!moe_int4_layout_supported(f)) {
        return refuse(MOE_INT4_REFUSE_LAYOUT, "region refused: activation layout (type, strides, alignment): ordinary path on Q4_0 conversions");
    }
    moe_int4_set c;
    if (!moe_int4_find_ffn(ctx.device, f, c) || !moe_int4_primed[ctx.device].load(std::memory_order_acquire)) {
        return refuse(MOE_INT4_REFUSE_STATE, "region refused: weights not packed or kernels not primed: ordinary path on Q4_0 conversions");
    }
    if (moe_int4_scratch_find(ctx) == nullptr) {
        if (moe_int4_capturing(ctx)) {
            return refuse(MOE_INT4_REFUSE_STATE, "region refused: no scratch for this stream and the graph is being captured: ordinary path");
        }
        moe_int4_scratch(ctx);
    }
    return true;
}

void ggml_cuda_moe_int4_run(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & f) {
    moe_int4_set c;
    moe_int4_scratch_buffers * s = moe_int4_scratch_find(ctx);
    GGML_ASSERT(moe_int4_find_ffn(ctx.device, f, c) && s != nullptr);
    cudaStream_t stream = ctx.stream();
    const int m       = (int) f.x->ne[2];
    const int n_slots = m * TOPK;
    const int64_t estride = (int64_t) (c.slot / sizeof(int4));
    const int4 * g = (const int4 *) f.gate->data, * u = (const int4 *) f.up->data, * d = (const int4 *) f.down->data;
    auto scales = [](const int4 * w) { return (const int4 *) ((const char *) w + MOE_INT4_CODES); };

    moe_int4_prep<<<1 + m * HIDDEN / (256 * 4), 256, 0, stream>>>((const float *) f.x->data, f.x->nb[2] / sizeof(float),
        (const int32_t *) f.ids->data, f.ids->nb[1] / sizeof(int32_t), (int) c.n_experts, m, s->xb, s->sorted, s->eids, s->ntpp, s->ctr);
    CUDA_CHECK(cudaGetLastError());
    moe_int4_w13_launch::run(ctx.device, stream, s->xb, HIDDEN, g, u, scales(g), scales(u), estride, s->sorted, s->eids, s->ntpp,
        (const float *) f.weights->data, n_slots, s->ctr, s->part, nullptr);
    moe_dec_act<KSPLIT13><<<n_slots, FF / 4, 0, stream>>>(s->part, n_slots, FF, f.limit, s->h);
    CUDA_CHECK(cudaGetLastError());
    moe_int4_w2_launch::run(ctx.device, stream, s->h, FF, d, nullptr, scales(d), nullptr, estride, s->sorted, s->eids, s->ntpp,
        (const float *) f.weights->data, n_slots, s->ctr, nullptr, s->c3);
    moe_int4_sum<<<dim3(HIDDEN / 8 / 128, m), 128, 0, stream>>>(s->c3, (float *) f.dst->data);
    CUDA_CHECK(cudaGetLastError());
    moe_int4_runs.fetch_add(1);
}

void ggml_cuda_moe_int4_shadow(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, ggml_cuda_pool_alloc<char> & storage,
        ggml_tensor & src0_q, ggml_tensor & dst_q) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_G128 && src0->ne[3] == 1 && src0->ne[0] % 128 == 0);
    GGML_ASSERT(src0->nb[1] == ggml_row_size(GGML_TYPE_Q4_G128, src0->ne[0]) && src0->nb[2] == src0->nb[1] * src0->ne[1]);
    const int64_t k = src0->ne[0], n = src0->ne[1], e = src0->ne[2];
    const size_t  dnb1 = (size_t) (k / 32) * sizeof(block_q4_0), dnb2 = dnb1 * n;
    char * q = storage.alloc(dnb2 * e);
    cudaStream_t stream = ctx.stream();
    if (moe_int4_is_packed(ctx.device, src0->data)) {
        GGML_ASSERT(n % 64 == 0 && e <= 65535);
        moe_int4_packed_to_q4_0<<<dim3((unsigned int) (n / 64), (unsigned int) e), MOE_INT4_CVT_THREADS, 0, stream>>>((const char *) src0->data, src0->nb[2], k, n, q, dnb1, dnb2);
    } else {
        const int64_t total = e * n * (k / 128);
        const unsigned int grid = (unsigned int) std::min<int64_t>((total + 255) / 256, 1 << 20);
        moe_int4_rowmajor_to_q4_0<<<grid, 256, 0, stream>>>((const char *) src0->data, src0->nb[1], src0->nb[2], k, n, e, q, dnb1, dnb2);
    }
    CUDA_CHECK(cudaGetLastError());

    src0_q = *src0;
    src0_q.type  = GGML_TYPE_Q4_0;
    src0_q.data  = q;
    src0_q.nb[0] = sizeof(block_q4_0);
    src0_q.nb[1] = dnb1;
    src0_q.nb[2] = dnb2;
    src0_q.nb[3] = dnb2 * e;
    dst_q = *dst;
    dst_q.src[0] = &src0_q;

    moe_int4_shadows.fetch_add(1);
    if (dst->ne[2] > MAX_M) {
        moe_int4_shadows_wide.fetch_add(1);
        moe_int4_log_once(MOE_INT4_REFUSE_COUNT + 1, "MUL_MAT_ID with INT4 weights and more than 16 rows: ordinary path on a Q4_0 conversion of the experts");
    } else {
        moe_int4_log_once(MOE_INT4_REFUSE_COUNT + 1, "MUL_MAT_ID with INT4 weights outside a matched region (fusion off, other layout): ordinary path on a Q4_0 conversion of the experts");
    }
}

void ggml_cuda_moe_int4_forget(int device, const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(moe_int4_mutex);
    for (size_t i = 0; i < moe_int4_sets.size();) {
        const moe_int4_set & c = moe_int4_sets[i];
        const size_t bytes = c.n_experts * c.slot;
        bool hit = false;
        for (const void * t : { c.gate, c.up, c.down }) {
            hit = hit || (c.device == device && (const char *) base < (const char *) t + bytes && (const char *) t < (const char *) base + size);
        }
        if (!hit) {
            ++i;
            continue;
        }
        moe_int4_sets.erase(moe_int4_sets.begin() + i);
    }
}

void ggml_cuda_moe_int4_check_not_packed(int device, const void * data, size_t size, const char * what) {
    std::lock_guard<std::mutex> lock(moe_int4_mutex);
    for (const moe_int4_set & c : moe_int4_sets) {
        const size_t bytes = c.n_experts * c.slot;
        for (const void * t : { c.gate, c.up, c.down }) {
            if (c.device == device && (const char *) data < (const char *) t + bytes && (const char *) t < (const char *) data + size) {
                GGML_ABORT("%s: the tensor was repacked in place for the INT4 MoE kernels (Q4_G128 weights); only those kernels and the "
                           "Q4_0 conversion of ggml_cuda_moe_int4_shadow read this layout", what);
            }
        }
    }
}

void ggml_cuda_moe_int4_free_scratch(const ggml_backend_cuda_context * ctx) {
    std::vector<moe_int4_scratch_buffers *> mine;
    {
        std::lock_guard<std::mutex> lock(moe_int4_scratch_mutex);
        for (auto it = moe_int4_scratch_map.begin(); it != moe_int4_scratch_map.end();) {
            if (it->first.first == ctx) {
                mine.push_back(it->second);
                it = moe_int4_scratch_map.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (moe_int4_scratch_buffers * s : mine) {
        ggml_cuda_set_device(s->device);
        CUDA_CHECK(cudaFree(s->base));
        moe_int4_scratch_bytes.fetch_sub(s->bytes);
        delete s;
    }
}

void ggml_backend_cuda_moe_int4_set_mode(int mode) {
    moe_int4_override.store(mode < 0 ? -1 : mode != 0, std::memory_order_relaxed);
}

void ggml_backend_cuda_moe_int4_get_stats(int64_t * out) {
    std::lock_guard<std::mutex> lock(moe_int4_mutex);
    int64_t bytes = 0;
    for (const moe_int4_set & c : moe_int4_sets) {
        bytes += 3 * c.n_experts * (int64_t) c.slot;
    }
    out[0]  = (int64_t) moe_int4_sets.size();
    out[1]  = moe_int4_packs.load();
    out[2]  = moe_int4_pack_us.load();
    out[3]  = bytes;
    out[4]  = moe_int4_runs.load();
    out[5]  = moe_int4_scratch_bytes.load();
    out[6]  = moe_int4_shadows.load();
    out[7]  = moe_int4_shadows_wide.load();
    out[8]  = moe_int4_refused[MOE_INT4_REFUSE_ROWS].load();
    out[9]  = moe_int4_refused[MOE_INT4_REFUSE_LAYOUT].load();
    out[10] = moe_int4_refused[MOE_INT4_REFUSE_WEIGHTS].load();
    out[11] = moe_int4_refused[MOE_INT4_REFUSE_STATE].load();
    out[12] = moe_int4_prepare_refused.load();
    out[13] = out[14] = out[15] = 0;
}

bool ggml_backend_cuda_moe_int4_get_pack(const void * data, int which, const void ** ptr, size_t * bytes, size_t * stride) {
    std::lock_guard<std::mutex> lock(moe_int4_mutex);
    const moe_int4_set * c = moe_int4_find_set(ggml_cuda_get_device(), data);
    if (c == nullptr) {
        return false;
    }
    *ptr    = which == 0 ? data : (const void *) ((const char *) data + MOE_INT4_CODES);
    *bytes  = which == 0 ? MOE_INT4_CODES : c->slot - MOE_INT4_CODES;
    *stride = c->slot;
    return true;
}

#else // GGML_CUDA_MOE_INT4_AVAILABLE

bool ggml_cuda_moe_int4_enabled() {
    return false;
}

bool ggml_cuda_moe_int4_type_supported(int device) {
    GGML_UNUSED(device);
    return false;
}

ggml_type ggml_cuda_moe_int4_dispatch_type(const ggml_tensor * src0) {
    return src0->type;
}

bool ggml_cuda_moe_int4_prepare(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn) {
    GGML_UNUSED_VARS(ctx, ffn);
    return false;
}

bool ggml_cuda_moe_int4_ready(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn) {
    GGML_UNUSED_VARS(ctx, ffn);
    return false;
}

void ggml_cuda_moe_int4_run(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_int4_ffn & ffn) {
    GGML_UNUSED_VARS(ctx, ffn);
    GGML_ABORT("the INT4 MoE decode path is not available on this backend");
}

void ggml_cuda_moe_int4_shadow(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, ggml_cuda_pool_alloc<char> & storage,
        ggml_tensor & src0_q, ggml_tensor & dst_q) {
    GGML_UNUSED_VARS(ctx, dst, storage, src0_q, dst_q);
    GGML_ABORT("Q4_G128 weights are not supported on this backend");
}

void ggml_cuda_moe_int4_forget(int device, const void * base, size_t size) {
    GGML_UNUSED_VARS(device, base, size);
}

void ggml_cuda_moe_int4_check_not_packed(int device, const void * data, size_t size, const char * what) {
    GGML_UNUSED_VARS(device, data, size, what);
}

void ggml_cuda_moe_int4_free_scratch(const ggml_backend_cuda_context * ctx) {
    GGML_UNUSED(ctx);
}

void ggml_backend_cuda_moe_int4_set_mode(int mode) {
    GGML_UNUSED(mode);
}

void ggml_backend_cuda_moe_int4_get_stats(int64_t * out) {
    for (int i = 0; i < 16; ++i) {
        out[i] = 0;
    }
}

bool ggml_backend_cuda_moe_int4_get_pack(const void * data, int which, const void ** ptr, size_t * bytes, size_t * stride) {
    GGML_UNUSED_VARS(data, which, ptr, bytes, stride);
    return false;
}

#endif // GGML_CUDA_MOE_INT4_AVAILABLE
